"""Whole-pipeline renders of the same clips by heads that never saw them (TEST) and heads that trained on them (TRAIN).

Clips: the 10 emotion picks of bgm.analysis.emotion_sections (one held-out test clip per emotion, not picked by rank).
Per clip, two renders with the full user-facing pipeline (VLM scene brief, pace match, length fit, silence, replay rule,
cue-switch decoder; Konjum files excluded):
  test   the run of the clip's TEST fold     -> heads, fusion and cue model never saw its film
  train  the run of the lowest fold that has the clip in its TRAIN split (not val) -> heads trained on this very clip
The scene brief is drafted once (test render) and reused by the train render, so the two differ only in the models.

    python -m bgm.analysis.pipeline_train_test      -> outputs/pipeline_train_test/
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bgm.analysis.emotion_sections import burn, overlay, picks

log = logging.getLogger(__name__)
OUT = Path("outputs/pipeline_train_test")
RENDERS = Path("renders/pipeline_train_test")
BIN = Path(sys.executable).parent
PIPE = ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "pipeline.dyn_variant=null",
        "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.pace_match.weight=1.0", "render.scene_brief.weight=1.0",
        "render.length_fit.weight=1.0", "render.silence.enabled=true", "render.replay_penalty=3.0", "render.max_plays=2",
        "render.max_windows_per_file=4", "render.audio_bitrate=320k", "render.exclude_music=[Konjum]",
        f"paths.renders={RENDERS}"]


def train_fold(folds: dict, pid: str, test_fold: int) -> int:
    """Lowest fold whose TRAIN split holds the clip (val clips steer early stopping, so they are skipped)."""
    ks = [int(k) for k, v in sorted(folds.items(), key=lambda kv: int(kv[0])) if pid in set(v["train"])]
    assert ks and test_fold not in ks, (pid, ks, test_fold)
    return ks[0]


def main() -> None:
    from bgm.analysis.poster_pipeline import film_of
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    cfg = load_config("configs/default.yaml", PIPE)
    store = FeatureStore(cfg)
    folds = json.loads(Path("data/manifests/folds_title.json").read_text())["folds"]
    OUT.mkdir(parents=True, exist_ok=True)
    RENDERS.mkdir(parents=True, exist_ok=True)
    film = lambda m: film_of(store, m)  # noqa: E731
    rows = []
    for i, r in picks().iterrows():
        src_dir = next(Path("outputs/emotion_sections").glob(f"{i + 1:02d}_{r.emotion}_*"))
        name = src_dir.name
        d = OUT / name
        (d / "logs").mkdir(parents=True, exist_ok=True)
        video = Path(f"data/interim/video/{r.pair_id}.mp4")
        k_test, k_train = int(r.fold), train_fold(folds, r.pair_id, int(r.fold))
        brief = RENDERS / f"{name}_test.brief.yaml"
        plans = {}
        for split, k in (("test", k_test), ("train", k_train)):
            stem = f"{name}_{split}"
            if split == "train":                                       # same features and the same brief as the test render
                (RENDERS / stem).mkdir(parents=True, exist_ok=True)
                for f in ("query.npz", "silent25.mp4"):
                    if not (RENDERS / stem / f).exists():
                        shutil.copy(RENDERS / f"{name}_test" / f, RENDERS / stem / f)
            plan_p = RENDERS / f"{stem}.plan.json"
            if not plan_p.exists():
                log.info("[%d/10] %s %s: fold %d", i + 1, name, split, k)
                extra = [f"render.scene_brief.file={brief}"] if split == "train" else []
                with open(d / "logs" / f"render_{split}.log", "w") as lf:
                    subprocess.run([str(BIN / "bgm"), "render", "--video", str(video), "--out", str(RENDERS / f"{stem}.mp4"),
                                    "--run", f"runs/titlesplit/pipeline_ens10/fold{k}/seed0", *PIPE, *extra], check=True,
                                   stdout=lf, stderr=subprocess.STDOUT)
            plans[split] = json.loads(plan_p.read_text())
            shutil.copy(plan_p, d / f"plan_{split}.json")
        shutil.copy(brief, d / "scene_brief.yaml")
        dur = float(plans["test"]["duration_s"])
        true_track = store.track[r.pair_id]
        for split, k in (("test", k_test), ("train", k_train)):
            if (d / f"{split}.mp4").exists():
                continue
            title = (f"TEST · fold {k} heads never saw this film" if split == "test"
                     else f"TRAIN · fold {k} heads were trained on this clip")
            ass = d / "logs" / f"overlay_{split}.ass"
            ass.write_text(overlay(plans[split], title, dur, film, true_track, store.music_track, r.emotion))
            burn(Path(plans[split]["output"]), ass, d / f"{split}.mp4")
        if not (d / "original.mp4").exists():
            shutil.copy(src_dir / "original.mp4", d / "original.mp4")    # already labelled, same encoding settings
        if not (d / "compare.mp4").exists():
            lst = d / "logs" / "concat.txt"
            lst.write_text("".join(f"file '{(d / f'{k}.mp4').resolve()}'\n" for k in ("original", "test", "train")))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                            "-movflags", "+faststart", str(d / "compare.mp4")], check=True)

        def summary(split):
            segs = plans[split]["segments"]
            hit = sum(s["end_s"] - s["start_s"] for s in segs if store.music_track[s["music_id"]] == true_track)
            return {f"{split}_fold": k_test if split == "test" else k_train,
                    f"{split}_switches": sum(a["music_id"] != b["music_id"] for a, b in zip(segs, segs[1:])),
                    f"{split}_true_music_s": round(hit, 1), f"{split}_true_music_frac": round(hit / dur, 2),
                    f"{split}_music": " → ".join(dict.fromkeys(film(s["music_id"]) for s in segs))}
        emo = [b.get("emotion") for b in plans["test"].get("scene_brief", [])]
        rows.append({"folder": name, "emotion_tag": r.emotion, "film": r.film, "pair_id": r.pair_id, "duration_s": round(dur, 1),
                     "segments": len(plans["test"]["segments"]),
                     "vlm_emotions": " ".join(dict.fromkeys(e for e in emo if e)), **summary("test"), **summary("train")})
        pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    log.info("done: %d clips", len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    main()
