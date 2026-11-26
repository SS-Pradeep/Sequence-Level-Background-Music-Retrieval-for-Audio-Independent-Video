"""M12 listening-study kit (SPEC M12): stimuli, randomized pairwise sessions, instructions, rating template.

* 20 test videos stratified by duration and source: 10 natural test clips (long / medium / short, several sources) and
  10 test concatenations (3-4 clips with different tracks). Each is rendered with its own fold's run (seed 0), so no
  condition has seen the video in training.
* 4 conditions, all through the same renderer (same loudness target, crossfades, limiter):
    ours      coarse -> fine -> Viterbi with the run's tuned penalties
    greedy    the same candidates, best window per segment
    baseline  single-track semantic retrieval: the top whole-video match of the semantic heads, from its start
    original  the video's own soundtrack(s) at their aligned offsets
* Stimuli are anonymized (kit/stimuli/v<NN>_<letter>.mp4, letters shuffled per video); kit/key.csv maps them back.
* Pairwise comparisons: ours vs greedy, ours vs baseline, ours vs original; a randomized trial order and left/right
  placement per rater (kit/sessions/rater_<RR>.csv); kit/ratings_template.csv; kit/instructions.md.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig

from bgm.data.prep import interim_music_path, interim_video_path
from bgm.eval.store import FeatureStore, fold_ids
from bgm.render.render import SR, Run, render_soundtrack, runs_from_path
from bgm.retrieval.index import FoldModels, MusicIndex, concat_query, pair_query
from bgm.retrieval.pipeline import coarse_shortlist, fine, fuse, zstats
from bgm.retrieval.stitch_eval import build_lattice, chosen_candidates, decode_lattice, greedy_lattice, read_concat_sets

log = logging.getLogger(__name__)
KIT = Path("kit")
CONDITIONS = ("ours", "greedy", "baseline", "original")
COMPARISONS = (("ours", "greedy"), ("ours", "baseline"), ("ours", "original"))
DIMENSIONS = ("content", "rhythm", "mood", "transitions")


def choose_videos(cfg, store: FeatureStore, n_natural: int = 10, n_concat: int = 10, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    folds = [fold_ids(cfg, k) for k in range(cfg.splits.n_folds)]
    test = pd.DataFrame([{"pair_id": p, "fold": k, "dur": store.duration("video", p), "source": store.pairs.loc[p, "meta_source"]}
                         for k, f in enumerate(folds) for p in f["test"]])
    test["bucket"] = pd.cut(test.dur, [0, 20, 40, 1e9], labels=["short", "medium", "long"])
    per_bucket = {"long": 4, "medium": 3, "short": n_natural - 7}
    out = []
    for b, n in per_bucket.items():
        sub = test[test.bucket == b]
        picked = []
        for src in rng.permutation(sub.source.unique()):          # spread over sources first
            if len(picked) < n:
                picked.append(sub[sub.source == src].sample(1, random_state=int(rng.integers(1e9))).iloc[0])
        rest = sub[~sub.pair_id.isin([r.pair_id for r in picked])]
        picked += [r for _, r in rest.sample(n - len(picked), random_state=int(rng.integers(1e9))).iterrows()] if n > len(picked) else []
        out += [{"kind": "natural", "fold": int(r.fold), "parts": [r.pair_id], "bucket": b} for r in picked]
    sets = read_concat_sets(cfg)
    for i in range(n_concat):
        k = i % cfg.splits.n_folds
        out.append({"kind": "concat", "fold": k, "parts": sets[str(k)]["test"][int(rng.integers(len(sets[str(k)]["test"])))], "bucket": "concat"})
    return out


def silent_video(cfg, parts: list[str], dst: Path) -> None:
    """The query video: the interim silent video, or the parts joined end to end (scaled to 640x360)."""
    if dst.exists():
        return
    if len(parts) == 1:
        dst.write_bytes(interim_video_path(cfg, parts[0]).read_bytes())
        return
    inputs = sum([["-i", str(interim_video_path(cfg, p))] for p in parts], [])
    filt = "".join(f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v{i}];"
                   for i in range(len(parts))) + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[out]", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], check=True)


def condition_runs(cfg, store, item, models, index, fusion, stitch) -> dict[str, list[Run]]:
    vq = pair_query(store, item["parts"][0]) if item["kind"] == "natural" else concat_query(store, item["parts"])
    vq.prepare(models, store.min_span)
    short = coarse_shortlist(index, vq, fusion["w_coarse"], fusion["K"])
    res = fine(index, vq, short, fusion["window_hop_s"])
    lat = build_lattice(index, vq, res, fusion["w_fine"], fusion["J"])
    segs = [(r.a, r.b) for r in res]
    out = {"ours": runs_from_path(chosen_candidates(index, lat, decode_lattice(lat, stitch["penalties"], stitch["beam"], stitch["tol_s"])), segs),
           "greedy": runs_from_path(chosen_candidates(index, lat, greedy_lattice(lat)), segs)}
    gv, pv = vq.pooled(0.0, vq.duration)
    raw = index.file_scores(gv, pv)
    top = index.items[int(np.argmax(fuse(raw, zstats(raw), {"sem": 1.0})))]
    out["baseline"] = [Run(top.music_id, 0.0, 0.0, vq.duration, top.tempo)]
    parts = vq.meta["parts"] if item["kind"] == "concat" else [(item["parts"][0], 0.0, vq.duration)]
    out["original"] = [Run(store.music_of[p], 0.0, t0, t0 + d, float(store.music(store.music_of[p])["tempo_bpm"])) for p, t0, d in parts]
    return out, vq.duration


def build_kit(cfg: DictConfig, n_raters: int = 20, seed: int = 0, exp: str = "pipeline") -> Path:
    store = FeatureStore(cfg)
    items = choose_videos(cfg, store, seed=seed)
    (KIT / "stimuli").mkdir(parents=True, exist_ok=True)
    (KIT / "work").mkdir(parents=True, exist_ok=True)
    (KIT / "sessions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    key, cache = [], {}
    for n, item in enumerate(items):
        k = item["fold"]
        if k not in cache:
            run = Path(cfg.paths.runs) / exp / f"fold{k}" / "seed0"
            models = FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant)
            cache[k] = (models, MusicIndex(store, models, list(dict.fromkeys(store.pairs.music_id))),
                        json.loads((run / "fusion.json").read_text()), json.loads((run / "stitch.json").read_text()))
        models, index, fusion, stitch = cache[k]
        runs, dur = condition_runs(cfg, store, item, models, index, fusion, stitch)
        vid = KIT / "work" / f"v{n:02d}_silent.mp4"
        silent_video(cfg, item["parts"], vid)
        letters = list(rng.permutation(list("ABCD")))
        for cond, letter in zip(CONDITIONS, letters):
            mix, _ = render_soundtrack(runs[cond], dur, lambda m: interim_music_path(cfg, m), cfg.render)
            wav = KIT / "work" / f"v{n:02d}_{letter}.wav"
            sf.write(wav, np.stack([mix, mix], 1) / np.sqrt(2), SR)
            out = KIT / "stimuli" / f"v{n:02d}_{letter}.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(vid), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
            key.append({"video": f"v{n:02d}", "letter": letter, "condition": cond, "kind": item["kind"], "bucket": item["bucket"],
                        "fold": k, "parts": ";".join(item["parts"]), "duration_s": round(dur, 2),
                        "pieces": len(runs[cond]), "tracks": ";".join(dict.fromkeys(r.music_id for r in runs[cond]))})
        log.info("kit: v%02d (%s, %.0f s) rendered", n, item["kind"], dur)
    key = pd.DataFrame(key)
    key.to_csv(KIT / "key.csv", index=False)
    write_sessions(key, n_raters, seed)
    (KIT / "instructions.md").write_text(INSTRUCTIONS)
    return KIT


def write_sessions(key: pd.DataFrame, n_raters: int, seed: int) -> None:
    lookup = {(r.video, r.condition): r.letter for r in key.itertuples()}
    videos = sorted(key.video.unique())
    template = []
    for rid in range(n_raters):
        rng = np.random.default_rng([seed, rid])
        trials = [(v, a, b) for v in videos for a, b in COMPARISONS]
        rows = []
        for t, i in enumerate(rng.permutation(len(trials))):
            v, a, b = trials[i]
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            rows.append({"rater": f"R{rid:02d}", "trial": t + 1, "video": v, "left": f"{v}_{lookup[(v, left)]}.mp4",
                         "right": f"{v}_{lookup[(v, right)]}.mp4", "preferred": "",
                         **{f"left_{d}": "" for d in DIMENSIONS}, **{f"right_{d}": "" for d in DIMENSIONS}})
        pd.DataFrame(rows).to_csv(KIT / "sessions" / f"rater_{rid:02d}.csv", index=False)
        template += rows
    pd.DataFrame(template).to_csv(KIT / "ratings_template.csv", index=False)


INSTRUCTIONS = """# Listening study: background music for silent videos

Thank you for taking part. You will watch short videos (10 s to about 2 min) that each come with a background-music
track. Every trial shows the SAME video twice, with two different soundtracks: LEFT and RIGHT.

For each trial, in your own sheet (`sessions/rater_XX.csv`, one row per trial, in order):

1. Watch LEFT, then RIGHT (headphones, a quiet room, the same volume throughout; you may replay).
2. `preferred`: write `L` or `R` for the soundtrack that suits the video better overall. You must choose one.
3. Rate EACH soundtrack from 1 (poor) to 5 (excellent) on:
   - `content`: does the music fit what is happening on screen?
   - `rhythm`: does the music's pace and energy follow the cuts and motion?
   - `mood`: does the music match the emotional tone?
   - `transitions`: are changes between music pieces smooth (5 if there are none and nothing jars)?

Please do not discuss the videos with other raters, and do not look up the films. File names are codes; they say
nothing about how the music was chosen. The study takes about 60-90 minutes; take a break whenever you like.
"""


if __name__ == "__main__":
    import argparse

    from bgm.config import load_config
    from bgm.log import setup_logging
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raters", type=int, default=20)
    ap.add_argument("--exp", default="pipeline")
    a = ap.parse_args()
    print(build_kit(load_config(), a.raters, exp=a.exp))
