"""Best held-out clip per emotion for the final system (user request after D106/D107).

Every clip outside the train-only group is a TEST clip of exactly one title-split fold, so each clip is scored by its own
fold's final system (10-head ensemble, head seeds 0-9, runs/titlesplit/semantic/siglip-both/fold<k>) and never by heads
that trained on it. Emotion (the 10 of D106) is a zero-shot estimate, not a label: SigLIP on the clip's frames (mean
over seconds of cos to "a {emotion} scene ..." prompts) plus CLAP on the clip's TRUE music (cos to the D106 music texts),
each z-scored per emotion over all clips and summed. A clip belongs to its argmax emotion when both sides are above
their mean for it. "Best" = the true music ranked highest by the model against the whole 1,248-file library, then
within the fold's test files, then the largest score margin of the true file over the best wrong one. One film group per
emotion.

    python -m bgm.analysis.emotion_showcase      -> outputs/emotion_showcase/scores.csv, picks.csv
"""
from __future__ import annotations

import json
import re
import subprocess
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bgm.analysis.emotion_variants import MUSIC
from bgm.config import load_config
from bgm.eval.metrics import rank_of_first_hit
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("outputs/emotion_showcase")
OVER = ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "pipeline.dyn_variant=null"]
MEMBERS = list(range(10))
VIDEO_TEXT = {
    "heroic": "a heroic mass hero entry", "action": "a fast violent fight or chase", "tension": "a tense suspenseful confrontation",
    "fear": "a scary horror scene", "sad": "a sad emotional crying moment", "romantic": "a romantic moment between lovers",
    "happy": "a happy festive celebration", "comic": "a funny comedy scene", "peaceful": "a calm peaceful landscape",
    "wonder": "a magical awe-inspiring sight"}
V_TEMPLATES = ["{} in an Indian film", "a film scene of {}", "a video showing {}"]


def emotion_scores(cfg, store: FeatureStore, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """[N, E] raw mean cosines for video (SigLIP) and the true music (CLAP), E = len(MUSIC)."""
    from bgm.features.hf import ClapEncoder, SiglipEncoder
    emos = list(MUSIC)
    sig = SiglipEncoder(cfg.features.sem_video.model, "cuda")
    tv = np.stack([sig.encode_text([t.format(VIDEO_TEXT[e]) for t in V_TEMPLATES]).mean(0) for e in emos])
    del sig
    clap = ClapEncoder(cfg.features.sem_music.clap_model, "cuda")
    tm = clap.encode_text([MUSIC[e] for e in emos])
    del clap
    torch.cuda.empty_cache()
    tv, tm = tv / np.linalg.norm(tv, axis=1, keepdims=True), tm / np.linalg.norm(tm, axis=1, keepdims=True)
    V, M = [], []
    for p in ids:
        v = np.load(f"data/features/video/{p}.npz")["sem_siglip_1hz"]
        m = np.load(f"data/features/music/{store.music_of[p]}.npz")["sem_clap_1hz"]
        v, m = v / np.linalg.norm(v, axis=1, keepdims=True), m / np.linalg.norm(m, axis=1, keepdims=True)
        V.append((v @ tv.T).mean(0)), M.append((m @ tm.T).mean(0))
    return np.array(V), np.array(M)


def retrieval(cfg, store: FeatureStore) -> pd.DataFrame:
    """One row per test clip: true-track rank in the whole library and in the fold's test files, top-1 file, margin."""
    from bgm.analysis.full_dataset import load_scorer
    from bgm.person.evaluate import MeanScorer
    lib = list(dict.fromkeys(store.pairs.music_id))
    ltrack = np.array([store.music_track[m] for m in lib])
    rows = []
    for k in range(cfg.splits.n_folds):
        test = fold_ids(cfg, k)["test"]
        sc = MeanScorer([load_scorer(Path(cfg.paths.runs) / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}", store, "cuda")
                         for s in MEMBERS])
        tset = set(store.music_of[p] for p in test)
        in_test = np.array([m in tset for m in lib])
        with torch.no_grad():
            for p in test:
                s = np.asarray(sc.track(p, lib), dtype=np.float64)
                hits = ltrack == store.track[p]
                best_wrong = s[~hits].max()
                rows.append({"pair_id": p, "fold": k, "rank_all": rank_of_first_hit(s, hits),
                             "rank_test": rank_of_first_hit(s[in_test], hits[in_test]), "n_test": int(in_test.sum()),
                             "top1_music": lib[int(s.argmax())], "margin": float(s[hits].max() - best_wrong)})
        log.info("fold %d: %d clips, R@1 all %.3f", k, len(test), np.mean([r["rank_all"] == 1 for r in rows if r["fold"] == k]))
        del sc
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> Path:
    cfg = load_config("configs/default.yaml", OVER)
    store = FeatureStore(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    df = retrieval(cfg, store)
    V, M = emotion_scores(cfg, store, df.pair_id.tolist())
    z = lambda x: (x - x.mean(0)) / x.std(0)  # noqa: E731
    zv, zm = z(V), z(M)
    emos = list(MUSIC)
    comb = zv + zm
    df["emotion"] = [emos[i] for i in comb.argmax(1)]
    df["emo_score"] = comb.max(1)
    df["both_agree"] = [zv[i, j] > 0 and zm[i, j] > 0 for i, j in enumerate(comb.argmax(1))]
    for j, e in enumerate(emos):
        df[f"zv_{e}"], df[f"zm_{e}"] = zv[:, j].round(3), zm[:, j].round(3)
    pr = store.pairs.loc[df.pair_id]
    titles = pd.read_csv("outputs/film_titles.csv")
    tmap = dict(zip(titles.source + ":" + titles.source_id.astype(str), titles.title))
    df["film"] = (pr.meta_source + ":" + pr.meta_film.astype(str)).map(tmap).fillna(pr.meta_film).values
    df["duration_s"] = pr.video_duration_s.values
    df["top1_correct"] = df.rank_all == 1
    df.to_csv(OUT / "scores.csv", index=False)
    used, picks = set(), []
    for e in emos:
        c = df[(df.emotion == e) & df.both_agree & (df.duration_s >= 8)]
        c = c[~c.film.isin(used)].sort_values(["rank_all", "rank_test", "margin"], ascending=[True, True, False])
        if c.empty:
            log.warning("%s: no candidate", e)
            continue
        r = c.iloc[0]
        used.add(r.film)
        picks.append({**r[["pair_id", "fold", "film", "duration_s", "rank_all", "rank_test", "n_test", "margin", "emo_score",
                           "top1_music"]].to_dict(), "emotion": e, "n_candidates": len(c)})
    pk = pd.DataFrame(picks)
    pk.to_csv(OUT / "picks.csv", index=False)
    log.info("overall R@1 all-library %.3f over %d clips", df.top1_correct.mean(), len(df))
    print(pk.to_string())
    return OUT


# Final choice after a contact-sheet check of the automatic picks (zero-shot tags are noisy): action, sad and peaceful
# take the strongest-emotion clip whose true music is the model's top pick among the fold's test files; the rest keep
# the automatic pick. Wonder stays weak: no clearly awe-inspiring clip among those the model gets right.
FINAL = {"heroic": "combined_full_bgm_dataset_youtube__135__135_bgm_0000", "action": "sunnxt_141_141_bgm_0000",
         "tension": "combined_full_bgm_dataset_youtube__11__11_bgm_0001", "fear": "sunnxt_213_213_bgm_0004",
         "sad": "combined_full_bgm_dataset_youtube__120__120_bgm_0001", "romantic": "sunnxt_272_272_bgm_0003",
         "happy": "combined_full_bgm_dataset__karnan__karnan_bgm_0003", "comic": "combined_full_bgm_dataset_youtube__82__82_bgm_0002",
         "peaceful": "combined_full_bgm_dataset_youtube__76__76_bgm_0001", "wonder": "combined_full_bgm_dataset_youtube__64__64_bgm_0011"}
RENDER = ["pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.audio_bitrate=320k", "render.length_fit.weight=1.0",
          "render.replay_penalty=3.0", "render.max_plays=2", "render.max_windows_per_file=4", "paths.renders=renders/emotion_showcase"]


def upscale(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(src), "-vf", "scale=-2:1080:flags=lanczos", "-c:v", "libx264",
                    "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(dst)], check=True)


def render_all() -> Path:
    """Per pick: model.mp4 (the clip's own fold's final system on the silent interim video; clips < 60 s, so no
    cross-fold long-video shortlist) and original.mp4 (its true music), both at 1080 lines."""
    from bgm.analysis.cue_demo import original_soundtrack
    from bgm.render.run import render_video
    cfg = load_config("configs/default.yaml", OVER + RENDER)
    store = FeatureStore(cfg)
    sc = pd.read_csv(OUT / "scores.csv").set_index("pair_id")
    film_of = dict(zip(store.pairs.music_id, sc.reindex(store.pairs.pair_id).film.fillna(store.pairs.meta_film).values))
    raw = Path("renders/emotion_showcase")
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for emo, p in FINAL.items():
        r = sc.loc[p]
        assert r.duration_s < 60, p
        name = f"{emo}_{re.sub(r'[^A-Za-z0-9]+', '_', r.film).strip('_').lower()}"
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        video = Path(f"data/interim/video/{p}.mp4")
        plan_path = render_video(cfg, str(video), str(raw / f"{name}.mp4"), f"runs/titlesplit/pipeline_ens10/fold{int(r.fold)}/seed0")
        plan = json.loads(Path(plan_path).read_text())
        upscale(raw / f"{name}.mp4", d / "model.mp4")
        orig = raw / f"{name}_original.mp4"
        frames = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                     "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True,
                                    check=True).stdout.strip())
        original_soundtrack(cfg, store, [p], [frames], video, orig)
        upscale(orig, d / "original.mp4")
        used = list(dict.fromkeys(x["music_id"] for x in plan["pieces"]))
        rows.append({"emotion": emo, "folder": name, "pair_id": p, "film": r.film, "fold": int(r.fold), "duration_s": round(r.duration_s, 1),
                     "rank_all": int(r.rank_all), "rank_test": int(r.rank_test), "n_test": int(r.n_test),
                     "render_true_music": store.music_track[used[0]] == store.track[p] if used else False,
                     "render_music": " → ".join(f"{film_of.get(m, '?')} ({m.rsplit('_', 1)[-1]})" for m in used)})
        log.info("%s: %s", emo, rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "index.csv", index=False)
    L = ["# Best held-out clip per emotion (final system)", "",
         "Generated by `python -m bgm.analysis.emotion_showcase --render`. Each clip is a TEST clip of its title-split fold and is "
         "scored and rendered by that fold's final system only (10-head ensemble, never trained on the clip's film). Emotion is a "
         "zero-shot estimate (SigLIP on the frames + CLAP on the true music), checked on contact sheets, not a human label. "
         "rank = rank of the clip's true music by the model: among all 1,248 library files / among the fold's ~195 test files. "
         "model.mp4 = the render (length fit, reprise rule; no scene brief), original.mp4 = the clip's own music. 1080 lines, 320k audio.", "",
         "| emotion | folder | film (fold) | length | rank, library | rank, fold test | render plays |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['emotion']} | {r['folder']}/ | {r['film']} ({r['fold']}) | {r['duration_s']} s | {r['rank_all']} / 1248 | "
                 f"{r['rank_test']} / {r['n_test']} | {r['render_music']} |")
    (OUT / "index.md").write_text("\n".join(L) + "\n")
    return OUT


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    render_all() if "--render" in sys.argv else main()
