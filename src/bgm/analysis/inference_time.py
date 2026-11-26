"""Inference time on test videos (one video at a time, GPU, frozen models; no training).

Run: python -m bgm.analysis.inference_time [n_videos]   -> results/inference_time.md, reports/inference_time.json

End to end, per video, from its original file (the soundtrack is dropped at the first step, as in `bgm render`):
silent 25 fps copy -> shots -> SigLIP 1 Hz + mood logits -> 25 Hz motion curves -> segments -> retrieval (coarse,
fine, lattice, Viterbi) against the whole library -> beat sync + soundtrack render -> mux. Timed on a random sample of
fold-0 test videos with the fold-0 / seed-0 models (every stage synchronized on the GPU). One-time costs (loading the
encoders and heads, indexing the library) are reported separately. Retrieval alone is also timed on every fold-0 test
video from the cached features, against the test-fold gallery (as in evaluation) and the whole library (as in `bgm render`).
ImageBind features (used only by `bgm check`) are not part of inference and are not timed.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from bgm.config import load_config
from bgm.data.media import ffprobe
from bgm.data.prep import interim_music_path
from bgm.data.segments import make_segments
from bgm.data.shots import detect_cuts
from bgm.eval.store import FeatureStore, fold_ids
from bgm.features.dynamics import video_curves
from bgm.features.extract import EncoderBank, frames_1hz, word_embeddings
from bgm.features.timing import n_1hz
from bgm.render.query import as_video_query, silent_copy
from bgm.render.render import SR, Run, beat_sync_runs, fill_short_runs, mux, render_soundtrack, runs_from_path
from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
from bgm.retrieval.pipeline import coarse_shortlist, fine
from bgm.retrieval.stitch_eval import build_lattice, chosen_candidates, decode_lattice

log = logging.getLogger(__name__)
FOLD, SEED = 0, 0


class Clock:
    def __init__(self):
        self.t: dict[str, float] = {}

    def __call__(self, name: str, fn, *a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
        return out


def retrieve(index, vq, fusion, stitch):
    short = coarse_shortlist(index, vq, fusion["w_coarse"], fusion["K"])
    results = fine(index, vq, short, fusion["window_hop_s"])
    lat = build_lattice(index, vq, results, fusion["w_fine"], fusion["J"])
    picks = decode_lattice(lat, stitch["penalties"], stitch["beam"], stitch["tol_s"])
    return results, chosen_candidates(index, lat, picks)


def end_to_end(cfg, store, bank, models, index, fusion, stitch, src: Path, work: Path) -> tuple[dict, float, int]:
    c = Clock()
    silent = work / "silent25.mp4"
    c("1 silent 25 fps copy", silent_copy, src, silent, cfg)
    info = ffprobe(silent)
    dur = info["video"]["duration_s"] if np.isfinite(info["video"]["duration_s"]) else info["duration_s"]
    cuts = np.array(c("2 shot detection", detect_cuts, silent, cfg.shots.adaptive_threshold), float)
    frames = c("3a decode 1 Hz frames", frames_1hz, silent, n_1hz(dur))
    sig = c("3b SigLIP encode", bank.frames, "siglip", frames)
    a = cfg.features.attributes
    W = word_embeddings(bank, "siglip", list(a.moods), list(a.video_templates))
    scale, bias = bank.logit_params("siglip")
    attr = c("3c mood logits", lambda: (scale * (sig @ W.T) + bias).astype(np.float32))
    d = cfg.features.dynamics
    dyn = c("4 motion curves (25 Hz)", video_curves, silent, cuts, d.impulse_sigma_s, d.flow_short_side, d.affine_grid)
    segs = c("5 segmentation", make_segments, cuts, dur, sig, cfg.segments.merge_cosine, cfg.segments.max_len_s, cfg.segments.min_shot_s)
    feat = {"sem_siglip_1hz": sig, "attr_logits_1hz": attr, "dyn_25hz": dyn, "cut_times_s": cuts.astype(np.float32),
            "segments": np.array([(s, e) for s, e, _ in segs], np.float32), "meta": {"duration_s": dur}}
    vq = as_video_query(feat)
    c("6a query heads", vq.prepare, models, store.min_span)
    results, chosen = c("6b retrieval (coarse + fine + Viterbi)", retrieve, index, vq, fusion, stitch)

    def soundtrack():
        runs = runs_from_path(chosen, [(r.a, r.b) for r in results])
        runs, _ = beat_sync_runs(runs, cuts, lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m),
                                 cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
        runs, _ = fill_short_runs(runs, lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m))
        mix, _ = render_soundtrack(runs, dur, lambda m: interim_music_path(cfg, m), cfg.render)
        sf.write(work / "bgm.wav", np.stack([mix, mix], 1) / np.sqrt(2), SR)
    c("7 soundtrack render", soundtrack)
    c("8 mux to MP4", mux, src, work / "bgm.wav", work / "out.mp4", 0.0)
    return c.t, dur, len(segs)


def main(n_videos: int = 20) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", [])
    run = Path(cfg.paths.runs) / "pipeline" / f"fold{FOLD}" / f"seed{SEED}"
    fusion, stitch = json.loads((run / "fusion.json").read_text()), json.loads((run / "stitch.json").read_text())
    once = Clock()
    store = once("load feature store", FeatureStore, cfg)
    bank = once("load SigLIP encoder", EncoderBank, cfg, False)
    models = once("load trained heads + calibration", FoldModels.load, cfg, store, FOLD, SEED)
    lib = list(dict.fromkeys(store.pairs.music_id))
    index = once(f"index library ({len(lib)} songs)", MusicIndex, store, models, lib)
    test = fold_ids(cfg, FOLD)["test"]
    index_test = once("index test-fold gallery", MusicIndex, store, models, list(dict.fromkeys(store.music_of[p] for p in test)))

    # retrieval only, every fold-0 test video, cached features
    ret = {"library": [], "test_fold": []}
    for p in test:
        vq = pair_query(store, p)
        vq.prepare(models, store.min_span)
        for key, ix in (("library", index), ("test_fold", index_test)):
            c = Clock()
            c("r", retrieve, ix, vq, fusion, stitch)
            ret[key].append(c.t["r"])

    # end to end on a random sample
    rng = np.random.default_rng(0)
    sample = [test[i] for i in rng.choice(len(test), n_videos + 1, replace=False)]
    rows = []
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        for k, p in enumerate(sample):
            src = Path(store.pairs.loc[p, "video_path"])
            work = Path(tmp) / f"v{k}"
            work.mkdir()
            t, dur, n_seg = end_to_end(cfg, store, bank, models, index, fusion, stitch, src, work)
            if k == 0:
                log.info("warm-up video done (not counted)")
                continue
            rows.append({"pair_id": p, "duration_s": dur, "segments": n_seg, **t, "total": sum(t.values())})
            log.info("%d/%d %s: %.1f s video, %.2f s total", k, n_videos, p, dur, rows[-1]["total"])
    out = {"device": torch.cuda.get_device_name(0), "fold": FOLD, "seed": SEED, "one_time_s": once.t, "end_to_end": rows,
           "retrieval_only_s": {k: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                                for k, v in ret.items()}}
    Path("reports/inference_time.json").write_text(json.dumps(out, indent=1))
    Path("results/inference_time.md").write_text(markdown(out))
    log.info("wrote results/inference_time.md")


def markdown(out: dict) -> str:
    rows = out["end_to_end"]
    stages = [k for k in rows[0] if k[0].isdigit()]
    dur = np.array([r["duration_s"] for r in rows])
    tot = np.array([r["total"] for r in rows])
    L = ["# Inference time", "", f"{out['device']}; fold {out['fold']} / seed {out['seed']} models; one video at a time; "
         f"end to end on {len(rows)} random fold-0 test videos (after one warm-up video), from the original file to the output MP4. "
         f"Video length median {np.median(dur):.1f} s (range {dur.min():.1f}-{dur.max():.1f} s).", "",
         "## Per video, end to end", "", "| stage | mean s | median s | share | s per minute of video |", "| --- | --- | --- | --- | --- |"]
    for s in stages + ["total"]:
        v = np.array([r[s] for r in rows])
        L.append(f"| {'**total**' if s == 'total' else s} | {v.mean():.2f} | {np.median(v):.2f} | {v.sum() / tot.sum():.0%} | {60 * v.sum() / dur.sum():.2f} |")
    rt = dur.sum() / tot.sum()
    L += ["", f"Real-time factor: **{rt:.1f}×** faster than real time ({tot.sum():.0f} s of processing for {dur.sum():.0f} s of video).", "",
          "## Retrieval only (cached features, every fold-0 test video)", "", "| gallery | videos | mean ms | median ms | p95 ms |",
          "| --- | --- | --- | --- | --- |"]
    for k, v in out["retrieval_only_s"].items():
        L.append(f"| {'whole library (1,248 songs)' if k == 'library' else 'test fold (~250 songs)'} | {v['n']} | {1000 * v['mean']:.0f} | "
                 f"{1000 * v['median']:.0f} | {1000 * v['p95']:.0f} |")
    L += ["", "## One-time costs (per session)", "", "| step | s |", "| --- | --- |"] + [f"| {k} | {v:.1f} |" for k, v in out["one_time_s"].items()]
    L += ["", "Videos of 60 s or more also run the 15-model chunked ensemble for the coarse shortlist in `bgm render`; test clips here "
          "are shorter, so it is not timed. The library music features are computed offline once (not part of inference)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
