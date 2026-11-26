"""Compute cost of the final system (cleaned data, title split, 10-head ensemble, Viterbi; PROGRESS.md D116).

Run: python -m bgm.analysis.compute_cost train|infer|report|all
  -> runs/compute_cost/ (retrained heads and pipeline), reports/cleaned/compute_cost.json, results/cleaned/compute_cost.md

train: retrains ensemble 0 (heads 0-9) of every fold, then fusion and stitch tuning, one job at a time, and checks the
       test metrics against the saved D116 runs. Saved runs are read only.
infer: times the saved D116 ensembles from the original video file to the output MP4, plus retrieval alone on every
       test video. ImageBind and the motion curves are separate stages: `bgm render` computes both, the ensemble's
       retrieval score uses neither.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from bgm.analysis.inference_time import Clock, retrieve
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
from bgm.render.render import beat_sync_runs, fill_short_runs, mux, render_soundtrack, runs_from_path, SR
from bgm.retrieval.fuse import run_fuse
from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
from bgm.retrieval.tune_stitch import run_tune_stitch
from bgm.train.semantic import run_train_semantic

log = logging.getLogger(__name__)
SAVED, NEW = Path("runs/cleaned"), Path("runs/compute_cost")
JSON_OUT, MD_OUT = Path("reports/cleaned/compute_cost.json"), Path("results/cleaned/compute_cost.md")
FOLDS, HEADS = range(5), list(range(10))
N_E2E_PER_FOLD = 10
DATA = ["splits.folds_file=folds_title.json", "paths.manifests=cleaned_dataset/manifests", "paths.interim=cleaned_dataset/interim",
        "paths.features=cleaned_dataset/features", "eval.concat_sets_file=concat_sets_title.json"]
HEAD_ARGS = ["train.semantic.max_clips_per_film_per_epoch=30", "train.semantic.music_feature=both"]
ENS = ["pipeline.dyn_variant=null", "pipeline.exp=pipeline_ens10", f"pipeline.sem_seeds=[{','.join(map(str, HEADS))}]"]


def config(root: Path, *extra: str):
    return load_config("configs/default.yaml", DATA + [f"paths.runs={root}", f"paths.results={root}/results", f"paths.reports={root}/reports", *extra])


def timed(fn) -> tuple[float, float]:
    """(wall seconds, peak GPU GiB allocated by this process) of fn()."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def same(a, b, tol: float = 1e-9) -> bool:
    """Nested metric dictionaries equal within tol."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(abs(a - b) <= tol or (np.isnan(a) and np.isnan(b)))
    return a == b


def phase_train() -> dict:
    heads, pipes = [], []
    for k in FOLDS:
        for s in HEADS:
            cfg = config(NEW, f"fold={k}", f"seed={s}", *HEAD_ARGS)
            sec, gib = timed(lambda: run_train_semantic(cfg, [k], [s], "siglip-both"))
            rel = Path("semantic/siglip-both") / f"fold{k}" / f"seed{s}" / "metrics.json"
            new, old = json.loads((NEW / rel).read_text()), json.loads((SAVED / rel).read_text())
            heads.append({"fold": k, "seed": s, "seconds": sec, "peak_gib": gib, "epochs": new["val"]["epochs_run"],
                          "reproduced": same(new["test"], old["test"])})
            log.info("head fold %d seed %d: %.1f s, %.2f GiB, reproduced %s", k, s, sec, gib, heads[-1]["reproduced"])
        cfg = config(NEW, f"fold={k}", "seed=0", *ENS)
        fuse_s, fuse_gib = timed(lambda: run_fuse(cfg, [k], [0]))
        stitch_s, stitch_gib = timed(lambda: run_tune_stitch(cfg, [k], [0]))
        rel = Path("pipeline_ens10") / f"fold{k}" / "seed0"
        a, b = json.loads((NEW / rel / "stitch.json").read_text()), json.loads((SAVED / rel / "stitch.json").read_text())
        pipes.append({"fold": k, "fuse_s": fuse_s, "stitch_s": stitch_s, "peak_gib": max(fuse_gib, stitch_gib),
                      "reproduced": same(a["penalties"], b["penalties"])})
        log.info("pipeline fold %d: fuse %.1f s, stitch %.1f s, reproduced %s", k, fuse_s, stitch_s, pipes[-1]["reproduced"])
    return {"heads": heads, "pipeline": pipes}


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
    c("3d ImageBind encode (render only)", bank.frames, "imagebind", frames)
    d = cfg.features.dynamics
    dyn = c("4 motion curves (render only)", video_curves, silent, cuts, d.impulse_sigma_s, d.flow_short_side, d.affine_grid)
    segs = c("5 segmentation", make_segments, cuts, dur, sig, cfg.segments.merge_cosine, cfg.segments.max_len_s, cfg.segments.min_shot_s)
    feat = {"sem_siglip_1hz": sig, "attr_logits_1hz": attr, "dyn_25hz": dyn, "cut_times_s": cuts.astype(np.float32),
            "segments": np.array([(s, e) for s, e, _ in segs], np.float32), "meta": {"duration_s": dur}}
    vq = as_video_query(feat)
    c("6a ensemble query heads", vq.prepare, models, store.min_span)
    results, chosen = c("6b retrieval + Viterbi", retrieve, index, vq, fusion, stitch)

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


def retrieval_ms(store, models, index, ids, fusion, stitch) -> list[float]:
    out = []
    for p in ids:
        vq = pair_query(store, p)
        c = Clock()
        c("r", lambda: (vq.prepare(models, store.min_span), retrieve(index, vq, fusion, stitch)))
        out.append(1000 * c.t["r"])
    return out


def phase_infer() -> dict:
    cfg = config(SAVED, *ENS)
    single = config(SAVED, "pipeline.dyn_variant=null", "pipeline.sem_seeds=null")
    once = Clock()
    store = once("load feature store", FeatureStore, cfg)
    bank = once("create encoder bank", EncoderBank, cfg, False)
    lib = list(dict.fromkeys(store.pairs.music_id))
    rows, ret, load = [], {"ensemble_library": [], "ensemble_test_fold": [], "single_library": []}, []
    with tempfile.TemporaryDirectory(dir=os.environ.get("BGM_TMP") or None) as tmp:
        for k in FOLDS:
            run = SAVED / "pipeline_ens10" / f"fold{k}" / "seed0"
            fusion, stitch = json.loads((run / "fusion.json").read_text()), json.loads((run / "stitch.json").read_text())
            c = Clock()
            models = c("load", FoldModels.load, cfg, store, k, 0, "siglip-both", None)
            index = c("index", MusicIndex, store, models, lib)
            load.append({"fold": k, "load_heads_s": c.t["load"], "index_library_s": c.t["index"]})
            test = list(fold_ids(cfg, k)["test"])
            gallery = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in test)))
            ret["ensemble_library"] += retrieval_ms(store, models, index, test, fusion, stitch)
            ret["ensemble_test_fold"] += retrieval_ms(store, models, gallery, test, fusion, stitch)
            one = FoldModels.load(single, store, k, 0, "siglip-both", None)
            ret["single_library"] += retrieval_ms(store, one, MusicIndex(store, one, lib), test, fusion, stitch)
            rng = np.random.default_rng(1000 + k)
            sample = [test[i] for i in rng.choice(len(test), min(N_E2E_PER_FOLD + 1, len(test)), replace=False)]
            for j, p in enumerate(sample):
                work = Path(tmp) / f"f{k}_{j}"
                work.mkdir()
                t, dur, n_seg = end_to_end(cfg, store, bank, models, index, fusion, stitch, Path(store.pairs.loc[p, "video_path"]), work)
                if j:   # the first video of each fold is a warm-up
                    rows.append({"fold": k, "pair_id": p, "duration_s": dur, "segments": n_seg, **t, "total": sum(t.values())})
                    log.info("fold %d video %d: %.1f s video, %.2f s", k, j, dur, rows[-1]["total"])
    return {"one_time_s": once.t, "fold_load": load, "end_to_end": rows, "retrieval_ms": ret}


def logged_features() -> list[dict]:
    """Feature-extraction costs from the D116 log; partly cached stages are scaled to all items."""
    text = Path("reports/cleaned/extract.log").read_text(errors="replace")
    out = []
    for name, kind, n, sec in re.findall(r"extract (\w+)/(\w+): (\d+) items in ([\d.]+) s", text):
        n, sec = int(n), float(sec)
        out.append({"stage": f"{name}/{kind}", "items": n, "seconds": sec, "seconds_all_1011": sec * 1011 / n})
    return out


def summary(v) -> dict:
    v = np.asarray(v, float)
    return {"n": len(v), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95))}


def markdown(o: dict) -> str:
    h, p, e = o["train"]["heads"], o["train"]["pipeline"], o["infer"]["end_to_end"]
    head_s = np.array([r["seconds"] for r in h])
    per_fold = np.array([sum(r["seconds"] for r in h if r["fold"] == k) + q["fuse_s"] + q["stitch_s"] for k, q in zip(FOLDS, p)])
    feat = o["features"]
    feat_s = sum(r["seconds_all_1011"] for r in feat)
    L = ["# Compute cost of the final system", "",
         f"Measured {o['date']} on {o['device']}, {o['cpu_threads']} CPU threads, PyTorch {o['torch']}, commit {o['commit'][:7]}"
         f"{' plus uncommitted changes' if o['dirty'] else ''}. System: D116 cleaned data, title split, 10-head ensemble, Viterbi. "
         "The measured jobs ran one after another in a single session. The machine was not reserved, so other work may have overlapped and these times are upper-end estimates for this hardware.", "",
         "## Training", "", "| step | scope | wall-clock |", "| --- | --- | --- |",
         f"| Feature extraction (logged in D116) | all 1,011 pairs | {feat_s / 60:.1f} min |",
         f"| One semantic head | mean ± std over {len(h)} heads | {head_s.mean():.1f} ± {head_s.std(ddof=1):.1f} s |",
         f"| Ten heads + fusion + stitch tuning | one fold, mean | {per_fold.mean() / 60:.1f} min |",
         f"| **Full 5-fold ensemble system** | 50 heads + 5 pipelines, serial | **{per_fold.sum() / 60:.1f} min** |",
         f"| Rule 11 protocol (extrapolated) | 150 heads + 15 pipelines, serial | {3 * per_fold.sum() / 60:.1f} min |", "",
         f"Peak GPU memory allocated: {max(r['peak_gib'] for r in h):.2f} GiB (head), {max(r['peak_gib'] for r in p):.2f} GiB (fusion / stitching). "
         f"Reproduced saved test metrics: {sum(r['reproduced'] for r in h)} of {len(h)} heads; saved stitch penalties: "
         f"{sum(r['reproduced'] for r in p)} of {len(p)} folds.", "",
         "## Inference, per video, end to end", "",
         f"{len(e)} random test videos ({N_E2E_PER_FOLD} per fold, after one warm-up each), median length "
         f"{np.median([r['duration_s'] for r in e]):.1f} s.", "",
         "| stage | mean s | median s | share | s per minute of video |", "| --- | --- | --- | --- | --- |"]
    dur, tot = sum(r["duration_s"] for r in e), sum(r["total"] for r in e)
    stages = [k for k in e[0] if k[0].isdigit()]
    for s in stages + ["total"]:
        v = np.array([r[s] for r in e])
        L.append(f"| {'**total**' if s == 'total' else s} | {v.mean():.3f} | {np.median(v):.3f} | {v.sum() / tot:.1%} | {60 * v.sum() / dur:.2f} |")
    skip = sum(r[s] for r in e for s in stages if "render only" in s)
    L += ["", f"As `bgm render` runs today: **{dur / tot:.2f}× real time**. Without the two stages the ensemble's score does not use: "
          f"**{dur / (tot - skip):.2f}× real time** (projected from the stage times, not run as a separate pipeline).", "",
          "## Retrieval alone (cached features, every test video)", "", "| system | gallery | videos | mean ms | median ms | p95 ms |",
          "| --- | --- | --- | --- | --- | --- |"]
    names = {"ensemble_library": ("10-head ensemble", "all 1,011 files"), "ensemble_test_fold": ("10-head ensemble", "test fold"),
             "single_library": ("single head", "all 1,011 files")}
    for k, v in o["infer"]["retrieval_ms"].items():
        q = summary(v)
        L.append(f"| {names[k][0]} | {names[k][1]} | {q['n']} | {q['mean']:.1f} | {q['median']:.1f} | {q['p95']:.1f} |")
    load = o["infer"]["fold_load"]
    L += ["", "## One-time inference setup", "", "| step | mean s |", "| --- | --- |",
          f"| load 10 heads + calibration | {np.mean([r['load_heads_s'] for r in load]):.2f} |",
          f"| index all music files | {np.mean([r['index_library_s'] for r in load]):.2f} |", "",
          "Not timed: videos of 60 s or more also use a chunked shortlist in `bgm render`, so these numbers cover the clip path only.",
          "Not timed: loading the SigLIP and ImageBind encoders, which happens during each fold's uncounted warm-up video. A cold `bgm render` pays this once.",
          "The faster projection does not apply to scene-aware renders: the pace option (D100) reads the motion curves.",
          "The single-head row reuses the ensemble's fusion and stitch settings, so it compares scoring cost only.",
          "Feature extraction is logged from the D116 run, which may have shared the machine with other jobs. It excludes data prep and shot detection."]
    return "\n".join(L) + "\n"


def meta() -> dict:
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"date": time.strftime("%Y-%m-%d"), "device": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__, "python": platform.python_version(), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "src", "configs"))}


def main(stage: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    o = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    if stage != "report":   # keep the commit and date of the measurement when only the report is rebuilt
        o.update(meta())
    if stage in ("train", "all"):
        o["train"] = phase_train()
        JSON_OUT.write_text(json.dumps(o, indent=1))
    if stage in ("infer", "all"):
        o["infer"] = phase_infer()
        JSON_OUT.write_text(json.dumps(o, indent=1))
    if stage in ("report", "all"):
        o["features"] = logged_features()
        JSON_OUT.write_text(json.dumps(o, indent=1))
        MD_OUT.write_text(markdown(o))
        log.info("wrote %s", MD_OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
