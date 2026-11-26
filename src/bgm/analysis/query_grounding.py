"""Grounding with chance levels on 1.mp4 (one silent query video from outside the dataset).

Run: python -m bgm.analysis.query_grounding   -> results/query_1_grounding.md, reports/query_1_grounding.json

Query: media/inputs/1.mp4 (silent). Its own music: the audio of media/inputs/1_BGM_Ori.mp4, the same video with its
original soundtrack. Alignment (checked 2026-09-26): the two files hold identical frames (pixel error 0 at lag 0; only
the silent file's container start differs, 0.041 s), and 1_BGM_Ori's audio and video both start at 0, so music time =
video time. The soundtrack matches no library file (Chromaprint, PROGRESS.md), so every trained model is leak-free here.

Features: the real pipeline stages (prep, shots, extract, segment) into data/interim/query/grounding/. No training:
learned dynamics (shiftneg and noshift, 15 checkpoints each), fused M9 (15 runs, their validation-tuned w_fine; the mood
calibration is fit on each run's training fold, as in M9) and the hand-crafted baseline score every 25 Hz offset of the
soundtrack for each segment; rules and chance levels as in results/dynamics.md (ties share the hit, D34).
Per segment: hits averaged over the 15 models of a system. One video, so no track clusters: CIs resample segments
(plain bootstrap), which treats neighbouring segments as independent; they are descriptive.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bgm.config import load_config
from bgm.eval.metrics import GROUNDING_RULES, RULE_LABELS, cluster_bootstrap
from bgm.eval.protocol import grounding, grounding_queries, score_predictions
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
ROOT = Path("data/interim/query/grounding")
VIDEO, ORIGINAL = "media/inputs/1.mp4", "media/inputs/1_BGM_Ori.mp4"
PAIR, MUSIC = "query_1", "query_1_bgm_ori"
SYSTEMS = {"shiftneg": "learned dynamics, shifted negatives (M7)", "noshift": "learned dynamics, no shifted negatives (M7)",
           "fused": "fused semantic + dynamics + mood (M9)", "hand": "hand-crafted dynamics (cut-on-beat + motion/energy)"}


def query_config(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"paths.{k}={ROOT / k}" for k in ("manifests", "interim", "features", "reports")]))


def build_features(qcfg) -> None:
    """Manifest, prep, shots, extract and segments for the one query pair (each stage skips finished work)."""
    from bgm.data import manifest
    from bgm.data.audit import probe_row
    from bgm.data.prep import interim_video_path, prep_pair
    from bgm.data.segments import run_segment
    from bgm.data.shots import detect_cuts
    from bgm.features.extract import run_extract
    row = probe_row({"pair_id": PAIR, "video_path": VIDEO, "music_path": ORIGINAL, "music_id": MUSIC})
    df = pd.DataFrame([{**row, "track_id": "query_track_1", "meta_source": "query"}])
    manifest.write_pairs(df, qcfg)
    manifest.write_tracks(pd.DataFrame({"track_id": df.track_id, "music_ids": df.music_id, "evidence": "none"}), qcfg)
    info = prep_pair(row, qcfg)
    assert info["offset_s"] == 0.0, info                       # separate music file: no A/V offset applied
    cuts = detect_cuts(interim_video_path(qcfg, PAIR), qcfg.shots.adaptive_threshold)
    manifest.write_shots(pd.DataFrame({"pair_id": [PAIR], "cut_times_s": [" ".join(f"{t:.4f}" for t in cuts)]}), qcfg)
    run_extract(qcfg, "sem_video,sem_music,attr,dyn_video,dyn_music,key", workers=4)
    run_segment(qcfg)


def checkpoints(cfg, kind: str, variant: str) -> list[tuple[int, int, Path]]:
    """(fold, seed, run dir) for runs/<kind>/<variant>/fold*/seed* (variant "" for runs/pipeline)."""
    return [(k, s, Path(cfg.paths.runs) / kind / variant / f"fold{k}" / f"seed{s}") for k in range(cfg.splits.n_folds) for s in cfg.train.seeds]


@torch.no_grad()
def score_dynamics(cfg, qstore, variant: str) -> list:
    from bgm.train.dynamics import DynModel, offset_scorer
    out = []
    for k, s, d in checkpoints(cfg, "dynamics", variant):
        model = DynModel(cfg, "cuda")
        model.load(torch.load(d / "ckpt.pt", map_location="cuda")["state"])
        model.eval()
        out.append(grounding(qstore, [PAIR], offset_scorer(model, qstore), cfg))
    return out


@torch.no_grad()
def score_fused(cfg, store, qstore, queries) -> list:
    from bgm.eval.rescore_grounding import fused_offset
    from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
    out = []
    for k, s, run in checkpoints(cfg, "pipeline", ""):
        w = json.loads((run / "fusion.json").read_text())["w_fine"]
        models = FoldModels.load(cfg, store, k, s, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=fold_ids(cfg, k)["train"])
        index = MusicIndex(qstore, models, [MUSIC])
        vq = pair_query(qstore, PAIR)
        vq.prepare(models, qstore.min_span)
        res = [fused_offset(index, vq.segment(q.a, q.b), q, w) for q in queries]
        out.append(score_predictions(qstore, queries, [p for p, _ in res], cfg, [t for _, t in res]))
        del models, index
        torch.cuda.empty_cache()
    return out


def summarize(results: list, n_boot: int) -> dict:
    """results: per-model GroundingResults over the same segments. Per segment: hits averaged over models."""
    idx = np.arange(len(results[0].err_s))
    out = {"n_models": len(results), "n_segments": int(len(idx)),
           "median_abs_err_s": float(np.median(np.abs(np.concatenate([g.err_s for g in results])))),
           "tied_share": float(np.mean(np.concatenate([g.n_tied > 1 for g in results])))}
    for r in GROUNDING_RULES:
        hit = np.mean([g.hits[r] for g in results], 0)
        ch = results[0].chance[r]
        out[r] = {"accuracy": cluster_bootstrap(hit, idx, n_boot), "chance": cluster_bootstrap(ch, idx, n_boot),
                  "excess": cluster_bootstrap(hit - ch, idx, n_boot),
                  "per_model_accuracy": [float(g.hits[r].mean()) for g in results]}
        out[f"_excess_{r}"] = hit - ch
    return out


def _ci(b: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(b['mean'])} [{f.format(b['lo'])}, {f.format(b['hi'])}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", [])
    qcfg = query_config(cfg)
    build_features(qcfg)
    store, qstore = FeatureStore(cfg), FeatureStore(qcfg)
    queries = grounding_queries(qstore, [PAIR])
    segs = qstore.segments[PAIR]
    log.info("1.mp4: %d segments, %d grounding queries; soundtrack %.1f s", len(segs), len(queries), qstore.duration("music", MUSIC))
    res = {v: score_dynamics(cfg, qstore, v) for v in ("shiftneg", "noshift")}
    res["fused"] = score_fused(cfg, store, qstore, queries)
    from bgm.eval.rescore_grounding import hand_predictions
    hp, ht = hand_predictions(cfg, qstore, queries)
    res["hand"] = [score_predictions(qstore, queries, hp, cfg, ht)]
    for name, gs in res.items():
        for g in gs:
            assert np.array_equal(g.true, np.array([q.true for q in queries])), name
    n_boot = cfg.eval.bootstrap_resamples
    S = {n: summarize(res[n], n_boot) for n in SYSTEMS}
    idx = np.arange(len(queries))
    paired = {f"{a} − hand": {r: cluster_bootstrap(S[a][f"_excess_{r}"] - S["hand"][f"_excess_{r}"], idx, n_boot) for r in GROUNDING_RULES}
              for a in ("shiftneg", "noshift", "fused")}
    per_seg = [{"seg": q.seg_idx, "start_s": round(q.a, 2), "len_s": round(q.b - q.a, 2), "n_offsets": int(len(qstore.dyn("music", MUSIC)) - q.L + 1),
                **{f"chance_{r}": float(res["hand"][0].chance[r][i]) for r in GROUNDING_RULES},
                **{f"{n}_{r}": float(np.mean([g.hits[r][i] for g in res[n]])) for n in SYSTEMS for r in GROUNDING_RULES},
                **{f"{n}_err_s": float(np.median([g.err_s[i] for g in res[n]])) for n in SYSTEMS}} for i, q in enumerate(queries)]
    out = {"n_segments": len(segs), "n_queries": len(queries), "soundtrack_s": qstore.duration("music", MUSIC),
           "systems": {n: {k: v for k, v in S[n].items() if not k.startswith("_")} for n in SYSTEMS}, "paired": paired, "segments": per_seg}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/query_1_grounding.json").write_text(json.dumps(out, indent=1))
    L = ["# 1.mp4: grounding with chance levels", "",
         f"Silent query media/inputs/1.mp4 ({qstore.duration('video', PAIR):.1f} s); its own soundtrack from media/inputs/1_BGM_Ori.mp4 "
         f"(verified aligned: identical frames, audio lag 0). {len(queries)} segments (cut-based, the pipeline's segmentation), each "
         "against every 25 Hz offset of the soundtrack. No training; the soundtrack is in no library file, so all 15 models per "
         "system are leak-free. Hits averaged over the 15 models per segment; 95% CIs resample segments (one video, so they "
         "treat neighbouring segments as independent and are descriptive).", "",
         "| system | rule | accuracy | chance | accuracy − chance (95% CI) | per-model accuracy (min–max) |", "| --- | --- | --- | --- | --- | --- |"]
    for n in SYSTEMS:
        for r in GROUNDING_RULES:
            s = S[n][r]
            pm = s["per_model_accuracy"]
            L.append(f"| {SYSTEMS[n]} | {RULE_LABELS[r]} | {_ci(s['accuracy'])} | {s['chance']['mean']:.3f} | **{_ci(s['excess'], True)}** | "
                     f"{min(pm):.3f}–{max(pm):.3f} |")
    L += ["", "Paired, same segments, on (hit − chance):", ""]
    for name, p in paired.items():
        L.append(f"- {name}: " + "; ".join(f"{RULE_LABELS[r].split(' (')[0]} {_ci(p[r], True)}" for r in GROUNDING_RULES))
    L += ["", "Median absolute error (s): " + ", ".join(f"{n} {S[n]['median_abs_err_s']:.1f}" for n in SYSTEMS)
          + f". Fused queries with tied top offsets: {100 * S['fused']['tied_share']:.0f}%.", ""]
    Path("results/query_1_grounding.md").write_text("\n".join(L) + "\n")
    log.info("wrote results/query_1_grounding.md")


if __name__ == "__main__":
    main()
