"""`bgm fuse` (SPEC M9): fusion weights per fold and seed on val, then test pipeline accuracy.

Per (fold, seed), with SigLIP + CLAP+MERT semantic heads, the shift-negative dynamics encoders and the fold's
attribute calibration (all fit on train only):
  1. coarse weights (sem, attr), 11-point grid, maximizing val track R@10 (val gallery);
  2. fine weights (sem, dyn, attr), 66-point grid, maximizing val pipeline window@1 (windows of the shortlist);
  3. test pipeline accuracy for each branch alone, fused, and fused without the coarse stage.
Outputs: runs/pipeline/fold<k>/seed<s>/{fusion.json, metrics.json, perquery.npz, config.yaml, ...}.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
from omegaconf import DictConfig

from bgm.eval.metrics import rank_of_first_hit
from bgm.eval.store import FeatureStore, fold_ids
from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
from bgm.retrieval.pipeline import fuse, rule_chance, run_video, segment_hits, segment_offset_rules, top1_hits, zstats
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)


def weight_grid(branches, step: float = 0.1) -> list[dict]:
    n = int(round(1 / step))
    out = []

    def rec(prefix, left, k):
        if k == 1:
            out.append(prefix + [left])
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k - 1)
    rec([], n, len(branches))
    return [{b: v / n for b, v in zip(branches, ws)} for ws in out]


def prepared_queries(store, ids, models):
    out = {}
    for p in ids:
        vq = pair_query(store, p)
        vq.prepare(models, store.min_span)
        out[p] = vq
    return out


def coarse_grid(step: float, attr_weight: float | None) -> list[dict]:
    """Coarse-stage grid over (sem, attr). With `fusion.attr_weight` set (D56: mood fusion null, pinned to 0,
    so the shortlist is semantic-only), there is nothing to tune; null tunes attr on val as before."""
    if attr_weight is None:
        return weight_grid(["sem", "attr"], step)
    return [{"sem": 1.0 - attr_weight, "attr": float(attr_weight)}]


def tune_coarse(store, index: MusicIndex, queries: dict, step: float, attr_weight: float | None = None) -> tuple[dict, float]:
    gtrack = np.array([store.music_track[m] for m in index.ids])
    raws = []
    for p, vq in queries.items():
        gv, pv = vq.pooled(0.0, vq.duration)
        raw = index.file_scores(gv, pv)
        raws.append((raw, zstats(raw), gtrack == store.track[p]))
    best, best_w = -1.0, None
    for w in coarse_grid(step, attr_weight):
        r10 = np.mean([rank_of_first_hit(fuse(raw, st, w), hits) <= 10 for raw, st, hits in raws])
        if r10 > best:
            best, best_w = float(r10), w
    return best_w, best


def fine_grid(step: float, dyn_weight: float | None, attr_weight: float | None = None) -> list[dict]:
    """Fine-stage weight grid. Pinned branches (`fusion.dyn_weight`, D54: dynamics at chance;
    `fusion.attr_weight`, D56: mood fusion null) are fixed and the free branches share the remaining
    mass; with both pinned the grid is the single semantic-only point. null tunes a branch as before."""
    pinned = {b: float(w) for b, w in (("dyn", dyn_weight), ("attr", attr_weight)) if w is not None}
    free = [b for b in ("sem", "dyn", "attr") if b not in pinned]
    rest = 1.0 - sum(pinned.values())
    return [{**{b: w[b] * rest for b in free}, **pinned} for w in weight_grid(free, step)]


def tune_fine(store, index, queries, results, J, step, dyn_weight: float | None = None,
              attr_weight: float | None = None) -> tuple[dict, dict]:
    best, best_w = None, None
    for w in fine_grid(step, dyn_weight, attr_weight):
        h = [segment_hits(index, store, queries[p], r, w, J) for p, rs in results.items() for r in rs]
        key = (np.mean([x["window@1"] for x in h]), np.mean([x["track@1"] for x in h]))
        if best is None or key > best:
            best, best_w = key, w
    return best_w, {"window@1": float(best[0]), "track@1": float(best[1])}


def fuse_one(cfg: DictConfig, store: FeatureStore, fold: int, seed: int) -> dict:
    t0 = time.time()
    r = cfg.retrieval
    K, J, hop, step = r.coarse_top_k, r.fine_top_j, r.window_hop_s, cfg.fusion.grid_step
    f = fold_ids(cfg, fold)
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    assert models.dyn is not None or cfg.fusion.get("dyn_weight", None) == 0, \
        "pipeline.dyn_variant=null needs fusion.dyn_weight=0: without a dynamics model that branch scores 0 everywhere"
    # ---- val: weights
    v_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["val"])))
    v_q = prepared_queries(store, f["val"], models)
    w_coarse, val_r10 = tune_coarse(store, v_index, v_q, step, cfg.fusion.get("attr_weight", None))
    v_res = {p: run_video(v_index, vq, w_coarse, K, hop) for p, vq in v_q.items()}
    w_fine, val_fine = tune_fine(store, v_index, v_q, v_res, J, step, cfg.fusion.get("dyn_weight", None),
                                 cfg.fusion.get("attr_weight", None))
    fusion = {"w_coarse": w_coarse, "w_fine": w_fine, "K": K, "J": J, "window_hop_s": hop,
              "val": {"track_R@10": val_r10, **val_fine}}
    # ---- test: pipeline accuracy
    t_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    t_q = prepared_queries(store, f["test"], models)
    systems = {"sem": {"sem": 1.0}, "dyn": {"dyn": 1.0}, "attr": {"attr": 1.0}, "fused": w_fine}
    if models.dyn is None:
        del systems["dyn"]
    per = {f"{s}/{m}": [] for s in list(systems) + ["fused_no_coarse"] for m in ("track@1", "window@1", "window@J")}
    per["dur"], per["clusters"], per["pair"] = [], [], []
    gtrack = np.array([store.music_track[m] for m in t_index.ids])
    off = {}                                             # offset precision (D60): tie-shared top-1 hits and chance
    for p, vq in t_q.items():
        res = run_video(t_index, vq, w_coarse, K, hop)
        res_nc = run_video(t_index, vq, w_coarse, K, hop, use_coarse=False)
        for rr, rn in zip(res, res_nc):
            for s, w in systems.items():
                h = segment_hits(t_index, store, vq, rr, w, J)
                for m in ("track@1", "window@1", "window@J"):
                    per[f"{s}/{m}"].append(h[m])
            h = segment_hits(t_index, store, vq, rn, w_fine, J)
            for m in ("track@1", "window@1", "window@J"):
                per[f"fused_no_coarse/{m}"].append(h[m])
            # rn holds every window of every gallery file: the set a random choice draws from
            all_rules = segment_offset_rules(t_index, store, vq, rn, gtrack)
            rows = {"offset/fused": top1_hits(segment_offset_rules(t_index, store, vq, rr, gtrack), fuse(rr.raw, rr.stats, w_fine)),
                    "offset/fused_no_coarse": top1_hits(all_rules, fuse(rn.raw, rn.stats, w_fine)),
                    "offset/chance": rule_chance(all_rules)}
            for name, row in rows.items():
                for k, v in row.items():
                    off.setdefault(f"{name}/{k}", []).append(v)
            per["dur"].append(rr.b - rr.a), per["clusters"].append(store.track[p]), per["pair"].append(p)
    per = {k: np.array(v) for k, v in (per | off).items()}
    test = {s: {m: float(per[f"{s}/{m}"].mean()) for m in ("track@1", "window@1", "window@J")} for s in list(systems) + ["fused_no_coarse"]}
    test["offset"] = {k[len("offset/"):]: float(np.nanmean(v)) for k, v in per.items() if k.startswith("offset/")}
    d = run_dir(cfg, cfg.pipeline.exp, fold, seed)
    (d / "fusion.json").write_text(json.dumps(fusion, indent=1))
    np.savez(d / "perquery.npz", **per)
    write_run(d, cfg, {"fusion": fusion, "test": test, "n_segments": int(len(per["dur"])), "seconds": time.time() - t0}, seed)
    log.info("fuse fold %d seed %d: coarse %s fine %s | val window@1 %.3f | test window@1 fused %.3f (no coarse %.3f), track@1 %.3f  [%.0f s]",
             fold, seed, w_coarse, w_fine, val_fine["window@1"], test["fused"]["window@1"], test["fused_no_coarse"]["window@1"],
             test["fused"]["track@1"], time.time() - t0)
    return {"fusion": fusion, "test": test}


def run_fuse(cfg: DictConfig, folds=None, seeds=None) -> list[dict]:
    store = FeatureStore(cfg)
    return [fuse_one(cfg, store, k, s) for k in (range(cfg.splits.n_folds) if folds is None else folds)
            for s in (list(cfg.train.seeds) if seeds is None else seeds)]
