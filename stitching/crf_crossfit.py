"""Phase 1, step 5: CRF rows 8, 9, 10, 13 retrained on the cross-fitted training episodes (3 seeds each).
Val (early stopping, gamma) and test episodes, L, rho, tau, features and recipe are exactly those of the original rows.
Row 13's (L, rho) are counted on the cross-fitted gold paths.
`.venv/bin/python -m stitching.crf_crossfit --fold k [--check-only]` -> stitching/results/fold<k>_xf.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from stitching.crf import CRF, ctx_features
from stitching.crossfit import load as load_xf
from stitching.data import load_fold
from stitching.evaluate import CRF_SEEDS, ensemble_tau, gold_stats
from stitching.gold_prior import gold_counts
from stitching.metrics import evaluate_path, fold_entropy

log = logging.getLogger(__name__)
OUT = Path("stitching/results")
XF_ROWS = [("crf_noctx_xf", "CRF, no context, cross-fitted"), ("crf_ctx_xf", "CRF, with context, cross-fitted"),
           ("crf_ctx_prior_xf", "CRF, with context + derived prior, cross-fitted"),
           ("crf_ctx_prior_gold_xf", "CRF, context + prior from gold paths, cross-fitted")]


def main(k: int, check_only: bool, device: str) -> Path:
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    t0 = time.time()
    F = load_fold(k)
    va, te = F["episodes"]["val"], F["episodes"]["test"]
    tr = load_xf(k)["episodes"]
    out = {"fold": k, "gold_stats": {"xtrain": gold_stats(tr), "train_insample": gold_stats(F["episodes"]["train"]),
                                     "val": gold_stats(va), "test": gold_stats(te)}, "methods": {}, "crf_val": {}}
    g = out["gold_stats"]
    ratio = g["xtrain"]["gold_in_candidates"] / max(g["test"]["gold_in_candidates"], 1e-9)
    out["xtrain_over_test_ratio"] = ratio
    log.info("fold %d gold-in-candidates: cross-fitted train %.3f, in-sample train %.3f, val %.3f, test %.3f (ratio %.2f)", k,
             g["xtrain"]["gold_in_candidates"], g["train_insample"]["gold_in_candidates"], g["val"]["gold_in_candidates"],
             g["test"]["gold_in_candidates"], ratio)
    path = OUT / f"fold{k}_xf.json"
    if check_only:
        path.write_text(json.dumps(out, default=float))
        return path
    cs = json.loads((OUT / "cue_stats.json").read_text())["folds"][str(k)]
    tau = ensemble_tau(load_config("configs/default.yaml", CUE_OVERRIDES), k)
    gc = gold_counts(tr)
    out["gold_prior_xf"] = gc
    dev = device if torch.cuda.is_available() else "cpu"
    specs = (("crf_noctx_xf", False, (0.0,), cs["L_s"], cs["rho"]), ("crf_ctx_xf", True, (0.0,), cs["L_s"], cs["rho"]),
             ("crf_ctx_prior_xf", True, (0.01, 0.1, 1.0), cs["L_s"], cs["rho"]),
             ("crf_ctx_prior_gold_xf", True, (0.01, 0.1, 1.0), gc["L_s"], gc["rho"]))
    for key, ctx, gam, L, rho in specs:
        recs = []
        for s in CRF_SEEDS:
            st = CRF(tau, float(L), float(rho), ctx, gam, seed=s, device=dev, name=key)
            params = st.fit(tr, va)
            paths = [st.timed(e) for e in te]
            rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
            recs.append({"name": dict(XF_ROWS)[key], "params": params, "rows": rows, "entropy": fold_entropy(te, paths, F["gallery_test"]), "seed": s})
            log.info("fold %d %-22s seed %d | test acc %.3f F1 %.3f | alpha %.2f lam %.2f gamma %s", k, key, s,
                     np.mean([r["track_acc"] for r in rows]), np.mean([r["switch_f1"] for r in rows]), params["alpha"], params["lam"], params["gamma"])
            if s == 0:
                vp = [st.decode(e) for e in va]
                vr = [evaluate_path(e, p) for e, p in zip(va, vp)]
                marg = [st.switch_marginals(e) for e in va]
                sig = [np.array([ctx_features(e, t)[0] for t in range(1, e.T)]) for e in va]
                lab = [np.array([any(abs(b - x) < 1e-6 for x in e.true_sw) for b in e.btimes], float) for e in va]
                out["crf_val"][key] = {"pred_switches_per_min": float(np.mean([r["switches_per_min"] for r in vr])),
                                       "true_switches_per_min": float(np.mean([r["true_switches_per_min"] for r in vr])),
                                       "mean_posterior_switch_prob": float(np.mean(np.concatenate(marg))),
                                       "true_change_rate_per_boundary": float(np.mean(np.concatenate(lab))),
                                       "marginals": np.concatenate(marg).tolist(), "siglip_dist": np.concatenate(sig).tolist(),
                                       "label": np.concatenate(lab).tolist()}
        out["methods"][key] = {"name": dict(XF_ROWS)[key], "seeds": recs}
        path.write_text(json.dumps(out, default=float))
    log.info("fold %d done [%.0f s]", k, time.time() - t0)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.fold, a.check_only, a.device)
