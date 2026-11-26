"""Fit every method on (train, val) of one fold, decode the fold's test episodes once, save everything.

`.venv/bin/python -m stitching.evaluate --fold k [--device cuda]` -> stitching/results/fold<k>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from stitching.crf import CRF
from stitching.data import load_fold
from stitching.dtw_single import PretetDTW
from stitching.greedy import Greedy
from stitching.hmm_derived import HandSet, HMMDerived
from stitching.metrics import evaluate_path, fold_entropy
from stitching.smooth import Smooth
from stitching.two_stage import TwoStage

log = logging.getLogger(__name__)
OUT = Path("stitching/results")
ROWS = [("greedy", "Greedy top-1"), ("smooth", "Greedy + smoothing"), ("two_stage", "Two-stage change-point + retrieve"),
        ("dtw", "Prétet-style DTW (one track)"), ("handset", "Viterbi, hand-set costs (existing)"),
        ("hmm_l1", "HMM, derived costs, λ = 1"), ("hmm_tuned", "HMM, derived costs, λ tuned"),
        ("crf_noctx", "CRF, no context"), ("crf_ctx", "CRF, with context"), ("crf_ctx_prior", "CRF, with context + derived prior")]
CRF_SEEDS = (0, 1, 2)
TIMELINE = ("greedy", "dtw", "hmm_tuned", "crf_ctx_prior")


def ensemble_tau(cfg, k: int) -> float:
    taus = []
    for s in cfg.pipeline.sem_seeds:
        ck = torch.load(Path(cfg.paths.runs) / "semantic" / cfg.pipeline.sem_variant / f"fold{k}" / f"seed{s}" / "ckpt.pt",
                        map_location="cpu", weights_only=False)
        taus.append(float(torch.exp(ck["state"]["log_tau"])))
    return float(np.mean(taus))


def gold_stats(eps) -> dict:
    """How often the true window is a candidate, and the true window's similarity percentile among the candidates
    (train vs val vs test: shows how much the frozen heads favour their own training clips)."""
    inc, pct = [], []
    for e in eps:
        for t, c in enumerate(e.cands):
            inc.append(e.gold[t] >= 0)
            pct.append(float((c.sim < e.gold_win[t]["sim"]).mean()))
    return {"gold_in_candidates": float(np.mean(inc)), "gold_sim_percentile": float(np.mean(pct)), "n_scenes": len(inc)}


def main(k: int, device: str) -> Path:
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    t0 = time.time()
    cfg = load_config("configs/default.yaml", CUE_OVERRIDES)
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    cs = json.loads((OUT / "cue_stats.json").read_text())["folds"][str(k)]
    L, rho = float(cs["L_s"]), float(cs["rho"])
    tau = ensemble_tau(cfg, k)
    mu = float(F["d75"]["penalties"]["mu"])
    out = {"fold": k, "tau": tau, "L_s": L, "rho": rho, "theta_segmenter": F["theta"], "gallery": {s: F[f"gallery_{s}"] for s in ("train", "val", "test")},
           "n_episodes": {"train": len(tr), "val": len(va), "test": len(te)},
           "gold_stats": {s: gold_stats(e) for s, e in (("train", tr), ("val", va), ("test", te))},
           "shortlist": {"track_R@10": float(np.mean(np.concatenate([e.shortlist_rank for e in te]) <= 10)),
                         "track_MRR": float(np.mean(1.0 / np.concatenate([e.shortlist_rank for e in te])))},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "methods": {}, "timelines": {},
           "test_meta": {str(i): {"segs": e.segs.tolist(), "true_sw": list(e.true_sw), "truth_files": [e.files[g["mid"]] for g in e.gold_win]}
                         for i, e in enumerate(te)}}
    gal = F["gallery_test"]

    def run(key, st, seed=None):
        t1 = time.time()
        params = st.fit(tr, va)
        fit_s = time.time() - t1
        paths = [st.timed(e) for e in te]
        rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
        rec = {"name": ROWS[[r[0] for r in ROWS].index(key)][1], "params": params, "fit_s": fit_s, "rows": rows,
               "entropy": fold_entropy(te, paths, gal), "seed": seed}
        m = {x: float(np.nanmean([r[x] for r in rows])) for x in ("track_acc", "switch_f1", "TBIoU", "switches_per_min")}
        log.info("fold %d %-14s seed %s fit %.0f s | test %s | %s", k, key, seed, fit_s, {a: round(b, 3) for a, b in m.items()},
                 {a: b for a, b in params.items() if a not in ("history", "val_objective", "base_w", "val_nll_by_gamma")})
        if seed in (None, 0) and key in TIMELINE:
            out["timelines"][key] = [[e.files[m] for m in p.mid] for e, p in zip(te, paths)]
        return rec, st

    for key, st in (("greedy", Greedy()), ("smooth", Smooth()), ("two_stage", TwoStage()), ("dtw", PretetDTW()),
                    ("handset", HandSet(cfg, mu)), ("hmm_l1", HMMDerived(tau, L, rho, 1.0)),
                    ("hmm_tuned", HMMDerived(tau, L, rho, tune_lam=True))):
        out["methods"][key], _ = run(key, st)
    # reference check: the hand-set row must reproduce D143's combo/own R0_tuned (same lattices, same tuning)
    ref = json.loads(Path(f"runs/titlesplit/seg_retrieval/fold{k}/eval.json").read_text())["arms"]["combo/own"]["systems"]["R0_tuned"]
    mine = out["methods"]["handset"]
    out["reference_check_handset_vs_D143"] = {
        "params_equal": {a: b for a, b in mine["params"]["penalties"].items()} == ref["params"],
        "max_abs_diff_objective": float(np.max(np.abs(np.array([r["objective"] for r in mine["rows"]]) - np.array([r["objective"] for r in ref["rows"]]))))}
    log.info("fold %d reference check: %s", k, out["reference_check_handset_vs_D143"])

    dev = device if torch.cuda.is_available() else "cpu"
    out["crf_val"] = {}
    for key, ctx, gam in (("crf_noctx", False, (0.0,)), ("crf_ctx", True, (0.0,)), ("crf_ctx_prior", True, (0.01, 0.1, 1.0))):
        recs = []
        for s in CRF_SEEDS:
            rec, st = run(key, CRF(tau, L, rho, ctx, gam, seed=s, device=dev, name=key), s)
            recs.append(rec)
            if s == 0:     # shortcut guard + plot data on val
                vp = [st.decode(e) for e in va]
                vr = [evaluate_path(e, p) for e, p in zip(va, vp)]
                marg = [st.switch_marginals(e) for e in va]
                from stitching.crf import ctx_features
                sig = [np.array([ctx_features(e, t)[0] for t in range(1, e.T)]) for e in va]
                lab = [np.array([any(abs(b - x) < 1e-6 for x in e.true_sw) for b in e.btimes], float) for e in va]
                out["crf_val"][key] = {"pred_switches_per_min": float(np.mean([r["switches_per_min"] for r in vr])),
                                       "true_switches_per_min": float(np.mean([r["true_switches_per_min"] for r in vr])),
                                       "pred_switch_rate_per_boundary": float(np.mean(np.concatenate([np.diff(p.mid) != 0 for p in vp if len(p.mid) > 1]))),
                                       "mean_posterior_switch_prob": float(np.mean(np.concatenate(marg))),
                                       "true_change_rate_per_boundary": float(np.mean(np.concatenate(lab))),
                                       "derived_p_change_mean": float(np.mean(np.concatenate([1 - np.exp(-e.scene_dur[1:] / L) for e in va]))),
                                       "marginals": np.concatenate(marg).tolist(), "siglip_dist": np.concatenate(sig).tolist(),
                                       "label": np.concatenate(lab).tolist()}
                log.info("fold %d %s val switch guard: %s", k, key, {a: round(b, 3) for a, b in out["crf_val"][key].items() if not isinstance(b, list)})
        out["methods"][key] = {"name": recs[0]["name"], "seeds": recs}
    p = OUT / f"fold{k}.json"
    p.write_text(json.dumps(out, default=float))
    log.info("fold %d done -> %s [%.0f s]", k, p, time.time() - t0)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.fold, a.device)
