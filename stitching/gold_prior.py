"""Extra rows: the transition prior counted from the TRAINING gold paths instead of the BGM cue statistics.

`.venv/bin/python -m stitching.gold_prior --fold k` -> stitching/results/fold<k>_gold.json

On every boundary of the training episodes' gold paths (true window per scene, inserted into the candidates) the
gold transition is a continue, a jump or a switch. With the same parametrisation as the derived HMM:
  P(change | d_t) = 1 - exp(-d_t / L)   -> L by maximum likelihood (grid 0.5 .. 500 s, log-spaced, 2,000 points)
  rho = #jumps / (#jumps + #switches)
Also reported: the raw counts and pi_c, pi_j, pi_s as plain frequencies.
Rows: HMM (lambda = 1, lambda tuned on val) and CRF context + prior (gamma on val, 3 seeds), each with (L, rho) from
the gold paths; everything else identical to evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from stitching.base import relation
from stitching.crf import CRF
from stitching.data import load_fold
from stitching.evaluate import CRF_SEEDS, ensemble_tau
from stitching.hmm_derived import HMMDerived
from stitching.metrics import evaluate_path, fold_entropy

log = logging.getLogger(__name__)
OUT = Path("stitching/results")
GOLD_ROWS = [("hmm_gold_l1", "HMM, π from gold paths, λ = 1"), ("hmm_gold_tuned", "HMM, π from gold paths, λ tuned"),
             ("crf_ctx_prior_gold", "CRF, context + prior from gold paths")]


def gold_counts(train) -> dict:
    d, kind = [], []
    for e in train:
        g = e.with_gold()
        for t in range(1, g.T):
            c, j, s = relation(g, t)
            a, b = g.gold[t - 1], g.gold[t]
            kind.append(0 if c[a, b] else (1 if j[a, b] else 2))
            d.append(g.scene_dur[t])
    d, kind = np.array(d), np.array(kind)
    ch = kind > 0
    Ls = np.exp(np.linspace(np.log(0.5), np.log(500), 2000))
    p = 1 - np.exp(-d[None, :] / Ls[:, None])
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = (np.log(p) * ch + np.log(1 - p) * ~ch).sum(1)
    L = float(Ls[np.argmax(ll)])
    n = np.bincount(kind, minlength=3)
    rho = float(n[1] / max(n[1] + n[2], 1))
    return {"L_s": L, "rho": rho, "n_continue": int(n[0]), "n_jump": int(n[1]), "n_switch": int(n[2]),
            "pi_c": float(n[0] / n.sum()), "pi_j": float(n[1] / n.sum()), "pi_s": float(n[2] / n.sum()),
            "mean_scene_dur_s": float(d.mean()), "n_boundaries": int(len(d))}


def main(k: int, device: str) -> Path:
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    t0 = time.time()
    cfg = load_config("configs/default.yaml", CUE_OVERRIDES)
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    tau = ensemble_tau(cfg, k)
    gc = gold_counts(tr)
    L, rho = gc["L_s"], gc["rho"]
    log.info("fold %d gold-path prior: %s", k, gc)
    gal = F["gallery_test"]
    out = {"fold": k, "gold_prior": gc, "tau": tau, "methods": {}}

    def run(key, st, seed=None):
        t1 = time.time()
        params = st.fit(tr, va)
        paths = [st.timed(e) for e in te]
        rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
        log.info("fold %d %-18s seed %s | test acc %.3f F1 %.3f | %s", k, key, seed, np.mean([r["track_acc"] for r in rows]),
                 np.mean([r["switch_f1"] for r in rows]), {a: b for a, b in params.items() if a in ("lam", "gamma", "alpha")})
        return {"name": dict(GOLD_ROWS)[key], "params": params, "fit_s": time.time() - t1, "rows": rows,
                "entropy": fold_entropy(te, paths, gal), "seed": seed}

    out["methods"]["hmm_gold_l1"] = run("hmm_gold_l1", HMMDerived(tau, L, rho, 1.0))
    out["methods"]["hmm_gold_tuned"] = run("hmm_gold_tuned", HMMDerived(tau, L, rho, tune_lam=True))
    dev = device if torch.cuda.is_available() else "cpu"
    recs = [run("crf_ctx_prior_gold", CRF(tau, L, rho, True, (0.01, 0.1, 1.0), seed=s, device=dev, name="crf_ctx_prior_gold"), s)
            for s in CRF_SEEDS]
    out["methods"]["crf_ctx_prior_gold"] = {"name": dict(GOLD_ROWS)["crf_ctx_prior_gold"], "seeds": recs}
    p = OUT / f"fold{k}_gold.json"
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
