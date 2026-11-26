"""Loss-function ablation of the semantic heads on the cleaned title split (PROGRESS.md D124).

python -m bgm.analysis.loss_compare   -> results/cleaned/loss_compare.md

Each variant differs from runs/cleaned/semantic/siglip-both only in train.semantic.loss. Rows: single heads (seeds
0-9, hits averaged over seeds per query) and the 10-head ensemble (run seed 0 = members 0-9, the shipped D75 rule).
Deltas are paired per query against siglip-both with the same seeds / members; 95% film-group cluster bootstrap,
10,000 resamples. Same test queries in every run (asserted via the cluster arrays).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bgm.eval.metrics import cluster_bootstrap

log = logging.getLogger(__name__)
ROOT = Path("runs/cleaned")
OUT = Path("results/cleaned/loss_compare.md")
LOSSES = {"siglip-both": "InfoNCE (shipped)", "siglip-both-sigmoid": "Sigmoid (SigLIP)", "siglip-both-hnnce": "Hard-negative InfoNCE (β 0.5)",
          "siglip-both-dcl": "Decoupled contrastive (DCL)", "siglip-both-vsemax": "VSE++ max-violation hinge (m 0.2)"}
SECTIONS = {"track_test_fold": "Track R@10", "segment_all": "Segment R@10"}
STATS = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
SEEDS = range(10)


def _load(kind: str, variant: str, seeds) -> dict:
    """{section: (ranks [n_seeds, n_queries], clusters [n_queries])}, test queries of all 5 folds concatenated."""
    out = {}
    for sec in SECTIONS:
        R, C = [], []
        for k in range(5):
            rk, ck = [], None
            for s in seeds:
                with np.load(ROOT / kind / variant / f"fold{k}" / f"seed{s}" / "perquery.npz") as z:
                    rk.append(z[sec])
                    ck = z[f"{sec}/clusters"] if ck is None else ck
                    assert np.array_equal(ck, z[f"{sec}/clusters"])
            R.append(np.stack(rk))
            C.append(ck)
        out[sec] = (np.concatenate(R, 1), np.concatenate(C))
    return out


def _cells(new: dict, base: dict | None, sec: str) -> list[str]:
    r, cl = new[sec]
    cells = [f"{STATS[m](r).mean():.3f}" for m in STATS] + [f"{np.median(r.mean(0)):.1f}"]
    if base is None:
        return cells + ["—"]
    rb, clb = base[sec]
    assert np.array_equal(cl, clb)
    d = cluster_bootstrap(STATS["R@10"](r).mean(0) - STATS["R@10"](rb).mean(0), cl)
    sig = "" if d["lo"] <= 0 <= d["hi"] else " *"
    return cells + [f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{sig}"]


def _val(variant: str) -> str:
    import json
    v = [json.load(open(ROOT / "semantic" / variant / f"fold{k}" / f"seed{s}" / "metrics.json"))["val"]["segment_R@10"]
         for k in range(5) for s in SEEDS]
    return f"{np.mean(v):.3f}"


def write_report() -> Path:
    single = {v: _load("semantic", v, SEEDS) for v in LOSSES}
    ens = {v: _load("semantic_ens", f"{v}-ens10", [0]) for v in LOSSES}
    L = ["# Loss functions for the semantic heads (D124)", "",
         "Cleaned dataset, title split, 894 test clips (2,117 segments). Everything except `train.semantic.loss` is the",
         "shipped D75 recipe (siglip-both, CLAP+MERT, film cap 30, early stopping on val segment R@10). Single = seeds 0–9",
         "per fold, hits averaged over seeds per query; ens10 = members 0–9 (the shipped system's rule). Δ R@10 is paired",
         "per query vs InfoNCE with the same seeds; 95% film-group cluster bootstrap; * = CI clear of 0.", ""]
    for sec, title in SECTIONS.items():
        for name, rows, base_key in (("single heads, 10 seeds × 5 folds", single, "siglip-both"), ("10-head ensemble", ens, "siglip-both")):
            L += [f"## {title.split()[0]} retrieval, {name}", "",
                  "| Loss | R@1 | R@10 | MRR | Median rank | Δ R@10 vs InfoNCE [95% CI] |" + (" Val seg R@10 |" if rows is single and sec == "segment_all" else ""),
                  "|---|---|---|---|---|---|" + ("---|" if rows is single and sec == "segment_all" else "")]
            for v, label in LOSSES.items():
                cells = _cells(rows[v], None if v == base_key else rows[base_key], sec)
                if rows is single and sec == "segment_all":
                    cells.append(_val(v))
                L.append(f"| {label} | " + " | ".join(cells) + " |")
            L.append("")
    OUT.write_text("\n".join(L) + "\n")
    log.info("-> %s", OUT)
    return OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    write_report()
