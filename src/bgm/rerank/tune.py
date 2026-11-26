"""Grid search of the re-scoring parameters on a fold's validation split only (D140).

Objective: val segment MRR averaged over the 3 ensembles (D136's rule: val splits hold ~100 segments, too few for
R@10 to separate nearby settings); the first grid value wins ties.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgm.rerank.evaluate import m2v_ranks, v2m_ranks
from bgm.rerank.normalize import dsl, inverted_softmax, mutual_nn, qbnorm

TS = (0.01, 0.02, 0.05, 0.1, 0.2)
BETAS = (0.5, 1.0)
MS = (5, 10)
ETAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
QB_K = 1

GRIDS: dict[str, list[tuple]] = {
    "base": [()],
    "dsl": [(T,) for T in TS],
    "is": [(T, b) for T in TS for b in BETAS],
    "qb": [(T, b) for T in TS for b in BETAS],
    "mnn": [(m, eta) for m in MS for eta in ETAS],
}


@dataclass(frozen=True)
class Mats:
    """One ensemble's masked matrices for one split: v2m [Q, Q], m2v [Q, Q], bank [B, Q]."""
    v2m: np.ndarray
    m2v: np.ndarray
    bank: np.ndarray


def masked(z, s: str, e: int) -> Mats:
    """Mats of ensemble e for split s from a scores_fold<k>.npz; same-track pairs set to -inf in both directions."""
    m = z[f"mask_{s}"]
    return Mats(np.where(m, -np.inf, z[f"v2m_{s}"][e].astype(np.float64)),
                np.where(m, -np.inf, z[f"m2v_{s}"][e].astype(np.float64)), z[f"bank_{s}"][e].astype(np.float64))


def rescore(method: str, p: tuple, x: Mats) -> np.ndarray:
    """[Q, Q] V2M scores of a method; M2V uses the columns of the same matrix (m2v_scores)."""
    if method == "base":
        return x.v2m
    if method == "dsl":
        return dsl(x.v2m, x.m2v, p[0])
    if method == "is":
        return inverted_softmax(x.v2m, x.bank, *p)
    if method == "qb":
        return qbnorm(x.v2m, x.bank, *p, k=QB_K)
    if method == "mnn":
        return mutual_nn(x.v2m, x.m2v, *p)
    raise ValueError(method)


def m2v_scores(method: str, p: tuple, x: Mats) -> np.ndarray:
    """[Q, Q] scores ranked by columns for M2V: the plain M2V matrix, or DSL (same formula, read by columns)."""
    return x.m2v if method == "base" else dsl(x.v2m, x.m2v, p[0])


def mrr(r: np.ndarray) -> float:
    return float((1.0 / np.asarray(r, float)).mean())


def choose(method: str, val: list[Mats], direction: str = "v2m") -> tuple[tuple, list[float]]:
    """(best parameters, val MRR per grid point) for one fold."""
    grid = GRIDS[method]
    if direction == "v2m":
        curve = [float(np.mean([mrr(v2m_ranks(rescore(method, p, x))) for x in val])) for p in grid]
    else:
        curve = [float(np.mean([mrr(m2v_ranks(m2v_scores(method, p, x))) for x in val])) for p in grid]
    return grid[int(np.argmax(curve))], curve
