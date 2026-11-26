"""Ranks and recall metrics for both directions on a masked square score matrix (D140).

S [Q, Q]: row i = scene i, column j = window j; the correct pair is the diagonal; masked entries are -inf.
"""
from __future__ import annotations

import numpy as np

KS = (1, 3, 5, 10)


def v2m_ranks(S: np.ndarray) -> np.ndarray:
    """[Q] 1-based rank of window i in row i: 1 + unmasked windows scored strictly above it (rank_of_first_hit rule)."""
    S = np.asarray(S, np.float64)
    d = np.diag(S)
    assert np.isfinite(d).all(), "a correct pair is masked or non-finite"
    above = S > d[:, None]
    np.fill_diagonal(above, False)
    return 1 + above.sum(1)


def m2v_ranks(S: np.ndarray) -> np.ndarray:
    """[Q] 1-based rank of scene j in column j (window j as the query, the fold's scenes as the gallery)."""
    return v2m_ranks(np.asarray(S).T)


def summary(ranks: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks, float)
    return {**{f"R@{k}": float((r <= k).mean()) for k in KS}, "MedR": float(np.median(r)), "MRR": float((1.0 / r).mean())}
