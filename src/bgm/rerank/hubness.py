"""k-occurrence hubness diagnostic (D140): how often each gallery window appears in some query's top k."""
from __future__ import annotations

import numpy as np

from bgm.analysis.hubness import skewness, top10_counts


def k_occurrence(S: np.ndarray, k: int = 10) -> np.ndarray:
    """[G] N_k: number of queries with window c in their top k (masked -inf entries sort last)."""
    return top10_counts(np.asarray(S, np.float64), k)


def hub_stats(S: np.ndarray, k: int = 10) -> dict[str, float]:
    n = k_occurrence(S, k)
    return {"N10_skew": skewness(n), "N10_max": float(n.max())}
