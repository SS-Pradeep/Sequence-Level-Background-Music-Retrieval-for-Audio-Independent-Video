"""Paired Wilcoxon signed-rank helper (D112)."""
import numpy as np
from scipy.stats import wilcoxon

from bgm.analysis.wilcoxon import f1_counts, holm, paired_wilcoxon


def test_matches_scipy_and_drops_nan_and_ties():
    rng = np.random.default_rng(0)
    a, b = rng.random(40), rng.random(40)
    a[3] = np.nan
    b[5] = a[5]                                            # a tie: dropped
    r = paired_wilcoxon(a, b)
    d = (a - b)[np.isfinite(a)]
    d = d[d != 0]
    assert r["n"] == 39 and r["n_nonzero"] == 38
    assert np.isclose(r["p"], wilcoxon(d).pvalue) and np.isclose(r["W"], wilcoxon(d).statistic)


def test_effect_size_and_direction():
    a = np.arange(1, 21, dtype=float)
    r = paired_wilcoxon(a, np.zeros(20))
    assert r["r"] == 1.0 and r["wins"] == 20 and r["p"] < 1e-4
    assert paired_wilcoxon(np.zeros(20), a)["r"] == -1.0
    assert paired_wilcoxon(a, a)["p"] == 1.0


def test_clusters_average_first():
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 2.0])
    r = paired_wilcoxon(a, b, np.array(["x", "x", "x", "y"]))
    assert r["n"] == 2 and r["wins"] == 1 and r["losses"] == 1 and np.isclose(r["mean"], (1 - 2) / 2)


def test_holm_and_f1():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    f = f1_counts(np.array([[1, 1, 0], [0, 0, 0]]))
    assert np.isclose(f[0], 2 / 3) and np.isnan(f[1])
