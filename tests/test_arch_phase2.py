"""Phase 2 envelope-test scorers (src/bgm/analysis/arch_phase2.py) on synthetic curves."""
import numpy as np
import pytest

from bgm.analysis.arch_phase2 import _rank, lag_corr, pearson, to_1hz, window_corr, zscore


def test_to_1hz_means_whole_seconds():
    x = np.arange(60, dtype=float)                       # 2.4 s at 25 Hz
    out = to_1hz(x)
    assert out.shape == (2,)
    assert np.allclose(out, [np.mean(range(25)), np.mean(range(25, 50))])


def test_pearson_constant_is_zero():
    assert pearson(np.ones(10), np.arange(10.0)) == 0.0
    assert pearson(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)


def test_lag_corr_recovers_small_lag():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(60)
    v = m[2:40]                                          # video = music shifted by +2 s
    assert lag_corr(v, m) == pytest.approx(1.0)
    assert lag_corr(v, m, max_lag=1) < 0.99              # outside the tolerance the match is gone
    assert lag_corr(v[:4], m) == -1.0                    # below min_overlap


def test_window_corr_finds_embedded_window():
    rng = np.random.default_rng(1)
    m = rng.standard_normal(120)
    v = m[57:92]
    assert window_corr(v, m) == pytest.approx(1.0)
    assert lag_corr(v, m) < 0.9                          # the aligned scorer must NOT find a window 57 s in
    brute = max(pearson(v, m[s : s + len(v)]) for s in range(len(m) - len(v) + 1))
    w = window_corr(rng.standard_normal(35), m)          # vectorised scan == brute force on a non-planted query
    assert w <= 1.0 and window_corr(v, m) >= brute - 1e-9


def test_rank_shares_ties():
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    relevant = np.array([True, False, False, False])
    assert _rank(scores, relevant) == 2.5                # one above, tied with one other -> (2 + 3) / 2
    assert _rank(np.array([0.9, 0.5]), np.array([True, False])) == 1.0


def test_zscore():
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
