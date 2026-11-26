"""Chunked querying and score ensembles."""
import numpy as np
import pytest

from bgm.retrieval.chunked import chunked_track_scores, ensemble, equal_chunks, window_starts, zscore


def test_equal_chunks_tile():
    c = equal_chunks(347.3)
    assert len(c) == 17 and c[0][0] == 0 and c[-1][1] == pytest.approx(347.3)
    assert all(abs((b - a) - 347.3 / 17) < 1e-9 for a, b in c)
    assert equal_chunks(8.0) == [(0.0, 8.0)]


def test_window_starts():
    assert window_starts(15.0).tolist() == [0.0]
    assert window_starts(32.0).tolist() == [0.0, 5.0, 10.0, 12.0]
    assert window_starts(20.0).tolist() == [0.0]


def test_chunked_score_is_mean_of_best_windows():
    rng = np.random.default_rng(0)
    u = lambda n: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.standard_normal((n, 8)))  # noqa: E731
    C, W1, W2 = u(3), u(4), u(1)
    s = chunked_track_scores(C, [W1, W2])
    assert s[0] == pytest.approx((C @ W1.T).max(1).mean()) and s[1] == pytest.approx((C @ W2.T).mean())


def test_ensemble_standardizes_scales():
    a = np.array([1.0, 2.0, 3.0])
    b = 1000 * np.array([3.0, 2.0, 1.0])                   # a huge-scale model must not dominate
    assert np.allclose(ensemble([a, b]), 0)
    assert np.allclose(zscore(np.array([[1.0, 1.0]])), 0)
