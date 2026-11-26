"""D138 Step 2: inverted softmax and CSLS on synthetic score matrices."""
import numpy as np

from bgm.analysis.hubness import choose, csls, inverted_softmax, rescore, top10_counts, track_ranks


def _hubby(seed=0, n=40):
    """n queries x n files, query i's right file is i; file 0 is a hub (high score for every query)."""
    rng = np.random.default_rng(seed)
    S = rng.normal(0, 0.05, (n, n)) + 0.3 * np.eye(n)
    S[:, 0] += 0.35
    return S


def test_inverted_softmax_and_csls_shapes_and_ranking_invariance_to_query_constants():
    S, B = _hubby(), _hubby(1)
    for f in (lambda X: inverted_softmax(X, B, 10.0), lambda X: csls(X, B, 10)):
        A = f(S)
        assert A.shape == S.shape
        # adding a per-query constant never changes that query's ranking
        assert np.array_equal(np.argsort(-A, 1), np.argsort(-f(S + np.arange(len(S))[:, None] * 0.01), 1))


def test_hub_is_demoted_and_recall_improves():
    S, B = _hubby(), _hubby(1)
    tr = np.arange(len(S))
    base = top10_counts(S)
    assert base[0] == len(S)                                  # the hub is in every top-10 list
    for m, p in (("is", 20.0), ("csls", 10)):
        A = rescore(S, B, m, p)
        assert top10_counts(A)[0] <= base[0]
        assert (track_ranks(A, tr) <= 1).mean() > (track_ranks(S, tr) <= 1).mean()
    assert top10_counts(rescore(S, B, "is", 20.0))[0] < base[0]


def test_csls_matches_definition():
    rng = np.random.default_rng(3)
    S, B = rng.normal(size=(5, 7)), rng.normal(size=(20, 7))
    r = np.array([np.sort(B[:, j])[-4:].mean() for j in range(7)])
    assert np.allclose(csls(S, B, 4), 2 * S - r)


def test_inverted_softmax_beta_zero_keeps_nothing_and_large_beta_matches_definition():
    rng = np.random.default_rng(4)
    S, B = rng.normal(size=(3, 6)), rng.normal(size=(10, 6))
    want = np.log(np.exp(2.0 * S) / np.exp(2.0 * B).sum(0))
    assert np.allclose(inverted_softmax(S, B, 2.0), want)


def test_choose_prefers_first_on_ties_and_returns_curve():
    S = _hubby()
    tr = np.arange(len(S))
    p, curve = choose("csls", (5, 10), [S, S], lambda e: _hubby(1), tr)
    assert p in (5, 10) and len(curve) == 2
    p, curve = choose("is", (1.0, 1.0), [S], lambda e: _hubby(1), tr)
    assert p == 1.0 and curve[0] == curve[1]
