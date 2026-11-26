"""Unit tests for the re-rankers (synthetic matrices only). `.venv/bin/python -m pytest -q exp_supp/rerank/test_methods.py`"""
import numpy as np

from exp_supp.rerank import methods as M
from exp_supp.rerank.cache import ranks_segment, ranks_track


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def setup(seed=0, nq=6, ng=30, nb=40, d=8):
    rng = np.random.default_rng(seed)
    Q, Gm, Bq = unit(rng.normal(size=(nq, d))), unit(rng.normal(size=(ng, d))), unit(rng.normal(size=(nb, d)))
    return Q, Gm, Bq, M.Block(Q @ Gm.T, np.ones(ng, bool), Gm @ Gm.T, Bq @ Gm.T, np.arange(ng) % 7)


def test_identity_reproduces_base_ranks():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(25, 25))
    valid = rng.random((25, 25)) > 0.2
    np.fill_diagonal(valid, True)
    want = ranks_segment(S, valid)
    got = np.array([M.final_ranks(S[i:i + 1], S[i:i + 1], valid[i], np.eye(25, dtype=bool)[i:i + 1])[0] for i in range(25)])
    assert np.array_equal(got, want)
    hits = rng.random((10, 25)) > 0.85
    hits[:, 0] = True
    S2 = rng.normal(size=(10, 25))
    assert np.array_equal(M.final_ranks(S2, S2, np.ones(25, bool), hits), ranks_track(S2, hits))


def test_rerank_only_moves_inside_shortlist():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(40, 60))
    hits = np.zeros((40, 60), bool)
    hits[np.arange(40), rng.integers(0, 60, 40)] = True
    R = rng.normal(size=S.shape)
    base = ranks_track(S, hits)
    got = M.final_ranks(S, R, np.ones(60, bool), hits, K=20)
    assert np.all(got[base > 20] == base[base > 20])
    assert np.all(got[base <= 20] <= 20)


def test_aqe_matches_embedding_space():
    Q, Gm, _, b = setup()
    R = M.aqe(b, 3)
    for i in range(len(Q)):
        nn = np.argsort(-b.S[i])[:3]
        q2 = Q[i] + Gm[nn].sum(0)
        assert np.array_equal(np.argsort(-R[i]), np.argsort(-(Gm @ q2)))


def test_alpha_qe_weights():
    Q, Gm, _, b = setup(3)
    R = M.alpha_qe(b, 2, 3.0)
    i = 0
    nn = np.argsort(-b.S[i])[:2]
    q2 = Q[i] + (np.maximum(b.S[i, nn], 0) ** 3)[:, None].__mul__(Gm[nn]).sum(0)
    np.testing.assert_allclose(R[i], Gm @ q2, atol=1e-10)


def test_dba_matches_embedding_space():
    Q, Gm, _, b = setup(4)
    R = M.dba(b, 3)
    G2 = np.stack([Gm[np.argsort(-(Gm @ g))[:3]].sum(0) for g in Gm])
    np.testing.assert_allclose(R, Q @ unit(G2).T, atol=1e-10)


def test_dual_softmax_matches_explicit():
    Q, Gm, Bq, b = setup(5)
    T = 0.1
    R = M.dual_softmax(b, T)
    for i in range(len(Q)):
        allq = np.vstack([Bq, Q[i:i + 1]]) @ Gm.T / T
        col = np.exp(allq[-1]) / np.exp(allq).sum(0)
        row = np.exp(allq[-1]) / np.exp(allq[-1]).sum()
        assert np.array_equal(np.argsort(-R[i]), np.argsort(-(row * col)))


def test_dedup_demotes_repeats_only():
    S = np.array([[0.9, 0.8, 0.7, 0.6]])
    b = M.Block(S, np.ones(4, bool), np.eye(4), np.zeros((2, 4)), np.array([0, 0, 1, 2]))
    R = M.dedup(b)
    assert list(np.argsort(-R[0])) == [0, 2, 3, 1]


def test_track_agg_file_pooling():
    S = np.zeros((1, 2))
    W = np.array([[0.1, 0.5, 0.2, 0.3]])
    b = M.Block(S, np.ones(2, bool), np.eye(2), np.zeros((1, 2)), np.arange(2), {"W": W, "w_file": np.array([0, 0, 0, 1])})
    np.testing.assert_allclose(M.track_agg(b, "max")[0], [0.5, 0.3])
    np.testing.assert_allclose(M.track_agg(b, "mean")[0], [0.8 / 3, 0.3])
    np.testing.assert_allclose(M.track_agg(b, "top3")[0], [0.8 / 3, 0.3])


def test_key_and_tempo_distances():
    assert M.key_dist(0, 1, 0, 1) == 0                   # C major = C major
    assert M.key_dist(0, 1, 9, 0) == 0                   # A minor = relative of C major
    assert M.key_dist(0, 1, 7, 1) == 1 / 6               # one fifth
    assert M.key_dist(0, 1, 6, 1) == 1.0                 # tritone
    assert M.tempo_dist(120, 60) == 0                    # octave error folded
    assert abs(M.tempo_dist(120, 90) - np.log2(120 / 90)) < 1e-12


def test_mutual_knn_puts_mutual_first():
    Q, Gm, Bq, b = setup(6)
    R = M.mutual_knn(b, 5)
    top = np.argsort(-R, 1)[:, 0]
    assert R.shape == b.S.shape and np.all(np.isfinite(R[np.arange(len(R)), top]))


def test_k_reciprocal_lambda_one_is_identity():
    _, _, _, b = setup(7)
    R = M.k_reciprocal(b, 5, 1, 1.0)
    sl = M.shortlist(b.S, b.valid)
    for i in range(len(R)):
        c = np.where(sl[i])[0]
        assert np.array_equal(np.argsort(-R[i, c]), np.argsort(-b.S[i, c]))


def test_permute_shortlist_keeps_values_and_outside():
    from exp_supp.rerank.hmm import permute_shortlist
    sim = np.array([0.9, 0.8, 0.7, 0.1])
    sl = np.array([True, True, True, False])
    R = np.array([0.0, 0.0, 5.0, 9.0])                    # item 2 promoted; item 3 outside the shortlist
    out = permute_shortlist(sim, R, sl)
    np.testing.assert_allclose(out, [0.8, 0.7, 0.9, 0.1])
