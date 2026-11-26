"""bgm.tagging.score / windows / evaluate: Jaccard bounds, group decomposition, window overlap, tie-shared ranks."""
from __future__ import annotations

import numpy as np

from bgm.rerank.evaluate import v2m_ranks
from bgm.tagging.evaluate import combine, group_matrix, overlap_test, per_query, random_floor, tag_minmax
from bgm.tagging.score import tag_score, tag_scores
from bgm.tagging.tune import ALL, choose, only, weight_grid
from bgm.tagging.vectors import WindowTags
from bgm.tagging.vocab import GROUPS
from bgm.tagging.windows import grid_starts, overlap_weights

V = group_matrix().shape[0]


def test_identical_one_disjoint_zero():
    a = np.zeros(V)
    a[[0, 5, 20]] = [0.9, 0.4, 0.7]
    b = np.zeros(V)
    b[[1, 6]] = 0.8
    assert abs(tag_score(a, a) - 1.0) < 1e-6
    assert tag_score(a, b) == 0.0
    assert tag_score(np.zeros(V), np.zeros(V)) == 0.0
    assert abs(tag_scores(a[None], a[None])[0] - 1.0) < 1e-6


def test_group_weights_match_dimension_weights():
    rng = np.random.default_rng(0)
    ts, tm = rng.random((4, V)) * (rng.random((4, V)) > 0.7), rng.random((4, V)) * (rng.random((4, V)) > 0.7)
    w = np.array([1.0, 0.0, 0.5, 1.0, 0.0])
    G = group_matrix()
    MIN = np.stack([(np.minimum(ts, tm) @ G)[:, g] for g in range(len(GROUPS))])[:, :, None]
    MAX = np.stack([(np.maximum(ts, tm) @ G)[:, g] for g in range(len(GROUPS))])[:, :, None]
    got = combine(MIN, MAX, w)[:, 0]
    want = tag_scores(ts, tm, G @ w)
    assert np.allclose(got, want, atol=1e-6)


def test_grid_covers_file_end():
    assert list(grid_starts(10.2)) == [0.0, 0.2]
    assert list(grid_starts(20.0)) == [0.0, 5.0, 10.0]
    assert list(grid_starts(23.0)) == [0.0, 5.0, 10.0, 13.0]
    assert list(grid_starts(7.0)) == [0.0]


def test_overlap_weights():
    s = grid_starts(20.0)
    w = overlap_weights(s, 10.0, np.array([0.0, 12.0, 30.0]), np.array([4.0, 6.0, 3.0]))
    assert np.allclose(w[0], [4, 0, 0]) and np.allclose(w[1], [0, 3, 6])
    assert w[2].sum() == 1.0 and w[2, 2] == 1.0                              # outside the file -> nearest window


def test_window_tags_mean():
    a, b = np.eye(V)[0].astype(np.float32), np.eye(V)[1].astype(np.float32)
    wt = WindowTags({("m", 0.0): a, ("m", 5.0): b, ("m", 10.0): b}, 10.0)
    v = wt.vectors("m", np.array([0.0, 2.5]), np.array([5.0, 5.0]))
    assert np.allclose(v[0, :2], [1.0, 0.0]) and np.allclose(v[1, :2], [5 / 7.5, 2.5 / 7.5])


def test_ties_are_shared_not_won():
    S = np.zeros((4, 4))                                                     # all tied: rank uniform over 1..4
    pq = per_query(S)
    assert np.allclose(pq["R@1"], 0.25) and np.allclose(pq["R@3"], 0.75)
    assert np.allclose(pq["RR"], (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    assert np.allclose(pq["midrank"], 2.5)
    assert v2m_ranks(S).tolist() == [1, 1, 1, 1]                             # what a strict-above rank would claim
    rng = np.random.default_rng(1)
    S = rng.normal(size=(30, 30))                                            # no ties: matches v2m_ranks
    S[0, 3] = -np.inf
    pq = per_query(S)
    r = v2m_ranks(S)
    assert np.allclose(pq["midrank"], r) and np.allclose(pq["RR"], 1 / r) and np.allclose(pq["R@5"], r <= 5)
    fl = random_floor(np.zeros((10, 10)))
    assert abs(fl["R@1"] - 0.1) < 1e-9 and abs(fl["R@10"] - 1.0) < 1e-9


def test_tag_minmax_and_tuning_find_planted_signal():
    rng = np.random.default_rng(0)
    Q = 40
    sv = (rng.random((Q, V)) > 0.85) * rng.random((Q, V))
    sv = sv.astype(np.float32)
    vecs = {(f"m{j}", 0.0): sv[j] for j in range(Q)}                          # window j carries scene j's tags
    wt = WindowTags(vecs, 10.0)
    music = [f"m{j}" for j in range(Q)]
    durs = np.full(Q, 5.0)
    MIN, MAX = tag_minmax(sv, wt, music, np.zeros(Q), durs, {m: 10.0 for m in music})
    T = combine(MIN, MAX, ALL)
    assert np.allclose(np.diag(T), 1.0, atol=1e-5)
    S = [rng.normal(0, 0.1, (Q, Q)) for _ in range(3)]
    lam, w, curve = choose(S, MIN, MAX, [0.0, 0.05, 0.5], weight_grid())
    assert lam > 0 and len(curve) == 3 * 31
    lam0, _, _ = choose(S, MIN * 0, MAX, [0.0, 0.5], [ALL])                    # no signal -> λ = 0 kept (first wins ties)
    assert lam0 == 0.0
    assert only("tempo") == (0.0, 0.0, 0.0, 0.0, 1.0)


def test_overlap_test_auc():
    rng = np.random.default_rng(0)
    n = 30
    sv = ((rng.random((n, V)) > 0.85) * rng.random((n, V))).astype(np.float32)
    music_of = {f"p{i}": f"m{i}" for i in range(n)}
    film = {f"p{i}": f"f{i // 2}" for i in range(n)}
    wt = WindowTags({(f"m{i}", 0.0): sv[i] for i in range(n)}, 10.0)
    scenes = [(f"p{i}", 0, 0.0, 5.0) for i in range(n)]
    o = overlap_test(scenes, {(f"p{i}", 0): sv[i] for i in range(n)}, wt, music_of, {f"m{i}": 10.0 for i in range(n)}, film,
                     n_random=20)
    assert o["auc"].mean() > 0.9 and (o["true"] > o["rand_mean"]).all()
    o2 = overlap_test(scenes, {(f"p{i}", 0): sv[i] for i in range(n)}, wt, music_of, {f"m{i}": 10.0 for i in range(n)}, film,
                      n_random=20)
    assert np.array_equal(o["auc"], o2["auc"])                               # per-item seeding: deterministic
