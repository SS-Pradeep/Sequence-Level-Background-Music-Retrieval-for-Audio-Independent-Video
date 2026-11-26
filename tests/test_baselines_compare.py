import numpy as np

from bgm.analysis.baselines_compare import evaluate_matrix, first_hit_ranks, pairwise_rows
from bgm.eval.metrics import rank_of_first_hit


def test_first_hit_ranks_matches_scalar_version():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(7, 7))
    hit = rng.random((7, 7)) < 0.3
    hit[np.arange(7), np.arange(7)] = True
    want = [rank_of_first_hit(S[i], hit[i]) for i in range(7)]
    assert first_hit_ranks(S, hit).tolist() == want


def test_pairwise_ties_count_half():
    S = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    mismatch = ~np.eye(3, dtype=bool)
    assert pairwise_rows(S, mismatch).tolist() == [0.75, 0.5, 0.5]


def test_evaluate_matrix_track_vs_exact():
    # video 0's best music is file 1, which shares its track: track rank 1, exact rank 2
    S = np.array([[0.5, 0.9, 0.1], [0.2, 0.8, 0.1], [0.0, 0.1, 0.7]])
    track = np.array(["a", "a", "b"])
    IB = np.arange(9.0).reshape(3, 3)
    r = evaluate_matrix(S, track, IB, IB)
    assert r["v2m/track_rank"].tolist() == [1, 1, 1]
    assert r["v2m/exact_rank"].tolist() == [2, 1, 1]
    assert r["m2v/exact_rank"].tolist() == [1, 2, 1]
    assert r["IB@1"].tolist() == [1.0, 4.0, 8.0]
