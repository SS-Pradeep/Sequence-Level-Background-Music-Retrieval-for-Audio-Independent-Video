import numpy as np

from bgm.analysis.zs_blend import blend, choose_w, track_ranks, zrow, zs_matrix


def test_zrow_is_per_row_and_floors_flat_rows():
    S = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    Z = zrow(S)
    assert np.allclose(Z[0], (S[0] - 2) / S[0].std()) and np.allclose(Z[1], 0)


def test_w0_keeps_the_heads_ranking():
    rng = np.random.default_rng(0)
    H, Zs = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
    track = np.array(list("abcdef"))
    assert np.array_equal(track_ranks(blend(H, zrow(Zs), 0.0), track), track_ranks(H, track))


def test_ib_plus_lb_sums_the_z_scores():
    IB, LB = np.arange(9.0).reshape(3, 3), np.eye(3)
    assert np.allclose(zs_matrix("ib+lb", IB, LB), zrow(IB) + zrow(LB))


def test_choose_w_picks_a_helpful_zero_shot_and_breaks_ties_low():
    # heads rank the right file last; the zero-shot scores rank it first -> a positive w raises MRR
    n = 5
    H = -np.eye(n)
    track = np.arange(n)
    w, mrr = choose_w([H], zrow(np.eye(n)), track, grid=np.array([0.0, 1.0, 2.0, 3.0]))
    assert w == 1.0 and mrr[0] < mrr[1] == mrr[3]         # from w = 1 the right file is first (ties count for the hit)
    w0, _ = choose_w([H], np.zeros((n, n)), track, grid=np.array([0.0, 1.0]))
    assert w0 == 0.0                                      # a useless zero-shot never moves w off 0
