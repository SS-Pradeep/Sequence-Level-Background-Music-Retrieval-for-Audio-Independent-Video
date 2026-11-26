"""Pure metrics of the evaluation suite (D111)."""
import numpy as np

from bgm.analysis.eval_suite import (beat_alignment, chance_ndcg, diversity, dtw_cost, kl, ndcg_at, nearest,
                                     tbiou)
from bgm.features.panns_tags import SR, frames_1hz


def test_ndcg_single_and_multi_hit():
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert ndcg_at(s, np.array([1, 0, 0, 0], bool)) == 1.0
    assert np.isclose(ndcg_at(s, np.array([0, 1, 0, 0], bool)), 1 / np.log2(3))
    # two relevant at ranks 1 and 3: (1 + 1/log2(4)) / (1 + 1/log2(3))
    assert np.isclose(ndcg_at(s, np.array([1, 0, 1, 0], bool)), (1 + 0.5) / (1 + 1 / np.log2(3)))
    # ties: hit first (the rank_of_first_hit convention)
    assert ndcg_at(np.zeros(3), np.array([0, 0, 1], bool)) == 1.0


def test_chance_ndcg_matches_simulation():
    rng = np.random.default_rng(0)
    N, h = 30, 2
    sims = []
    for _ in range(20000):
        hits = np.zeros(N, bool)
        hits[rng.choice(N, h, replace=False)] = True
        sims.append(ndcg_at(rng.random(N), hits))
    assert abs(np.mean(sims) - chance_ndcg(N, h)) < 0.005


def test_nearest_and_beat_alignment():
    assert np.allclose(nearest(np.array([0.0, 1.04, 5.0]), np.array([1.0, 2.0])), [1.0, 0.04, 3.0])
    assert np.isinf(nearest(np.array([1.0]), np.array([]))).all()
    rng = np.random.default_rng(0)
    cuts = np.array([1.0, 2.0, 3.0])
    a = beat_alignment(cuts, np.array([1.0, 2.0, 3.0]), 10.0, rng)
    assert a["LB"] == 1.0 and np.isclose(a["twLB"], 1.0)
    b = beat_alignment(cuts, np.array([1.06, 2.5]), 10.0, rng)
    assert np.isclose(b["LB"], 1 / 3)
    assert np.isclose(b["twLB"], (np.exp(-0.06 ** 2 / (2 * 0.07 ** 2)) + 2 * np.exp(-0.5 ** 2 / (2 * 0.07 ** 2))) / 3)
    assert np.isnan(beat_alignment(np.array([]), np.array([1.0]), 10.0, rng)["LB"])


def test_tbiou():
    assert tbiou([5.0], [5.0], 10.0) == 1.0
    assert np.isclose(tbiou([5.0], [], 10.0), 0.5)               # [0,5) and [5,10) vs [0,10): IoU 0.5 each
    assert np.isclose(tbiou([5.0], [6.0], 10.0), (5 / 6 + 4 / 5) / 2)


def test_dtw_and_kl_and_diversity():
    X = np.eye(3)
    assert np.isclose(dtw_cost(X, X), 0.0)
    assert np.isclose(dtw_cost(X, X[[0, 0, 1, 2]]), 0.0)          # warping absorbs a repeated frame
    assert dtw_cost(X, X[::-1]) > 0
    p = np.array([0.2, 0.8])
    assert np.isclose(kl(p, p), 0.0) and kl(p, p[::-1]) > 0
    d = diversity(["a", "b", "c", "d"], 4)
    assert np.isclose(d["entropy"], 1.0) and d["coverage"] == 1.0 and d["top5_share"] == 1.0
    d = diversity(["a"] * 9 + ["b"], 100)
    assert d["coverage"] == 0.02 and d["entropy"] < 0.1


def test_panns_frames_centred():
    y = np.arange(3 * SR, dtype=np.float32)
    fr = frames_1hz(y, 3, context_s=2.0)
    assert fr.shape == (3, 2 * SR)
    assert fr[1, SR] == 1.5 * SR                                # centre sample of frame 1 is t = 1.5 s
    assert fr[0, 0] == 0 and fr[0, SR // 2 + 1] == 1           # zero padding before the file start, then y
