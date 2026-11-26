"""D127 loss diagnostics: per-negative push shares and uniformity on toy similarity matrices (no dataset)."""
import numpy as np
import pytest

from bgm.analysis.loss_why import push_weights, uniformity


def _toy():
    s = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3], [0.4, 0.35, 0.1]])
    neg = ~np.eye(3, dtype=bool)
    return s, neg


def test_push_rows_sum_to_one_on_negatives():
    s, neg = _toy()
    W = push_weights(s, neg, 0.07)
    for name in ("info_nce", "hn_nce", "sigmoid", "dcl"):
        assert np.allclose(W[name].sum(1), 1.0), name
        assert np.all(W[name][~neg] == 0), name


def test_hn_nce_is_sharper_than_info_nce():
    s, neg = _toy()
    W = push_weights(s, neg, 0.07)
    # row 0: the hardest negative (0.5) gets more of HN-NCE's push than of InfoNCE's
    assert W["hn_nce"][0, 1] > W["info_nce"][0, 1]
    # far below the bias (here tau = 1) the sigmoid weights are proportional to exp(s / tau), as InfoNCE's
    W1 = push_weights(s, neg, 1.0)
    assert np.allclose(W1["sigmoid"], W1["info_nce"], atol=1e-3)


def test_vse_max_one_hot_and_inactive_rows():
    s, neg = _toy()
    W = push_weights(s, neg, 0.07)
    assert W["vse_max"][0].sum() == 0.0                    # 0.2 - 0.9 + 0.5 < 0: hinge inactive, no push
    assert W["vse_max"][1].sum() == 0.0                    # 0.2 - 0.8 + 0.3 < 0
    assert W["vse_max"][2].sum() == 1.0 and W["vse_max"][2, 0] == 1.0     # 0.2 - 0.1 + 0.4 > 0


def test_uniformity_bounds():
    x = np.eye(4)
    assert uniformity(x) == pytest.approx(-4.0)            # all pairs at squared distance 2
    y = np.tile([1.0, 0.0], (5, 1))
    assert uniformity(y) == pytest.approx(0.0)             # collapsed
