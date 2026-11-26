"""D140 segmentation study: probabilities -> segments; joins are never crossed; lengths in [2, 20] s; metrics."""
import numpy as np

from bgm.data.segments import make_segments
from bgm.segmentation import evaluate as ev
from bgm.segmentation.segment import cuts_from_probs, segments_of, siglip_merge_cuts, uniform_cuts


def _check(cuts, T, joins):
    b = [0.0] + list(cuts) + [T]
    L = np.diff(b)
    assert np.all(L >= 2.0 - 1e-6) and np.all(L <= 20.0 + 1e-6), L
    for a, e in zip(b[:-1], b[1:]):
        assert not any(a + 1e-6 < j < e - 1e-6 for j in joins)


def test_prob_segments_never_cross_joins():
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = float(rng.uniform(60, 180))
        bt = np.sort(rng.uniform(0.3, T - 0.3, int(rng.integers(5, 80))))
        joins = sorted(rng.choice(bt, int(rng.integers(1, 6)), replace=False).tolist())
        # clips are >= 9.8 s in the dataset; keep joins >= 4 s apart and from the ends
        joins = [j for j in joins if j > 4 and T - j > 4]
        joins = [j for i, j in enumerate(joins) if i == 0 or j - joins[i - 1] > 4]
        p = rng.uniform(0, 1, len(bt))
        cuts = cuts_from_probs(bt, p, float(rng.uniform(0.1, 0.9)), T, forced=joins)
        assert set(np.round(joins, 6)) <= set(np.round(cuts, 6))
        _check(cuts, T, joins)
        segs = segments_of(cuts, T, bt, p)
        assert segs[0][0] == 0 and abs(segs[-1][1] - T) < 1e-3


def test_unforced_and_long_split():
    bt = np.array([1.0, 5.0, 9.0, 30.0, 31.0])
    p = np.array([0.9, 0.95, 0.2, 0.1, 0.99])
    cuts = cuts_from_probs(bt, p, 0.5, 50.0)
    assert 31.0 in cuts and 5.0 in cuts and 1.0 not in cuts          # 1.0 would make a 1 s segment
    _check(cuts, 50.0, [])
    assert cuts_from_probs(np.array([]), np.array([]), 0.5, 45.0) == [15.0, 30.0]


def test_uniform_and_merge_forced():
    c = uniform_cuts(23.0, forced=[12.0])
    assert c == [5.0, 10.0, 12.0, 17.0]
    _check(uniform_cuts(61.0), 61.0, [])
    emb = np.random.default_rng(1).normal(size=(30, 8))
    cent = np.arange(30) + 0.5
    shot = [3.0, 6.0, 12.0, 15.0, 22.0]
    cuts = siglip_merge_cuts(np.array(shot), 30.0, emb, cent, forced=[12.0])
    assert 12.0 in cuts
    _check(cuts, 30.0, [12.0])
    # one piece without forced cuts = make_segments
    one = siglip_merge_cuts(np.array(shot), 30.0, emb, cent)
    ref = [s[0] for s in make_segments(shot, 30.0, emb)[1:]]
    assert np.allclose(one, ref)


def test_metrics():
    assert ev.average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1])) == (1 + 2 / 3) / 2
    rec = {"preds": [10.2, 20.0, 40.0], "gt": [10.0, 30.0, 40.5], "gt_film": [10.0], "cross": [30.0, 40.5],
           "gt_edge": [10.0], "edge_lo": 15.0, "edge_hi": 100.0}
    c = ev.seq_counts(rec)
    assert c["all/1"] == (2, 1, 1) and c["all/0.5"] == (2, 1, 1)
    assert c["in_film/1"] == (1, 1, 0)                                  # 40.0 is dropped (near the cross join 40.5)
    assert c["edge/1"] == (1, 0, 0)
    bs = ev.Bootstrap(["a", "b", "b", "c"], 200, 0)
    f = bs.f1(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [2, 0, 0]], float))
    assert f.shape == (200,) and np.all((f >= 0) & (f <= 1))
    a = bs.ap(np.array([0.9, 0.1, 0.5, 0.4]), np.array([1, 0, 1, 0]), np.array([0, 1, 2, 3]))
    assert a.shape == (200,) and np.all(a <= 1 + 1e-6)


def test_segment_metrics():
    m = ev.segment_metrics([10.0], [10.0], 20.0)
    assert m["purity"] == 1 and m["coverage"] == 1 and m["miou"] == 1 and m["pk"] == 0 and m["windowdiff"] == 0
    over = ev.segment_metrics([5.0, 10.0, 15.0], [10.0], 20.0)          # every cue cut in half
    assert over["purity"] == 1 and over["coverage"] == 0.5 and over["miou"] == 0.5
    under = ev.segment_metrics([], [10.0], 20.0)                         # two cues in one segment
    assert under["coverage"] == 1 and under["purity"] == 0.5 and under["pk"] > 0
