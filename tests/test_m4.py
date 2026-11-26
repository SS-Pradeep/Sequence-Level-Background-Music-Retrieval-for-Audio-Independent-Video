"""M4: segments tile each video exactly and follow the length rules; the excess cut-on-beat metric."""
from __future__ import annotations

import numpy as np
import pytest

from bgm.data.segments import make_segments, span_embedding
from bgm.eval.metrics import cut_on_beat_rate, excess_cut_on_beat


def emb_for(shot_bounds, labels, dim=16, seed=0):
    """1 Hz embeddings where each shot's seconds carry its label's direction (+ a little noise)."""
    rng = np.random.default_rng(seed)
    dirs = {l: rng.standard_normal(dim) for l in set(labels)}
    T1 = int(shot_bounds[-1])
    e = np.zeros((T1, dim))
    for k in range(T1):
        i = int(np.searchsorted(shot_bounds, k + 0.5, side="right")) - 1
        e[k] = dirs[labels[min(i, len(labels) - 1)]] + 0.05 * rng.standard_normal(dim)
    return e


def assert_tiles(segs, duration):
    assert segs[0][0] == 0.0 and segs[-1][1] == pytest.approx(duration)
    for (a, b, _), (c, d, _) in zip(segs, segs[1:]):
        assert b == pytest.approx(c) and b > a


@pytest.mark.parametrize("seed", range(20))
def test_segments_tile_and_respect_lengths(seed):
    rng = np.random.default_rng(seed)
    dur = float(rng.uniform(3, 120))
    cuts = np.sort(rng.uniform(0, dur, rng.integers(0, 40)))
    bounds = np.r_[0, cuts, dur]
    labels = list(rng.integers(0, 4, len(bounds) - 1))
    segs = make_segments(cuts, dur, emb_for(bounds, labels, seed=seed) if dur >= 1 else np.zeros((1, 16)))
    assert_tiles(segs, dur)
    L = np.array([b - a for a, b, _ in segs])
    assert np.all(L <= 20.0 + 1e-6)
    if dur >= 2.0:
        # units under 2 s were folded; only an equal split of a long unit could produce one (never: parts >= 10 s)
        assert np.all(L >= 2.0 - 1e-9) or len(segs) == 1


def test_similar_shots_merge_and_different_ones_do_not():
    bounds = np.array([0, 4, 8, 12, 16.0])
    segs = make_segments(bounds[1:-1], 16.0, emb_for(bounds, ["a", "a", "b", "b"]))
    assert [(round(a), round(b)) for a, b, _ in segs] == [(0, 8), (8, 16)]
    assert [n for *_, n in segs] == [2, 2]


def test_short_shot_folds_into_more_similar_neighbour():
    bounds = np.array([0, 6, 7, 13.0])                      # a 1 s shot between two different 6 s shots
    segs = make_segments(bounds[1:-1], 13.0, emb_for(bounds, ["a", "b", "b"]))
    assert [(round(a), round(b)) for a, b, _ in segs] == [(0, 6), (6, 13)]


def test_long_single_shot_is_split_and_short_clip_is_one_segment():
    segs = make_segments([], 45.0, np.ones((45, 4)))
    assert len(segs) == 3 and all(abs((b - a) - 15) < 1e-9 for a, b, _ in segs)
    assert make_segments([0.7], 1.5, np.ones((1, 4))) == [(0.0, 1.5, 2)]


def test_span_embedding_uses_nearest_vector_when_empty():
    e = np.eye(4)
    assert np.allclose(span_embedding(e, 1.6, 1.9), e[1])          # centre 1.5 is nearest to 1.75


def test_excess_cut_on_beat():
    rng = np.random.default_rng(0)
    beats = np.arange(0.25, 60, 0.5)
    cuts = beats[4:100:7] + 0.01
    rate, shifted, excess = excess_cut_on_beat(cuts, beats, 60.0, rng)
    assert rate == 1.0 and excess > 0.5
    random_cuts = rng.uniform(1, 59, 40)
    _, _, ex = excess_cut_on_beat(random_cuts, beats, 60.0, rng, draws=200)
    assert abs(ex) < 0.15
    assert np.isnan(excess_cut_on_beat([0.2, 59.8], beats, 60.0, rng)[2])      # no eligible cut
    assert cut_on_beat_rate([1.0], []) == 0.0
