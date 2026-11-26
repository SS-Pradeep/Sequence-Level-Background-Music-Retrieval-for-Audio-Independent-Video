"""D138 Step 1: error classification, path cost and the forced-candidate lattice on a toy library."""
from types import SimpleNamespace

import numpy as np

from bgm.analysis.error_sources import CAUSES, build_lattice_forced, classify, path_cost
from bgm.retrieval.pipeline import SegmentResult, zstats
from bgm.retrieval.stitch_eval import build_lattice, decode_lattice

PEN = {"lam_switch": 1.0, "lam_jump": 0.5, "mu": 0.0}


class ToyIndex:
    """Three 60 s files; a window's raw semantic score is a fixed function of (file, start, segment start)."""

    def __init__(self):
        self.items = [SimpleNamespace(duration=60.0, key=k, tempo=100.0 + 10 * k) for k in range(3)]

    def window_scores(self, q, fi, st, d):
        return {"sem": np.sin(0.7 * np.asarray(fi) + 0.13 * np.asarray(st) + q)}


def _setup():
    idx = ToyIndex()
    vq = SimpleNamespace(segment=lambda a, b: a)
    res = []
    for a, b in ((0.0, 4.0), (4.0, 9.0), (9.0, 12.0)):
        fi = np.repeat(np.arange(3), 20)
        st = np.tile(np.arange(20) * 2.0, 3)
        raw = idx.window_scores(a, fi, st, b - a)
        res.append(SegmentResult(a, b, fi, st, raw, zstats(raw)))
    return idx, vq, res


def test_classify():
    tr = np.array(["A", "B", "B"])
    cost = np.array([-2.0, -1.0, -0.5])
    assert classify(tr, cost, "A", "A", -2.0) is None
    assert classify(tr, cost, "C", "A", -2.0) == CAUSES[0]
    assert classify(tr, cost, "B", "A", -2.0) == CAUSES[1]               # best true-song window -1.0 is worse than -2.0
    assert classify(tr, cost, "A", "B", -1.0) == CAUSES[2]               # the true song scored better; moves decided


def test_forced_lattice_equals_build_lattice_when_the_window_is_already_on_top():
    idx, vq, res = _setup()
    w = {"sem": 1.0}
    ref = build_lattice(idx, vq, res, w, 5)
    forced = [(int(L.mid[0]), float(L.off[0])) for L in ref]
    got = build_lattice_forced(idx, vq, res, w, 5, forced)
    for a, b in zip(ref, got):
        assert a.n_top == b.n_top and np.array_equal(a.mid, b.mid) and np.allclose(a.off, b.off) and np.allclose(a.cost, b.cost)


def test_forced_window_is_added_scored_with_segment_stats_and_decoded_cost_matches():
    idx, vq, res = _setup()
    w = {"sem": 1.0}
    ref = build_lattice(idx, vq, res, w, 3)
    forced = [(2, 38.0), (2, 0.0), (0, 10.0)]
    lat = build_lattice_forced(idx, vq, res, w, 3, forced)
    for L, R, (m, o), r in zip(lat, ref, forced, res):
        if L.n_top == R.n_top + 1:
            k = L.n_top - 1
            assert L.mid[k] == m and L.off[k] == o
            mu, sd = r.stats["sem"]
            assert np.isclose(-L.cost[k], (idx.window_scores(r.a, [m], [o], 0)["sem"][0] - mu) / sd)
    picks = decode_lattice(lat, PEN, None)
    # the decoded path is the cheapest: no single-candidate change lowers its cost
    c0 = path_cost(lat, picks, PEN, 0.04)
    for i, L in enumerate(lat):
        for j in range(len(L.mid)):
            alt = list(picks)
            alt[i] = j
            assert path_cost(lat, alt, PEN, 0.04) >= c0 - 1e-9
