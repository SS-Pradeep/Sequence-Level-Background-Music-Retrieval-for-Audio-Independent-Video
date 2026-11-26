"""M5: span rule, O(1) span means, s_attr, segment-gallery rules, hand-crafted dynamics scoring."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from bgm.eval.attributes import s_attr, softmax
from bgm.eval.protocol import segment_gallery
from bgm.eval.store import Series1Hz, apply_span_rule


def test_span_rule():
    assert apply_span_rule(3.0, 4.0, 20.0, 5.0) == (1.0, 6.0)                 # widened symmetrically
    assert apply_span_rule(0.5, 1.5, 20.0, 5.0) == (0.0, 5.0)                 # clamped at the start
    assert apply_span_rule(18.0, 19.5, 20.0, 5.0) == (15.0, 20.0)             # clamped at the end
    assert apply_span_rule(1.0, 2.0, 3.5, 5.0) == (0.0, 3.5)                  # file shorter than the span
    assert apply_span_rule(2.0, 9.0, 20.0, 5.0) == (2.0, 9.0)


def test_series_span_mean_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 3))
    s = Series1Hz(x)
    for _ in range(200):
        a = rng.uniform(0, 29)
        b = a + rng.uniform(0.2, 12)
        centres = np.arange(30) + 0.5
        inside = (centres >= a) & (centres < b)
        want = x[inside].mean(0) if inside.any() else x[int(np.clip(np.floor((a + b) / 2), 0, 29))]
        assert np.allclose(s.span_mean(a, b), want)


def test_s_attr_matches_scipy_squared_jensenshannon():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = softmax(rng.standard_normal(16), 1.0), softmax(rng.standard_normal(16), 0.5)
        assert s_attr(p, q) == pytest.approx(1 - jensenshannon(p, q, base=2) ** 2, abs=1e-9)
    assert s_attr(p, p) == pytest.approx(1.0)
    assert s_attr(np.eye(4)[0], np.eye(4)[1]) == pytest.approx(0.0, abs=1e-6)


class FakeStore:
    def __init__(self):
        self.track = {"A": "t1", "B": "t1", "C": "t2"}          # A and B share a track
        self.music_of = {"A": "mA", "B": "mB", "C": "mC"}
        self.durs = {"mA": 20.0, "mB": 20.0, "mC": 8.0}

    def duration(self, side, m):
        return self.durs[m]


def test_segment_gallery_rules():
    st = FakeStore()
    segs = [("A", 0, 0.0, 6.0), ("A", 1, 6.0, 20.0), ("B", 0, 0.0, 20.0), ("C", 0, 0.0, 4.0), ("C", 1, 4.0, 8.0)]
    wins, hit = segment_gallery(st, segs, ("A", 0, 0.0, 6.0), "all")
    assert [w[0] for w in wins] == ["mA", "mA", "mC", "mC"] and hit == 0     # B (same track, other video) left out
    assert wins[3] == ("mC", 2.0, 6.0)                                        # shifted back to fit the 8 s file
    wins, hit = segment_gallery(st, segs, ("A", 0, 0.0, 6.0), "other")
    assert [w[0] for w in wins] == ["mA", "mC", "mC"] and hit == 0             # own video's other window left out


def test_hand_dynamics_finds_the_aligned_offset():
    """Motion that follows the music's energy, and cuts on its beats, should peak at the true offset."""
    from types import SimpleNamespace
    from bgm.eval.zeroshot import ZeroShot
    rng = np.random.default_rng(0)
    n = 1500
    rms = np.convolve(rng.standard_normal(n), np.ones(25) / 25, "same")
    beats = np.arange(0.5, n / 25, 0.52)
    off = 400
    v = np.zeros((700, 5), np.float32)
    v[:, 0] = rms[off:off + 700] + 0.1 * rng.standard_normal(700)
    m = np.zeros((n, 5), np.float32)
    m[:, 2] = rms
    cuts = beats[(beats > off / 25 + 1) & (beats < (off + 700) / 25 - 1)][::5] - off / 25
    st = SimpleNamespace(dyn=lambda side, i: v if side == "video" else m,
                         video=lambda q: {"cut_times_s": cuts}, music=lambda i: {"beat_times_s": beats})
    zs = ZeroShot.__new__(ZeroShot)
    zs.store, zs.cfg = st, SimpleNamespace(eval=SimpleNamespace(cut_beat_tol_s=0.07))
    s = zs.hand_offsets("q", 0.0, 20.0, "m")
    assert abs(int(np.argmax(s)) - off) <= 1
