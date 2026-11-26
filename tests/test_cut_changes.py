import numpy as np
import pandas as pd

from bgm.cue.cut_model import boundary_f1, best_threshold
from bgm.cue.cuts import clip_cuts, join_distance, silence_drop, since_music_start, span_mean_1hz


def test_clip_cuts_drops_edges():
    assert clip_cuts(np.array([1.0, 3.0, 9.5]), 10.0, 2.0) == [(1, 3.0)]


def test_silence_drop_needs_a_long_enough_dip():
    rms = np.full(250, -20.0)
    rms[120:125] = -60.0                       # 0.2 s dip
    assert not silence_drop(rms, 5.0, 1.0, 20.0, 0.3)
    rms[120:135] = -60.0                       # 0.6 s dip
    assert silence_drop(rms, 5.0, 1.0, 20.0, 0.3)


def test_join_distance_separates_different_music():
    a = np.tile([1.0, 0.0], (5, 1)); b = np.tile([0.0, 1.0], (5, 1))
    assert join_distance(a, a, 2) < 1e-9 and join_distance(a, b, 2) > 0.9


def test_span_mean_and_since_start():
    x = np.arange(10, dtype=float)[:, None]
    assert span_mean_1hz(x, 2.0, 5.0)[0] == 3.0
    df = pd.DataFrame({"pair": ["a", "a", "a"], "t": [2.0, 5.0, 9.0], "label": [0, 1, 0]})
    np.testing.assert_allclose(since_music_start(df), [2.0, 5.0, 4.0])


def test_boundary_f1_tolerance_and_threshold():
    df = pd.DataFrame({"pair": ["a"] * 5, "idx": range(5), "label": [0, 1, 0, 0, 1]})
    assert boundary_f1(df, np.array([0, 0, 1, 0, 0])) == (1, 0, 1)        # off by one still counts
    assert boundary_f1(df, np.array([1, 0, 0, 0, 0]), tol=0) == (0, 1, 2)
    s = np.array([0.1, 0.9, 0.2, 0.8]); th = best_threshold(s, np.array([0, 1, 0, 1]))
    assert ((s >= th) == np.array([0, 1, 0, 1], bool)).all()


def _two_layer_lattice():
    from bgm.retrieval.stitch_eval import Layer
    # segment 0: file 0 at 0 s; segment 1: file 0 continuing at 5 s (index 0) or file 1 at 0 s (index 1, slightly better fit)
    L0 = Layer(np.array([0]), np.array([0.0]), np.array([5.0]), np.zeros(1, int), np.zeros(1), np.array([0.0]), 1)
    L1 = Layer(np.array([0, 1]), np.array([5.0, 0.0]), np.array([5.0, 5.0]), np.zeros(2, int), np.zeros(2), np.array([0.0, -0.2]), 2,
               cont_of=np.array([0]))
    return [L0, L1]


def test_decode_learned_follows_change_probability():
    from bgm.cue.cut_hook import decode_learned, path_changes
    lat = _two_layer_lattice()
    pen = {"lam_switch": 0.5, "lam_jump": 1.0, "mu": 0.0}
    assert decode_learned(lat, pen, [None, None], 0.0, None) == [0, 0]                 # switch costs 0.5 > fit gain 0.2
    hi = decode_learned(lat, pen, [None, np.array([0.99])], 1.0, None)
    lo = decode_learned(lat, pen, [None, np.array([0.01])], 1.0, None)
    assert hi == [0, 1] and lo == [0, 0]
    assert path_changes(lat, hi).tolist() == [1] and path_changes(lat, lo).tolist() == [0]
