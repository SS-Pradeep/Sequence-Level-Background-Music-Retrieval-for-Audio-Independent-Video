"""D137: per-window key / tempo and the switch cost that compares the end of one window with the start of the next."""
import numpy as np

from bgm.features.keys import KK_MAJOR, KK_MINOR, chroma_1hz, estimate_key, kk_key, window_key, window_tempo
from bgm.retrieval.stitch_eval import Layer, fifths_matrix, tempo_matrix, transition_matrix

SR = 22050


def test_window_tempo():
    beats = np.arange(0, 20, 0.5)                              # 120 BPM
    beats = np.r_[beats[beats < 10], 10 + np.arange(0, 10, 0.75)]   # then 80 BPM
    assert np.isclose(window_tempo(beats, 0, 5), 120)
    assert np.isclose(window_tempo(beats, 12, 18), 80)
    assert window_tempo(beats, 3.1, 4.4) == -1.0               # 3 beats < min_beats


def test_window_key_relative_major_and_silence():
    T = 10
    rms = np.full(T, 0.1, np.float32)
    ch = np.zeros((T, 12), np.float32)
    ch[:5] = np.roll(KK_MAJOR, 7)                              # G major
    ch[5:] = np.roll(KK_MINOR, 9)                              # A minor -> relative major C
    assert window_key(ch, rms, 0, 4) == 7
    assert window_key(ch, rms, 6, 10) == 0
    rms[6:] = 0.0
    assert window_key(ch, rms, 6, 10) == -1                    # all seconds silent
    assert window_key(ch, rms, 0, 0.5) == 7                    # partial second counts as its whole second


def test_kk_key_matches_estimate_key():
    t = np.arange(3 * SR) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0)).astype(np.float32) * 0.2   # C major triad
    c, rms = chroma_1hz(y, SR)
    assert c.shape == (3, 12) and rms.shape == (3,) and np.all(rms > 0.1)
    assert set(np.argsort(-c.mean(0))[:3]) == {0, 4, 7}
    assert kk_key(c.mean(0))[0] == estimate_key(y, SR)[0] == 0


def _layer(mid, key, tempo, **kw):
    n = len(mid)
    return Layer(np.array(mid), np.zeros(n), np.full(n, 5.0), np.array(key), np.array(tempo, float), np.zeros(n), n, None, **kw)


def test_transition_uses_window_ends():
    pen = {"lam_switch": 1.0, "lam_jump": 0.5, "mu": 2.0}
    P = _layer([0, 1], [0, 7], [120, 90])
    C = _layer([2, 3], [2, -1], [60, 100])
    base = transition_matrix(P, C, pen)
    assert np.allclose(base, 1.0 + 2.0 * (fifths_matrix(P.key, C.key) + tempo_matrix(P.tempo, C.tempo)))
    # window values replace the file values: outgoing end of P against incoming start of C
    Pw = _layer([0, 1], [0, 7], [120, 90], key_out=np.array([2, 2]), tempo_out=np.array([60.0, 60.0]))
    Cw = _layer([2, 3], [2, -1], [60, 100], key_in=np.array([2, 2]), tempo_in=np.array([120.0, 120.0]))
    assert np.allclose(transition_matrix(Pw, Cw, pen), 1.0)    # same key, double time -> no extra cost
    # file values of P are still used when only C has window values
    Cw_only = _layer([2, 3], [2, -1], [60, 100], key_in=np.array([0, 0]), tempo_in=np.array([120.0, 120.0]))
    assert np.allclose(transition_matrix(P, Cw_only, pen)[0], 1.0)
    # subset keeps the window fields
    s = Pw.subset(np.array([1]))
    assert s.key_out.tolist() == [2] and s.tempo_out.tolist() == [60.0] and s.key_in is None
