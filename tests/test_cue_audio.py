"""D98 audio teacher: music distance, sequence clock, boundary times, calibration, and that only training draws use it."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bgm.cue.audio_teacher import (SEP, AudioTeacher, Calibration, boundary_times, music_distance, part_music_1hz,
                                   sequence_music_1hz)
from bgm.cue.tokens import Scaler, segment_features
from bgm.cue.train import FoldData


def _two_cue(n_a: int, n_b: int, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=d), rng.normal(size=d)
    return np.stack([a + 0.05 * rng.normal(size=d) for _ in range(n_a)] + [b + 0.05 * rng.normal(size=d) for _ in range(n_b)])


def test_music_distance_peaks_at_the_change():
    e = _two_cue(10, 10)
    d = music_distance(e, np.arange(1, 20, dtype=float), half_s=3)
    assert np.argmax(d) == 9                         # t = 10 s: the first second of the second cue
    assert d[9] > 0.5 and np.median(np.delete(d, np.arange(6, 13))) < 0.05
    assert music_distance(e, np.array([0.0, 20.0]), 3).tolist() == [0.0, 0.0]     # an empty side gives 0


def test_sequence_clock_follows_concat_rule():
    a = np.arange(3, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)          # part A: 3.4 s, rows 0..2
    b = 10 + np.arange(3, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)     # part B: 2.6 s, rows 10..12
    e = sequence_music_1hz([(a, 85), (b, 65)])                                         # 3.4 s + 2.6 s = 6 s
    assert e.shape == (6, 4)
    assert e[:, 0].tolist() == [0, 1, 2, 10, 11, 12]       # centre 3.5 s is in B at 0.1 s -> B row 0


def test_boundary_times_match_segment_starts():
    segs_a, segs_b = [(0.0, 2.0), (2.0, 5.5)], [(0.0, 3.0), (3.0, 4.0)]
    sem = np.ones((6, 8), np.float32)
    ta = segment_features(sem, np.zeros(138), segs_a)
    tb = segment_features(sem, np.zeros(100), segs_b)
    raw = np.concatenate([ta, tb])
    bt = boundary_times(raw, [0, 0, 1, 1], [138, 100])
    np.testing.assert_allclose(bt, [2.0, 5.52, 8.52], atol=1e-4)


def test_calibration_is_monotone_and_separates():
    rng = np.random.default_rng(0)
    d = np.concatenate([rng.uniform(0.01, 0.05, 200), rng.uniform(0.08, 0.3, 200)])
    y = np.r_[np.zeros(200), np.ones(200)]
    c = Calibration.fit(d, y)
    p = c(np.array([0.01, 0.05, 0.1, 0.3]))
    assert np.all(np.diff(p) > 0) and p[0] < 0.2 and p[-1] > 0.8


def _fake_store():
    m = {"mA": _two_cue(6, 6, seed=1), "mB": _two_cue(12, 0, seed=2)}
    return SimpleNamespace(music_of={"A": "mA", "B": "mB"}, track={"A": "t1", "B": "t2"},
                           music=lambda k: {"sem_mert_1hz": m[k]},
                           video=lambda p: {"dyn_25hz": np.zeros((300, 5))})


def test_splice_parts_take_the_kept_music():
    st = _fake_store()
    full, n = part_music_1hz(st, "A")
    a, na = part_music_1hz(st, f"A{SEP}4-7A")
    b, nb = part_music_1hz(st, f"A{SEP}4-7B")
    assert (n, na, nb) == (300, 100, 125)
    np.testing.assert_array_equal(a, full[:4])
    np.testing.assert_array_equal(b, full[7:])


def test_teacher_only_on_training_draws():
    st = _fake_store()
    sem = np.ones((12, 8), np.float32)
    tok = {p: segment_features(sem, np.zeros(300), [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0)]) for p in "AB"}
    fd = object.__new__(FoldData)
    fd.store, fd.tokens, fd.syn, fd.fg = st, tok, {}, {"A": "f1", "B": "f1"}
    fd.scaler = Scaler.fit(list(tok.values()))
    fd.teacher = AudioTeacher(st, 3, Calibration(4.0, 10.0))
    _, y0, typ = fd.xy(["A", "B"])
    _, y1, _ = fd.xy(["A", "B"], teacher=True)
    assert y0.tolist() == [0, 0, 0, 1, 0, 0, 0] and typ[3] == "film"
    assert y1.dtype == np.float32 and y1.shape == y0.shape
    assert y1[1] > 0.5 and y1[3] > 0.5               # A's music changes at 6 s inside the clip (label 0 above); the join
    assert max(y1[0], y1[2], y1[4], y1[5], y1[6]) < 0.01
