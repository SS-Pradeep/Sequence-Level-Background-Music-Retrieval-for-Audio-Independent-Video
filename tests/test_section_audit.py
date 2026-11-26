"""Section-level audit helpers."""
import numpy as np

from bgm.analysis.section_audit import excess, hit_rate, loudness_changes, novelty_peaks


def test_hit_rate_and_excess():
    b = np.array([5.0, 10.0])
    assert hit_rate(b, np.array([5.5, 30.0]), 1.0) == 0.5 and hit_rate(b, np.zeros(0), 1.0) == 0.0
    rng = np.random.default_rng(0)
    r, sh, ex = excess(np.arange(5.0, 55.0, 5.0), np.arange(5.0, 55.0, 5.0) + 0.3, 60.0, rng, draws=200)
    assert r == 1.0 and ex > 0.3                                   # aligned events beat circular shifts


def test_loudness_change_found():
    x = np.r_[np.full(250, -30.0), np.full(250, -12.0)]            # +18 dB step at 10 s
    t = loudness_changes(x)
    assert len(t) == 1 and abs(t[0] - 10.0) <= 0.08


def test_novelty_peak_at_section_change():
    rng = np.random.default_rng(0)
    E = np.vstack([np.tile(rng.standard_normal(32), (15, 1)), np.tile(rng.standard_normal(32), (15, 1))])
    E += 0.05 * rng.standard_normal(E.shape)
    pk = novelty_peaks(E, 4)
    assert np.any(np.abs(pk - 15.0) <= 1.0)
