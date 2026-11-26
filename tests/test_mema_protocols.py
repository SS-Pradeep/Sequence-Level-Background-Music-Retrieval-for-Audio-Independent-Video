import numpy as np
import soundfile as sf

from bgm.analysis.mema_protocols import degrade_audio, pairwise_accuracy, relative_drop


def test_pairwise_accuracy_counts_ties_half_and_ignores_non_mismatches():
    s = np.array([0.9, 0.5, 0.9, 0.95, 0.1])
    mismatch = np.array([False, True, True, False, True])       # index 3 (a same-track file) is not a mismatch
    assert pairwise_accuracy(s, 0, mismatch) == (1 + 0.5 + 1) / 3


def test_relative_drop():
    assert abs(relative_drop(0.4, 0.3) - 0.25) < 1e-12
    assert np.isnan(relative_drop(0.0, 0.1))


def test_degrade_audio_keeps_length_rate_and_alignment(tmp_path):
    sr = 48000
    rng = np.random.default_rng(0)
    t = np.arange(3 * sr) / sr
    y = (0.2 * np.sin(2 * np.pi * 440 * t) * (rng.random(len(t)) > 0.5)).astype(np.float32)   # gated tone: sharp onsets
    src, dst = tmp_path / "a.flac", tmp_path / "out" / "a.flac"
    sf.write(src, y, sr, subtype="PCM_24")
    degrade_audio(src, dst, 64, tmp_path / "tmp")
    z, sr2 = sf.read(dst, dtype="float32")
    assert sr2 == sr and len(z) == len(y)
    X = np.fft.irfft(np.fft.rfft(z, 2 * len(y)) * np.conj(np.fft.rfft(y, 2 * len(y))))
    lag = int(np.argmax(X))
    lag = lag if lag < len(y) else lag - 2 * len(y)
    assert lag == 0                                                # MP3 encoder delay removed: timing rule holds
