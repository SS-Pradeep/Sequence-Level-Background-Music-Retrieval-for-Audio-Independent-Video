import numpy as np

from bgm.analysis.bgm_presence import file_verdict, rms_db_1hz, second_categories


def test_second_categories_order():
    pm = np.array([0.9, 0.9, 0.1, 0.3, 0.05, 0.9])
    ps = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
    db = np.array([-20, -20, -20, -20, -20, -70.0])
    assert list(second_categories(pm, ps, db)) == ["music", "music+speech", "speech", "weak music", "other sound", "silent"]


def test_file_verdict():
    assert file_verdict(np.array(["music"] * 10))[1] == "clean BGM"
    assert file_verdict(np.array(["music"] * 9 + ["speech"]))[1] == "mostly BGM"     # any speech second: not clean
    s, v = file_verdict(np.array(["music"] * 4 + ["other sound"] * 6))
    assert v == "little BGM" and abs(s - 0.4) < 1e-9
    assert file_verdict(np.array([], dtype=object)) == (0.0, "little BGM")


def test_rms_db_1hz():
    sr = 1000
    y = np.concatenate([np.full(sr, 0.1), np.zeros(sr)]).astype(np.float32)
    db = rms_db_1hz(y, sr, 3)
    assert abs(db[0] - 20 * np.log10(0.1)) < 1e-6 and db[1] < -200 and db[2] == -120.0


def test_highpass_normalize_removes_dc_and_hum_and_sets_level():
    """Konjum salangai's case: a +0.084 DC offset, quiet music and a 100 Hz hum about as loud as the music."""
    from bgm.features.panns_tags import SR, highpass_normalize
    t = np.arange(2 * SR) / SR
    dc, hum, tone = 0.084, 0.002 * np.sin(2 * np.pi * 100 * t), 0.002 * np.sin(2 * np.pi * 1000 * t)
    y = highpass_normalize((dc + hum + tone).astype(np.float32))
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 20) < 0.05           # file RMS = -20 dBFS
    core = y[SR // 2:-SR // 2]                                                  # away from the filter's edge transient
    assert abs(core.mean()) < 1e-3                                              # DC gone
    X = np.abs(np.fft.rfft(core)) ** 2
    f = np.fft.rfftfreq(len(core), 1 / SR)
    assert X[np.abs(f - 100) < 5].sum() / X.sum() < 0.01                       # hum down by > 20 dB
    assert X[np.abs(f - 1000) < 5].sum() / X.sum() > 0.95                      # the music is what is left
