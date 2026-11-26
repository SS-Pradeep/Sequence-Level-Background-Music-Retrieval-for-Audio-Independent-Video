"""M8: sem_music on pool clips cut just short of a whole second (u01126_00: MERT 20 frames, CLAP 19)."""
from __future__ import annotations

import numpy as np
import pytest

from bgm.config import load_config
from bgm.features import extract as X
from bgm.tools import REPO_ROOT


class _Bank:
    """Stub encoders: CLAP returns one vector per window, MERT a fixed number of 1 Hz frames."""

    def __init__(self, n_mert: int, layers: bool):
        self.n_mert, self.layers = n_mert, layers

    def audio(self, name, windows, sr):
        assert name == "clap" and sr == 48000
        return np.ones((len(windows), 512), np.float32)

    def mert_track(self, y):
        m = np.ones((self.n_mert, 1024), np.float32)
        return (m, np.ones((25, self.n_mert, 1024), np.float32)) if self.layers else m


def _run(monkeypatch, n48: int, n_mert: int, layers: bool) -> dict:
    ex = X.Extractor.__new__(X.Extractor)
    ex.cfg = load_config(str(REPO_ROOT / "configs" / "default.yaml"), [])
    ex.music = {"u": "u.flac"}
    ex.bank = _Bank(n_mert, layers)
    out = {}
    ex.update = lambda side, i, group, arr: out.update(arr)
    monkeypatch.setattr(X, "load_music", lambda path, sr: np.zeros(n48 * sr // 48000, np.float32))
    ex._sem_music("music", ["u"])
    return out


@pytest.mark.parametrize("layers", [False, True])
def test_mert_one_frame_longer_is_cropped(monkeypatch, layers):
    n48 = 20 * 48000 - 1                         # 19.99998 s: T1 = 19 at 48 kHz; the 24 kHz copy can round up to 20 s
    out = _run(monkeypatch, n48, 20, layers)
    assert out["sem_clap_1hz"].shape == (19, 512)
    assert out["sem_mert_1hz"].shape == (19, 1024)
    if layers:
        assert out["sem_mert_layers_1hz"].shape == (25, 19, 1024)


def test_other_length_mismatches_still_fail(monkeypatch):
    with pytest.raises(AssertionError):
        _run(monkeypatch, 20 * 48000 - 1, 21, False)     # two frames longer: not a rounding difference
    with pytest.raises(AssertionError):
        _run(monkeypatch, 20 * 48000 - 1, 18, False)     # shorter
