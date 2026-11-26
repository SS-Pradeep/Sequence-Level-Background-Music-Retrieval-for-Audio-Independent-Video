"""Frozen-encoder interfaces and deterministic fakes.

Every pretrained encoder sits behind one of three small interfaces:

* `FrameEncoder.encode_frames(frames)`: RGB frames [N, H, W, 3] uint8 -> [N, dim] float32, unit norm.
* `AudioEncoder.encode_audio(windows, sr)`: mono windows [N, S] float32 at `sr` Hz -> [N, dim] float32,
  unit norm. Implementations assert `sr` equals their native rate (CLAUDE.md rule 7).
* `TextEncoder.encode_text(texts)`: list of N strings -> [N, dim] float32, unit norm.

The fakes are fixed random projections of simple content statistics (seeded, CPU only), so tests and
`bgm smoke --fake` exercise the pipeline in seconds. They are content-dependent on purpose: a frame's
colour and a window's spectrum move their embeddings, so synthetic pairs are learnable.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameEncoder(Protocol):
    dim: int

    def encode_frames(self, frames: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AudioEncoder(Protocol):
    dim: int
    sr: int

    def encode_audio(self, windows: np.ndarray, sr: int) -> np.ndarray: ...


@runtime_checkable
class TextEncoder(Protocol):
    dim: int

    def encode_text(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-normalize along `axis`; all-zero rows stay zero."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def _projection(seed: int, d_in: int, d_out: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((d_in, d_out)).astype(np.float32) / np.sqrt(d_in)


class FakeFrameEncoder:
    """Mean-pool each frame to a 6 x 8 x 3 grid of [0, 1] colours, then a fixed random projection."""

    GRID = (6, 8)

    def __init__(self, dim: int = 1152, seed: int = 0):
        self.dim = dim
        self._w = _projection(seed, self.GRID[0] * self.GRID[1] * 3, dim)

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """frames: [N, H, W, 3] uint8 -> [N, dim] unit norm."""
        assert frames.ndim == 4 and frames.shape[-1] == 3, f"expected [N, H, W, 3], got {frames.shape}"
        n, h, w, _ = frames.shape
        gh, gw = self.GRID
        x = frames[:, : h - h % gh, : w - w % gw].astype(np.float32) / 255.0
        x = x.reshape(n, gh, (h - h % gh) // gh, gw, (w - w % gw) // gw, 3).mean((2, 4))
        return l2_normalize((x.reshape(n, -1) - 0.5) @ self._w)


class FakeAudioEncoder:
    """Log band energies (64 bands of the magnitude spectrum), mean-removed, then a fixed random projection."""

    N_BANDS = 64

    def __init__(self, dim: int = 512, sr: int = 48000, seed: int = 1):
        self.dim, self.sr = dim, sr
        self._w = _projection(seed, self.N_BANDS, dim)

    def encode_audio(self, windows: np.ndarray, sr: int) -> np.ndarray:
        """windows: [N, S] mono float at `sr` Hz -> [N, dim] unit norm."""
        assert sr == self.sr, f"{type(self).__name__} expects {self.sr} Hz audio, got {sr} Hz"
        windows = np.atleast_2d(np.asarray(windows, np.float32))
        spec = np.abs(np.fft.rfft(windows, axis=1))
        edges = np.linspace(0, spec.shape[1], self.N_BANDS + 1).astype(int)
        bands = np.stack([spec[:, a:max(b, a + 1)].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
        feats = np.log1p(bands)
        return l2_normalize((feats - feats.mean(1, keepdims=True)) @ self._w)


class FakeTextEncoder:
    """A stable pseudo-random unit vector per string (seeded from its SHA-1)."""

    def __init__(self, dim: int = 512, seed: int = 2):
        self.dim, self.seed = dim, seed

    def encode_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            h = int.from_bytes(hashlib.sha1(f"{self.seed}:{t}".encode()).digest()[:8], "little")
            out.append(np.random.default_rng(h).standard_normal(self.dim))
        return l2_normalize(np.array(out, np.float32))


def fake_encoders() -> dict[str, object]:
    """The fake counterparts of the frozen encoders, keyed by role (dims match the real models)."""
    return {
        "siglip_image": FakeFrameEncoder(1152, seed=10),
        "siglip_text": FakeTextEncoder(1152, seed=11),
        "clap_audio": FakeAudioEncoder(512, sr=48000, seed=12),
        "clap_text": FakeTextEncoder(512, seed=13),
        "mert_audio": FakeAudioEncoder(1024, sr=24000, seed=14),
        "imagebind_image": FakeFrameEncoder(1024, seed=15),
        "imagebind_audio": FakeAudioEncoder(1024, sr=16000, seed=16),
    }
