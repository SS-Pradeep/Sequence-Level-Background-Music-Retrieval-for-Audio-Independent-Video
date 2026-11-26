"""Timelines (SPEC §4.2). 1 Hz sample k is centred at k + 0.5 s; 25 Hz sample t sits at t / 25 s."""
from __future__ import annotations

import math

import numpy as np

RATE = 25


def n_1hz(duration_s: float) -> int:
    """T1 = floor(duration_s)."""
    return int(math.floor(duration_s + 1e-9))


def n_25hz(duration_s: float) -> int:
    """T25 = round(25 x duration_s)."""
    return int(round(RATE * duration_s))


def upsample_1hz_to_25hz(x, n25):
    """(tested) x: [T1] or [T1, C] with sample k centred at k + 0.5 s -> [n25(, C)] at t / 25 s."""
    t = np.arange(n25) / 25.0
    src = np.arange(len(x)) + 0.5
    if x.ndim == 1:
        return np.interp(t, src, x)
    return np.stack([np.interp(t, src, x[:, c]) for c in range(x.shape[1])], axis=1)


def fit_len(x: np.ndarray, n: int) -> np.ndarray:
    """Crop or edge-pad the first axis to n."""
    if len(x) >= n:
        return x[:n]
    if len(x) == 0:
        return np.zeros((n,) + x.shape[1:], x.dtype)
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), axis=0)])
