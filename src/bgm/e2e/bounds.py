"""Boundary features for the cross-modal switch cost (D141). Pure functions.

Music, per file on its 25 Hz grid, evaluated where a candidate window is LEFT (its end) or ENTERED (its start):
    0 beat      distance to the nearest beat / beat period, x2 -> [0, 1] (0 = on a beat)
    1 downbeat  distance to the nearest downbeat / bar, x2 (bar = 4 beats; phase = the beat-index residue mod 4 with the
                highest mean onset strength, a downbeat proxy: no downbeat tracker is cached)
    2 phrase    distance to the nearest 4-bar boundary / 16 beats, x2 (same phase)
    3 novelty   'novelty' curve, mean over +-0.5 s, rank within the file -> [0, 1]
    4 onset     'log_onset' curve, mean over +-0.2 s, rank within the file
    5 rms       'rms_db', mean over +-0.5 s, rank within the file
    6 rms_slope mean rms over [t, t + 1 s) minus over [t - 1 s, t), rank within the file
Leak diagnostic only (user choice: file-edge cues blocked in the main model):
    7 to_end    min(file end - t, 10 s) / 10
    8 from_start min(t, 10 s) / 10
Ranks are per file, so a feature says "how unusual is this point for THIS piece", never its position in the file.

Video, per boundary between segment i and i + 1 at time t:
    0 siglip    1 - cos(SigLIP mean over [t - 2, t), mean over [t, t + 2))
    1-5 dyn     |mean video curve over [t - 1, t) - over [t, t + 1)|, per channel
    6 cut       a shot cut within 0.1 s of t
    7-8 dur     log duration of the left / right segment (the clip-join shortcut of D81; dropped in the nodur variant)
"""
from __future__ import annotations

import numpy as np

RATE = 25
MUSIC_NAMES = ("beat", "downbeat", "phrase", "novelty", "onset", "rms", "rms_slope", "to_end", "from_start")
N_MUSIC_MAIN = 7
VIDEO_NAMES = ("siglip", "dyn0", "dyn1", "dyn2", "dyn3", "dyn4", "cut", "dur_left", "dur_right")
CH = {"log_onset": 0, "beat_impulse": 1, "rms_db": 2, "centroid_change": 3, "novelty": 4}


def _win_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """[T] -> [T] mean of x over frames [k + lo, k + hi) (clipped to the array, at least one frame)."""
    T = len(x)
    cs = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    k = np.arange(T)
    a = np.clip(k + lo, 0, T - 1)
    b = np.clip(k + hi, a + 1, T)
    return (cs[b] - cs[a]) / (b - a)


def rank01(x: np.ndarray) -> np.ndarray:
    """Within-array rank scaled to [0, 1] (ties keep stable order)."""
    if len(x) < 2:
        return np.full(len(x), 0.5)
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1)


def grid_distance(t: np.ndarray, grid: np.ndarray, period: float) -> np.ndarray:
    """Distance from each t to the nearest grid point / period, x2, clipped to [0, 1]; 0.5 when there is no grid."""
    if len(grid) == 0 or not np.isfinite(period) or period <= 0:
        return np.full(len(t), 0.5)
    j = np.clip(np.searchsorted(grid, t), 1, len(grid) - 1) if len(grid) > 1 else np.zeros(len(t), int)
    d = np.minimum(np.abs(t - grid[j]), np.abs(t - grid[np.maximum(j - 1, 0)]))
    return np.clip(2.0 * d / period, 0.0, 1.0)


def bar_phase(beats: np.ndarray, onset: np.ndarray, per_bar: int = 4) -> int:
    """Beat-index residue mod per_bar whose beats have the highest mean onset strength (downbeat proxy)."""
    if len(beats) < per_bar:
        return 0
    f = np.clip(np.round(beats * RATE).astype(int), 0, len(onset) - 1)
    s = onset[f]
    return int(np.argmax([s[p::per_bar].mean() for p in range(per_bar)]))


def music_frame_features(beats: np.ndarray, dyn: np.ndarray, duration: float) -> np.ndarray:
    """[T25, 9] boundary features of every 25 Hz frame of one music file (see the module docstring).
    beats: beat times, s; dyn: [T25, 5] normalized music curves (M_CHANNELS order)."""
    T = len(dyn)
    t = np.arange(T) / RATE
    beats = np.sort(np.asarray(beats, float))
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else np.nan
    ph = bar_phase(beats, dyn[:, CH["log_onset"]])
    down, phrase = beats[ph::4], beats[ph::16]
    rms = _win_mean(dyn[:, CH["rms_db"]], -12, 13)
    after, before = _win_mean(dyn[:, CH["rms_db"]], 0, RATE), _win_mean(dyn[:, CH["rms_db"]], -RATE, 0)
    return np.stack([
        grid_distance(t, beats, ibi),
        grid_distance(t, down, 4 * ibi),
        grid_distance(t, phrase, 16 * ibi),
        rank01(_win_mean(dyn[:, CH["novelty"]], -12, 13)),
        rank01(_win_mean(dyn[:, CH["log_onset"]], -5, 6)),
        rank01(rms),
        rank01(after - before),
        np.minimum(duration - t, 10.0) / 10.0,
        np.minimum(t, 10.0) / 10.0,
    ], 1).astype(np.float32)


def at_times(table: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rows of a [T25, F] frame table at times t (s), nearest frame, clipped."""
    return table[np.clip(np.round(np.asarray(t) * RATE).astype(int), 0, len(table) - 1)]


def video_boundary_features(sem_1hz: np.ndarray, dyn_25hz: np.ndarray, cut_times: np.ndarray,
                            segments: list[tuple[float, float]]) -> np.ndarray:
    """[N - 1, 9] features of each boundary between consecutive segments (see the module docstring)."""
    out = []
    T1 = len(sem_1hz)
    for (a0, b0), (a1, b1) in zip(segments, segments[1:]):
        t = a1
        k = int(round(t))
        l = sem_1hz[max(0, k - 2):max(k, 1)].mean(0)
        r = sem_1hz[min(k, T1 - 1):max(min(k + 2, T1), min(k, T1 - 1) + 1)].mean(0)
        cos = float(l @ r / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-9))
        f = int(round(t * RATE))
        dl = dyn_25hz[max(0, f - RATE):max(f, 1)].mean(0)
        dr = dyn_25hz[min(f, len(dyn_25hz) - 1):max(min(f + RATE, len(dyn_25hz)), min(f, len(dyn_25hz) - 1) + 1)].mean(0)
        cut = float(len(cut_times) > 0 and np.min(np.abs(np.asarray(cut_times) - t)) <= 0.1)
        out.append(np.r_[1.0 - cos, np.abs(dl - dr), cut, np.log(max(b0 - a0, 0.04)), np.log(max(b1 - a1, 0.04))])
    return np.array(out, np.float32).reshape(-1, len(VIDEO_NAMES))
