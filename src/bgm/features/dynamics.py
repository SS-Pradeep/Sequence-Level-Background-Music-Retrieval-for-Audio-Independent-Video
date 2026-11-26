"""Dynamics curves at 25 Hz (SPEC §5.2). Curves are stored raw; `normalize_dyn` runs at load time.

Video channels: log_motion, visual_onset, cut_impulse, luma_change, camera_motion.
Music channels: log_onset, beat_impulse, rms_db, centroid_change, novelty.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from bgm.data.media import ffprobe
from bgm.features.timing import RATE, fit_len, n_25hz, upsample_1hz_to_25hz

V_CHANNELS = ("log_motion", "visual_onset", "cut_impulse", "luma_change", "camera_motion")
M_CHANNELS = ("log_onset", "beat_impulse", "rms_db", "centroid_change", "novelty")
IMPULSE_CHANNELS = {"cut_impulse", "beat_impulse"}
LIBROSA_SR = 22050


def gaussian_impulses(times_s, n: int, sigma_s: float = 0.04, rate: int = RATE) -> np.ndarray:
    """[n] sum of Gaussians (peak 1 each) centred at `times_s`, sampled at t / rate."""
    t = np.arange(n) / rate
    out = np.zeros(n, np.float64)
    for c in np.asarray(times_s, float):
        lo, hi = max(0, int((c - 5 * sigma_s) * rate)), min(n, int((c + 5 * sigma_s) * rate) + 2)
        out[lo:hi] += np.exp(-0.5 * ((t[lo:hi] - c) / sigma_s) ** 2)
    return out.astype(np.float32)


def normalize_dyn(dyn: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    """Load-time normalization: z-score continuous channels over the whole file; impulses stay in [0, 1]."""
    out = dyn.astype(np.float32).copy()
    for c, name in enumerate(channels):
        if name not in IMPULSE_CHANNELS:
            out[:, c] = (out[:, c] - out[:, c].mean()) / (out[:, c].std() + 1e-6)
    return out


def _mask_cuts(x: np.ndarray, cut_frames: list[int], k: int = 5) -> np.ndarray:
    """At each cut frame c and c + 1, replace x with the median of the k frames on each side."""
    x = x.copy()
    n = len(x)
    for c in cut_frames:
        idx = [i for i in list(range(c - k, c)) + list(range(c + 2, c + 2 + k)) if 0 <= i < n]
        if idx:
            med = float(np.median(x[idx]))
            for i in (c, c + 1):
                if 0 <= i < n:
                    x[i] = med
    return x


def duplicate_frames(motion: np.ndarray, rel: float = 0.3, floor: float = 0.05, width: int = 25, max_run: int = 2) -> np.ndarray:
    """Boolean [T25]: frames whose flow collapses below `rel` x the local median (window `width` frames) while the
    local median is above `floor` px/frame. These are repeated or near-identical frames from frame-rate conversion
    (e.g. one duplicate per second when 24 fps becomes 25 fps), not real stillness: only dips of at most `max_run`
    consecutive frames count, so motion that really stops is left alone (PROGRESS.md D27)."""
    from scipy.ndimage import label, median_filter
    med = median_filter(np.asarray(motion, float), size=width, mode="nearest")
    dip = (med > floor) & (motion < rel * med)
    runs, n = label(dip)                                    # duplicates are isolated: keep dips of at most `max_run`
    sizes = np.bincount(runs.ravel())
    return dip & (sizes[runs] <= max_run)


def repair_duplicate_frames(motion: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate raw motion and camera motion across duplicate frames (see `duplicate_frames`)."""
    bad = duplicate_frames(motion)
    if not bad.any() or bad.all():
        return motion, camera
    t, good = np.arange(len(motion)), ~bad
    return np.interp(t, t[good], motion[good]), np.interp(t, t[good], camera[good])


def video_curves(video_path: str | Path, cut_times_s, sigma_s: float = 0.04, short_side: int = 256,
                 grid: int = 16) -> np.ndarray:
    """[T25, 5] raw video curves from the silent 25 fps interim video. Refuses a file with an audio stream
    (CLAUDE.md rule 8: video features never see the soundtrack)."""
    import cv2
    cv2.setNumThreads(1)
    info = ffprobe(video_path)
    assert info["audio"] is None, f"video extractor refuses {video_path}: it has an audio stream"
    v = info["video"]
    assert abs(v["fps"] - RATE) < 1e-3, f"{video_path} is not 25 fps interim video"
    w, h = v["width"], v["height"]
    if min(w, h) > short_side:
        w, h = (int(round(w * short_side / h / 2)) * 2, short_side) if h <= w else (short_side, int(round(h * short_side / w / 2)) * 2)
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-an", "-map", "0:v:0",
                             "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    gy, gx = np.meshgrid(np.linspace(0.5, grid - 0.5, grid) * h / grid, np.linspace(0.5, grid - 0.5, grid) * w / grid, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    pi = np.clip(pts.astype(int), 0, [w - 1, h - 1])
    motion, camera, luma = [], [], []
    prev = None
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        g = np.frombuffer(buf, np.uint8).reshape(h, w)
        luma.append(float(g.mean()))
        if prev is None:
            motion.append(0.0)
            camera.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion.append(float(np.linalg.norm(flow, axis=-1).mean()))
            dst = pts + flow[pi[:, 1], pi[:, 0]]
            M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.RANSAC)
            camera.append(float(np.linalg.norm(pts @ M[:, :2].T + M[:, 2] - pts, axis=1).mean()) if M is not None else 0.0)
        prev = g
    proc.stdout.close()
    proc.wait()
    n = n_25hz(info["video"]["duration_s"] if np.isfinite(info["video"]["duration_s"]) else info["duration_s"])
    motion, camera, luma = fit_len(np.array(motion), n), fit_len(np.array(camera), n), fit_len(np.array(luma), n)
    motion, camera = repair_duplicate_frames(motion, camera)
    cut_frames = [int(round(t * RATE)) for t in cut_times_s]
    log_motion = _mask_cuts(np.log1p(motion), cut_frames)
    camera = _mask_cuts(camera, cut_frames)
    visual_onset = np.maximum(0.0, np.diff(log_motion, prepend=log_motion[:1]))
    luma_change = np.abs(np.diff(luma, prepend=luma[:1])) / 255.0
    cut = gaussian_impulses(cut_times_s, n, sigma_s)
    return np.stack([log_motion, visual_onset, cut, luma_change, camera], 1).astype(np.float32)


def music_curves(y: np.ndarray, sr: int, sigma_s: float = 0.04, centroid_floor_hz: float = 20.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Raw music curves [T25, 5] (novelty left at 0, filled from MERT by `add_novelty`), beat times (s),
    tempo (BPM, <= 0 when unknown). y: mono at 22,050 Hz."""
    import librosa
    assert sr == LIBROSA_SR, f"librosa features expect {LIBROSA_SR} Hz, got {sr}"
    n = n_25hz(len(y) / sr)
    t25 = np.arange(n) / RATE
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    ft = librosa.frames_to_time(np.arange(len(oenv)), sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0]) if len(beats) >= 2 else -1.0
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    lc = np.log(np.maximum(cent, centroid_floor_hz))
    cchange = np.abs(np.diff(lc, prepend=lc[:1]))
    interp = lambda x: np.interp(t25, librosa.frames_to_time(np.arange(len(x)), sr=sr), x)  # noqa: E731
    dyn = np.stack([np.interp(t25, ft, np.log1p(oenv)), gaussian_impulses(beats, n, sigma_s), interp(rms),
                    interp(cchange), np.zeros(n)], 1).astype(np.float32)
    return dyn, np.asarray(beats, np.float32), tempo


def foote_novelty(emb_1hz: np.ndarray, half_width: int = 8) -> np.ndarray:
    """[T1] Foote novelty on the cosine self-similarity of 1 Hz embeddings, Gaussian-tapered checkerboard
    kernel of half-width `half_width` samples centred on each sample, zero-padded at the edges."""
    e = emb_1hz.astype(np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    S = e @ e.T
    L = half_width
    x = np.arange(-L, L + 1)
    g = np.sign(x) * np.exp(-0.5 * (x / (L / 2)) ** 2)
    K = np.outer(g, g)
    Sp = np.pad(S, L)
    return np.array([(K * Sp[k:k + 2 * L + 1, k:k + 2 * L + 1]).sum() for k in range(len(e))], np.float32)


def add_novelty(dyn: np.ndarray, mert_1hz: np.ndarray, half_width: int = 8) -> np.ndarray:
    out = dyn.copy()
    out[:, M_CHANNELS.index("novelty")] = upsample_1hz_to_25hz(foote_novelty(mert_1hz, half_width), len(dyn))
    return out
