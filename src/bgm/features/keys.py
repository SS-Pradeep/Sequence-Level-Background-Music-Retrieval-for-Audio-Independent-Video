"""Key / mode estimation and the stitching key comparison (SPEC §5.4)."""
from __future__ import annotations

import numpy as np

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(y: np.ndarray, sr: int) -> tuple[int, int]:
    """Krumhansl-Kessler correlation on mean chroma_cqt (Essentia is not installed): (key_pc, key_mode),
    mode 0 = major, 1 = minor; (-1, -1) for silence."""
    import librosa
    if not np.any(np.abs(y) > 1e-6):
        return -1, -1
    return kk_key(librosa.feature.chroma_cqt(y=y, sr=sr).mean(1))


def kk_key(chroma: np.ndarray) -> tuple[int, int]:
    """Krumhansl-Kessler key of a mean chroma vector [12]: (key_pc, key_mode); (-1, -1) if it is flat."""
    if chroma.std() < 1e-9:
        return -1, -1
    best, arg = -np.inf, (-1, -1)
    for mode, prof in enumerate((KK_MAJOR, KK_MINOR)):
        for k in range(12):
            r = np.corrcoef(chroma, np.roll(prof, k))[0, 1]
            if r > best:
                best, arg = r, (k, mode)
    return arg


def chroma_1hz(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-second chroma_cqt and RMS (D137): ([T1, 12] mean chroma of the frames in [k, k + 1) s, [T1] RMS of y
    there), T1 = floor(duration). librosa features run at 22,050 Hz."""
    import librosa
    from bgm.features.timing import n_1hz
    assert sr == 22050, f"librosa features need sr=22050, got {sr}"
    T1 = n_1hz(len(y) / sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)                     # [12, F]
    sec = np.floor(librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)).astype(int)
    out = np.zeros((T1, 12), np.float32)
    for k in range(T1):
        if np.any(sec == k):
            out[k] = chroma[:, sec == k].mean(1)
    rms = np.array([np.sqrt(np.mean(y[k * sr:(k + 1) * sr] ** 2)) for k in range(T1)], np.float32)
    return out, rms


def window_key(chroma: np.ndarray, rms: np.ndarray, t0: float, t1: float, silence_rms: float = 1e-3) -> int:
    """Stitch key (relative major, as `stitch_key`) of the music in [t0, t1] s from the per-second chroma [T1, 12]:
    KK on the mean chroma of the whole seconds overlapping the window that are not silent; -1 if none."""
    a, b = max(int(np.floor(t0)), 0), min(int(np.ceil(t1)), len(chroma))
    sel = np.arange(a, b)
    sel = sel[rms[sel] >= silence_rms] if len(sel) else sel
    if not len(sel):
        return -1
    pc, mode = kk_key(chroma[sel].mean(0))
    return stitch_key(pc, mode, -1, False)


def window_tempo(beats: np.ndarray, t0: float, t1: float, min_beats: int = 4) -> float:
    """BPM of the music in [t0, t1] s from the track's beat times: 60 / median inter-beat interval of the beats inside
    the window; -1 (unknown) with fewer than `min_beats` beats."""
    b = np.sort(beats[(beats >= t0) & (beats <= t1)])
    if len(b) < min_beats:
        return -1.0
    return float(60.0 / np.median(np.diff(b)))


def stitch_key(key_pc, key_mode, tonic_pc, use_tonic):
    """(tested) Pitch class compared across tracks: the tonic when use_tonic, else the relative-major
    key (minor -> +3 semitones). -1 when unknown. Never compare a tonic with a key."""
    if use_tonic:
        return tonic_pc
    if key_pc < 0:
        return -1
    return (key_pc + 3) % 12 if key_mode == 1 else key_pc


def fifths_distance(a, b):
    """(tested) 0 = same, 1/6 = a fifth apart (C-G, C-F), 1 = tritone; 0.5 if either is unknown."""
    if a < 0 or b < 0:
        return 0.5
    d = abs((7 * a) % 12 - (7 * b) % 12)
    return min(d, 12 - d) / 6.0
