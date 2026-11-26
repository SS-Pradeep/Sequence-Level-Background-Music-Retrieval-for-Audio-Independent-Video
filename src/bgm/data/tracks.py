"""Track grouping by Chromaprint (SPEC §3.2 step 2): reused songs must never span train and test.

`fpcalc -raw` gives ~8 sub-fingerprints (32-bit ints) per second. Candidate pairs come from an inverted index
over sub-fingerprint values (exact matches voting for a relative offset); each candidate is verified by the bit
error rate (BER) at its best offset over at least `min_overlap_s` of overlap, and linked if BER <= `ber_max`.
Union-find over the links gives the groups.
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from bgm.tools import find_tool


@dataclass(frozen=True)
class Fingerprint:
    values: np.ndarray       # [n] uint32 sub-fingerprints
    rate_hz: float           # sub-fingerprints per second of audio


@dataclass(frozen=True)
class PairCheck:
    i: int
    j: int
    offset: int              # j's item index = i's item index + offset
    ber: float
    overlap_s: float
    spec_corr: float = float("nan")   # log-mel correlation at the matched offset (verification step)


FP_SR = 11025                        # Chromaprint's internal rate
FP_ITEM_RATE = FP_SR / (4096 / 3)    # one sub-fingerprint per 4096/3-sample hop: 8.07 / s, independent of length


def fingerprint(path: str, highpass_hz: float = 0.0) -> Fingerprint:
    """Chromaprint sub-fingerprints of a file. With highpass_hz > 0 the audio is high-passed first (ffmpeg,
    two 2-pole stages): a strong mains hum (e.g. 100 Hz in old film prints) otherwise dominates the chroma
    and makes unrelated clips look identical (PROGRESS.md, M1 finding 3)."""
    fpcalc = find_tool("fpcalc")
    assert fpcalc, "fpcalc not found (see `bgm doctor`)"
    af = f"highpass=f={highpass_hz}:poles=2,highpass=f={highpass_hz}:poles=2" if highpass_hz > 0 else "anull"
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-af", af, "-ac", "1", "-ar", str(FP_SR),
                          "-f", "s16le", "-"], capture_output=True)
    if pcm.returncode != 0 or not pcm.stdout:
        raise RuntimeError(f"ffmpeg failed on {path}: {pcm.stderr.decode(errors='replace').strip()}")
    out = subprocess.run([fpcalc, "-raw", "-length", "0", "-format", "s16le", "-rate", str(FP_SR), "-channels", "1", "-"],
                         input=pcm.stdout, capture_output=True)
    vals = None
    for line in out.stdout.decode().splitlines():
        if line.startswith("FINGERPRINT="):
            vals = np.array([int(v) for v in line[12:].split(",") if v], np.int64).astype(np.uint32)
    if vals is None or not len(vals):
        raise RuntimeError(f"fpcalc failed on {path}: {out.stderr.decode(errors='replace').strip()}")
    # items / duration would under-estimate the rate on short clips (edge frames are dropped), mis-scaling offsets
    return Fingerprint(vals, FP_ITEM_RATE)


def fingerprint_all(paths: list[str], workers: int = 16, highpass_hz: float = 0.0) -> list[Fingerprint]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: fingerprint(p, highpass_hz), paths))


def bit_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of differing bits between equal-length uint32 arrays."""
    assert a.shape == b.shape and a.dtype == np.uint32 and b.dtype == np.uint32
    return float(np.unpackbits(np.bitwise_xor(a, b).view(np.uint8)).mean()) if len(a) else 1.0


def ber_at_offset(a: np.ndarray, b: np.ndarray, offset: int) -> tuple[float, int]:
    """BER over the overlap when b[k + offset] is aligned with a[k]; returns (ber, overlap length)."""
    a0, b0 = max(0, -offset), max(0, offset)
    n = min(len(a) - a0, len(b) - b0)
    if n <= 0:
        return 1.0, 0
    return bit_error_rate(a[a0:a0 + n], b[b0:b0 + n]), n


def candidates(fps: list[Fingerprint], min_votes: int) -> dict[tuple[int, int], int]:
    """(i, j), i < j -> the relative offset with the most exact sub-fingerprint matches (>= min_votes)."""
    index = defaultdict(list)
    for j, f in enumerate(fps):
        for p, v in enumerate(f.values):
            index[int(v)].append((j, p))
    out = {}
    for i, f in enumerate(fps):
        votes = defaultdict(int)
        for p, v in enumerate(f.values):
            for j, q in index.get(int(v), ()):
                if j > i:
                    votes[(j, q - p)] += 1
        best = {}
        for (j, off), n in votes.items():
            if n >= min_votes and n > best.get(j, (0, 0))[0]:
                best[j] = (n, off)
        out.update({(i, j): off for j, (_, off) in best.items()})
    return out


def check_candidates(fps: list[Fingerprint], cands: dict[tuple[int, int], int], search: int = 2) -> list[PairCheck]:
    """BER at the best offset within +/- `search` items of the voted offset."""
    out = []
    for (i, j), off in cands.items():
        a, b = fps[i].values, fps[j].values
        ber, n, best_off = min((*ber_at_offset(a, b, o), o) for o in range(off - search, off + search + 1))
        out.append(PairCheck(i, j, best_off, ber, n / fps[i].rate_hz))
    return out


SPEC_SR, SPEC_HOP = 16000, 320     # 50 log-mel frames / s


def log_mel(path: str, fmin: float = 300.0, fmax: float = 3800.0) -> np.ndarray:
    """[48, frames] log-mel over fmin..fmax (hum excluded), z-scored per band over the file; 50 frames / s."""
    import librosa
    y, sr = librosa.load(str(path), sr=SPEC_SR, mono=True)
    assert sr == SPEC_SR
    m = np.log(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=SPEC_HOP, n_mels=48,
                                              fmin=fmin, fmax=fmax) + 1e-6)
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)


def spectral_corr(a: np.ndarray, b: np.ndarray, offset_s: float, search_s: float = 0.3, min_overlap_s: float = 5.0) -> float:
    """Max Pearson correlation of two log-mel images when b[t + offset] is aligned with a[t], searching
    +/- search_s around offset_s. Robust to phase, gain and small drift between transfers of one recording,
    unlike waveform correlation; -1 if the overlap is shorter than min_overlap_s."""
    fps = SPEC_SR / SPEC_HOP
    best = -1.0
    lag0 = int(round(offset_s * fps))
    for lag in range(lag0 - int(search_s * fps), lag0 + int(search_s * fps) + 1):
        aa, bb = (a, b[:, lag:]) if lag >= 0 else (a[:, -lag:], b)
        n = min(aa.shape[1], bb.shape[1])
        if n < min_overlap_s * fps:
            continue
        x, y = aa[:, :n].ravel(), bb[:, :n].ravel()
        best = max(best, float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y) + 1e-12)))
    return best


def verify_spectral(checks: list[PairCheck], fps: list[Fingerprint], paths: list[str], workers: int = 8) -> list[PairCheck]:
    """Attach spec_corr to each check (offset converted from fingerprint items to seconds)."""
    need = sorted({k for c in checks for k in (c.i, c.j)})
    with ThreadPoolExecutor(workers) as ex:
        mels = dict(zip(need, ex.map(lambda k: log_mel(paths[k]), need)))
    from dataclasses import replace
    return [replace(c, spec_corr=spectral_corr(mels[c.i], mels[c.j], c.offset / fps[c.i].rate_hz)) for c in checks]


def union_find(n: int, links: list[tuple[int, int]]) -> np.ndarray:
    """Component label per node (the smallest node index in its component)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(x) for x in range(n)])


def group_tracks(fps: list[Fingerprint], ber_max: float, min_overlap_s: float, min_votes: int,
                 paths: list[str] | None = None, spec_min: float | None = None):
    """Returns (component label per file, all candidate checks, the accepted links). A link needs
    BER <= ber_max over >= min_overlap_s and, when `paths` and `spec_min` are given, a log-mel correlation
    >= spec_min at the matched offset (Chromaprint alone is fooled by mains hum; PROGRESS.md M1)."""
    checks = check_candidates(fps, candidates(fps, min_votes))
    if paths is not None and spec_min is not None:
        checks = verify_spectral(checks, fps, paths)
    links = [c for c in checks if c.ber <= ber_max and c.overlap_s >= min_overlap_s
             and (spec_min is None or c.spec_corr >= spec_min)]
    return union_find(len(fps), [(c.i, c.j) for c in links]), checks, links
