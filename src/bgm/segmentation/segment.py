"""Boundary scores -> cuts -> segments (D140). All times are float seconds in sequence (or clip) time.

Rules for the probability variants:
  1. Forced cuts (clip joins, for the segments handed to retrieval) are taken first.
  2. Candidates with p >= theta are taken in order of decreasing p when they keep every segment >= min_len.
  3. A segment longer than max_len is split at its highest-p internal boundary that keeps both parts >= min_len,
     recursively; with no such boundary it is split into equal parts <= max_len (PROGRESS.md D13).
"""
from __future__ import annotations

import math

import numpy as np

from bgm.data.segments import make_segments


def _fits(t: float, cuts: list[float], T: float, min_len: float) -> bool:
    return t >= min_len - 1e-9 and T - t >= min_len - 1e-9 and all(abs(t - c) >= min_len - 1e-9 for c in cuts)


def cuts_from_probs(btimes: np.ndarray, p: np.ndarray, theta: float, T: float, min_len: float = 2.0,
                    max_len: float = 20.0, forced=()) -> list[float]:
    """Sorted cut times (0 and T excluded)."""
    cuts = sorted(float(f) for f in forced)
    for k in np.argsort(-p, kind="stable"):
        if p[k] < theta:
            break
        t = float(btimes[k])
        if all(abs(t - c) > 1e-6 for c in cuts) and _fits(t, cuts, T, min_len):
            cuts.append(t)
    return split_long(sorted(cuts), btimes, p, T, min_len, max_len)


def split_long(cuts: list[float], btimes: np.ndarray, p: np.ndarray, T: float, min_len: float, max_len: float) -> list[float]:
    bounds = [0.0] + list(cuts) + [float(T)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out += _split(a, b, btimes, p, min_len, max_len)
    return sorted(set(round(c, 6) for c in out + list(cuts)))


def _split(a: float, b: float, btimes, p, min_len, max_len) -> list[float]:
    if b - a <= max_len + 1e-9:
        return []
    inside = (btimes >= a + min_len - 1e-9) & (btimes <= b - min_len + 1e-9)
    if inside.any():
        k = np.flatnonzero(inside)[int(np.argmax(p[inside]))]
        t = float(btimes[k])
        return _split(a, t, btimes, p, min_len, max_len) + [t] + _split(t, b, btimes, p, min_len, max_len)
    n = math.ceil((b - a) / max_len - 1e-9)
    return [a + (b - a) * j / n for j in range(1, n)]


def uniform_cuts(T: float, step: float = 5.0, min_len: float = 2.0, forced=()) -> list[float]:
    """Every `step` s from the start (restarting after each forced cut); a last piece under min_len joins the one
    before it."""
    bounds = [0.0] + sorted(float(f) for f in forced) + [float(T)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = list(np.arange(a + step, b - 1e-9, step))
        if c and b - c[-1] < min_len:
            c.pop()
        out += [float(x) for x in c]
    return sorted(out + [float(f) for f in forced])


def siglip_merge_cuts(shot_cuts: np.ndarray, T: float, emb_1hz: np.ndarray, centres: np.ndarray, merge_cosine: float = 0.85,
                      max_len: float = 20.0, min_shot: float = 2.0, forced=()) -> list[float]:
    """The current method (bgm.data.segments.make_segments) on a whole sequence; with forced cuts it runs on each
    piece between them separately (exactly the per-clip segmentation when forced = the clip joins)."""
    bounds = [0.0] + sorted(float(f) for f in forced) + [float(T)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = (centres >= a) & (centres < b)
        e, c = (emb_1hz[sel], centres[sel] - a) if sel.any() else (emb_1hz, centres - a)
        cuts = [float(t) - a for t in shot_cuts if a < t < b]
        segs = make_segments(cuts, b - a, e, merge_cosine, max_len, min_shot, centres=c)
        out += [a + s[0] for s in segs[1:]]
    return sorted(out + [float(f) for f in forced])


def segments_of(cuts: list[float], T: float, btimes: np.ndarray | None = None, p: np.ndarray | None = None) -> list[list[float]]:
    """[(start_s, end_s, p at its end boundary)]; p is nan where the end is not a scored boundary (or for the last)."""
    b = [0.0] + list(cuts) + [float(T)]
    out = []
    for a, e in zip(b[:-1], b[1:]):
        pe = float("nan")
        if btimes is not None and p is not None and len(btimes) and e < T:
            k = int(np.argmin(np.abs(btimes - e)))
            pe = float(p[k]) if abs(btimes[k] - e) < 1e-3 else float("nan")
        out.append([round(a, 4), round(e, 4), pe])
    return out
