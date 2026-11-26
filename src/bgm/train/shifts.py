"""Shifted negatives for dynamics training (SPEC §5.6), and the matching "in sync" test used by the grounding metric."""
from __future__ import annotations

import numpy as np


def sample_negative_shifts(onset, beats, start, T, rng, k=4, fps=25, min_s=0.5, max_s=4.0,
                           guard_s=0.1, max_ac=0.5, ctx=25, max_tries=200):
    """(tested) Start frames of k distinct shifted copies of the aligned music window [start, start + T),
    shifted min_s..max_s seconds either way, keeping only shifts at which the music no longer lines up
    with itself. Both guards must pass:
      1. beat grid: the window's own beats, shifted by d, must not land (median) within guard_s of a beat;
         skipped when fewer than 2 beats fall in the window;
      2. onset autocorrelation: the z-scored onset curve over the window +/- ctx frames, correlated with
         itself at lag d, must stay below max_ac. This catches tempo octave errors, drift and unknown tempo.
    onset: [frames] z-scored log_onset of the file; beats: [n] beat times (s) of the file;
    rng: np.random.Generator seeded per item, e.g. np.random.default_rng([seed, epoch, item_idx]).
    Returns a list of start frames; slice them from the long encoded music sequence."""
    n = len(onset)
    lo, hi = max(0, start - ctx), min(n, start + T + ctx)
    ref = onset[lo:hi]
    local = beats[(beats >= start / fps) & (beats <= (start + T) / fps)]
    out = []
    for _ in range(max_tries):
        d = int(round(rng.uniform(min_s, max_s) * fps)) * int(rng.choice([-1, 1]))
        s = start + d
        if s in out or s < 0 or s + T > n or lo + d < 0 or hi + d > n:
            continue
        if len(local) >= 2:
            near = np.min(np.abs((local + d / fps)[:, None] - beats[None, :]), axis=1)
            if np.median(near) < guard_s:
                continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ac = np.corrcoef(ref, onset[lo + d:hi + d])[0, 1]
        if np.isfinite(ac) and ac < max_ac:
            out.append(s)
            if len(out) == k:
                break
    return out


def shift_in_sync(onset, beats, start, T, d, fps=25, guard_s=0.1, max_ac=0.5, ctx=25) -> bool:
    """True when a shift of d frames would be REJECTED by `sample_negative_shifts`' guards, i.e. the shifted
    music still lines up with itself (the grounding "sync" criterion, SPEC §5.12). Shifts whose context
    falls off the file are judged by the guards that can be evaluated (beat grid only)."""
    n = len(onset)
    lo, hi = max(0, start - ctx), min(n, start + T + ctx)
    local = beats[(beats >= start / fps) & (beats <= (start + T) / fps)]
    if len(local) >= 2:
        near = np.min(np.abs((local + d / fps)[:, None] - beats[None, :]), axis=1)
        if np.median(near) < guard_s:
            return True
    if lo + d < 0 or hi + d > n:
        return False
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.corrcoef(onset[lo:hi], onset[lo + d:hi + d])[0, 1]
    return bool(np.isfinite(ac) and ac >= max_ac)
