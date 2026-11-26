"""Segmental (semi-Markov) Viterbi: exact joint DP over the segmentation of the scene sequence into music runs and
the music of each run.

A run = scenes [a, b) played on ONE file as one contiguous excerpt (window of scene t starts at s0 + (start_t - start_a)).
    run score  = (1 - m) * sum_t z_t(file, contiguous window)  +  m * sum_t max_window z_t(file)
                 (z_t = scene t's raw similarities standardised over all windows of the shortlisted files, as the
                 existing pipeline z-scores; m mixes in the run's track-level evidence)
    F[b] = max_{a < b} F[a] + max_file run(a, b, file) - lam_sw * [a > 0] + beta * vis(a) * [a > 0]
    vis(a) = cosine distance between the projected embeddings of scenes a-1 and a
Exact (all segmentations, all files of the shortlist, all start offsets on the 0.5 s window grid). lam_sw, beta, m
tuned on val (objective = track acc + switch F1, as every other row). Two-stage change-point + retrieve is the special
case with the segmentation fixed by the visual change alone.
"""
from __future__ import annotations

import itertools

import numpy as np

from stitching.base import Episode, Path, Stitcher
from stitching.metrics import evaluate_path

HOP = 0.5


def run_tables(ep: Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every run [a, b): best contiguous score C[a, b], its file / start, and track evidence B[a, b] of that file
    ... computed for every file; returns C_f [F, T, T+1], B_f [F, T, T+1], start_f [F, T, T+1], files."""
    files = sorted(ep.dense)
    T = ep.T
    # per-scene standardisation over every window of every shortlisted file
    allv = [np.concatenate([ep.dense[f][1][t][np.isfinite(ep.dense[f][1][t])] for f in files]) for t in range(T)]
    mu = np.array([v.mean() if len(v) else 0.0 for v in allv])
    sd = np.array([max(v.std(), 1e-3) if len(v) else 1.0 for v in allv])
    starts = ep.segs[:, 0]
    C = np.full((len(files), T, T + 1), -np.inf)
    B = np.full((len(files), T, T + 1), -np.inf)
    S = np.zeros((len(files), T, T + 1))
    for fi, f in enumerate(files):
        grid, tab = ep.dense[f]
        z = (tab - mu[:, None]) / sd[:, None]                    # [T, W], -inf invalid
        W = len(grid)
        best = np.max(z, 1)                                      # [T]
        for a in range(T):
            shift = np.round((starts[a:] - starts[a]) / HOP).astype(int)      # [T - a]
            idx = np.arange(W)[None, :] + shift[:, None]                     # [T - a, W]
            ok = idx < W
            vals = np.where(ok, z[np.arange(a, T)[:, None], np.minimum(idx, W - 1)], -np.inf)
            cum = np.cumsum(vals, 0)                                         # -inf propagates
            j = np.argmax(cum, 1)
            C[fi, a, a + 1:] = cum[np.arange(T - a), j]
            S[fi, a, a + 1:] = grid[j]
            B[fi, a, a + 1:] = np.cumsum(best[a:])
    return C, B, S, np.array(files)


class SemiViterbi(Stitcher):
    name = "Segmental Viterbi (semi-Markov)"

    def __init__(self, lam: float = 2.0, beta: float = 0.0, m: float = 0.0):
        self.lam, self.beta, self.m = lam, beta, m
        self._cache = {}

    def _tables(self, ep):
        if ep.key not in self._cache:
            self._cache[ep.key] = run_tables(ep)
        return self._cache[ep.key]

    def fit(self, train, val):
        grid = list(itertools.product((0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0), (0.0, 2.0, 5.0, 10.0, 20.0), (0.0, 0.5, 1.0)))
        scores = {}
        for lam, beta, m in grid:
            self.lam, self.beta, self.m = lam, beta, m
            scores[(lam, beta, m)] = float(np.mean([evaluate_path(e, self.decode(e))["objective"] for e in val]))
        self.lam, self.beta, self.m = max(scores, key=scores.get)
        return {"lam_sw": self.lam, "beta": self.beta, "m": self.m, "val_objective": max(scores.values())}

    def decode(self, ep: Episode) -> Path:
        C, B, S, files = self._tables(ep)
        T = ep.T
        R = (1 - self.m) * C + self.m * np.where(np.isfinite(C), B, -np.inf)    # only runs that can be laid
        fbest = np.argmax(R, 0)                                                # [T, T+1]
        rbest = np.max(R, 0)
        E = ep.scene_emb
        vis = np.r_[0.0, 1 - (E[1:] * E[:-1]).sum(1)]
        F = np.full(T + 1, -np.inf)
        F[0] = 0.0
        arg = np.zeros(T + 1, int)
        for b in range(1, T + 1):
            a = np.arange(b)
            v = F[a] + rbest[a, b] + np.where(a > 0, -self.lam + self.beta * vis[a], 0.0)
            k = int(np.argmax(v))
            F[b], arg[b] = v[k], k
        mid, off = np.zeros(T, int), np.zeros(T)
        b = T
        if not np.isfinite(F[T]):                     # no layable run covers some scene: per-scene best
            for t in range(T):
                j = int(np.argmax(ep.cands[t].sim))
                mid[t], off[t] = ep.cands[t].mid[j], ep.cands[t].off[j]
            return Path(mid, off)
        while b > 0:
            a = arg[b]
            f = files[fbest[a, b]]
            s0 = S[fbest[a, b], a, b]
            mid[a:b] = f
            off[a:b] = s0 + (ep.segs[a:b, 0] - ep.segs[a, 0])
            b = a
        return Path(mid, off)
