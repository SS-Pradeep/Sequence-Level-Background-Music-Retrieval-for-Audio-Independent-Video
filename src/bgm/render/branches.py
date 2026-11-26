"""Render-only extra scoring branches (D100 pace, D101 scene brief) layered over a MusicIndex.

Each layer wraps the index below it, adds one branch to every window's raw scores and to the file-level scores of the
coarse shortlist, and delegates everything else. The pipeline standardizes each branch over the segment's window set
and fuses it with its weight, like the trained branches (SPEC §5.8).
"""
from __future__ import annotations

import numpy as np

from bgm.retrieval.index import SegmentQuery, VideoQuery
from bgm.retrieval.pipeline import fuse, zstats


class BranchIndex:
    name = ""

    def __init__(self, index):
        self._index = index

    def __getattr__(self, name):
        return getattr(self._index, name)

    def window_branch(self, a: float, b: float, fi: np.ndarray, st: np.ndarray, d: float) -> np.ndarray:
        """Raw score [W] of windows (file index fi, start st s, length d s) for video span [a, b] s."""
        raise NotImplementedError

    def file_branch(self, a: float, b: float) -> np.ndarray:
        """Raw score [G] of every index file for video span [a, b] s."""
        raise NotImplementedError

    def window_scores(self, q: SegmentQuery, fi: np.ndarray, st: np.ndarray, d: float) -> dict[str, np.ndarray]:
        raw = self._index.window_scores(q, fi, st, d)
        raw[self.name] = self.window_branch(q.a, q.b, fi, st, d)
        return raw

    def file_branches(self, a: float, b: float) -> dict[str, np.ndarray]:
        inner = self._index.file_branches(a, b) if isinstance(self._index, BranchIndex) else {}
        return {**inner, self.name: self.file_branch(a, b)}


def branch_shortlist(index: BranchIndex, vq: VideoQuery, w_coarse: dict, K: int) -> list[int]:
    """coarse_shortlist with the extra file-level branches: union of the whole-video and each segment's top K files."""
    keep = set()
    for a, b in [(0.0, vq.duration)] + list(vq.segments):
        gv, pv = vq.pooled(a, b)
        raw = {**index.file_scores(gv, pv), **index.file_branches(a, b)}
        keep.update(np.argsort(-fuse(raw, zstats(raw), w_coarse))[:K].tolist())
    return sorted(keep)


class LengthIndex(BranchIndex):
    """D105: the "length" branch, -(share of the segment the window's file cannot cover without looping). Long single
    shots (20 s +) otherwise get short files that the renderer loops, restart after restart."""
    name = "length"

    def window_branch(self, a, b, fi, st, d):
        dur = np.array([self.items[int(i)].duration for i in fi])
        return -np.maximum(0.0, d - (dur - np.minimum(st, dur))) / max(d, 1e-6)

    def file_branch(self, a, b):
        dur = np.array([it.duration for it in self.items])
        return -np.maximum(0.0, (b - a) - dur) / max(b - a, 1e-6) if (b - a) < 60 else np.zeros(len(dur))
