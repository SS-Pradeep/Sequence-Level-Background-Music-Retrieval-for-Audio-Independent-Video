"""Track grouping at scale (PROGRESS.md D139): the vote-and-verify rule of data/tracks.py, vectorised for ~10^5 files.

data/tracks.candidates builds a Python dict over every sub-fingerprint, which cannot hold the ~270M of the YouTube
list. Here all (value, file, position) triples are sorted once; values seen more than `max_occ` times (silence,
loops, generic patterns) carry no identity and are dropped, as Chromaprint lookups do. Every remaining value
group votes for (i, j, offset) once per matching position pair, the best offset per file pair with >= min_votes
becomes a candidate, and candidates are verified by BER over the overlap exactly as group_tracks does.
"""
from __future__ import annotations

import logging

import numpy as np

from bgm.data.tracks import FP_ITEM_RATE, ber_at_offset, union_find

log = logging.getLogger(__name__)
OFF_BITS = 20                  # offsets in items, |offset| < 2^19 items (~18 h at 8 items/s)


def _pair_votes(g_file: np.ndarray, g_pos: np.ndarray, size: int) -> np.ndarray:
    """All within-group pairs of G groups of `size` occurrences ([G, size] arrays) -> packed keys of (i, j, j_pos - i_pos)
    with i < j; pairs inside one file are skipped."""
    a, b = np.triu_indices(size, 1)
    fi, fj = g_file[:, a].ravel(), g_file[:, b].ravel()
    pi, pj = g_pos[:, a].ravel(), g_pos[:, b].ravel()
    keep = fi != fj
    fi, fj, pi, pj = fi[keep], fj[keep], pi[keep], pj[keep]
    swap = fi > fj
    i, j = np.where(swap, fj, fi).astype(np.int64), np.where(swap, fi, fj).astype(np.int64)
    off = np.where(swap, pi - pj, pj - pi).astype(np.int64)
    return (i << 18 + OFF_BITS) | (j << OFF_BITS) | (off + (1 << OFF_BITS - 1))


def candidates_np(fps: list[np.ndarray], min_votes: int, max_occ: int = 32, chunk_pairs: int = 50_000_000) -> dict[tuple[int, int], int]:
    """(i, j), i < j -> the offset (j's item index - i's item index) with the most exact sub-fingerprint matches, if
    >= min_votes. Same rule as data/tracks.candidates except values occurring more than max_occ times are ignored.
    Supports up to 2^18 files."""
    n = len(fps)
    assert n < 1 << 18, "more than 2^18 files: widen the key packing"
    lens = np.array([len(f) for f in fps], np.int64)
    vals = np.concatenate(fps).astype(np.uint32) if n else np.zeros(0, np.uint32)
    file = np.repeat(np.arange(n, dtype=np.int32), lens)
    pos = np.concatenate([np.arange(k, dtype=np.int32) for k in lens]) if n else np.zeros(0, np.int32)
    order = np.argsort(vals, kind="stable")
    vals, file, pos = vals[order], file[order], pos[order]
    starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
    sizes = np.diff(np.r_[starts, len(vals)])
    keep = (sizes >= 2) & (sizes <= max_occ)
    log.info("fingerprint index: %d items, %d distinct values, %d shared by 2..%d items", len(vals), len(starts), keep.sum(), max_occ)

    keys_acc, counts_acc = [], []

    def flush(parts):
        k, c = np.unique(np.concatenate(parts), return_counts=True)
        keys_acc.append(k)
        counts_acc.append(c)

    parts, pending = [], 0
    for s in np.unique(sizes[keep]):
        st = starts[keep & (sizes == s)]
        per = s * (s - 1) // 2
        step = max(1, chunk_pairs // per)
        for c in range(0, len(st), step):
            idx = st[c:c + step, None] + np.arange(s)[None]
            parts.append(_pair_votes(file[idx], pos[idx], int(s)))
            pending += len(parts[-1])
            if pending >= chunk_pairs:
                flush(parts)
                parts, pending = [], 0
    if parts:
        flush(parts)
    if not keys_acc:
        return {}
    keys = np.concatenate(keys_acc)
    cnts = np.concatenate(counts_acc)
    order = np.argsort(keys, kind="stable")
    keys, cnts = keys[order], cnts[order]
    st = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    keys, cnts = keys[st], np.add.reduceat(cnts, st)
    m = cnts >= min_votes
    keys, cnts = keys[m], cnts[m]
    pair = keys >> OFF_BITS
    off = (keys & ((1 << OFF_BITS) - 1)) - (1 << OFF_BITS - 1)
    # best offset per pair: most votes, ties to the smallest offset (data/tracks keeps the first one it meets)
    order = np.lexsort((off, -cnts, pair))
    pair, off = pair[order], off[order]
    first = np.r_[True, pair[1:] != pair[:-1]]
    pair, off = pair[first], off[first]
    i, j = pair >> 18, pair & ((1 << 18) - 1)
    return {(int(a), int(b)): int(o) for a, b, o in zip(i, j, off)}


def group_fingerprints(fps: list[np.ndarray], ber_max: float, min_overlap_s: float, min_votes: int, max_occ: int = 32,
                       search: int = 2) -> tuple[np.ndarray, list[tuple[int, int, int, float, float]]]:
    """Component label per file and the accepted links (i, j, offset, ber, overlap_s). A link needs BER <= ber_max at
    the best offset within +/- `search` items of the voted one, over >= min_overlap_s."""
    links = []
    for (i, j), off in candidates_np(fps, min_votes, max_occ).items():
        ber, n, o = min((*ber_at_offset(fps[i], fps[j], o), o) for o in range(off - search, off + search + 1))
        if ber <= ber_max and n / FP_ITEM_RATE >= min_overlap_s:
            links.append((i, j, o, ber, n / FP_ITEM_RATE))
    return union_find(len(fps), [(a, b) for a, b, *_ in links]), links
