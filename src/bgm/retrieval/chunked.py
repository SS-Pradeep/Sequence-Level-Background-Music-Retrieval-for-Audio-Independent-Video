"""Chunked querying for long videos, and score ensembles.

A long query is split into equal chunks of about `chunk_s`; each library track is represented by its `window_s`
windows (hop `hop_s`; a track no longer than `window_s` is one window). A track scores the mean, over the query's
chunks, of its best window's score. Ensembles z-score each model's scores across the gallery per query, then average.
"""
from __future__ import annotations

import numpy as np


def equal_chunks(duration_s: float, chunk_s: float = 20.0) -> list[tuple[float, float]]:
    """[(start_s, end_s)] of max(1, round(duration / chunk_s)) equal chunks tiling [0, duration_s]."""
    n = max(1, int(round(duration_s / chunk_s)))
    e = np.linspace(0.0, duration_s, n + 1)
    return [(float(a), float(b)) for a, b in zip(e[:-1], e[1:])]


def window_starts(duration_s: float, window_s: float = 20.0, hop_s: float = 5.0) -> np.ndarray:
    """Start times (s) of `window_s` windows every `hop_s`, the last one flush with the end; [0] if the track is short."""
    if duration_s <= window_s:
        return np.array([0.0])
    s = np.arange(0.0, duration_s - window_s + 1e-9, hop_s)
    if duration_s - window_s - s[-1] > 1e-6:
        s = np.r_[s, duration_s - window_s]
    return s


def chunked_track_scores(chunk_emb: np.ndarray, window_emb: list[np.ndarray]) -> np.ndarray:
    """chunk_emb [C, D]; window_emb: per track [W_t, D] (unit vectors) -> [G] mean over chunks of max over windows."""
    return np.array([float((chunk_emb @ W.T).max(1).mean()) for W in window_emb])


def zscore(scores: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Standardize a query's scores across its gallery (last axis)."""
    s = np.asarray(scores, float)
    return (s - s.mean(-1, keepdims=True)) / np.maximum(s.std(-1, keepdims=True), floor)


def ensemble(score_list: list[np.ndarray]) -> np.ndarray:
    """Mean of per-model standardized scores (each [G] or [Q, G])."""
    return np.mean([zscore(s) for s in score_list], 0)
