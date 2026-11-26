"""Chained shot sequences and their music-change labels (D140).

Sequences are the cue-switch predictor's (bgm.cue.sequences): training sequences are drawn per item with
`sample_sequence(..., 'blocked')` (60-180 s, blocks of 2-8 clips of one film, never the same film twice in a row);
val / test are the frozen title-split sets (data/manifests/cue_sequences_title.json). Here a sequence is cut at shot
boundaries instead of D75 segments: the shots of its clips, concatenated in order, so a clip join is the boundary
between the last shot of one clip and the first shot of the next.

Boundary after shot s (s = 0 .. S - 2), labels from `boundary_labels`:
  inside a clip                -> 0, type 'clip'
  join, same film              -> 1 if the track groups differ else 0, type 'film'
  join, different films        -> 1, type 'cross'
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bgm.cue.sequences import boundary_labels


@dataclass
class ClipShots:
    pair_id: str
    shots: np.ndarray            # [S, 2] start_s, end_s, tiling [0, dur]
    feats: np.ndarray | None     # [S, 2048] BaSSL shot features (None for detectors without them)
    flow: np.ndarray | None      # [S] mean log motion of each shot (dyn_25hz column 0)


@dataclass
class Chain:
    parts: list[str]
    shot_part: np.ndarray        # [S] index into parts
    shots: np.ndarray            # [S, 2] in sequence time, s
    btimes: np.ndarray           # [S - 1] boundary times, s
    labels: np.ndarray           # [S - 1] int8
    types: list[str]             # [S - 1]
    is_join: np.ndarray          # [S - 1] bool
    joins: np.ndarray            # [n_parts - 1] join times, s
    duration: float
    feats: np.ndarray | None = None
    flow: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def chain(parts: list[str], clips: dict[str, ClipShots], track: dict[str, str], film: dict[str, str]) -> Chain:
    off, shots, sp, feats, flow, joins = 0.0, [], [], [], [], []
    for k, p in enumerate(parts):
        c = clips[p]
        shots.append(c.shots + off)
        sp += [k] * len(c.shots)
        if c.feats is not None:
            feats.append(c.feats)
        if c.flow is not None:
            flow.append(c.flow)
        off += float(c.shots[-1, 1])
        joins.append(off)
    shots = np.concatenate(shots)
    sp = np.array(sp)
    lab, typ = boundary_labels(list(sp), parts, track, film)
    return Chain(parts, sp, shots, shots[:-1, 1].copy(), lab, typ, sp[1:] != sp[:-1], np.array(joins[:-1]), off,
                 np.concatenate(feats) if feats else None, np.concatenate(flow) if flow else None)


def scalar_features(ch: Chain) -> np.ndarray:
    """[S - 1, 4]: log duration of the shot before / after each boundary, mean log motion before / after."""
    d = np.log(np.maximum(ch.shots[:, 1] - ch.shots[:, 0], 0.04))
    f = ch.flow if ch.flow is not None else np.zeros(len(ch.shots))
    return np.stack([d[:-1], d[1:], f[:-1], f[1:]], 1).astype(np.float32)


def edge_region(ch: Chain, n: int = 8) -> tuple[float, float]:
    """Times t <= lo or t >= hi lie where a boundary's +-n shot window is clamped at the sequence edges."""
    B = len(ch.btimes)
    if B == 0:
        return ch.duration, 0.0
    return float(ch.btimes[min(n - 1, B - 1)]), float(ch.btimes[max(B - n + 1, 0)])


def _cos_means(cs: np.ndarray, a: np.ndarray, m: np.ndarray, b: np.ndarray) -> np.ndarray:
    """cs: cumulative sums [N + 1, D] of unit vectors; cosine of mean(rows a..m-1) vs mean(rows m..b-1)."""
    left = cs[m] - cs[a]
    right = cs[b] - cs[m]
    num = (left * right).sum(1)
    return num / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-9)


def shot_contrast(feats: np.ndarray, ks=(1, 2, 4, 8)) -> np.ndarray:
    """D142: [S - 1, len(ks)] cosine between the mean of the k shots before each boundary and the k shots after it
    (fewer at the sequence edges). feats [S, D] shot features."""
    S = len(feats)
    u = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cs = np.concatenate([np.zeros((1, u.shape[1])), np.cumsum(u, 0)])
    m = np.arange(1, S)                                     # boundary after shot m - 1
    return np.stack([_cos_means(cs, np.maximum(m - k, 0), m, np.minimum(m + k, S)) for k in ks], 1).astype(np.float32)


def time_contrast(emb: np.ndarray, centres: np.ndarray, btimes: np.ndarray, ws=(2.0, 4.0)) -> np.ndarray:
    """D142: [B, len(ws)] cosine between the mean 1 Hz vector centred in [t - w, t) and in [t, t + w) for each boundary
    time t (at least one vector on each side, clamped to the sequence)."""
    T = len(emb)
    u = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cs = np.concatenate([np.zeros((1, u.shape[1])), np.cumsum(u, 0)])
    m = np.clip(np.searchsorted(centres, btimes), 1, T - 1)
    out = []
    for w in ws:
        a = np.minimum(np.searchsorted(centres, btimes - w), m - 1)
        b = np.maximum(np.searchsorted(centres, btimes + w), m + 1)
        out.append(_cos_means(cs, a, m, np.minimum(b, T)))
    return np.stack(out, 1).astype(np.float32)
