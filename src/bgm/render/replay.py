"""Replay penalty for renders (D101): do not play the same passage of a music file again and again.

The stitcher is first-order (each segment only sees the previous one), so with a uniform brief one short file can win
every segment and restart from its top many times. After a decode, each segment's candidates are charged
`penalty` x (fraction of their window already played by EARLIER segments of the same file); the lattice is decoded
again and the charges add up over rounds, until no segment replays or `rounds` is reached. Continuations are never
charged: their seconds are new. Opt-in: `render.replay_penalty` (0 = off); `render.max_plays` passes allowed (2 = one reprise).
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

log = logging.getLogger(__name__)


def covered(a: float, b: float, spans: list[tuple[float, float]], times: int = 1) -> float:
    """Seconds of [a, b] covered by at least `times` of `spans` (which may overlap one another)."""
    ev = sorted([(max(a, s), 1) for s, e in spans if min(b, e) > max(a, s)] + [(min(b, e), -1) for s, e in spans if min(b, e) > max(a, s)])
    tot, depth, last = 0.0, 0, a
    for t, dv in ev:
        if depth >= times:
            tot += t - last
        depth, last = depth + dv, t
    return tot


def replay_fracs(layers, picks: list[int], max_plays: int = 1) -> list[np.ndarray]:
    """Per layer [n_cand]: fraction of each candidate's window already played >= max_plays times by the chosen windows
    of earlier layers (max_plays 2: a passage may come back once, as a reprise)."""
    played: dict[int, list[tuple[float, float]]] = {}
    out = []
    for L, j in zip(layers, picks):
        out.append(np.array([covered(o, o + d, played.get(int(m), []), max_plays) / max(d, 1e-6) for m, o, d in zip(L.mid, L.off, L.dur)]))
        played.setdefault(int(L.mid[j]), []).append((float(L.off[j]), float(L.off[j] + L.dur[j])))
    return out


def decode_without_replays(decode, layers, penalty: float, max_plays: int = 1, rounds: int = 20, min_frac: float = 0.5):
    """decode(layers) -> picks, re-run with growing replay charges. Returns (picks, layers with charged costs, info)."""
    base = [L.cost.copy() for L in layers]
    charge = [np.zeros(len(L.cost)) for L in layers]
    picks = decode(layers)
    n0 = None
    for r in range(rounds + 1):
        fr = replay_fracs(layers, picks, max_plays)
        n = sum(f[j] >= min_frac for f, j in zip(fr, picks))
        n0 = n if n0 is None else n0
        if n == 0 or r == rounds:
            break
        charge = [c + penalty * f for c, f in zip(charge, fr)]
        layers = [replace(L, cost=b + c) for L, b, c in zip(layers, base, charge)]
        picks = decode(layers)
    log.info("replay penalty %.1f (plays allowed %d): segments over it %d -> %d after %d round(s)", penalty, max_plays, n0, n, r)
    return picks, layers, {"penalty": penalty, "max_plays": max_plays, "replays_before": int(n0), "replays_after": int(n), "rounds": r}
