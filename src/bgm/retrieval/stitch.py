"""Stitching (SPEC §5.10): candidates, continuations, beam, Viterbi. Reference code marked (tested) is ported as-is."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgm.features.keys import fifths_distance


@dataclass(frozen=True)
class Candidate:
    music_id: str     # the music FILE (not the track_id song group)
    offset: float     # window start in the file, s
    dur: float        # segment duration, s
    key: int          # stitch_key(...) pitch class, -1 unknown
    tempo: float      # BPM, <= 0 unknown
    cost: float       # unary cost = -S_fine (this segment's z-statistics)


def transition_cost(a, b, lam_jump, lam_switch, mu, tol_s=0.04):
    """(tested) Cost of moving from candidate a (segment i) to b (segment i + 1)."""
    if a.music_id == b.music_id:
        return 0.0 if abs(b.offset - (a.offset + a.dur)) <= tol_s else lam_jump
    if a.tempo > 0 and b.tempo > 0:
        r = abs(np.log2(b.tempo / a.tempo))
        tempo_d = 2.0 * abs(r - round(r))          # 0 = same, double or half time; 1 = worst
    else:
        tempo_d = 0.5
    return lam_switch + mu * (fifths_distance(a.key, b.key) + tempo_d)


def viterbi(cands, **tc):
    """(tested) cands: list over segments of lists of Candidate. Returns one index per segment minimising
    sum(cost) + sum(transition_cost). tc: lam_jump, lam_switch, mu, tol_s (finite values only)."""
    D = np.array([c.cost for c in cands[0]], dtype=float)
    back = []
    for i in range(1, len(cands)):
        C = np.array([[transition_cost(a, b, **tc) for b in cands[i]] for a in cands[i - 1]])
        tot = D[:, None] + C                        # [J_prev, J_cur]
        back.append(tot.argmin(axis=0))
        D = tot.min(axis=0) + np.array([c.cost for c in cands[i]])
    assert np.isfinite(D.min()), "path cost is not finite: use large finite penalties, not inf"
    path = [int(D.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]


def add_continuations(prev, cur, dur, file_len, score_fn):
    """(tested) Add each previous candidate's contiguous continuation to the current list.
    file_len: {music_id: seconds}; score_fn(music_id, offset, dur) -> unary cost using the SAME
    z-statistics as the current segment's other candidates."""
    seen = {(c.music_id, round(c.offset / 0.04)) for c in cur}
    out = list(cur)
    for a in prev:
        off = a.offset + a.dur
        key = (a.music_id, round(off / 0.04))
        if off + dur <= file_len[a.music_id] and key not in seen:
            seen.add(key)
            out.append(Candidate(a.music_id, off, dur, a.key, a.tempo, score_fn(a.music_id, off, dur)))
    return out


def forward_costs(prev_cands, prev_D, cur, **tc) -> np.ndarray:
    """Accumulated Viterbi cost of each current candidate given the previous segment's accumulated costs."""
    C = np.array([[transition_cost(a, b, **tc) for b in cur] for a in prev_cands])
    return (prev_D[:, None] + C).min(0) + np.array([c.cost for c in cur])


def greedy(cands) -> list[int]:
    return [int(np.argmin([c.cost for c in cs])) for cs in cands]
