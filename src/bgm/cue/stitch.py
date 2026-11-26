"""Viterbi with per-boundary transition costs from the cue-switch predictor (Step 3; PROGRESS.md D78).

At boundary i (between segment i and i + 1) with predicted change probability p_i, as additive costs (the negated
scores of the brief):
    continue (same file, contiguous)       -alpha * log(1 - p_i)
    switch   (another file)                -alpha * log(p_i) + mu * (fifths + tempo distance)   (the existing term)
    jump     (same file, not contiguous)   -alpha * log(p_i) + lam_jump
Every path crosses each boundary exactly once, so subtracting the continue cost at every boundary changes neither the
best path nor the beam's ranking. What is left is the existing `transition_matrix` with
    lam_switch_i = alpha * (log(1 - p_i) - log(p_i)) = -alpha * logit_i,    lam_jump_i = lam_switch_i + lam_jump.
With a constant logit = -lam_switch / alpha and lam_jump := lam_jump_fixed - lam_switch this IS the fixed-penalty
decoder (tests/test_cue.py). lam_jump may be negative (fold 0's tuned penalties need -0.75, user 2026-10-19).

Inside a segment nothing changes (candidates and unary costs are the emission lattice's). `stitch_eval.decode_lattice`
is not modified; this is a separate decoder with the same beam rule.
"""
from __future__ import annotations

import numpy as np

from bgm.retrieval.stitch_eval import Layer, transition_matrix


def boundary_penalties(logits: np.ndarray, alpha: float, lam_jump: float, mu: float) -> list[dict]:
    s = -float(alpha) * np.asarray(logits, float)
    return [{"lam_switch": float(si), "lam_jump": float(si) + float(lam_jump), "mu": float(mu)} for si in s]


def decode_lattice_cue(layers: list[Layer], logits: np.ndarray, alpha: float, lam_jump: float, mu: float,
                       beam: int | None, tol_s: float = 0.04) -> list[int]:
    """As stitch_eval.decode_lattice, with the penalties of boundary i - 1 between layers i - 1 and i.
    logits: [len(layers) - 1] predicted change logits."""
    assert len(logits) == len(layers) - 1, (len(logits), len(layers))
    pens = boundary_penalties(logits, alpha, lam_jump, mu)
    act = np.arange(len(layers[0].mid))
    D = layers[0].cost[act].copy()
    acts, backs = [act], []
    for i in range(1, len(layers)):
        L = layers[i]
        conts = L.cont_of[acts[-1]]
        cur = np.unique(np.r_[np.arange(L.n_top), conts[conts >= 0]])
        T = transition_matrix(layers[i - 1].subset(acts[-1]), L.subset(cur), pens[i - 1], tol_s)
        tot = D[:, None] + T
        bk = tot.argmin(0)
        Dc = tot.min(0) + L.cost[cur]
        if beam is not None and len(cur) > beam:
            keep = np.sort(np.argsort(Dc)[:beam])
            cur, Dc, bk = cur[keep], Dc[keep], bk[keep]
        acts.append(cur), backs.append(bk)
        D = Dc
    assert np.isfinite(D.min()), "path cost is not finite: use large finite penalties, not inf"
    j = int(np.argmin(D))
    path = [j]
    for bk in reversed(backs):
        j = int(bk[j])
        path.append(j)
    path = path[::-1]
    return [int(acts[i][path[i]]) for i in range(len(layers))]


def constant_equivalent(pen: dict, n_boundaries: int, alpha: float = 1.0) -> tuple[np.ndarray, float, float, float]:
    """(logits, alpha, lam_jump, mu) that reproduce a fixed penalty set exactly."""
    return (np.full(n_boundaries, -pen["lam_switch"] / alpha), alpha, pen["lam_jump"] - pen["lam_switch"], pen["mu"])


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)
