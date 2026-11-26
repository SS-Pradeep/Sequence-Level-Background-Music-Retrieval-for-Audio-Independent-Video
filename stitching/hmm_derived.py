"""Method 5: HMM with costs derived from cue statistics, plus the existing hand-set Viterbi through the same interface.

Emission E[t, k] = sim[t, k] / tau (tau = the trained InfoNCE temperature, ensemble mean).
Transition j (scene t-1) -> k (scene t), counts N inside scene t's candidate set:
    continue  log pi_c
    jump      log pi_j - log N_jump(j, t)
    switch    log pi_s - log N_switch(j, t)
p_change(t) = 1 - exp(-d_t / L); pi_c = 1 - p, pi_j = rho p, pi_s = (1 - rho) p.
A category with no member in the candidate set (no continuation because the file ends, no other window of the same
file, ...) is dropped and the remaining pi are renormalised, so sum_k exp(A[j, k]) = 1 for every j.
Decode: argmax sum E + lam sum A, exact Viterbi (no beam).
"""
from __future__ import annotations

import itertools

import numpy as np

from stitching.base import Episode, Path, Stitcher, path_from_picks, relation, viterbi_ragged
from stitching.metrics import evaluate_path, val_objective

EPS = 1e-9


def derived_transition(ep: Episode, t: int, L: float, rho: float) -> np.ndarray:
    """A [K_{t-1}, K_t] log-probabilities (rows sum to 1 in probability)."""
    cont, jump, sw = relation(ep, t)
    d = ep.scene_dur[t]
    p = float(np.clip(1.0 - np.exp(-d / L), 1e-6, 1 - 1e-6))
    pi = np.array([1 - p, rho * p, (1 - rho) * p])
    n = np.stack([cont.sum(1), jump.sum(1), sw.sum(1)], 1).astype(float)      # [Kp, 3]
    avail = n > 0
    Z = (pi[None, :] * avail).sum(1, keepdims=True)
    logp = np.log(np.maximum(pi[None, :] / Z, EPS)) - np.log(np.maximum(n, 1))  # per member of each category
    A = np.full(cont.shape, -np.inf)
    for c, m in enumerate((cont, jump, sw)):
        A = np.where(m, logp[:, c:c + 1], A)
    return A


class HMMDerived(Stitcher):
    def __init__(self, tau: float, L: float, rho: float, lam: float = 1.0, tune_lam: bool = False):
        self.tau, self.L, self.rho, self.lam, self.tune_lam = tau, L, rho, lam, tune_lam
        self.name = "HMM, derived costs, λ tuned" if tune_lam else f"HMM, derived costs, λ = {lam:g}"

    def fit(self, train, val):
        if not self.tune_lam:
            return {"lam": self.lam, "tau": self.tau, "L": self.L, "rho": self.rho}
        grid = [float(x) for x in 2.0 ** np.linspace(-2, 2, 9)]
        scores = {}
        for lam in grid:
            self.lam = lam
            scores[lam] = val_objective(self, val)
        self.lam = max(scores, key=scores.get)
        return {"lam": self.lam, "tau": self.tau, "L": self.L, "rho": self.rho, "val_objective": scores}

    def decode(self, ep: Episode) -> Path:
        emit = [c.sim / self.tau for c in ep.cands]
        trans = [self.lam * derived_transition(ep, t, self.L, self.rho) for t in range(1, ep.T)]
        trans = [np.where(np.isfinite(A), A, -1e9) for A in trans]
        picks, _ = viterbi_ragged(emit, trans)
        return path_from_picks(ep, picks)


class HandSet(Stitcher):
    """Row 5: the existing hand-set costs (bgm.retrieval.stitch_eval.decode_lattice: unary -S_fused, continue 0,
    jump lam_jump, switch lam_switch + mu (key + tempo), beam 5 x J), lam_switch x lam_jump re-tuned on val over the
    existing grid with mu = the fold's D75 value -- exactly R0_tuned of D121 / D143."""
    name = "Viterbi, hand-set costs (existing)"

    def __init__(self, cfg, mu: float):
        self.cfg, self.mu = cfg, mu
        self.beam = int(cfg.stitch.beam_factor * cfg.retrieval.fine_top_j)
        self.pen = None

    def fit(self, train, val):
        g = self.cfg.stitch.grid
        scores = {}
        for a, b in itertools.product(g.lam_switch, g.lam_jump):
            self.pen = {"lam_switch": float(a), "lam_jump": float(b), "mu": self.mu}
            scores[(float(a), float(b))] = float(np.mean([evaluate_path(e, self.decode(e))["objective"] for e in val]))
        a, b = max(scores, key=scores.get)        # first max in grid order, as bgm.segmentation.retrieval_eval.tune
        self.pen = {"lam_switch": a, "lam_jump": b, "mu": self.mu}
        return {"penalties": self.pen, "val_objective": scores[(a, b)]}

    def decode(self, ep: Episode) -> Path:
        from bgm.retrieval.stitch_eval import decode_lattice
        return path_from_picks(ep, decode_lattice(ep.layers, self.pen, self.beam, float(self.cfg.stitch.tol_s)))
