"""Method 6: learnable CRF over the shared candidate lattice (built on stitching/learnable_viterbi.py).

emission    alpha * sim, alpha = exp(log_alpha) initialised to 1 / tau
transition  lam * (A_derived + A_theta), lam initialised to 1; A_derived = the step-5 HMM transition (fixed);
            A_theta = TransitionScorer(pair, ctx) (base linear + context gate), initialised to 0, so training starts
            at the derived HMM
pair feats  is_continue, is_jump, is_switch, -log N_jump (on jumps), -log N_switch (on switches), key distance (circle
            of fifths, per-file key), |log2 tempo ratio| folded at octaves, |loudness out(j) - in(k)| / 10 dB,
            1 - cos(MERT last 2 s of j, MERT first 2 s of k), beat-phase error (circular, end of j vs start of k)
ctx feats   SigLIP cosine distance between scenes t-1 and t, mood shift (half L1 of the calibrated mood
            distributions), log d_t, log d_{t-1}, t / T. No film / track ID, no absolute position.
            Without context: the ctx vector is all zeros (the gate reduces to a constant).
loss        CRF NLL (forward algorithm) per scene + gamma * ||theta||^2 (theta = the learned transition parameters;
            theta = 0 is the derived prior); Adam lr 1e-2, weight decay 1e-3, batches of 8 episodes, early stopping on
            val NLL (patience 5, max 40 epochs). Gold is inserted into the candidates of train and val-NLL episodes only.
Encoders and projection heads are frozen (sims are precomputed).
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch

from bgm.retrieval.stitch_eval import fifths_matrix, tempo_matrix
from stitching.base import Episode, Path, Stitcher, path_from_picks, relation, viterbi_ragged
from stitching.hmm_derived import derived_transition
from stitching.learnable_viterbi import LearnableStitcher

log = logging.getLogger(__name__)
PAIR_NAMES = ("is_continue", "is_jump", "is_switch", "-logN_jump", "-logN_switch", "key_dist", "tempo_dist",
              "loudness_diff_10dB", "mert_dist", "beat_phase_err")
CTX_NAMES = ("siglip_dist", "mood_shift", "log_d_t", "log_d_prev", "t_over_T")


def pair_features(ep: Episode, t: int) -> np.ndarray:
    P, C = ep.cands[t - 1], ep.cands[t]
    cont, jump, sw = relation(ep, t)
    nj = np.maximum(jump.sum(1), 1)[:, None]
    ns = np.maximum(sw.sum(1), 1)[:, None]
    ph = np.abs(P.ph_out[:, None] - C.ph_in[None, :])
    f = [cont, jump, sw, jump * -np.log(nj), sw * -np.log(ns),
         fifths_matrix(ep.file_key[P.mid], ep.file_key[C.mid]), tempo_matrix(ep.file_tempo[P.mid], ep.file_tempo[C.mid]),
         np.abs(P.rms_out[:, None] - C.rms_in[None, :]) / 10.0, 1.0 - P.emb_out @ C.emb_in.T, np.minimum(ph, 1 - ph)]
    return np.stack([np.broadcast_to(x, cont.shape).astype(np.float32) for x in f], -1)


def ctx_features(ep: Episode, t: int) -> np.ndarray:
    d = ep.scene_dur
    return np.array([1 - float(ep.scene_sig[t] @ ep.scene_sig[t - 1]), 0.5 * float(np.abs(ep.scene_mood[t] - ep.scene_mood[t - 1]).sum()),
                     np.log(max(d[t], 0.1)), np.log(max(d[t - 1], 0.1)), t / ep.T], np.float32)


class Tensors:
    """One episode on the device."""

    def __init__(self, ep: Episode, L: float, rho: float, use_ctx: bool, dev):
        self.T = ep.T
        self.sim = [torch.tensor(c.sim, dtype=torch.float32, device=dev) for c in ep.cands]
        self.pair = [torch.from_numpy(pair_features(ep, t)).to(dev) for t in range(1, ep.T)]
        ctx = [ctx_features(ep, t) if use_ctx else np.zeros(len(CTX_NAMES), np.float32) for t in range(1, ep.T)]
        self.ctx = [torch.from_numpy(c).to(dev) for c in ctx]
        self.ctx_np = np.array([ctx_features(ep, t) for t in range(1, ep.T)]) if ep.T > 1 else np.zeros((0, len(CTX_NAMES)))
        self.A = [torch.tensor(derived_transition(ep, t, L, rho), dtype=torch.float32, device=dev) for t in range(1, ep.T)]
        self.gold = None if ep.gold is None else torch.tensor(ep.gold, device=dev)


class CRFModel(torch.nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.core = LearnableStitcher(len(PAIR_NAMES), len(CTX_NAMES))
        with torch.no_grad():
            self.core.log_alpha.fill_(float(np.log(1.0 / tau)))
            self.core.log_lam.fill_(0.0)
            self.core.trans.base.weight.zero_(), self.core.trans.base.bias.zero_()
            self.core.trans.ctx_gate[-1].weight.zero_(), self.core.trans.ctx_gate[-1].bias.zero_()

    def theta(self):
        return list(self.core.trans.parameters())

    def scores(self, x: Tensors):
        a, lam = self.core.log_alpha.exp(), self.core.log_lam.exp()
        emit = [a * s for s in x.sim]
        trans = [lam * (A + self.core.trans(p[None], c[None])[0]) for A, p, c in zip(x.A, x.pair, x.ctx)]
        return emit, trans

    def nll(self, x: Tensors) -> torch.Tensor:
        emit, trans = self.scores(x)
        a = emit[0]
        for t in range(1, x.T):
            a = torch.logsumexp(a[:, None] + trans[t - 1], 0) + emit[t]
        g = x.gold
        gold = sum(emit[t][g[t]] for t in range(x.T)) + sum(trans[t - 1][g[t - 1], g[t]] for t in range(1, x.T))
        return (torch.logsumexp(a, 0) - gold) / x.T

    @torch.no_grad()
    def switch_marginals(self, ep: Episode, x: Tensors) -> np.ndarray:
        """[T-1] posterior P(file changes at boundary t) by forward-backward."""
        emit, trans = self.scores(x)
        T = x.T
        al = [emit[0]]
        for t in range(1, T):
            al.append(torch.logsumexp(al[-1][:, None] + trans[t - 1], 0) + emit[t])
        be = [torch.zeros_like(emit[-1])]
        for t in range(T - 1, 0, -1):
            be.insert(0, torch.logsumexp(trans[t - 1] + (emit[t] + be[0])[None, :], 1))
        logz = torch.logsumexp(al[-1], 0)
        out = []
        for t in range(1, T):
            _, _, sw = relation(ep, t)
            j = al[t - 1][:, None] + trans[t - 1] + (emit[t] + be[t])[None, :] - logz
            m = torch.tensor(sw, device=j.device)
            out.append(float(torch.exp(torch.logsumexp(j[m], 0)).clamp(0, 1)))
        return np.array(out)


class CRF(Stitcher):
    def __init__(self, tau: float, L: float, rho: float, use_ctx: bool, gammas=(0.0,), seed: int = 0, device: str = "cuda",
                 name: str = "CRF"):
        self.tau, self.L, self.rho, self.use_ctx, self.gammas, self.seed, self.dev = tau, L, rho, use_ctx, gammas, seed, device
        self.name = name
        self.model = None

    def _x(self, ep):
        return Tensors(ep, self.L, self.rho, self.use_ctx, self.dev)

    def _train(self, tr: list[Tensors], va: list[Tensors], gamma: float) -> tuple[CRFModel, float, list]:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng([self.seed, 17])
        m = CRFModel(self.tau).to(self.dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)

        def val_nll():
            with torch.no_grad():
                return float(np.mean([m.nll(x).item() for x in va]))
        best, best_state, bad, hist = val_nll(), copy.deepcopy(m.state_dict()), 0, []
        hist.append({"epoch": 0, "val_nll": best})
        for ep_i in range(1, 41):
            order = rng.permutation(len(tr))
            tl = []
            for b in range(0, len(order), 8):
                loss = sum(m.nll(tr[i]) for i in order[b:b + 8]) / len(order[b:b + 8])
                if gamma > 0:
                    loss = loss + gamma * sum((p ** 2).sum() for p in m.theta())
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl.append(loss.item())
            v = val_nll()
            hist.append({"epoch": ep_i, "train_loss": float(np.mean(tl)), "val_nll": v,
                         "alpha": float(m.core.log_alpha.exp()), "lam": float(m.core.log_lam.exp())})
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= 5:
                    break
        m.load_state_dict(best_state)
        return m, best, hist

    def fit(self, train, val):
        tr = [self._x(e.with_gold()) for e in train if e.T > 1]
        va = [self._x(e.with_gold()) for e in val if e.T > 1]
        runs = {g: self._train(tr, va, g) for g in self.gammas}
        g = min(runs, key=lambda k: runs[k][1])
        self.model, vbest, hist = runs[g]
        self.gamma = g
        c = self.model.core
        return {"gamma": g, "val_nll_by_gamma": {str(k): v[1] for k, v in runs.items()}, "val_nll": vbest,
                "alpha": float(c.log_alpha.exp()), "lam": float(c.log_lam.exp()), "alpha_init": 1 / self.tau,
                "base_w": dict(zip(PAIR_NAMES, c.trans.base.weight.detach().cpu().numpy()[0].round(4).tolist())),
                "epochs": len(hist) - 1, "history": hist, "n_train": len(tr), "n_val": len(va), "seed": self.seed}

    def decode(self, ep: Episode) -> Path:
        if ep.T == 1:
            return path_from_picks(ep, [int(np.argmax(ep.cands[0].sim))])
        with torch.no_grad():
            emit, trans = self.model.scores(self._x(ep))
        picks, _ = viterbi_ragged([e.cpu().numpy() for e in emit], [t.cpu().numpy() for t in trans])
        return path_from_picks(ep, picks)

    def switch_marginals(self, ep: Episode) -> np.ndarray:
        return self.model.switch_marginals(ep, self._x(ep)) if ep.T > 1 else np.zeros(0)
