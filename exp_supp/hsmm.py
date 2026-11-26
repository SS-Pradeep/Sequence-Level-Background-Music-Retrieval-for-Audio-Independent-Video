"""Hidden semi-Markov (cue-level) decoder on the Table 3 lattices (stitching/ harness, imported read-only).

Hidden unit = a cue u = (a, b, j): scenes a..b-1 played as ONE continuous stretch of one track. It starts at candidate
j of scene a (any candidate of the augmented set C~_a, so exactly the Table 3 candidates) and follows that window's
continuation chain c+ = (k, tau + d) through C~_{a+1}, ... (Eq. (15)); a cue cannot run past its track's end because
the chain stops there.

Score of a soundtrack u_1..u_R (maximised exactly by a segmental Viterbi DP):
    sum_r  E(u_r) + gamma * log P_dur(D(u_r)) + alpha * sum_{internal boundaries i of u_r} log(1 - p_i)
  - sum_{r>1} seam(u_{r-1}, u_r)
  E(a, b, j)  = sum_{t=a}^{b-1} [(1 - m) e_t(chain_j(t)) + m max_{c in C~_t, file(c) = file(j)} e_t(c)]
                e_t(c) = sim_t(c) / tau: the Table 3 HMM emission (same ensemble cosine, same temperature)
                m pools track-level evidence over the cue (m = 0: the stretch's own windows only)
  P_dur       = log-normal fitted (MLE) to the cue lengths of the fold's TRAINING films' BGM (stitching.cue_stats
                segmentation: Foote novelty on chroma + MFCC, file ends count as cue ends)
  seam        = Eq. (20) at the boundary into scene a:  -alpha log p_a + lambda_jump        (same file: jump)
                                                      -alpha log p_a + mu (d_key + d_tempo) (switch)
  p_i         = 1 - exp(-d_i / L), d_i = duration of the scene after boundary i, L = the final row's gold-path L
                (exactly the p_i of the final system, "HMM, prior from gold paths, alpha tuned")
alpha, lambda_jump, mu, gamma, m tuned per fold on the val sequences (objective = track acc. + switch F1, as every
Table 3 row). gamma = 0 and m = 0 is the final HMM restricted to the chain structure (without its -log N terms).

Known simplification: the DP state is the last cue's FILE, so a new cue that starts exactly at the previous cue's
continuation is charged as a jump; the metrics are computed on the decoded windows, where it is a continue.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from bgm.retrieval.stitch_eval import fifths_matrix, tempo_matrix
from stitching.base import Episode, Path, Stitcher, relation
from stitching.metrics import evaluate_path

NEG = -np.inf


# ------------------------------------------------------------------ duration prior
class LogNormal:
    def __init__(self, durations_s):
        x = np.log(np.maximum(np.asarray(durations_s, float), 1e-3))
        self.mu, self.sigma = float(x.mean()), float(max(x.std(), 1e-3))
        self.n = len(x)

    def logpdf(self, d):
        d = np.maximum(np.asarray(d, float), 1e-3)
        return -np.log(d * self.sigma * math.sqrt(2 * math.pi)) - (np.log(d) - self.mu) ** 2 / (2 * self.sigma ** 2)

    def describe(self) -> dict:
        return {"mu_log_s": self.mu, "sigma_log": self.sigma, "median_s": math.exp(self.mu),
                "mean_s": math.exp(self.mu + self.sigma ** 2 / 2), "n_cues": self.n}


def training_bgm_cue_lengths(k: int) -> np.ndarray:
    """Cue lengths (s) of fold k's training films' BGM, same segmentation and same files as stitching.cue_stats."""
    from bgm.config import load_config
    from bgm.data import manifest
    from bgm.eval.store import fold_ids
    from stitching.cue_stats import FR, extract_all, foote_boundaries
    feats = extract_all(1)                                   # cached: stitching/cache/cue_feats.npz
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    pairs = manifest.read_pairs(cfg).set_index("pair_id")
    files = sorted({pairs.loc[p, "music_id"] for p in fold_ids(cfg, k)["train"]})
    out = []
    for m in files:
        X, _, dur = feats[m]
        if not len(X):
            continue
        b = np.array([0] + foote_boundaries(X) + [len(X)], float) * FR
        b[-1] = dur
        out.extend(np.diff(b)[np.diff(b) > 0].tolist())
    return np.array(out)


# ------------------------------------------------------------------ per-episode tables (parameter-free)
class Tables:
    """Ec[a, j, b], Et[a, j, b]: summed own / track-level emission of the cue starting at candidate j of scene a and
    ending before scene b (-inf where the chain does not reach b-1). fid[a, j]: file. chain[a, j, t]: candidate index
    at scene t. D[a, b]: cue duration (s). inner[a, b] = sum_{i=a+1}^{b-1} log(1-p_i). logp[a] = log p_a."""

    def __init__(self, ep: Episode, tau: float, L: float):
        T = ep.T
        Ks = [len(c.mid) for c in ep.cands]
        K = max(Ks)
        e = [c.sim / tau for c in ep.cands]
        nxt = []
        for t in range(T - 1):
            cont = relation(ep, t + 1)[0]
            n = np.where(cont.any(1), cont.argmax(1), -1)
            nxt.append(n)
        # track-level evidence per scene and file
        F = len(ep.files)
        tmax = np.full((T, F), NEG)
        for t, c in enumerate(ep.cands):
            np.maximum.at(tmax[t], c.mid, e[t])
        self.fid = np.full((T, K), -1)
        self.chain = np.full((T, K, T), -1)
        self.Ec = np.full((T, K, T + 1), NEG)
        self.Et = np.full((T, K, T + 1), NEG)
        for a in range(T):
            ka = Ks[a]
            self.fid[a, :ka] = ep.cands[a].mid
            cur = np.arange(ka)
            sc, st = np.zeros(ka), np.zeros(ka)
            alive = np.ones(ka, bool)
            for t in range(a, T):
                self.chain[a, :ka, t] = np.where(alive, cur, -1)
                ci = np.where(alive, cur, 0)
                sc = np.where(alive, sc + e[t][ci], NEG)
                st = np.where(alive, st + tmax[t, self.fid[a, :ka]], NEG)
                self.Ec[a, :ka, t + 1], self.Et[a, :ka, t + 1] = sc, st
                if t < T - 1:
                    n = nxt[t][ci]
                    alive = alive & (n >= 0)
                    cur = np.where(alive, n, 0)
        s0, s1 = ep.segs[:, 0], ep.segs[:, 1]
        self.D = np.full((T, T + 1), np.nan)
        for a in range(T):
            self.D[a, a + 1:] = s1[a:] - s0[a]
        p = np.clip(1.0 - np.exp(-ep.scene_dur / L), 1e-6, 1 - 1e-6)
        self.logp = np.log(p)                                       # [T]; index 0 unused
        l1p = np.log(1 - p)
        cs = np.r_[0.0, np.cumsum(l1p)]                             # cs[i] = sum_{t<i} l1p[t]
        self.inner = np.full((T, T + 1), 0.0)
        for a in range(T):
            for b in range(a + 1, T + 1):
                self.inner[a, b] = cs[b] - cs[a + 1]               # boundaries a+1 .. b-1
        keys, tempos = np.asarray(ep.file_key), np.asarray(ep.file_tempo, float)
        self.delta = fifths_matrix(keys, keys) + tempo_matrix(tempos, tempos)    # [F, F]
        self.Ks, self.T, self.F = Ks, T, F
        # per start scene: candidates grouped by file (for the max over j within a file)
        self.groups = []
        for a in range(T):
            order = np.argsort(self.fid[a, :Ks[a]], kind="stable")
            f_sorted = self.fid[a, order]
            starts = np.r_[0, np.where(np.diff(f_sorted) != 0)[0] + 1]
            self.groups.append((order, starts, f_sorted[starts]))


def decode_tables(tb: Tables, ep: Episode, alpha, lam_jump, mu, gamma, m, dur: LogNormal, return_score=False):
    T, F = tb.T, tb.F
    LD = dur.logpdf(np.nan_to_num(tb.D, nan=1.0))                                          # [T, T+1]; b <= a masked below
    ok = np.isfinite(tb.Ec)
    R = np.where(ok, (1 - m) * np.where(ok, tb.Ec, 0.0) + m * np.where(ok, tb.Et, 0.0)
                 + (gamma * LD + alpha * tb.inner)[:, None, :], NEG)
    G = np.full((T + 1, F), NEG)
    H = np.full((T, F), NEG)
    Hprev = np.full((T, F), -1)
    pen = mu * tb.delta.copy()
    np.fill_diagonal(pen, lam_jump)
    for a in range(T):
        if a == 0:
            H[0] = 0.0
        else:
            cand = G[a][:, None] - pen                          # [F prev, F next]
            Hprev[a] = cand.argmax(0)
            H[a] = cand.max(0) + alpha * tb.logp[a]
        order, starts, files = tb.groups[a]
        v = H[a, tb.fid[a, order]][:, None] + R[a, order, a + 1:]      # [K_a, T - a]
        gm = np.maximum.reduceat(v, starts, axis=0)                      # [n_files_a, T - a]
        G[a + 1:, files] = np.maximum(G[a + 1:, files], gm.T)
    # backtrack (recompute the argmax along the path)
    b, f = T, int(np.argmax(G[T]))
    best = float(G[T, f])
    mid, off = np.zeros(T, int), np.zeros(T)
    while b > 0:
        found = None
        for a in range(b):
            js = np.where(tb.fid[a, :tb.Ks[a]] == f)[0]
            if not len(js) or not np.isfinite(H[a, f]):
                continue
            vals = H[a, f] + R[a, js, b]
            i = int(np.argmax(vals))
            if np.isclose(vals[i], G[b, f], rtol=0, atol=1e-9) and np.isfinite(vals[i]):
                found = (a, int(js[i]))
                break
        a, j = found
        for t in range(a, b):
            ci = tb.chain[a, j, t]
            mid[t], off[t] = ep.cands[t].mid[ci], ep.cands[t].off[ci]
        if a > 0:
            f = int(Hprev[a, f])
        b = a
    p = Path(mid, off)
    return (p, best) if return_score else p


GRID = {"alpha": (0.25, 0.5, 1.0, 2.0), "lam_jump": (0.0, 1.0, 2.0, 4.0, 8.0), "mu": (0.0, 1.0, 2.0),
        "gamma": (0.0, 0.5, 1.0, 2.0, 4.0), "m": (0.0, 0.5)}


class HSMM(Stitcher):
    name = "Hidden semi-Markov (cue-level)"

    def __init__(self, tau: float, L: float, dur: LogNormal, grid: dict | None = None):
        self.tau, self.L, self.dur = tau, L, dur
        self.grid = grid or GRID
        self.params = {"alpha": 1.0, "lam_jump": 1.0, "mu": 1.0, "gamma": 1.0, "m": 0.0}
        self._tb = {}

    def tables(self, ep):
        if ep.key not in self._tb:
            self._tb[ep.key] = Tables(ep, self.tau, self.L)
        return self._tb[ep.key]

    def fit(self, train, val):
        keys = list(self.grid)
        scores = {}
        for vals in itertools.product(*(self.grid[k] for k in keys)):     # grid order; first max wins
            self.params = dict(zip(keys, vals))
            scores[vals] = float(np.mean([evaluate_path(e, self.decode(e))["objective"] for e in val]))
        best = max(scores, key=scores.get)
        self.params = dict(zip(keys, best))
        return dict(self.params, val_objective=scores[best], tau=self.tau, L=self.L, dur=self.dur.describe(),
                    n_configs=len(scores))

    def decode(self, ep: Episode) -> Path:
        return decode_tables(self.tables(ep), ep, dur=self.dur, **self.params)
