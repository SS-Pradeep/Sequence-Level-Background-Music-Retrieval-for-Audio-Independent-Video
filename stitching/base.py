"""Common interface for every stitching method (ablation table).

Episode = one test / val / train sequence after step-1 segmentation (BaSSL C head, own cuts) and the shared
coarse-to-fine shortlist. The candidate set of scene t is ragged (K_t): the scene's top-J windows followed by the
contiguous next window of every candidate of scene t-1 ("shortlist augmentation", built once by the existing
`build_lattice`, identical for all methods). A candidate is a global window ID = (music file, start s); window
metadata (track group, file length, key, tempo) is looked up through `Episode.files`.

Greedy uses only the first n_top candidates of each scene (no augmentation); every other method sees all K_t.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

TOL_S = 0.04          # contiguity tolerance, as the existing stitcher


@dataclass
class Cands:
    """Candidates of one scene. sim = raw ensemble cosine (the emission before z-scoring)."""
    mid: np.ndarray        # [K_t] int, index into Episode.files
    off: np.ndarray        # [K_t] window start in the file, s
    sim: np.ndarray        # [K_t]
    n_top: int             # first n_top = the scene's own top-J shortlist
    # per-candidate music descriptors used by the CRF pair features
    emb_in: np.ndarray | None = None     # [K_t, D] unit MERT mean over the window's first 2 s
    emb_out: np.ndarray | None = None    # [K_t, D] ... last 2 s
    rms_in: np.ndarray | None = None     # [K_t] dB
    rms_out: np.ndarray | None = None
    ph_in: np.ndarray | None = None      # [K_t] beat phase in [0, 1) at window start
    ph_out: np.ndarray | None = None     # ... at window end

    def take(self, idx):
        g = lambda a: None if a is None else a[idx]  # noqa: E731
        return Cands(self.mid[idx], self.off[idx], self.sim[idx], 0, g(self.emb_in), g(self.emb_out), g(self.rms_in),
                     g(self.rms_out), g(self.ph_in), g(self.ph_out))


@dataclass
class Episode:
    key: str                         # "<fold>/<split>/<i>"
    cands: list[Cands]               # T
    segs: np.ndarray                 # [T, 2] scene start / end, s
    scene_emb: np.ndarray            # [T, 256] projected (ensemble head) video embedding, unit
    scene_sig: np.ndarray            # [T, 1152] raw SigLIP mean, unit (context feature only)
    scene_mood: np.ndarray           # [T, M] calibrated mood distribution
    files: list[str]                 # music ids of the episode's gallery subset (shortlist union + gold)
    file_track: np.ndarray           # [F] track group (object)
    file_len: np.ndarray             # [F] s
    file_key: np.ndarray             # [F] stitch key (-1 unknown)
    file_tempo: np.ndarray           # [F] BPM
    dense: dict = field(default_factory=dict)   # file idx -> (starts [W], sims [T, W], -inf where invalid): all windows
    gold: np.ndarray | None = None   # [T] index into cands[t] of the true window, -1 if not a candidate
    gold_win: list | None = None     # [T] (file idx, off, sim, descriptors...) of the true window (train / val only)
    # evaluation metadata (never shown to a method)
    pieces: tuple | None = None
    true_sw: list = field(default_factory=list)
    duration: float = 0.0
    films: list = field(default_factory=list)
    shortlist_rank: np.ndarray | None = None    # [T] rank of the true track in the shortlist (method-independent)
    layers: list | None = None       # the existing Layer objects (hand-set Viterbi only)

    @property
    def T(self) -> int:
        return len(self.cands)

    @property
    def scene_dur(self) -> np.ndarray:
        return self.segs[:, 1] - self.segs[:, 0]

    @property
    def btimes(self) -> np.ndarray:
        return self.segs[1:, 0]

    def with_gold(self) -> "Episode":
        """Copy with the true window inserted into every scene's candidates (training / val NLL only)."""
        import copy
        ep = copy.copy(self)
        cands, gold = [], []
        for t, c in enumerate(self.cands):
            gw = self.gold_win[t]
            hit = np.where((c.mid == gw["mid"]) & (np.abs(c.off - gw["off"]) <= TOL_S))[0]
            if len(hit):
                cands.append(c), gold.append(int(hit[0]))
                continue
            cat = lambda a, b: None if a is None else np.concatenate([a, np.asarray(b)[None]])  # noqa: E731
            cands.append(Cands(np.r_[c.mid, gw["mid"]], np.r_[c.off, gw["off"]], np.r_[c.sim, gw["sim"]], c.n_top,
                               cat(c.emb_in, gw["emb_in"]), cat(c.emb_out, gw["emb_out"]), cat(c.rms_in, gw["rms_in"]),
                               cat(c.rms_out, gw["rms_out"]), cat(c.ph_in, gw["ph_in"]), cat(c.ph_out, gw["ph_out"])))
            gold.append(len(c.mid))
        ep.cands, ep.gold = cands, np.array(gold)
        return ep


# ---------------------------------------------------------------- shared helpers
def relation(ep: Episode, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks [K_{t-1}, K_t] for the join from scene t-1 to t: (continue, jump, switch)."""
    P, C = ep.cands[t - 1], ep.cands[t]
    d_prev = ep.scene_dur[t - 1]
    same = P.mid[:, None] == C.mid[None, :]
    cont = same & (np.abs(C.off[None, :] - (P.off + d_prev)[:, None]) <= TOL_S)
    return cont, same & ~cont, ~same


@dataclass
class Path:
    """A decoded soundtrack: one window per scene."""
    mid: np.ndarray      # [T] file index into Episode.files
    off: np.ndarray      # [T] s
    runtime_s: float = 0.0


def path_from_picks(ep: Episode, picks: list[int]) -> Path:
    return Path(np.array([ep.cands[t].mid[j] for t, j in enumerate(picks)]),
                np.array([ep.cands[t].off[j] for t, j in enumerate(picks)], float))


class Stitcher:
    name = "base"

    def fit(self, train_episodes: list[Episode], val_episodes: list[Episode]) -> dict:
        """Returns the chosen hyperparameters (logged). No-op for untrained methods."""
        return {}

    def decode(self, ep: Episode) -> Path:
        raise NotImplementedError

    def timed(self, ep: Episode) -> Path:
        t0 = time.perf_counter()
        p = self.decode(ep)
        p.runtime_s = time.perf_counter() - t0
        return p


def viterbi_ragged(emit: list[np.ndarray], trans: list[np.ndarray]) -> tuple[list[int], float]:
    """Exact max-sum Viterbi. emit: T x [K_t]; trans: (T-1) x [K_{t-1}, K_t]. Returns (path, score)."""
    d = np.asarray(emit[0], float)
    back = []
    for t in range(1, len(emit)):
        s = d[:, None] + trans[t - 1]
        bp = s.argmax(0)
        d = s[bp, np.arange(s.shape[1])] + emit[t]
        back.append(bp)
    j = int(np.argmax(d))
    best = float(d[j])
    path = [j]
    for bp in reversed(back):
        j = int(bp[j])
        path.append(j)
    return path[::-1], best


def logz_ragged(emit: list[np.ndarray], trans: list[np.ndarray]) -> float:
    from scipy.special import logsumexp
    a = np.asarray(emit[0], float)
    for t in range(1, len(emit)):
        a = logsumexp(a[:, None] + trans[t - 1], axis=0) + emit[t]
    return float(logsumexp(a))
