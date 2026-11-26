"""Method 3: two-stage change-point + retrieve.
Stage 1 (scene groups): boundary where the cosine distance between consecutive projected scene embeddings > theta,
or kernel temporal segmentation (KTS: exact DP minimising within-group scatter + pen x #groups). Mode, theta / pen
tuned on val. Stage 2: per group, the file with the highest mean (over the group's scenes) of its best window
similarity (all windows of the shortlisted files); its windows laid contiguously from the start offset whose
contiguous run has the highest summed similarity."""
import numpy as np

from stitching.base import Episode, Path, Stitcher
from stitching.metrics import val_objective


def threshold_groups(E: np.ndarray, theta: float) -> list[int]:
    d = 1 - (E[1:] * E[:-1]).sum(1)
    return [0] + [t + 1 for t in np.where(d > theta)[0]]


def kts_groups(E: np.ndarray, pen: float) -> list[int]:
    """Exact DP: cost(seg) = sum ||x - mean||^2; minimise sum cost + pen * n_groups. Returns group start indices."""
    T = len(E)
    cs = np.vstack([np.zeros(E.shape[1]), np.cumsum(E, 0)])
    cq = np.r_[0, np.cumsum((E ** 2).sum(1))]

    def cost(a, b):  # [a, b)
        s = cs[b] - cs[a]
        return cq[b] - cq[a] - (s @ s) / (b - a)
    F = np.full(T + 1, np.inf)
    F[0] = 0
    arg = np.zeros(T + 1, int)
    for b in range(1, T + 1):
        for a in range(b):
            v = F[a] + cost(a, b) + pen
            if v < F[b]:
                F[b], arg[b] = v, a
    starts, b = [], T
    while b > 0:
        starts.append(arg[b])
        b = arg[b]
    return sorted(starts)


def lay(ep: Episode, f: int, ts: list[int]) -> tuple[float, np.ndarray]:
    """Best contiguous run of file f over scenes ts (consecutive). Returns (score, offsets)."""
    grid, tab = ep.dense[f]
    d = ep.scene_dur[ts]
    rel = np.r_[0, np.cumsum(d[:-1])]
    flen = ep.file_len[f]
    best, best_o = -np.inf, None
    for s0 in grid:
        offs = s0 + rel
        if offs[-1] + d[-1] > flen + 1e-6:
            break
        gi = np.clip(np.round(offs / 0.5).astype(int), 0, len(grid) - 1)
        v = tab[ts, gi]
        if not np.all(np.isfinite(v)):
            continue
        s = float(v.sum())
        if s > best:
            best, best_o = s, offs
    return best, best_o


class TwoStage(Stitcher):
    name = "Two-stage change-point + retrieve"

    def __init__(self, mode: str = "threshold", param: float = 0.3):
        self.mode, self.param = mode, param

    def fit(self, train, val):
        dists = np.concatenate([1 - (e.scene_emb[1:] * e.scene_emb[:-1]).sum(1) for e in val if e.T > 1])
        grid = [("threshold", float(q)) for q in np.quantile(dists, np.linspace(0.05, 0.95, 19))]
        grid += [("kts", float(p)) for p in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)]
        scores = {}
        for m, p in grid:
            self.mode, self.param = m, p
            scores[(m, p)] = val_objective(self, val)
        self.mode, self.param = max(scores, key=scores.get)
        return {"mode": self.mode, "param": self.param, "val_objective": max(scores.values())}

    def decode(self, ep: Episode) -> Path:
        E = ep.scene_emb
        starts = threshold_groups(E, self.param) if self.mode == "threshold" else kts_groups(E, self.param)
        return decode_groups(ep, starts)


def decode_groups(ep: Episode, starts: list[int]) -> Path:
    """Stage 2 for a given grouping (group start indices)."""
    bounds = list(starts) + [ep.T]
    mid, off = np.zeros(ep.T, int), np.zeros(ep.T)
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts = list(range(a, b))
        # file with the highest mean best-window similarity over the group
        means = {f: float(np.mean(np.max(tab[ts], 1))) for f, (_, tab) in ep.dense.items()}
        placed = False
        for f in sorted(means, key=means.get, reverse=True):
            if not np.isfinite(means[f]):
                continue
            sc, offs = lay(ep, f, ts)
            if offs is not None:
                mid[a:b], off[a:b] = f, offs
                placed = True
                break
        if not placed:          # group longer than every file: per-scene best window
            for t in ts:
                j = int(np.argmax(ep.cands[t].sim))
                mid[t], off[t] = ep.cands[t].mid[j], ep.cands[t].off[j]
    return Path(mid, off)
