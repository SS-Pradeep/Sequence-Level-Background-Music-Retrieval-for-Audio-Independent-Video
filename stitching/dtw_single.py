"""Method 4: "Prétet-style DTW" (one track + monotonic alignment; not a reproduction of their model).
Top-N files by mean over scenes of the best window similarity; for each, cost[t, i] = 1 - sim(scene t, window i)
over ALL 0.5 s-hop windows of the file (not only the shortlist); monotonic DTW: each scene takes one window, window
index non-decreasing; normalised by the path length (= T). Output: the lowest-cost file's path."""
import numpy as np

from stitching.base import Episode, Path, Stitcher


def monotonic_dtw(C: np.ndarray) -> tuple[float, np.ndarray]:
    """C [T, W] (inf = invalid). min sum_t C[t, i_t] s.t. i_t non-decreasing. Returns (cost / T, path)."""
    T, W = C.shape
    D = C[0].copy()
    back = []
    for t in range(1, T):
        pm = np.minimum.accumulate(D)
        arg = np.zeros(W, int)
        best = 0
        for i in range(W):                     # running argmin of D[:i+1]
            if D[i] < D[best]:
                best = i
            arg[i] = best
        back.append(arg)
        D = pm + C[t]
    i = int(np.argmin(D))
    cost = float(D[i])
    path = [i]
    for arg in reversed(back):
        i = int(arg[i])
        path.append(i)
    return cost / T, np.array(path[::-1])


class PretetDTW(Stitcher):
    name = "Prétet-style DTW (one track)"

    def __init__(self, n_files: int = 10):
        self.n = n_files

    def decode(self, ep: Episode) -> Path:
        fm = {f: float(np.mean(np.max(tab, 1))) for f, (_, tab) in ep.dense.items()}
        top = [f for f in sorted(fm, key=fm.get, reverse=True) if np.isfinite(fm[f])][:self.n]
        best = (np.inf, None, None)
        for f in top:
            grid, tab = ep.dense[f]
            C = 1.0 - tab.astype(float)
            C[~np.isfinite(C)] = np.inf
            if not np.all(np.isfinite(C).any(1)):
                continue
            c, p = monotonic_dtw(C)
            if c < best[0]:
                best = (c, f, grid[p])
        if best[1] is None:                      # no file can host every scene: greedy
            from stitching.greedy import Greedy
            return Greedy().decode(ep)
        return Path(np.full(ep.T, best[1]), best[2].astype(float))
