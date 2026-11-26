"""Method 2: greedy + sliding-window majority vote over the chosen file (w tuned on val), then windows inside the
voted file: the contiguous next window if it is a candidate, else that file's best candidate at the scene."""
from collections import Counter

import numpy as np

from stitching.base import TOL_S, Episode, Path, Stitcher
from stitching.greedy import Greedy
from stitching.metrics import val_objective


class Smooth(Stitcher):
    name = "Greedy + smoothing"

    def __init__(self, w: int = 5):
        self.w = w

    def fit(self, train, val):
        scores = {}
        for w in (3, 5, 7):
            self.w = w
            scores[w] = val_objective(self, val)
        self.w = max(scores, key=scores.get)
        return {"w": self.w, "val_objective": scores}

    def decode(self, ep: Episode) -> Path:
        g = Greedy().decode(ep)
        T, h = ep.T, self.w // 2
        voted = []
        for t in range(T):
            win = g.mid[max(0, t - h):t + h + 1]
            cnt = Counter(win.tolist())
            top = max(cnt.values())
            tied = [m for m, c in cnt.items() if c == top]
            # tie: the scene's own greedy file if tied, else the tied file with the higher summed similarity in the window
            if g.mid[t] in tied:
                voted.append(int(g.mid[t]))
            else:
                voted.append(max(tied, key=lambda m: sum(ep.cands[s].sim[ep.cands[s].mid == m].max(initial=-np.inf)
                                                         for s in range(max(0, t - h), min(T, t + h + 1)))))
        mid, off = np.zeros(T, int), np.zeros(T)
        for t in range(T):
            c, f = ep.cands[t], voted[t]
            sel = np.where(c.mid == f)[0]
            if not len(sel):                               # voted file not shortlisted here: keep greedy
                mid[t], off[t] = g.mid[t], g.off[t]
                continue
            if t and mid[t - 1] == f:
                nxt = sel[np.abs(c.off[sel] - (off[t - 1] + ep.scene_dur[t - 1])) <= TOL_S]
                if len(nxt):
                    mid[t], off[t] = f, c.off[nxt[0]]
                    continue
            j = sel[np.argmax(c.sim[sel])]
            mid[t], off[t] = f, c.off[j]
        return Path(mid, off)
