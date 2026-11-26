"""Method 1: greedy top-1 over each scene's own shortlist (no continuation augmentation)."""
import numpy as np

from stitching.base import Episode, Path, Stitcher, path_from_picks


class Greedy(Stitcher):
    name = "Greedy top-1"

    def decode(self, ep: Episode) -> Path:
        return path_from_picks(ep, [int(np.argmax(c.sim[:c.n_top])) for c in ep.cands])
