"""Phase 0 calibration rows on the same cached episodes.
  every_cut     change file at every boundary: top-1 per scene, and if that is the previous scene's file, the best
                candidate of a different file
  never_switch  one file for the whole video: the file with the highest mean (over scenes) best-window similarity,
                laid contiguously if it fits, else each scene on its best window of that file (jumps)
`.venv/bin/python -m stitching.phase0` -> stitching/results/fold<k>_phase0.json (k = 0..4)"""
import json
from pathlib import Path

import numpy as np

from stitching.base import Episode, Path as SPath, Stitcher
from stitching.data import load_fold
from stitching.metrics import evaluate_path, fold_entropy
from stitching.two_stage import lay

PHASE0_ROWS = [("every_cut", "Switch at every cut (calibration)"), ("never_switch", "Never switch (calibration)")]


class EveryCut(Stitcher):
    def decode(self, ep: Episode) -> SPath:
        mid, off = np.zeros(ep.T, int), np.zeros(ep.T)
        for t, c in enumerate(ep.cands):
            order = np.argsort(-c.sim[:c.n_top])
            j = order[0]
            if t and c.mid[j] == mid[t - 1]:
                other = [k for k in order if c.mid[k] != mid[t - 1]]
                j = other[0] if other else j
            mid[t], off[t] = c.mid[j], c.off[j]
        return SPath(mid, off)


class NeverSwitch(Stitcher):
    def decode(self, ep: Episode) -> SPath:
        ts = list(range(ep.T))
        means = {f: float(np.mean(np.max(tab, 1))) for f, (_, tab) in ep.dense.items()}
        f = max(means, key=means.get)
        _, offs = lay(ep, f, ts)
        if offs is not None:                      # the file can hold the whole video contiguously
            return SPath(np.full(ep.T, f), offs)
        grid, tab = ep.dense[f]                   # files (~20 s) are shorter than videos: reuse the file, each
        return SPath(np.full(ep.T, f), grid[np.argmax(tab, 1)].astype(float))   # scene on its best window


if __name__ == "__main__":
    for k in range(5):
        F = load_fold(k)
        te = F["episodes"]["test"]
        out = {"fold": k, "methods": {}}
        for key, st in (("every_cut", EveryCut()), ("never_switch", NeverSwitch())):
            paths = [st.timed(e) for e in te]
            rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
            out["methods"][key] = {"name": dict(PHASE0_ROWS)[key], "params": {}, "rows": rows, "entropy": fold_entropy(te, paths, F["gallery_test"])}
            print(k, key, round(np.mean([r["track_acc"] for r in rows]), 3), round(np.mean([r["switch_f1"] for r in rows]), 3))
        Path(f"stitching/results/fold{k}_phase0.json").write_text(json.dumps(out, default=float))
