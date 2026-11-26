"""D143: segment-level track R@k of the retrieval system on each step-1 segmenter's segments.

From runs/titlesplit/seg_retrieval/fold<k>/eval.json (nothing re-scored): per test segment, the rank of its true track
group among the coarse shortlist (top-K files) ordered by the best window's fused score (cue.evaluate.emission_rank);
not shortlisted = miss. A segment's true track = the clip at its midpoint (truth_for). R@k for k <= K only.
Mean over segments (and duration-weighted), cluster bootstrap over film groups (10,000), paired differences vs S0.

`python -m bgm.segmentation.seg_recall` -> results/seg_recall.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.segmentation import evaluate as ev
from bgm.segmentation.retrieval_eval import ARMS, EXP, arm_name

log = logging.getLogger(__name__)
KS = (1, 5, 10)


def main(n_boot: int = 10000) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    folds = [json.loads((Path("runs/titlesplit") / EXP / f"fold{k}" / "eval.json").read_text()) for k in range(5)]
    arms = [a for a in ARMS if all(a in f["arms"] for f in folds)]
    clusters = []
    for k, f in enumerate(folds):
        clusters += ev.seq_clusters([c["films"] for c in f["arms"][arms[0]]["cases"]], f"f{k}")
    bs = ev.Bootstrap(clusters, n_boot, 0)
    C = len(bs.uniq)

    def per_seq(a: str):
        """[n_seq, len(KS)] segment-mean hits, duration-weighted hits, mean rank-1 of hits; segments per sequence."""
        hit, whit, nseg = [], [], []
        for f in folds:
            for c in f["arms"][a]["cases"]:
                r = np.array(c["ranks"], float)
                w = np.array([b - x for x, b in c["segs"]])
                h = np.stack([r <= k for k in KS], 1).astype(float)
                hit.append(h.mean(0))
                whit.append((h * w[:, None]).sum(0) / w.sum())
                nseg.append(len(r))
        return np.array(hit), np.array(whit), np.array(nseg, float)

    def boot(x, n):
        """Segment-pooled mean per resample: x = per-sequence means, n = segments per sequence."""
        per, cnt = np.zeros((C, x.shape[1])), np.zeros(C)
        np.add.at(per, bs.inv, x * n[:, None])
        np.add.at(cnt, bs.inv, n)
        return (bs.M @ per) / (bs.M @ cnt)[:, None]

    def boot_w(x):
        per, cnt = np.zeros((C, x.shape[1])), np.zeros(C)
        np.add.at(per, bs.inv, x)
        np.add.at(cnt, bs.inv, 1.0)
        return (bs.M @ per) / (bs.M @ cnt)[:, None]

    V = {a: per_seq(a) for a in arms}
    rows, drows = [], []
    for a in arms:
        h, wh, n = V[a]
        bm, bw = boot(h, n), boot_w(wh)
        pm, pw = (h * n[:, None]).sum(0) / n.sum(), wh.mean(0)
        cells = [f"{pm[i]:.3f} [{np.quantile(bm[:, i], .025):.3f}, {np.quantile(bm[:, i], .975):.3f}]" for i in range(len(KS))]
        cells += [f"{pw[i]:.3f}" for i in range(len(KS))]
        rows.append(f"| {arm_name(a)} | {int(n.sum())} | " + " | ".join(cells) + " |")
    for base in [b for b in ("psd_siglip/own", "psd_siglip/forced") if b in arms]:
        drows.append(("base", base))
        for a in arms:
            if a == base:
                continue
            d = V[a][1] - V[base][1]          # duration-weighted: same timeline, so pairing is per sequence
            bd = boot_w(d)
            cells = []
            for i in range(len(KS)):
                lo, hi = np.quantile(bd[:, i], .025), np.quantile(bd[:, i], .975)
                s = f"{d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
                cells.append(f"**{s}**" if lo > 0 or hi < 0 else s)
            drows.append((a, cells))
    L = ["# Segment-level track R@k by step-1 segmenter (D143)", "",
         f"Title split, original dataset, frozen blocked test sequences, 5 folds pooled ({len(clusters)} sequences, {C} film clusters); "
         "D121 emission (10-head ensemble, gallery = the test fold's music). Rank of the segment's true track group among the "
         "coarse shortlist (top 10 files) by best-window fused score; outside the shortlist = miss. True track = clip at the "
         "segment's midpoint. Segment R@k: mean over segments, [95% CI] cluster bootstrap over film groups (10,000). "
         "Time R@k: share of each sequence's time covered by a hit, averaged over sequences.", "",
         "| segmenter | segments | segment R@1 | segment R@5 | segment R@10 | time R@1 | time R@5 | time R@10 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |", *rows, ""]
    for x in drows:
        if x[0] == "base":
            L += ["", f"Paired Δ time R@k vs {arm_name(x[1])} (per sequence; bold = CI excludes 0):", "",
                  "| segmenter | Δ R@1 | Δ R@5 | Δ R@10 |", "| --- | --- | --- | --- |"]
        else:
            L.append(f"| {arm_name(x[0])} | " + " | ".join(x[1]) + " |")
    out = Path("results/seg_recall.md")
    out.write_text("\n".join(L) + "\n")
    log.info("\n%s", out.read_text())
    return out


if __name__ == "__main__":
    main()
