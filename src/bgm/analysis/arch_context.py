"""Whole-video context for segment retrieval (PROGRESS.md D72), 1,248-pair film split. Training-free.

Run: python -m bgm.analysis.arch_context   -> reports/arch_context.md + ledger rows
A segment's score for a music window becomes
    cos(segment, window) + w * cos(whole video, the window's whole file),
with w = 1 (plain sum of two cosines from the same heads; nothing tuned) and, for reference, w = 0.5 and 2.
The whole query video is known at retrieval time, so this uses no extra information. Evaluated on segment_all exactly
as the heads are; single models (hits averaged over 10 seeds) and the 10-seed ensemble; paired Δ vs w = 0, 95% cluster
bootstrap over film groups.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_ensemble import fold_truth, member_scores
from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("reports/arch_context.md")
WS = (0.0, 0.5, 1.0, 2.0)
SEEDS = range(10)


def context_terms(store, f) -> list[tuple[int, np.ndarray]]:
    """Per test segment: (row of its video in the track score matrix, gallery column of each window's file)."""
    gal = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
    col, row = {m: i for i, m in enumerate(gal)}, {p: i for i, p in enumerate(f["test"])}
    segs = test_segments(store, f["test"])
    return [(row[q[0]], np.array([col[m] for m, _, _ in segment_gallery(store, segs, q, "all")[0]])) for q in segs]


def seg_ranks(T: np.ndarray, S: list[np.ndarray], ctx, shit, w: float) -> np.ndarray:
    return np.array([rank_of_first_hit(S[i] + w * T[r, c], shit[i]) for i, (r, c) in enumerate(ctx)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film = manifest.read_folds(cfg)["film_group"]
    store = FeatureStore(cfg)
    R = {(kind, w): [] for kind in ("single", "ens10") for w in WS}
    CL = []
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        _, shit, seg_pairs = fold_truth(store, f)
        ctx = context_terms(store, f)
        M = [member_scores(store, load_scorer(FILM_ROOT / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}", store, "cuda"), f) for s in SEEDS]
        Te, Se = np.mean([m[0] for m in M], 0), [np.mean([m[1][i] for m in M], 0) for i in range(len(shit))]
        for w in WS:
            R[("single", w)].append(np.stack([seg_ranks(m[0], m[1], ctx, shit, w) for m in M]))
            R[("ens10", w)].append(seg_ranks(Te, Se, ctx, shit, w))
        CL.append(np.array([film[p] for p in seg_pairs]))
        log.info("fold %d: segment R@10 single %.3f -> %.3f (w=1); ens10 %.3f -> %.3f", k, (R[('single', 0.0)][-1] <= 10).mean(),
                 (R[('single', 1.0)][-1] <= 10).mean(), (R[('ens10', 0.0)][-1] <= 10).mean(), (R[('ens10', 1.0)][-1] <= 10).mean())
    cl = np.concatenate(CL)
    pq = lambda key, f: np.concatenate([f(x).mean(0) if x.ndim == 2 else f(x) for x in R[key]])  # noqa: E731
    h10, h1, mrr = (lambda r: (r <= 10).astype(float)), (lambda r: (r <= 1).astype(float)), (lambda r: 1.0 / r)
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Whole-video context for segment retrieval (D72)", "",
         "Segment score = cos(segment, window) + w × cos(whole video, the window's whole music file). 1,248-pair film split, segment "
         f"retrieval over all windows (gallery 511 on average). Paired Δ vs w = 0; 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         "| heads | w | segment R@1 | segment R@10 | MRR | Δ R@10 vs w = 0 |", "| --- | --- | --- | --- | --- | --- |"]
    rows = []
    for kind, name in (("single", "single model (mean of 10 seeds)"), ("ens10", "10-seed ensemble")):
        for w in WS:
            a = pq((kind, w), h10)
            d = None if w == 0 else cluster_bootstrap(a - pq((kind, 0.0), h10), cl, B)
            L.append(f"| {name} | {w:g} | {pq((kind, w), h1).mean():.3f} | {a.mean():.3f} | {pq((kind, w), mrr).mean():.3f} | {'—' if d is None else fmt(d)} |")
            if w == 1.0:
                rows.append(f"| {date.today().isoformat()} | Whole-video context in the segment score (w = 1, training-free), {name} | film | 511 (474–546) windows | "
                            f"{pq((kind, w), h1).mean():.3f} | {a.mean():.3f} | {pq((kind, w), mrr).mean():.3f} | segment: {fmt(d)} | "
                            f"{'KEEP' if d['lo'] > 0 else ('REJECT (worse)' if d['hi'] < 0 else 'NULL')} (segment rule) |")
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
