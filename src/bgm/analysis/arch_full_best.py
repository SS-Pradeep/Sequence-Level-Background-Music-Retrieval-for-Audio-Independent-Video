"""The current best method on the full 4,246-pair dataset (PROGRESS.md D73): 10-seed ensemble of the siglip-both
heads and the whole-video context score, extended film split.

Run: python -m bgm.analysis.arch_full_best fold <k>     (one fold; writes reports/full/best_fold<k>.npz)
     python -m bgm.analysis.arch_full_best report       -> results/full_best.md + ledger rows
Sections as for the heads: track retrieval on the test-fold gallery, segment retrieval over all windows.
Systems: single models (hit@10 and hit@1 averaged over the 10 seeds), the 10-seed ensemble (mean cosine), and the
ensemble with context (segment score + cos(whole video, the window's whole file), w = 1; segments only).
Old clips = the 1,248 pairs of the earlier dataset; new clips = the Demucs-separated clips of the added films.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_context import context_terms, seg_ranks
from bgm.analysis.arch_ensemble import fold_truth, member_scores, ranks_of
from bgm.analysis.full_dataset import BASE_FOLDS, FULL_RUNS, full_cfg, load_scorer
from bgm.data import manifest
from bgm.eval.metrics import chance_recall, cluster_bootstrap
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
SEEDS = range(10)
OUT, B = Path("results/full_best.md"), 5000


def run_fold(k: int) -> None:
    cfg = full_cfg()
    store = FeatureStore(cfg)
    film = manifest.read_folds(cfg)["film_group"]
    f = fold_ids(cfg, k)
    thits, shit, seg_pairs = fold_truth(store, f)
    ctx = context_terms(store, f)
    M = []
    for s in SEEDS:
        M.append(member_scores(store, load_scorer(FULL_RUNS / f"fold{k}" / f"seed{s}", store, "cuda"), f))
        log.info("fold %d seed %d scored", k, s)
    single = [ranks_of([m], thits, shit) for m in M]
    ens = ranks_of(M, thits, shit)
    Te, Se = np.mean([m[0] for m in M], 0), [np.mean([m[1][i] for m in M], 0) for i in range(len(shit))]
    out = {"track/single": np.stack([r["track_test_fold"] for r in single]), "segment/single": np.stack([r["segment_all"] for r in single]),
           "track/ens": ens["track_test_fold"], "segment/ens": ens["segment_all"], "segment/ens_ctx": seg_ranks(Te, Se, ctx, shit, 1.0),
           "segment/single_ctx": np.stack([seg_ranks(m[0], m[1], ctx, shit, 1.0) for m in M]),
           "track/pair": np.array(f["test"]), "segment/pair": np.array(seg_pairs),
           "track/chance10": np.array([chance_recall(len(h), int(h.sum()), 10) for h in thits]),
           "segment/chance10": np.array([chance_recall(len(h), 1, 10) for h in shit]),
           "track/film": np.array([film[p] for p in f["test"]]), "segment/film": np.array([film[p] for p in seg_pairs])}
    p = Path("reports/full")
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / f"best_fold{k}.npz", **out)
    log.info("fold %d: track R@10 single %.3f ens %.3f | segment single %.3f ens %.3f ens+ctx %.3f", k, (out["track/single"] <= 10).mean(),
             (out["track/ens"] <= 10).mean(), (out["segment/single"] <= 10).mean(), (out["segment/ens"] <= 10).mean(), (out["segment/ens_ctx"] <= 10).mean())


def report() -> None:
    Z = [dict(np.load(f"reports/full/best_fold{k}.npz", allow_pickle=True)) for k in range(5)]
    cat = lambda key, f=None: np.concatenate([(f(z[key]).mean(0) if z[key].ndim == 2 else f(z[key])) if f else z[key] for z in Z])  # noqa: E731
    old_ids = set(json.loads(BASE_FOLDS.read_text())["film_group"])
    h10, h1 = (lambda r: (r <= 10).astype(float)), (lambda r: (r <= 1).astype(float))
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Best method on the full dataset: 10-seed ensemble and whole-video context (D73)", "",
         "All 4,246 pairs of `dataset_merged`, extended film split, heads `runs/full/semantic/siglip-both` seeds 0–9. Test-fold galleries: "
         "743–812 music files (track retrieval) and 2,228–2,993 windows (segment retrieval). Single-model rows average per-query hits over "
         f"the 10 seeds. Paired differences: 95% cluster bootstrap over film groups, {B:,} resamples.", ""]
    rows = []
    for sec, title, systems in (("track", "Track retrieval, test-fold gallery", (("single", "single model"), ("ens", "10-seed ensemble"))),
                                ("segment", "Segment retrieval, all windows", (("single", "single model"), ("single_ctx", "single model + context"),
                                                                              ("ens", "10-seed ensemble"), ("ens_ctx", "10-seed ensemble + context")))):
        pair, cl, ch = cat(f"{sec}/pair"), cat(f"{sec}/film"), cat(f"{sec}/chance10")
        is_old = np.array([p in old_ids for p in pair])
        L += [f"## {title}", "", f"Chance R@10 {ch.mean():.3f}. {len(pair):,} queries: {int(is_old.sum()):,} old-clip, {int((~is_old).sum()):,} new-clip.", "",
              "| system | R@1 | R@10 | Δ R@10 vs single model [95% CI] | R@10 old clips | R@10 new clips |", "| --- | --- | --- | --- | --- | --- |"]
        base = cat(f"{sec}/single", h10)
        for key, name in systems:
            a, a1 = cat(f"{sec}/{key}", h10), cat(f"{sec}/{key}", h1)
            d = None if key == "single" else cluster_bootstrap(a - base, cl, B)
            L.append(f"| {name} | {a1.mean():.3f} | {a.mean():.3f} | {'—' if d is None else fmt(d)} | {a[is_old].mean():.3f} | {a[~is_old].mean():.3f} |")
            if d is not None:
                rows.append(f"| {date.today().isoformat()} | Full dataset, {sec}: {name} vs single model | extended (4,246) | {'781 (743–812)' if sec == 'track' else '2,228–2,993 windows'} | "
                            f"{a1.mean():.3f} | {a.mean():.3f} | — | {fmt(d)} | {'KEEP' if d['lo'] > 0 else ('REJECT (worse)' if d['hi'] < 0 else 'NULL')} |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run_fold(int(sys.argv[2])) if sys.argv[1] == "fold" else report()
