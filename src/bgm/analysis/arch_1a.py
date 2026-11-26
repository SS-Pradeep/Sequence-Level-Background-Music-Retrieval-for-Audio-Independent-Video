"""Phase 1a evaluation (D53): learned MERT layer mixing vs the baseline heads, film split.

Run after the 15 mix runs finish: python -m bgm.analysis.arch_1a
Writes reports/arch_1a.md, appends the ledger row to results/arch_ablation.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, FILM_ROOT, _per_query, film_cfg
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import METRICS, Split, _cell, _gallery, _row, _runs

log = logging.getLogger(__name__)
VARIANT = "siglip-both-mix1a"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = film_cfg()
    split = Split(cfg, "folds_film.json", "film")
    base = _runs(FILM_ROOT, "siglip-both")
    mix = _runs(FILM_ROOT, VARIANT)
    common = sorted(set(base) & set(mix))
    assert len(common) == 5, f"folds present: {common}"
    stats = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
    out = {}
    for sec in ("track_test_fold", "segment_all"):
        row = {"gallery": _gallery(split, sec, False), "abs": _row(split, mix, sec, False), "base": _row(split, base, sec, False)}
        for m, f in stats.items():
            a, cl = _per_query(split, mix, sec, f)
            b, _ = _per_query(split, base, sec, f)
            row[m] = cluster_bootstrap(a - b, cl, B)
        out[sec] = row
    W = []
    for k in common:
        for s in (0, 1, 2):
            m = json.loads((FILM_ROOT / "semantic" / VARIANT / f"fold{k}" / f"seed{s}" / "metrics.json").read_text())
            W.append(m["mert_layer_weights"])
    W = np.array(W)
    keep = out["track_test_fold"]["R@10"]
    verdict = "KEEP" if keep["lo"] > 0 else ("REJECT (worse)" if keep["hi"] < 0 else "NULL")
    ci = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Phase 1a: learned MERT layer mixing (D53)", "",
         f"`train.semantic.mert_layer_mix=true`: softmax over the 25 stored per-layer MERT span means (uniform at init, "
         f"= the baseline's layer average up to per-second renormalization), {len(W)} runs. Paired Δ vs runs/filmsplit/"
         f"semantic/siglip-both, {B:,} cluster-bootstrap resamples over film groups.", ""]
    for sec, title in (("track_test_fold", "Track retrieval, test-fold gallery"), ("segment_all", "Segment retrieval, all windows")):
        r = out[sec]
        L += [f"## {title} (gallery {r['gallery']})", "",
              "| system | R@1 | R@10 | MRR |", "| --- | --- | --- | --- |",
              "| baseline heads | " + " | ".join(_cell(r["base"][m]) for m in ("R@1", "R@10", "MRR")) + " |",
              "| layer mix | " + " | ".join(_cell(r["abs"][m]) for m in ("R@1", "R@10", "MRR")) + " |",
              "", f"Δ mix − baseline: R@1 {ci(r['R@1'])}; R@10 {ci(r['R@10'])}; MRR {ci(r['MRR'])}", ""]
    top = np.argsort(-W.mean(0))[:8]
    L += [f"**Verdict (plan rule, track R@10 CI): {verdict}.**", "",
          "## Learned layer weights (softmax, mean ± std over runs; uniform = 0.040)", "",
          "| layer | " + " | ".join(str(int(i)) for i in top) + " |",
          "| --- | " + " | ".join("---" for _ in top) + " |",
          "| weight | " + " | ".join(f"{W.mean(0)[i]:.3f} ± {W.std(0)[i]:.3f}" for i in top) + " |", "",
          f"Entropy of the mean weights: {float(-(W.mean(0) * np.log(W.mean(0) + 1e-12)).sum()):.2f} nats (uniform = {np.log(25):.2f}).", ""]
    Path("reports/arch_1a.md").write_text("\n".join(L) + "\n")
    a = out["track_test_fold"]["abs"]
    led = (f"| 2026-10-02 | 1a MERT layer mix (`train.semantic.mert_layer_mix`) | film | {out['track_test_fold']['gallery']} | "
           f"{_cell(a['R@1'])} | {_cell(a['R@10'])} | {_cell(a['MRR'])} | {ci(out['track_test_fold']['R@10'])} | {verdict} |")
    with open("results/arch_ablation.md", "a") as f:
        f.write(led + "\n")
    log.info("verdict %s; wrote reports/arch_1a.md and the ledger row", verdict)


if __name__ == "__main__":
    main()
