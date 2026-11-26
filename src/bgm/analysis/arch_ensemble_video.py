"""Ensembles that mix SigLIP heads with heads trained on V-JEPA 2 features (PROGRESS.md D68), 1,248-pair film split.

Run: python -m bgm.analysis.arch_ensemble_video   -> reports/arch_ensemble_video.md + ledger rows
An ensemble averages its members' cosine scores. Mixed ensembles are compared with SigLIP-only ensembles of the SAME
number of models, so the comparison isolates the mix from the size.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_ensemble import fold_truth, member_scores, ranks_of
from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("reports/arch_ensemble_video.md")
MEMBERS = {"sig": ("siglip-both", range(10)), "sv": ("siglipvjepa-both", range(3)), "vj": ("vjepa-both", range(3))}
SYSTEMS = {  # name -> members
    "SigLIP ×3": [("sig", s) for s in range(3)],
    "SigLIP ×6": [("sig", s) for s in range(6)],
    "SigLIP ×3 + SigLIP+V-JEPA ×3": [("sig", s) for s in range(3)] + [("sv", s) for s in range(3)],
    "SigLIP ×9": [("sig", s) for s in range(9)],
    "SigLIP ×3 + SigLIP+V-JEPA ×3 + V-JEPA ×3": [("sig", s) for s in range(3)] + [("sv", s) for s in range(3)] + [("vj", s) for s in range(3)],
    "SigLIP ×10": [("sig", s) for s in range(10)],
    "SigLIP ×10 + SigLIP+V-JEPA ×3": [("sig", s) for s in range(10)] + [("sv", s) for s in range(3)],
    "SigLIP ×10 + SigLIP+V-JEPA ×3 + V-JEPA ×3": [("sig", s) for s in range(10)] + [("sv", s) for s in range(3)] + [("vj", s) for s in range(3)],
}
PAIRS = [("SigLIP ×3 + SigLIP+V-JEPA ×3", "SigLIP ×6"), ("SigLIP ×3 + SigLIP+V-JEPA ×3 + V-JEPA ×3", "SigLIP ×9"),
         ("SigLIP ×10 + SigLIP+V-JEPA ×3", "SigLIP ×10"), ("SigLIP ×10 + SigLIP+V-JEPA ×3 + V-JEPA ×3", "SigLIP ×10")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film = manifest.read_folds(cfg)["film_group"]
    store = FeatureStore(cfg)
    V = {n: {"track_test_fold": [], "segment_all": []} for n in SYSTEMS}
    CL = {"track_test_fold": [], "segment_all": []}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        thits, shit, seg_pairs = fold_truth(store, f)
        M = {(g, s): member_scores(store, load_scorer(FILM_ROOT / "semantic" / v / f"fold{k}" / f"seed{s}", store, "cuda"), f)
             for g, (v, seeds) in MEMBERS.items() for s in seeds}
        for n, mem in SYSTEMS.items():
            r = ranks_of([M[x] for x in mem], thits, shit)
            for sec in CL:
                V[n][sec].append((r[sec] <= 10).astype(float))
        CL["track_test_fold"].append(np.array([film[p] for p in f["test"]]))
        CL["segment_all"].append(np.array([film[p] for p in seg_pairs]))
        log.info("fold %d: track R@10 " + ", ".join(f"{n} {V[n]['track_test_fold'][-1].mean():.3f}" for n in ("SigLIP ×6", "SigLIP ×3 + SigLIP+V-JEPA ×3", "SigLIP ×10")), k)
    V = {n: {s: np.concatenate(x) for s, x in v.items()} for n, v in V.items()}
    cl = {s: np.concatenate(x) for s, x in CL.items()}
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Ensembles mixing SigLIP heads with V-JEPA 2 heads (D68)", "",
         f"1,248-pair film split, test-fold gallery 196 (195–199). Hit@10 per query; paired differences: 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         "| ensemble | models | track R@10 | segment R@10 |", "| --- | --- | --- | --- |"]
    L += [f"| {n} | {len(m)} | {V[n]['track_test_fold'].mean():.3f} | {V[n]['segment_all'].mean():.3f} |" for n, m in SYSTEMS.items()]
    L += ["", "| mixed ensemble | against | track Δ R@10 [95% CI] | segment Δ R@10 [95% CI] |", "| --- | --- | --- | --- |"]
    rows = []
    for a, b in PAIRS:
        dt = cluster_bootstrap(V[a]["track_test_fold"] - V[b]["track_test_fold"], cl["track_test_fold"], B)
        ds = cluster_bootstrap(V[a]["segment_all"] - V[b]["segment_all"], cl["segment_all"], B)
        L.append(f"| {a} | {b} | {fmt(dt)} | {fmt(ds)} |")
        verdict = "KEEP" if dt["lo"] > 0 else ("REJECT (worse)" if dt["hi"] < 0 else "NULL")
        rows.append(f"| {date.today().isoformat()} | Mixed ensemble: {a} vs {b} | film | 196 (195–199) | — | {V[a]['track_test_fold'].mean():.3f} | — | {fmt(dt)}; segment {fmt(ds)} | {verdict} |")
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("\n".join(rows) + "\n")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
