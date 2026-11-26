"""results/semantic.md (SPEC M6): semantic heads vs ImageBind zero-shot, 5 folds x 3 seeds.

Per-query metrics are averaged over seeds first; comparisons use a cluster bootstrap over track_id groups on the
paired per-query differences (SPEC §5.12). H1: the trained heads beat ImageBind on mean track R@10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.eval.metrics import cluster_bootstrap

SECTIONS = {"track_test_fold": "Track retrieval, test-fold gallery", "track_all": "Track retrieval, all-library gallery",
            "segment_all": "Segment retrieval, all windows", "segment_other": "Segment retrieval, other-video windows"}


def load_variant(cfg: DictConfig, variant: str):
    root = Path(cfg.paths.runs) / "semantic" / variant
    runs = sorted(root.glob("fold*/seed*/metrics.json"))
    assert runs, f"no runs under {root}"
    metrics = [(int(p.parent.parent.name[4:]), int(p.parent.name[4:]), json.loads(p.read_text())) for p in runs]
    per = {}
    for p in runs:
        k = int(p.parent.parent.name[4:])
        with np.load(p.parent / "perquery.npz") as z:
            per.setdefault(k, []).append({key: z[key] for key in z.files})
    return metrics, per


def write_semantic_report(cfg: DictConfig, variant: str) -> Path:
    metrics, per = load_variant(cfg, variant)
    zs = {k: dict(np.load(Path(cfg.paths.runs) / "zeroshot" / f"fold{k}" / "perquery.npz")) for k in per}
    L = [f"# Semantic heads: {variant}", "",
         f"{len(metrics)} runs ({len(per)} folds x {len(metrics) // max(len(per), 1)} seeds). Cells: mean ± std over runs; "
         "Δ vs ImageBind: per-query R@k averaged over seeds, paired, 95% cluster-bootstrap CI over track groups.", ""]
    verdict = None
    for sec, title in SECTIONS.items():
        L += [f"## {title}", "", "| system | R@1 | R@5 | R@10 | MedR | MRR |", "| --- | --- | --- | --- | --- | --- |"]
        row = lambda vals: " | ".join(f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}" for v in vals)  # noqa: E731
        ours = [[m["test"][sec][x] for _, _, m in metrics] for x in ("R@1", "R@5", "R@10", "MedR", "MRR")]
        L.append(f"| **heads ({variant})** | {row(ours)} |")
        ib_key = f"{sec}/imagebind"
        ib_folds = [json.loads((Path(cfg.paths.runs) / 'zeroshot' / f'fold{k}' / 'metrics.json').read_text()) for k in sorted(per)]
        ib = [[f[sec]["imagebind"][x] for f in ib_folds] for x in ("R@1", "R@5", "R@10", "MedR", "MRR")]
        L.append(f"| ImageBind zero-shot | {row(ib)} |")
        rnd = [[f[sec]["random"][x] for f in ib_folds] for x in ("R@1", "R@5", "R@10")]
        L.append(f"| random (exact) | {row(rnd)} | — | — |")
        L.append("")
        for kk in (1, 10):
            diff, cl = [], []
            for k in sorted(per):
                ours_hit = np.mean([(r[sec] <= kk).astype(float) for r in per[k]], 0)
                ib_hit = (zs[k][ib_key] <= kk).astype(float)
                diff.append(ours_hit - ib_hit)
                cl.append(per[k][0][f"{sec}/clusters"])
            b = cluster_bootstrap(np.concatenate(diff), np.concatenate(cl), cfg.eval.bootstrap_resamples)
            L.append(f"- Δ R@{kk} heads − ImageBind: **{b['mean']:+.3f}** [{b['lo']:+.3f}, {b['hi']:+.3f}] "
                     f"({b['n_queries']} queries, {b['n_clusters']} track groups)")
            if sec == "track_test_fold" and kk == 10:
                verdict = b
        L.append("")
    L += ["## H1", "", f"Trained heads beat ImageBind zero-shot on test-fold track R@10: "
          f"**{'yes' if verdict['lo'] > 0 else 'not shown'}** (Δ {verdict['mean']:+.3f}, 95% CI [{verdict['lo']:+.3f}, {verdict['hi']:+.3f}])."]
    val = [m["val"] for _, _, m in metrics]
    L += ["", f"Validation (early stopping): segment R@10 {np.mean([v['segment_R@10'] for v in val]):.3f} ± "
          f"{np.std([v['segment_R@10'] for v in val], ddof=1):.3f}; best epoch median {np.median([v['best_epoch'] for v in val]):.0f}."]
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"semantic_{variant}.md"
    path.write_text("\n".join(L) + "\n")
    pd.DataFrame([{"fold": k, "seed": s, **{f"{sec}/{x}": m["test"][sec][x] for sec in SECTIONS for x in ("R@1", "R@5", "R@10", "MedR", "MRR")}}
                  for k, s, m in metrics]).to_csv(out / f"semantic_{variant}.csv", index=False)
    return path
