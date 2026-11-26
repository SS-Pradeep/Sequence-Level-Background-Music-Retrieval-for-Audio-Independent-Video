"""`bgm eval --exp <name>`: evaluate an experiment on all 5 folds and aggregate (SPEC §5.12).

Per fold: runs/<exp>/fold<k>/{config.yaml, metrics.json, perquery.npz, git_commit.txt, seed.txt}.
Aggregate: results/<exp>.csv and results/<exp>.md (mean +- std over folds, cluster-bootstrap CIs over pooled queries).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.eval.metrics import GROUNDING_RULES, RULE_LABELS, chance_summary, cluster_bootstrap, recall_summary
from bgm.eval.protocol import grounding, segment_retrieval, track_retrieval
from bgm.eval.store import FeatureStore, fold_ids
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)


def evaluate_fold_zeroshot(cfg: DictConfig, store: FeatureStore, k: int) -> tuple[dict, dict]:
    from bgm.eval.zeroshot import ZeroShot
    f = fold_ids(cfg, k)
    test, train = f["test"], f["train"]
    zs = ZeroShot(store, cfg, train, tag=f"fold{k}")
    gal_test = list(dict.fromkeys(store.music_of[p] for p in test))
    gal_all = list(dict.fromkeys(store.pairs.music_id))
    systems = {"imagebind": (zs.imagebind_track, zs.imagebind_window),
               "text_bridge": (zs.bridge_track, zs.bridge_window),
               "attribute": (zs.attr_track, zs.attr_window)}
    metrics, per = {"track_test_fold": {}, "track_all": {}, "segment_all": {}, "segment_other": {}, "grounding": {}}, {}
    for name, (tf, wf) in systems.items():
        for key, gal in (("track_test_fold", gal_test), ("track_all", gal_all)):
            r = track_retrieval(store, test, gal, tf)
            metrics[key][name] = recall_summary(r.ranks, cfg.eval.ks)
            per[f"{key}/{name}"] = r.ranks
            if name == "imagebind":
                metrics[key]["random"] = chance_summary(r.N, r.h, cfg.eval.ks)
                per[f"{key}/clusters"] = r.clusters
        for key, variant in (("segment_all", "all"), ("segment_other", "other")):
            r = segment_retrieval(store, test, wf, variant)
            metrics[key][name] = recall_summary(r.ranks, cfg.eval.ks)
            per[f"{key}/{name}"] = r.ranks
            if name == "imagebind":
                metrics[key]["random"] = chance_summary(r.N, r.h, cfg.eval.ks)
                per[f"{key}/clusters"] = r.clusters
    from bgm.train.dynamics import grounding_arrays, grounding_summary
    g = grounding(store, test, zs.hand_offsets, cfg)
    metrics["grounding"]["hand_dynamics"] = grounding_summary(g)
    for k, v in grounding_arrays(g).items():
        per["grounding/clusters" if k == "clusters" else f"grounding/hand_dynamics/{k}"] = v
    metrics["calibration"] = {kind: c.temp for kind, c in zs.cal.items()}
    return metrics, per


def run_eval(cfg: DictConfig, exp: str) -> pd.DataFrame:
    assert exp == "zeroshot", f"`bgm eval --exp {exp}` is not built yet (zeroshot only, M5)"
    store = FeatureStore(cfg)
    folds, pers = [], []
    for k in range(cfg.splits.n_folds):
        m, per = evaluate_fold_zeroshot(cfg, store, k)
        d = run_dir(cfg, exp, k)
        write_run(d, cfg, m, seed=None)
        np.savez(d / "perquery.npz", **per)
        folds.append(m)
        pers.append(per)
        log.info("fold %d: track R@10 imagebind %.3f bridge %.3f random %.3f", k, m["track_test_fold"]["imagebind"]["R@10"],
                 m["track_test_fold"]["text_bridge"]["R@10"], m["track_test_fold"]["random"]["R@10"])
    return aggregate(cfg, exp, folds, pers)


def refresh_zeroshot_report(cfg: DictConfig) -> pd.DataFrame:
    """results/zeroshot.{csv,md} from the saved fold runs, with the hand-crafted grounding replaced by its re-scored
    arrays (runs/zeroshot/fold*/grounding_rescored.npz, `bgm eval --exp grounding`). Nothing is recomputed and the
    run directories are not modified."""
    from bgm.train.dynamics import grounding_summary
    from types import SimpleNamespace
    folds, pers = [], []
    for k in range(cfg.splits.n_folds):
        d = run_dir(cfg, "zeroshot", k)
        m, per = json.loads((d / "metrics.json").read_text()), dict(np.load(d / "perquery.npz"))
        z = dict(np.load(d / "grounding_rescored.npz"))
        assert np.array_equal(z["clusters"], per["grounding/clusters"]), d
        g = SimpleNamespace(err_s=z["err_s"], hits={r: z[r] for r in GROUNDING_RULES}, chance={r: z[f"{r}_chance"] for r in GROUNDING_RULES})
        m["grounding"]["hand_dynamics"] = grounding_summary(g)
        per = {key: v for key, v in per.items() if not key.startswith("grounding/hand_dynamics/")}
        per.update({f"grounding/hand_dynamics/{r}{sfx}": z[f"{r}{sfx}"] for r in GROUNDING_RULES for sfx in ("", "_chance")})
        folds.append(m)
        pers.append(per)
    return aggregate(cfg, "zeroshot", folds, pers)


def aggregate(cfg: DictConfig, exp: str, folds: list[dict], pers: list[dict]) -> pd.DataFrame:
    rows = []
    for section in ("track_test_fold", "track_all", "segment_all", "segment_other"):
        for sysname in folds[0][section]:
            for metric in ("R@1", "R@5", "R@10", "MedR", "MRR"):
                vals = [f[section][sysname].get(metric) for f in folds]
                if any(v is None for v in vals):
                    continue
                row = {"section": section, "system": sysname, "metric": metric, "mean": np.mean(vals), "std": np.std(vals, ddof=1)}
                if sysname != "random" and metric == "R@10":
                    ranks = np.concatenate([p[f"{section}/{sysname}"] for p in pers])
                    cl = np.concatenate([p[f"{section}/clusters"] for p in pers])
                    b = cluster_bootstrap((ranks <= 10).astype(float), cl, cfg.eval.bootstrap_resamples)
                    row.update(ci_lo=b["lo"], ci_hi=b["hi"])
                rows.append(row)
    cl = np.concatenate([p["grounding/clusters"] for p in pers])
    for sysname in folds[0]["grounding"]:
        for metric in [m for r in GROUNDING_RULES for m in (r, f"{r}_chance", f"{r}_excess")] + ["median_abs_err_s"]:
            base = metric.removesuffix("_excess")
            if metric != "median_abs_err_s" and f"grounding/{sysname}/{base}" not in pers[0]:
                continue
            if metric.endswith("_excess"):
                v = np.concatenate([p[f"grounding/{sysname}/{base}"] - p[f"grounding/{sysname}/{base}_chance"] for p in pers])
                vals = [float((p[f"grounding/{sysname}/{base}"] - p[f"grounding/{sysname}/{base}_chance"]).mean()) for p in pers]
            else:
                vals = [f["grounding"][sysname][metric] for f in folds]
                v = np.concatenate([p[f"grounding/{sysname}/{metric}"] for p in pers]) if metric != "median_abs_err_s" else None
            row = {"section": "grounding", "system": sysname, "metric": metric, "mean": np.mean(vals), "std": np.std(vals, ddof=1)}
            if v is not None:
                b = cluster_bootstrap(v, cl, cfg.eval.bootstrap_resamples)
                row.update(ci_lo=b["lo"], ci_hi=b["hi"])
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{exp}.csv", index=False)
    (out / f"{exp}.md").write_text(to_markdown(df, exp, folds))
    return df


def to_markdown(df: pd.DataFrame, exp: str, folds: list[dict]) -> str:
    def cell(r: dict) -> str:
        s = f"{r['mean']:.3f} ± {r['std']:.3f}"
        if pd.notna(r.get("ci_lo", np.nan)):
            s += f" [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        return s
    L = [f"# {exp}", "", f"Mean ± std over {len(folds)} folds; R@10 and grounding rates add a 95% cluster-bootstrap CI "
         "(resampling track_id groups over the pooled test queries).", ""]
    titles = {"track_test_fold": "Track retrieval, gallery = the test fold's music files",
              "track_all": "Track retrieval, gallery = all library music files",
              "segment_all": "Segment retrieval, all windows", "segment_other": "Segment retrieval, other-video windows only"}
    for section, title in titles.items():
        sub = df[df.section == section]
        if sub.empty:
            continue
        L += [f"## {title}", "", "| system | R@1 | R@5 | R@10 | MedR | MRR |", "| --- | --- | --- | --- | --- | --- |"]
        for sysname, g in sub.groupby("system", sort=False):
            cells = {r["metric"]: cell(r) for r in g.to_dict("records")}
            L.append(f"| {sysname} | " + " | ".join(cells.get(m, "—") for m in ("R@1", "R@5", "R@10", "MedR", "MRR")) + " |")
        L.append("")
    sub = df[df.section == "grounding"]
    if not sub.empty:
        L += ["## Grounding (own music file given, every 25 Hz offset; rules and chance levels as in results/dynamics.md)", "",
              "| system | rule | accuracy | chance | accuracy − chance |", "| --- | --- | --- | --- | --- |"]
        for sysname, g in sub.groupby("system", sort=False):
            c = {r["metric"]: cell(r) for r in g.to_dict("records")}
            for r in GROUNDING_RULES:
                if r in c:
                    L.append(f"| {sysname} | {RULE_LABELS[r]} | {c[r]} | {c.get(f'{r}_chance', '—')} | {c.get(f'{r}_excess', '—')} |")
            L.append(f"| {sysname} | median abs error s | {c.get('median_abs_err_s', '—')} | | |")
    L += ["", "Calibration temperatures per fold: " + json.dumps([f["calibration"] for f in folds])]
    return "\n".join(L) + "\n"
