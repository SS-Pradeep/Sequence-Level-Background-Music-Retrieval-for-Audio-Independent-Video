"""Shared helpers: run logging (exp_supp/results.csv + runs/exp_supp/...), clusters, paired cluster bootstrap, LaTeX."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "results.csv"
RUNS = ROOT.parent / "runs" / "exp_supp"
FIELDS = ["timestamp", "git_commit", "experiment", "method", "fold", "seed", "split", "config", "metrics"]


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT.parent)
    dirty = subprocess.run(["git", "status", "--porcelain", "exp_supp"], capture_output=True, text=True, cwd=ROOT.parent).stdout
    return r.stdout.strip() + ("+dirty-exp_supp" if dirty.strip() else "")


def log_run(experiment: str, method: str, fold: int, seed: int, split: str, config: dict, metrics: dict) -> None:
    """Append one run to exp_supp/results.csv and write runs/exp_supp/<experiment>/<method>/fold<k>/seed<s>/."""
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(), "experiment": experiment,
           "method": method, "fold": fold, "seed": seed, "split": split,
           "config": json.dumps(config, sort_keys=True, default=float), "metrics": json.dumps(metrics, sort_keys=True, default=float)}
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    d = RUNS / experiment / method / f"fold{fold}" / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    import yaml
    (d / "config.yaml").write_text(yaml.safe_dump(json.loads(row["config"]), sort_keys=True))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (d / "git_commit.txt").write_text(row["git_commit"] + "\n")
    (d / "seed.txt").write_text(f"{seed}\n")


def merge_clusters(groups: list[list[str]]) -> np.ndarray:
    """Union-find: units (e.g. sequences) that share any film group fall in one cluster. Returns [n] cluster ids."""
    par: dict = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, g in enumerate(groups):
        for f in set(g):
            par[find(("f", f))] = find(("u", i))
        find(("u", i))
    roots = [find(("u", i)) for i in range(len(groups))]
    first = {}
    for r in roots:                                  # deterministic ids: order of first appearance
        first.setdefault(r, len(first))
    return np.array([first[r] for r in roots])


def paired_ci(diff, clusters, n: int = 10000, seed: int = 0) -> dict:
    from bgm.eval.metrics import cluster_bootstrap
    return cluster_bootstrap(diff, clusters, n_resamples=n, seed=seed)


def fmt_ci(r: dict, nd: int = 3) -> str:
    s = lambda x: f"{x:+.{nd}f}".replace("-", "−")  # noqa: E731
    star = "*" if r["lo"] > 0 or r["hi"] < 0 else ""
    return f"{s(r['mean'])} [{s(r['lo'])}, {s(r['hi'])}]{star}"
