"""Train vs test retrieval for the title-split loss runs (user 2026-11-20: "title split and train").

Every saved checkpoint (runs/titlesplit/planned_losses/semantic/<config>/fold<k>/seed<s>/ckpt.pt) is re-scored with the
same M5 protocol (`bgm.train.semantic.evaluate`) on two clip sets of equal size:
  test  = folds[k].test (the fold's held-out title groups; numbers equal planned_losses_runs.csv)
  train = a random subset of folds[k].train with len(test) clips (rng [fold, 2026], the same subset for every config),
so the gallery size, and so chance level, matches. gap = train - test. Evaluation only, no training.

    python -m bgm.sweep.train_vs_test            # configs of the loss bank; cached per run
Outputs: results/titlesplit/train_vs_test{_runs.csv,.md}.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.sweep.titlesplit import BANK_ORDER, FOLDS, N_FOLDS, OUT, load_cfg, load_scorer

log = logging.getLogger("bgm.sweep.train_vs_test")
CSV = OUT / "train_vs_test_runs.csv"
FIELDS = ["config", "fold", "seed", "set", "n", "R1", "R10", "MRR", "track_R10"]


def main(seeds=tuple(range(10)), names=None) -> None:
    import torch
    from bgm.eval.store import FeatureStore
    from bgm.train.semantic import evaluate
    cfg = load_cfg()
    store = FeatureStore(cfg)
    split = json.loads(FOLDS.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV)
        done = set(zip(d.config, d.fold, d.seed, d["set"]))
    sets = {}
    for k in range(N_FOLDS):
        f = split["folds"][str(k)]
        sub = sorted(np.random.default_rng([k, 2026]).choice(f["train"], len(f["test"]), replace=False).tolist())
        sets[k] = {"test": f["test"], "train": sub}
    new = not CSV.exists()
    with open(CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for name in names or BANK_ORDER:
            for k in range(N_FOLDS):
                for s in seeds:
                    ck = Path(cfg.paths.runs) / "semantic" / name / f"fold{k}" / f"seed{s}" / "ckpt.pt"
                    if not ck.exists():
                        continue
                    sc = None
                    for which, ids in sets[k].items():
                        if (name, k, s, which) in done:
                            continue
                        sc = sc or load_scorer(cfg, store, name, k, s, device)
                        res, _ = evaluate(cfg, store, sc, ids)
                        sa, tr = res["segment_all"], res["track_test_fold"]
                        w.writerow({"config": name, "fold": k, "seed": s, "set": which, "n": len(ids), "R1": sa["R@1"],
                                    "R10": sa["R@10"], "MRR": sa["MRR"], "track_R10": tr["R@10"]})
                        fh.flush()
    if names is None:
        table()


def table() -> str:
    d = pd.read_csv(CSV)
    rows = []
    for name in BANK_ORDER:
        x = d[d.config == name]
        if x.empty:
            continue
        te, tr = x[x["set"] == "test"].set_index(["fold", "seed"]), x[x["set"] == "train"].set_index(["fold", "seed"])
        idx = te.index.intersection(tr.index)
        r = {"config": name, "runs": len(idx)}
        for c in ("R10", "track_R10"):
            a, b = tr.loc[idx, c].astype(float), te.loc[idx, c].astype(float)
            r[f"tr_{c}"], r[f"te_{c}"], r[f"gap_{c}"] = (a.mean(), a.std(ddof=1)), (b.mean(), b.std(ddof=1)), ((a - b).mean(), (a - b).std(ddof=1))
        rows.append(r)
    rows.sort(key=lambda r: -r["te_R10"][0])
    pm = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}"
    L = ["# Train vs test retrieval, title split",
         "",
         "Title split (folds_title.json), 5 folds × every trained seed, mean ± std over runs. Each checkpoint is scored with the M5 "
         "protocol on its fold's test films and on an equal-size random subset of its training clips (same subset for "
         "every config), so both galleries have the same size and the same chance level. gap = train − test (larger = "
         "more overfitting). No tags are used anywhere. Sorted by test segment R@10.",
         "",
         "| config | runs | train seg R@10 | test seg R@10 | gap | train track R@10 | test track R@10 | gap |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['runs']} | {pm(r['tr_R10'])} | {pm(r['te_R10'])} | {pm(r['gap_R10'])} | "
                 f"{pm(r['tr_track_R10'])} | {pm(r['te_track_R10'])} | {pm(r['gap_track_R10'])} |")
    md = "\n".join(L) + "\n"
    (OUT / "train_vs_test.md").write_text(md)
    print(md, flush=True)
    return md


def parallel(n_workers: int = 5) -> None:
    """Score BANK_ORDER in n_workers subprocesses (configs dealt round-robin), then write the table."""
    import os
    import subprocess
    import sys
    env = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    groups = [BANK_ORDER[i::n_workers] for i in range(n_workers)]
    procs = [subprocess.Popen([sys.executable, "-m", "bgm.sweep.train_vs_test", "--configs", *g], env=env) for g in groups]
    codes = [p.wait() for p in procs]
    assert not any(codes), codes
    table()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()
    main(names=a.configs) if a.configs else parallel()
