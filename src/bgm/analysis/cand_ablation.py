"""Candidate ablation of the stitching stage, cleaned title split (PROGRESS.md D138, Steps 4-5).

python -m bgm.analysis.cand_ablation  -> results/d138_candidates/{ablation,penalties}.csv, summary.md

One factor at a time around the shipped setting (K = 10 files per span, C = J = 20 windows per segment, 0.5 s windows):
C in {5, 10, 20, 40} (K = 10), K in {5, 10, 20} (C = 20), window step in {0.25, 0.5, 1, 2} s (K = 10, C = 20).
Each cell = fuse + tune-stitch of the 10-head ensembles (run seed e = heads 10e .. 10e + 9), 5 folds x run seeds 0-2;
inference only. Step 5: tune-stitch re-tunes lam_switch, lam_jump and mu for every cell on that fold's val
concatenations (grid of SPEC §6; objective = part track accuracy + switch F1, as the paper); the chosen values per fold
are listed. Cells already run by the D135 shortlist sweep (runs/cleaned/topk) are reused, not re-run.

Columns: switch F1, switches / min, part (song) track accuracy on the 50 test concatenations per fold; candidate recall
= window@C on the natural test videos (the true window among the segment's top C, fuse's metric); runtime = fuse
seconds per run (all test + val queries: shortlist, window scoring) and tune-stitch seconds per run (72-point val
grid + test decoding + rendering; timed by the D138 chain, so only for the new cells). Ground-truth switches / min = the concatenations' joins.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RUNS = Path("runs/cleaned")
OUT = Path("results/d138_candidates")
TIMES = Path("reports/cleaned/d138/cand/times.tsv")
FOLDS, SEEDS = range(5), range(3)
BASE = (10, 20, 0.5)
CELLS = ([("C", (10, c, 0.5)) for c in (5, 10, 20, 40)] + [("K", (k, 20, 0.5)) for k in (5, 10, 20)]
         + [("step", (10, 20, h)) for h in (0.25, 0.5, 1.0, 2.0)]
         + [("combined", (10, 40, h)) for h in (1.0, 2.0)])          # the best C and steps on val, together (Step 6)


def exp_name(K: int, J: int, hop: float) -> str:
    if (K, J, hop) == BASE:
        return "pipeline_ens10"
    if hop == 0.5 and J <= 20:                         # D135 shortlist sweep (K <= 20, J <= 20), reused
        return f"topk/K{K}_J{J}"
    return f"d138_cand/K{K}_J{J}_H{hop:g}"


def complete(cell) -> bool:
    return all((RUNS / exp_name(*cell) / f"fold{k}" / f"seed{s}" / "stitch_metrics.json").exists() for k in FOLDS for s in SEEDS)


def stitch_seconds() -> dict[tuple[str, int, int], float]:
    """{(exp, fold, seed): tune-stitch seconds} from the D138 chain's timing file (exp, fold, seed, fuse_s, stitch_s).
    Cells reused from the D135 sweep were not timed this way and are absent."""
    out = {}
    if TIMES.exists():
        for line in TIMES.read_text().splitlines():
            e, k, s, _, st = line.split("\t")
            out[(e, int(k), int(s))] = float(st)
    return out


def fold_stats(vals: dict[int, list[float]]) -> tuple[float, float]:
    F = np.array([np.mean(vals[k]) for k in FOLDS])
    return float(F.mean()), float(F.std(ddof=1))


def gt_switches_per_min() -> float:
    p = Path("results/d138_errors/sequences.csv")
    if not p.exists():
        return float("nan")
    import pandas as pd
    S = pd.read_csv(p).drop_duplicates(["fold", "seq"])
    return float((S.n_joins / (S.duration / 60)).mean())


def main() -> None:
    import pandas as pd
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    secs = stitch_seconds()
    rows, pens = [], []
    for factor, cell in CELLS:
        if not complete(cell):
            log.warning("cell %s %s not complete, skipped", factor, cell)
            continue
        e = exp_name(*cell)
        cols = {}
        for k in FOLDS:
            for s in SEEDS:
                d = RUNS / e / f"fold{k}" / f"seed{s}"
                m, st = json.loads((d / "metrics.json").read_text()), json.loads((d / "stitch_metrics.json").read_text())
                t = st["test"]
                r = {"switch_f1": t["concat/viterbi/switch_f1"], "sw_min": t["concat/viterbi/switches_per_min"],
                     "song_acc": t["concat/viterbi/part_track_acc"], "cand_recall": m["test"]["fused"]["window@J"],
                     "greedy_switch_f1": t["concat/greedy/switch_f1"], "greedy_sw_min": t["concat/greedy/switches_per_min"],
                     "fuse_s": m["seconds"], "stitch_s": secs.get((e, k, s), np.nan), "val_obj": st["val_objective"]}
                for c, v in r.items():
                    cols.setdefault(c, {}).setdefault(k, []).append(v)
                pens.append({"factor": factor, "K": cell[0], "C": cell[1], "step_s": cell[2], "fold": k, "seed": s, **st["penalties"]})
        row = {"factor": factor, "K": cell[0], "C": cell[1], "step_s": cell[2], "shipped": cell == BASE, "exp": e}
        for c, v in cols.items():
            row[c], row[c + "_std"] = fold_stats({k: [x for x in v[k] if np.isfinite(x)] or [np.nan] for k in FOLDS})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["gt_sw_min"] = gt_switches_per_min()
    P = pd.DataFrame(pens)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ablation.csv", index=False)
    P.to_csv(OUT / "penalties.csv", index=False)
    gt = df.gt_sw_min.iloc[0] if len(df) else float("nan")
    L = ["# Candidate ablation (D138, Step 4) and re-tuned penalties (Step 5)", "",
         "Cleaned title split, shipped 10-head ensembles, 50 test concatenations per fold, 5 folds × 3 run seeds; ± = std over "
         "the 5 fold means. One factor at a time; **bold** = shipped. Penalties re-tuned on val for every cell. "
         f"Ground-truth switches / min: {gt:.2f}. Development numbers.", "",
         "| factor | K | C | step (s) | switch F1 | switches / min | song acc. | cand. recall (window@C) | fuse s / run | tune-stitch s / run | val obj. |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        b = (lambda x: f"**{x}**") if r.shipped else (lambda x: x)
        L.append("| " + " | ".join(b(x) for x in (r.factor, f"{r.K}", f"{r.C}", f"{r.step_s:g}", f"{r.switch_f1:.3f} ± {r.switch_f1_std:.3f}",
                                                   f"{r.sw_min:.2f}", f"{r.song_acc:.3f} ± {r.song_acc_std:.3f}", f"{r.cand_recall:.3f}",
                                                   f"{r.fuse_s:.0f}", f"{r.stitch_s:.0f}" if np.isfinite(r.stitch_s) else "—", f"{r.val_obj:.3f}")) + " |")
    L += ["", "## Step 5: penalties chosen on val per fold (run seeds 0 / 1 / 2)", "",
          "| factor | K | C | step (s) | fold | λ_switch | λ_jump | μ |", "|---|---|---|---|---|---|---|---|"]
    for (f, K, C, h, k), g in P.groupby(["factor", "K", "C", "step_s", "fold"], sort=False):
        L.append(f"| {f} | {K} | {C} | {h:g} | {k} | " + " | ".join(" / ".join(f"{x:g}" for x in g[c]) for c in ("lam_switch", "lam_jump", "mu")) + " |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT / "summary.md")


if __name__ == "__main__":
    main()
