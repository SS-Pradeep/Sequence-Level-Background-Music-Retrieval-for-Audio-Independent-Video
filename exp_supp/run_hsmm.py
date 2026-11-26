"""Experiment 1: hidden semi-Markov decoder vs the final system's decoder, on the Table 3 lattices.

`.venv/bin/python -m exp_supp.run_hsmm` -> runs/exp_supp/hsmm/fold<k>.json, exp_supp/results.csv,
                                         exp_supp/hsmm_table.{md,tex}

Final system (Table 3) = "HMM, prior from gold paths, alpha tuned" (stitching/results/fold<k>_gold.json, read-only).
Before anything else the final row is re-decoded with its saved parameters and must reproduce its saved per-sequence
metrics exactly (same lattices). Tuning on val sequences only; test decoded once.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from exp_supp.common import RUNS, fmt_ci, log_run, merge_clusters, paired_ci
from exp_supp.hsmm import HSMM, LogNormal, training_bgm_cue_lengths
from stitching.data import load_fold
from stitching.hmm_derived import HMMDerived
from stitching.metrics import evaluate_path, fold_entropy
from stitching.report import _derive

log = logging.getLogger("exp_supp.hsmm")
GOLD = Path("stitching/results")
OUT = RUNS / "hsmm"
METRICS = ["track_acc", "switch_f1", "TBIoU", "switches_per_min", "jumps_per_min", "align_success"]


def fold_run(k: int) -> dict:
    t0 = time.time()
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    final = json.loads((GOLD / f"fold{k}_gold.json").read_text())["methods"]["hmm_gold_tuned"]
    fp = final["params"]
    # 1) reproduce the final row on these lattices
    hmm = HMMDerived(fp["tau"], fp["L"], fp["rho"], fp["lam"])
    rep = [evaluate_path(e, hmm.decode(e)) for e in te]
    diff = max(abs(a[m] - b[m]) for a, b in zip(rep, final["rows"]) for m in METRICS if np.isfinite(a[m]))
    assert diff < 1e-9, f"fold {k}: final row not reproduced (max diff {diff})"
    # 2) fit the HSMM on val, decode test once
    dur = LogNormal(training_bgm_cue_lengths(k))
    st = HSMM(fp["tau"], fp["L"], dur)
    params = st.fit(tr, va)
    paths = [st.timed(e) for e in te]
    rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
    ent = fold_entropy(te, paths, F["gallery_test"])
    mean = {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS}
    mean["ms_per_video"] = 1000 * float(np.mean([r["runtime_s"] for r in rows]))
    log.info("fold %d HSMM %s | test %s | final reproduced (max diff %.1e) [%.0f s]", k,
             {a: b for a, b in params.items() if a in ("alpha", "lam_jump", "mu", "gamma", "m")},
             {a: round(b, 3) for a, b in mean.items()}, diff, time.time() - t0)
    cfg = {"decoder": "hsmm", "lattices": f"stitching/cache/fold{k}.pkl (Table 3)", "grid": st.grid,
           **{a: b for a, b in params.items()}}
    log_run("exp1_hsmm", "hsmm", k, 0, "test", cfg, dict(mean, entropy=ent, val_objective=params["val_objective"]))
    rec = {"fold": k, "params": params, "rows": rows, "entropy": ent, "final_rows": final["rows"],
           "final_params": fp, "reproduction_max_diff": diff,
           "films": {e.key: sorted(set(e.films)) for e in te}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold{k}.json").write_text(json.dumps(rec, default=float))
    return rec


def table(recs: list[dict]) -> None:
    def fold_stats(key, m):
        per = [np.nanmean([_derive(r)[m] for r in rec[key]]) for rec in recs]
        return float(np.mean(per)), float(np.std(per))
    keys = [r["key"] for rec in recs for r in rec["rows"]]
    assert keys == [r["key"] for rec in recs for r in rec["final_rows"]]
    films = {k: v for rec in recs for k, v in rec["films"].items()}
    clusters = merge_clusters([films[k] for k in keys])
    H = [r for rec in recs for r in rec["rows"]]
    Fr = [r for rec in recs for r in rec["final_rows"]]
    deltas = {m: paired_ci(np.array([h[m] - f[m] for h, f in zip(H, Fr)], float), clusters) for m in METRICS}
    ms = lambda key: 1000 * np.mean([r["runtime_s"] for rec in recs for r in rec[key]])  # noqa: E731
    names = {"final_rows": "HMM, prior from gold paths, $\\alpha$ tuned (final)", "rows": "Hidden semi-Markov (cue level)"}
    L = ["# Experiment 1: hidden semi-Markov decoder (Table 3 protocol)", "",
         "112 test sequences (Table 3 lattices: BaSSL scenes, same candidates and emissions), 5 title folds. Mean ± std over",
         "folds (std for track acc. and switch F1, as Table 3). Δ = HSMM − final, per sequence, 95% cluster bootstrap over film groups",
         f"(sequences sharing a film group merged: {len(set(clusters))} clusters), 10,000 resamples. * = CI excludes 0.", "",
         "Protocol note: the Table 3 lattices were built by the stitching/ harness from data/manifests (1,248 pairs) with the",
         "film-title folds that cleaned_dataset/manifests/folds_title.json carries over unchanged (0 fold/split mismatches);",
         "121 of the 982 clips in these test sequences were later excluded by cleaning (77 of 112 sequences touched). Both rows",
         "use these identical lattices; the final row is re-decoded and reproduces its saved Table 3 values exactly.", "",
         "| Decoder | Track acc. | Switch F1 ±1 s | TBIoU | Switches/min | Jumps/min | Align. succ. | ms/video |",
         "|---|---|---|---|---|---|---|---|"]
    for key in ("final_rows", "rows"):
        a, b = fold_stats(key, "track_acc"), fold_stats(key, "switch_f1")
        L.append(f"| {names[key].replace('$\\alpha$', 'α')} | {a[0]:.3f} ± {a[1]:.3f} | {b[0]:.3f} ± {b[1]:.3f} | "
                 + " | ".join(f"{fold_stats(key, m)[0]:.{2 if 'min' in m else 3}f}" for m in METRICS[2:]) + f" | {ms(key):.1f} |")
    L += ["", "| Δ (HSMM − final) | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS),
          "| | " + " | ".join(fmt_ci(deltas[m], 2 if "min" in m else 3) for m in METRICS) + " |", "",
          "## Tuned parameters (val)", "", "| fold | α | λ_jump | μ | γ | m | val objective | cue prior median / mean (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for rec in recs:
        p = rec["params"]
        L.append(f"| {rec['fold']} | {p['alpha']} | {p['lam_jump']} | {p['mu']} | {p['gamma']} | {p['m']} | {p['val_objective']:.3f} | "
                 f"{p['dur']['median_s']:.1f} / {p['dur']['mean_s']:.1f} |")
    Path("exp_supp/hsmm_table.md").write_text("\n".join(L) + "\n")

    t = lambda s: s.replace("−", "-")  # noqa: E731
    d = deltas
    a, b = fold_stats("rows", "track_acc"), fold_stats("rows", "switch_f1")
    fa, fb = fold_stats("final_rows", "track_acc"), fold_stats("final_rows", "switch_f1")
    row = lambda nm, A, B, key: (f"{nm} & {A[0]:.3f} & {B[0]:.3f} & {fold_stats(key, 'TBIoU')[0]:.3f} & "  # noqa: E731
                                 f"{fold_stats(key, 'switches_per_min')[0]:.2f} & {fold_stats(key, 'jumps_per_min')[0]:.2f} & "
                                 f"{fold_stats(key, 'align_success')[0]:.3f} \\\\")
    sig = lambda r: r["lo"] > 0 or r["hi"] < 0  # noqa: E731
    ci = lambda r, nd=3: t((r"$\mathbf{%+.*f}$" if sig(r) else r"$%+.*f$") % (nd, r["mean"]))  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
           r"\caption{Cue-level hidden semi-Markov decoder vs.\ the final scene-level HMM on the 112 test sequences of",
           r"Tab.~3 (same scenes, candidates and emissions; parameters tuned on validation). Mean over five folds;",
           r"$\Delta$ = semi-Markov $-$ final with 95\% film-level cluster-bootstrap CI.}",
           r"\label{tab:supp_hsmm}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
           r"Decoder & Acc. & F1 & TBIoU & Sw./min & J./min & Align. \\", r"\midrule",
           row(r"HMM, gold prior (final)", fa, fb, "final_rows"), row(r"Semi-Markov (cue)", a, b, "rows"), r"\midrule",
           r"$\Delta$ & " + " & ".join(ci(d[m], 2 if "min" in m else 3) for m in METRICS) + r" \\",
           r"95\% CI & " + " & ".join(t(f"[{d[m]['lo']:+.{2 if 'min' in m else 3}f}, {d[m]['hi']:+.{2 if 'min' in m else 3}f}]") for m in METRICS) + r" \\",
           r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path("exp_supp/hsmm_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(L))


def main() -> None:
    recs = [fold_run(k) for k in range(5)]
    table(recs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
