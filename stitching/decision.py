"""Phase 3 decision (rule fixed before the results, user 2026-11-17) + Wilcoxon of every new row vs two-stage and greedy.

A new method is the paper's main method only if, against two-stage, (a) its mean track accuracy is higher with
Wilcoxon p < 0.05 AND (b) its switch F1 is not lower with p < 0.05 (fails (b) if mean switch F1 is lower and p < 0.05).
Two-stage reference: row 3 (as written in the rule) and the equal-budget re-tuned two-stage (768 configs for 2a / 2b /
my segmental Viterbi, 3,072 for 2c); a method must pass against both to count.
Paired over the 112 test sequences (5 folds pooled); CRF rows = mean over 3 seeds per sequence.
`.venv/bin/python -m stitching.decision` -> stitching/results/decision.md
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from stitching.report import OUT, per_episode

NEW = {"every_cut": [], "never_switch": [],
       "crf_noctx_xf": [], "crf_ctx_xf": [], "crf_ctx_prior_xf": [], "crf_ctx_prior_gold_xf": [],
       "semi_sum": ["two_stage_768"], "semi_pool": ["two_stage_768"], "semi_pool_cp": ["two_stage_3072"],
       "semi_viterbi": ["two_stage_768"], "two_stage_768": [], "two_stage_3072": []}
STATUS = (
    "**Status (2026-11-22, D150).** The rule above only decides whether a *new* decoder (semi-Markov 2a / 2b / 2c, "
    "segmental Viterbi, cross-fitted CRF) replaces two-stage. It did not pick the system used later: from D146 on, the "
    "supplementary experiments take row 12 (HMM, π from gold paths, λ tuned) as the 'final system' for their Δs. That "
    "label is not supported by this table. Two-stage (row 3) and the segmental Viterbi beat row 12 on track accuracy "
    "on both datasets: original 0.083 / 0.069 vs 0.044; cleaned 1,011 pairs 0.081 / 0.081 vs 0.048, Δ +0.033 "
    "[+0.012, +0.053] / [+0.013, +0.053], film-cluster bootstrap. Switch F1 is equal or higher (cleaned: +0.026 "
    "[−0.000, +0.052] / +0.036 [+0.008, +0.061]). Row 12 has the highest TBIoU and alignment success, because it "
    "switches more often (4.1 / min vs 3.4 true). Which decoder the paper presents as the main method is open; see "
    "docs/PROGRESS.md D150. Cleaned re-run: exp_supp/stitch_clean_table.md, "
    "runs/exp_supp/stitch_clean/results/{main_table,decision}.md.")
CANDIDATES = ("crf_noctx_xf", "crf_ctx_xf", "crf_ctx_prior_xf", "crf_ctx_prior_gold_xf", "semi_sum", "semi_pool", "semi_pool_cp", "semi_viterbi")


def load():
    folds = [json.loads((OUT / f"fold{k}.json").read_text()) for k in range(5)]
    for k, f in enumerate(folds):
        for suf in ("gold", "phase0", "xf", "phase2", "semi"):
            p = OUT / f"fold{k}_{suf}.json"
            if p.exists():
                f["methods"].update(json.loads(p.read_text())["methods"])
    names, pooled = {}, {}
    for f in folds:
        for key, m in f["methods"].items():
            rows, _ = per_episode(m)
            pooled.setdefault(key, []).extend(rows)
            names[key] = m["name"]
    return {k: v for k, v in pooled.items() if len(v) == 112}, names


def test(a, b, m):
    x = np.array([r[m] for r in a]) - np.array([r[m] for r in b])
    p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    return float(x.mean()), p


def main():
    P, N = load()
    L = ["# Phase 3: decision and paired tests", ""]
    verdicts = []
    for c in CANDIDATES:
        if c not in P:
            continue
        refs = ["two_stage"] + [r for r in NEW[c] if r in P]
        ok_all, parts = True, []
        for r in refs:
            da, pa = test(P[c], P[r], "track_acc")
            df, pf = test(P[c], P[r], "switch_f1")
            ok = (da > 0 and pa < 0.05) and not (df < 0 and pf < 0.05)
            ok_all &= ok
            parts.append(f"vs {N[r]}: Δacc {da:+.3f} (p {pa:.2g}), ΔF1 {df:+.3f} (p {pf:.2g}) → {'pass' if ok else 'fail'}")
        verdicts.append((c, ok_all, parts))
    winners = [c for c, ok, _ in verdicts if ok]
    if winners:
        L.insert(2, f"**Outcome: {', '.join(N[w] for w in winners)} meets the pre-registered rule against two-stage and is the main method.**")
    else:
        L.insert(2, "**Outcome: no new method meets the pre-registered rule against two-stage; two-stage change-point + retrieve is the "
                    "main method and the HMM / CRF / semi-Markov rows are ablations.**")
    L += ["", STATUS]
    L += ["", "## Rule check per candidate", ""]
    for c, ok, parts in verdicts:
        L.append(f"- **{N[c]}**: {'PASS' if ok else 'fail'}")
        L += [f"  - {x}" for x in parts]
    L += ["", "## Wilcoxon, every new row vs two-stage (row 3) and greedy (row 1)", "",
          "| row | vs | Δ track acc | p | Δ switch F1 | p | Δ precision | Δ recall |", "|---|---|---|---|---|---|---|---|"]
    for c in NEW:
        if c not in P:
            continue
        for r in ("two_stage", "greedy") + tuple(x for x in NEW[c] if x in P):
            da, pa = test(P[c], P[r], "track_acc")
            df, pf = test(P[c], P[r], "switch_f1")
            dp = np.nanmean(np.array([x["switch_precision"] for x in P[c]], float) - np.array([x["switch_precision"] for x in P[r]], float))
            dr = np.nanmean(np.array([x["align_success"] for x in P[c]], float) - np.array([x["align_success"] for x in P[r]], float))
            L.append(f"| {N[c]} | {N[r]} | {da:+.3f} | {pa:.2g} | {df:+.3f} | {pf:.2g} | {dp:+.3f} | {dr:+.3f} |")
    (OUT / "decision.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
