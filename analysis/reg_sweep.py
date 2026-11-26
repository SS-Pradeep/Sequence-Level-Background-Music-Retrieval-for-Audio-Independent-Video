"""Regularisation sweep for the semantic projection heads, cleaned title split (PROGRESS.md D138, Step 3; follows D134 / D135).

python -m bgm.analysis.reg_sweep overrides <variant>   -> dotted overrides of a variant (read by reports/cleaned/d138_chain.sh)
python -m bgm.analysis.reg_sweep select combo|ensemble -> the combination's variants / the setting to train as a 10-head ensemble
python -m bgm.analysis.reg_sweep report                -> results/d138_regularisation/{single_heads,ensembles}.csv, summary.md

Baseline = the shipped head (LayerNorm -> 512 -> GELU -> dropout 0.2 -> 256; AdamW lr 1e-3, wd 0.05, batch 128;
learnable τ from 0.07, floor 0.01; early stop on val R@10, patience 20) = the D134 baseline runs. One change at a time,
5 folds x seeds 0-2 each. The τ floor 0.05 is not trained: the learned τ never falls below 0.059 in the 15 baseline runs
(train_log.csv), so the clamp would never act and the run would equal the baseline. Rows the user asked for that D134 / D135 already trained under the same protocol are reused
(τ fixed at 0.07, linear head, dropout 0.5). Combination: the best change of each family (temperature, weight decay,
dropout, width, patience, noise) whose val R@10 beats the baseline, the top 3 families by val. The best single-head
setting on val (combination included) is trained with seeds 0-9 and scored as a 10-head ensemble against the shipped one.
Everything is chosen on validation segment R@10 (the early-stopping metric); the test fold is only reported.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.analysis.underfit_diag import SPLIT, complete, hits10, run_paths, summarize
from bgm.config import load_config
from bgm.eval.metrics import cluster_bootstrap

log = logging.getLogger(__name__)

OUT = Path("results/d138_regularisation")
REP = Path("reports/cleaned/d138")
SEL = REP / "selection.json"
ENS = Path("runs/cleaned/semantic_ens")
BASE = "d134-base"
SEM = "train.semantic."
NEW = {"d138-tau005": [SEM + "tau_learnable=false", SEM + "tau_init=0.05"],
       "d138-tau01": [SEM + "tau_learnable=false", SEM + "tau_init=0.1"],
       "d138-wd01": [SEM + "weight_decay=0.1"],
       "d138-wd03": [SEM + "weight_decay=0.3"],
       "d138-do03": [SEM + "dropout=0.3"],
       "d138-h256": [SEM + "hidden=256"],
       "d138-pat10": [SEM + "patience=10"],
       "d138-noise005": [SEM + "input_noise=0.05"]}
REUSED = {"d134-taufixed": [SEM + "tau_learnable=false"], "d134-linear": [SEM + "head_type=linear"],
          "d135-do05": [SEM + "dropout=0.5"]}
SINGLE = {**NEW, **REUSED}
FAMILY = {"temperature": ["d138-tau005", "d138-tau01", "d134-taufixed"],
          "weight decay": ["d138-wd01", "d138-wd03"], "dropout": ["d138-do03", "d135-do05"],
          "width": ["d138-h256", "d134-linear"], "patience": ["d138-pat10"], "noise": ["d138-noise005"]}
LABEL = {BASE: "baseline (shipped)", "d138-tau005": "τ fixed 0.05", "d138-tau01": "τ fixed 0.1",
         "d138-taumin005": "τ floor 0.05", "d134-taufixed": "τ fixed 0.07 (D134)", "d138-wd01": "weight decay 0.1",
         "d138-wd03": "weight decay 0.3", "d138-do03": "dropout 0.3", "d135-do05": "dropout 0.5 (D135)",
         "d138-h256": "hidden 256", "d134-linear": "linear head (D134)", "d138-pat10": "patience 10",
         "d138-noise005": "input noise σ 0.05", "d138-combo": "combination"}
SHIPPED_ENS = "siglip-both-ens10"


def val_of(variant: str) -> float:
    return float(np.mean([json.loads((p / "metrics.json").read_text())["val"]["segment_R@10"] for *_, p in run_paths(variant)]))


def selection() -> dict:
    return json.loads(SEL.read_text()) if SEL.exists() else {}


def overrides(variant: str) -> list[str]:
    if variant in SINGLE:
        return SINGLE[variant]
    if variant == "d138-combo":
        return [o for v in selection()["combo_parts"] for o in SINGLE[v]]
    if variant == BASE:
        return []
    raise KeyError(variant)


def pick_combo(val: dict[str, float], base: float, max_parts: int = 3) -> list[str]:
    """Best variant of each family that beats the baseline on val; the top max_parts families by that val."""
    best = {f: max(vs, key=val.get) for f, vs in FAMILY.items()}
    good = sorted((v for v in best.values() if val[v] > base), key=val.get, reverse=True)
    return good[:max_parts]


def select(stage: str) -> str:
    sel = selection()
    if stage == "combo":
        val = {v: val_of(v) for v in [BASE, *SINGLE]}
        parts = pick_combo(val, val[BASE])
        sel.update(val=val, combo_parts=parts)
        out = " ".join(parts) if len(parts) >= 2 else ""
    else:
        val = sel["val"] | ({"d138-combo": val_of("d138-combo")} if complete("d138-combo") else {})
        best = max(val, key=lambda v: (round(val[v], 4), v == BASE))
        sel.update(val=val, ensemble=best)
        out = best
    REP.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(sel, indent=1))
    log.info("selection: %s", sel)
    return out


def ens_name(variant: str) -> str:
    return SHIPPED_ENS if variant == BASE else f"{variant}-ens10"


def report() -> None:
    import pandas as pd
    from bgm.analysis.overfit_fix import ens_hits
    cfg = load_config(overrides=SPLIT)
    sel = selection()
    variants = [v for v in [BASE, *SINGLE, "d138-combo"] if complete(v)]
    base_h, groups = hits10(cfg, BASE)
    rows = []
    for v in variants:
        s = summarize(v)
        m, sd = s["mean"], s["std"]
        r = {"variant": v, "setting": LABEL[v] + (" (" + " + ".join(LABEL[p] for p in sel.get("combo_parts", [])) + ")" if v == "d138-combo" else ""),
             "source": "D138" if v.startswith("d138") else ("baseline" if v == BASE else v.split("-")[0].upper() + " (reused)"),
             "overrides": " ".join(overrides(v)), "train_R@10": m[1], "train_R@10_std": sd[1], "val_segR@10": s["val"],
             "test_R@1": m[2], "test_R@10": m[3], "test_R@10_std": sd[3], "test_MRR": m[4], "gap_train_test": m[1] - m[3],
             "gap_train_val": m[1] - s["val"], "best_epoch": s["best_epoch"]}
        if v != BASE:
            h, _ = hits10(cfg, v)
            ci = cluster_bootstrap(h - base_h, groups)
            r |= {"dtest_R@10": ci["mean"], "dtest_R@10_lo": ci["lo"], "dtest_R@10_hi": ci["hi"]}
        rows.append(r)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "single_heads.csv", index=False)
    L = ["# Regularisation sweep of the semantic heads (D138, Step 3)", "",
         "Cleaned title split, SigLIP → CLAP+MERT, InfoNCE. Single heads, 5 folds × seeds 0–2 per row, metrics at the "
         "best-on-validation epoch. train R@10 = track R@10 on a training subset the size of the test fold (D134); val R@10 = "
         "segment R@10 on the val clips (the early-stopping and selection metric; val galleries hold 44–69 files, so it is "
         "not on the same scale as train / test); test R@10 = track R@10, test-fold gallery. Gap = train − test track R@10. "
         "± = std over fold means; Δ = per test query, 95% film-group cluster bootstrap. Development numbers. "
         "τ floor 0.05 is identical to the baseline by construction: the learned τ stays ≥ 0.059 in every baseline run, "
         "so the floor never acts (not trained).", "",
         "| setting | source | train R@10 | val R@10 | test R@10 | gap (train − test) | Δ test R@10 vs baseline [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        d = "—" if r.variant == BASE else f"{r['dtest_R@10']:+.3f} [{r['dtest_R@10_lo']:+.3f}, {r['dtest_R@10_hi']:+.3f}]"
        L.append(f"| {r.setting} | {r.source} | {r['train_R@10']:.3f} ± {r['train_R@10_std']:.3f} | {r['val_segR@10']:.3f} | "
                 f"{r['test_R@10']:.3f} ± {r['test_R@10_std']:.3f} | {r.gap_train_test:.3f} | {d} |")
    best = sel.get("ensemble")
    if best:
        L += ["", f"Best on validation: **{LABEL.get(best, best)}**.", ""]
    if best and best != BASE and (ENS / ens_name(best) / "fold4" / "seed0" / "metrics.json").exists():
        er = []
        for nm, lab in ((SHIPPED_ENS, "shipped: baseline heads × 10"), (ens_name(best), f"{LABEL[best]} × 10")):
            ms = [json.loads((ENS / nm / f"fold{k}" / "seed0" / "metrics.json").read_text())["test"] for k in range(5)]
            tr = np.array([[m["track_test_fold"]["R@1"], m["track_test_fold"]["R@10"], m["track_test_fold"]["MRR"]] for m in ms])
            er.append({"ensemble": lab, "name": nm, "R@1": tr[:, 0].mean(), "R@10": tr[:, 1].mean(), "R@10_std": tr[:, 1].std(ddof=1), "MRR": tr[:, 2].mean()})
        hn, g, _ = ens_hits(cfg, ens_name(best))
        ho, _, _ = ens_hits(cfg, SHIPPED_ENS)
        ci = cluster_bootstrap(hn - ho, g)
        er[-1] |= {"dR@10": ci["mean"], "dR@10_lo": ci["lo"], "dR@10_hi": ci["hi"]}
        pd.DataFrame(er).to_csv(OUT / "ensembles.csv", index=False)
        L += ["## 10-head ensembles (seeds 0–9, one per fold)", "", "| ensemble | R@1 | R@10 | MRR | Δ R@10 [95% CI] |", "|---|---|---|---|---|"]
        for r in er:
            d = f"{r['dR@10']:+.3f} [{r['dR@10_lo']:+.3f}, {r['dR@10_hi']:+.3f}]" if "dR@10" in r else "—"
            L.append(f"| {r['ensemble']} | {r['R@1']:.3f} | {r['R@10']:.3f} ± {r['R@10_std']:.3f} | {r['MRR']:.3f} | {d} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT / "summary.md")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cmd = sys.argv[1]
    if cmd == "overrides":
        print(" ".join(overrides(sys.argv[2])))
    elif cmd == "select":
        print(select(sys.argv[2]))
    else:
        report()


if __name__ == "__main__":
    main()
