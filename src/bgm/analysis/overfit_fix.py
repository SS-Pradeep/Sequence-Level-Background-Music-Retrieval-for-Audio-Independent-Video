"""Reducing the semantic heads' overfitting, cleaned title split (PROGRESS.md D135; follows D134's diagnosis).

python -m bgm.analysis.overfit_fix overrides <variant>   -> dotted overrides of a variant (read by the run chain)
python -m bgm.analysis.overfit_fix select                -> combo settings, then the single-head setting to ensemble
python -m bgm.analysis.overfit_fix report                -> results/cleaned/overfit.md, reports/cleaned/d135/*.png

Stage 1, single heads (5 folds x seeds 0-2, train recall logged as in D134; baseline = the D134 baseline runs):
weight decay 0.3 + dropout 0.4 (D76's KEEP on the old film split), dropout 0.5, weight decay 1.0 + dropout 0.4,
input-feature dropout 0.2, EMA of the weights (0.95 per step), and distillation from the 10 same-fold baseline heads
(D76's other KEEP). Stage 2: a combination (the best weight-decay / dropout setting plus each other regulariser whose
val R@10 beats the baseline). Stage 3: the best single-head setting on val, trained with seeds 0-9 and evaluated as a
10-head ensemble against the shipped one (siglip-both-ens10, members 0-9) - the comparison that decides.
Everything is chosen on validation R@10; Δ = per test query (seeds averaged), 95% cluster bootstrap over film groups.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.analysis.underfit_diag import SPLIT, complete, hits10, plot_curves, run_paths, summarize
from bgm.config import load_config
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.store import fold_ids

log = logging.getLogger(__name__)

REP = Path("reports/cleaned/d135")
OUT_MD = Path("results/cleaned/overfit.md")
OUT_CFG = Path("configs/best_semantic.yaml")
ENS = Path("runs/cleaned/semantic_ens")
SEL = REP / "selection.json"
BASE = "d134-base"
WDDO = {"d135-wd03-do04": ["train.semantic.weight_decay=0.3", "train.semantic.dropout=0.4"],
        "d135-do05": ["train.semantic.dropout=0.5"],
        "d135-wd1-do04": ["train.semantic.weight_decay=1.0", "train.semantic.dropout=0.4"]}
OTHER = {"d135-indrop02": ["train.semantic.input_dropout=0.2"],
         "d135-ema095": ["train.semantic.ema_decay=0.95"]}
KD = {"d135-kd": ["train.semantic.distill_from_seeds=[0,1,2,3,4,5,6,7,8,9]"]}
STAGE1 = {**WDDO, **OTHER, **KD}
LABEL = {BASE: "baseline (wd 0.05, dropout 0.2)", "d135-wd03-do04": "weight decay 0.3 + dropout 0.4",
         "d135-do05": "dropout 0.5", "d135-wd1-do04": "weight decay 1.0 + dropout 0.4",
         "d135-indrop02": "input-feature dropout 0.2", "d135-ema095": "EMA of weights (0.95/step)",
         "d135-kd": "distillation from the 10-head ensemble", "d135-combo": "combination"}


def val_of(variant: str) -> float:
    return float(np.mean([json.loads((p / "metrics.json").read_text())["val"]["segment_R@10"] for *_, p in run_paths(variant)]))


def selection() -> dict:
    return json.loads(SEL.read_text()) if SEL.exists() else {}


def overrides(variant: str) -> list[str]:
    if variant in STAGE1:
        return STAGE1[variant]
    if variant == "d135-combo":
        return selection()["combo"]
    raise KeyError(variant)


def select(stage: str) -> str:
    """stage 'combo': the combination's overrides; stage 'ensemble': the single-head setting to train with seeds 0-9."""
    sel = selection()
    if stage == "combo":
        base = val_of(BASE)
        val = {v: val_of(v) for v in [BASE, *STAGE1]}
        wd = max(WDDO, key=val.get)
        combo = WDDO[wd] + [x for v, o in OTHER.items() if val[v] > base for x in o]
        sel.update(val=val, combo=combo, combo_parts=[wd] + [v for v in OTHER if val[v] > base])
    else:
        cands = [BASE, *WDDO, *OTHER, "d135-combo"]
        val = sel["val"] | {"d135-combo": val_of("d135-combo")}
        best = max(cands, key=lambda v: (round(val[v], 4), v == BASE))
        sel.update(val=val, ensemble=best)
    REP.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(sel, indent=1))
    log.info("selection: %s", sel)
    return " ".join(sel["combo"]) if stage == "combo" else sel["ensemble"]


def ens_hits(cfg, name: str, key: str = "track_test_fold", k_at: int = 10) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from bgm.data.manifest import read_film_groups
    film = read_film_groups(cfg)
    h, g, m = [], [], []
    for k in range(5):
        d = ENS / name / f"fold{k}" / "seed0"
        pq = np.load(d / "perquery.npz")
        h.append(pq[key] <= k_at)
        # perquery clusters are track ids; map them to film groups via the fold's track-level queries (one per test video)
        track_film = {t: film[p] for t, p in zip(pq["track_test_fold/clusters"], fold_ids(cfg, k)["test"])}
        g.extend(track_film[t] for t in pq[f"{key}/clusters"])
        m.append(json.loads((d / "metrics.json").read_text())["test"][key])
    return np.concatenate(h).astype(float), np.asarray(g), m


def fmt(m: float, s: float) -> str:
    return f"{m:.3f} ± {s:.3f}"


def report() -> None:
    cfg = load_config(overrides=SPLIT)
    sel = selection()
    variants = [v for v in [BASE, *STAGE1, "d135-combo"] if complete(v)]
    S = {v: summarize(v) for v in variants}
    base_h, groups = hits10(cfg, BASE)
    combo_label = " + ".join(LABEL[v] for v in sel.get("combo_parts", []))
    L = ["# Reducing overfitting of the semantic heads (D135)", "",
         "Cleaned title split, SigLIP → CLAP+MERT, InfoNCE. Single heads: 5 folds × seeds 0–2 per row, metrics at the "
         "best-on-validation epoch; train = a size-matched subset of each fold's training clips (as in D134). Every "
         "choice is made on validation R@10; the test fold is only reported. Δ = per test query, seeds averaged, 95% "
         "cluster bootstrap over film groups (10,000). Development numbers.", "",
         "## Single heads", "",
         "| run | train R@1 | train R@10 | test R@1 | test R@10 | test MRR | train − test R@10 | Δ test R@10 vs baseline [95% CI] | val R@10 | best epoch |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in variants:
        s = S[v]
        cells = [fmt(m, d) for m, d in zip(s["mean"], s["std"])]
        if v == BASE:
            delta = "—"
        else:
            h, _ = hits10(cfg, v)
            ci = cluster_bootstrap(h - base_h, groups)
            delta = f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]" + (" **↑**" if ci["lo"] > 0 else " **↓**" if ci["hi"] < 0 else "")
        name = LABEL[v] + (f" ({combo_label})" if v == "d135-combo" else "")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {s['mean'][1] - s['mean'][3]:.3f} | {delta} | {s['val']:.3f} | {s['best_epoch']:.0f} |")
    L += ["", "↑ / ↓ = the CI clears zero. Curves per fold: `reports/cleaned/d135/<run>_fold<k>.png`.", ""]
    best = sel.get("ensemble")
    name_new, name_old = f"{best}-ens10", "siglip-both-ens10"
    if best and (ENS / name_new / "fold4" / "seed0" / "metrics.json").exists():
        L += ["## Ensembles (what the system ships)", "",
              f"The best single-head setting on validation, **{LABEL[best]}**, trained with seeds 0–9 and scored as a "
              "10-head ensemble (mean cosine), against the shipped ensemble (baseline heads, seeds 0–9). One ensemble "
              "per fold; ± = std over the 5 folds.", "",
              "| ensemble | test R@1 | test R@10 | test MRR | segment R@10 |", "| --- | --- | --- | --- | --- |"]
        rows = {}
        for nm, lab in ((name_old, "shipped: baseline heads × 10"), (name_new, f"{LABEL[best]} × 10")):
            ms = [json.loads((ENS / nm / f"fold{k}" / "seed0" / "metrics.json").read_text())["test"] for k in range(5)]
            tr = np.array([[m["track_test_fold"]["R@1"], m["track_test_fold"]["R@10"], m["track_test_fold"]["MRR"],
                            m["segment_all"]["R@10"]] for m in ms])
            rows[nm] = tr
            L.append(f"| {lab} | " + " | ".join(fmt(a, b) for a, b in zip(tr.mean(0), tr.std(0, ddof=1))) + " |")
        L.append("")
        for key, k_at, lab in (("track_test_fold", 10, "track R@10"), ("track_test_fold", 1, "track R@1"), ("segment_all", 10, "segment R@10")):
            hn, g, _ = ens_hits(cfg, name_new, key, k_at)
            ho, _, _ = ens_hits(cfg, name_old, key, k_at)
            ci = cluster_bootstrap(hn - ho, g)
            L.append(f"- Δ {lab}, new − shipped: {ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
            if lab == "track R@10":
                ens_ci = ci
        L.append("")
        win = ens_ci["lo"] > 0
        L += ["## Recommendation", "",
              (f"- **Adopt {LABEL[best]}**: its 10-head ensemble beats the shipped one on test track R@10 with a CI clear of zero. "
               f"Saved as `{OUT_CFG}`." if win else
               f"- The ensemble of {LABEL[best]} heads does not beat the shipped ensemble with a CI clear of zero, so the "
               f"shipped configuration stays; `{OUT_CFG}` is unchanged.")]
        if win:
            write_config(overrides(best) if best != BASE else [], best)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    for v in variants:
        if v != BASE:
            plot_curves(v, REP / v.removeprefix("d135-"), LABEL[v])
    log.info("wrote %s", OUT_MD)


def write_config(cfg_overrides: list[str], variant: str) -> None:
    from omegaconf import OmegaConf
    cfg = load_config(overrides=SPLIT + cfg_overrides)
    head = (f"# D135 recommended semantic-head configuration ({variant}: {LABEL[variant]}), chosen on validation and\n"
            f"# confirmed as a 10-head ensemble against the shipped one. = configs/default.yaml + the cleaned title split\n"
            f"# + {' '.join(cfg_overrides) or 'no changes'}. Use: bgm train semantic --config {OUT_CFG}\n")
    OUT_CFG.write_text(head + OmegaConf.to_yaml(cfg))


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
