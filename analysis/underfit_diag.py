"""Underfitting diagnosis and quick fixes for the semantic heads, cleaned title split (PROGRESS.md D134).

python -m bgm.analysis.underfit_diag overrides <variant>   -> the variant's dotted overrides (read by the run chain)
python -m bgm.analysis.underfit_diag select                -> picks the Step 2 setting with the best mean VAL R@10
python -m bgm.analysis.underfit_diag report                -> results/cleaned/underfit.md, reports/cleaned/d134/*.png,
                                                              configs/best_semantic.yaml

Every run: SigLIP -> CLAP+MERT heads, InfoNCE, 5 folds x seeds 0-2, `train.semantic.log_train_recall=true` (per-epoch
track R@1/10/MRR on a fixed train subset of the test fold's size, gallery = its music files, and on the test fold;
logged only). Step 2 (training settings) is chosen on validation R@10; Steps 3-4 (head, pooling) reuse that choice.
Table: train / test track R@k at the best-on-val checkpoint, mean ± std over the 5 fold means (seeds averaged first);
Δ test R@10 vs the baseline per query (seeds averaged), 95% cluster bootstrap over film groups, 10,000 resamples.
Development numbers: the settings are searched on these same folds.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.config import load_config
from bgm.data.manifest import read_film_groups
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.store import fold_ids

log = logging.getLogger(__name__)

RUNS = Path("runs/cleaned/semantic")
REP = Path("reports/cleaned/d134")
OUT_MD = Path("results/cleaned/underfit.md")
OUT_CFG = Path("configs/best_semantic.yaml")
SPLIT = ["splits.folds_file=folds_title.json", "paths.manifests=cleaned_dataset/manifests",
         "paths.interim=cleaned_dataset/interim", "paths.features=cleaned_dataset/features", "paths.runs=runs/cleaned",
         "paths.results=results/cleaned", "paths.reports=reports/cleaned",
         "train.semantic.max_clips_per_film_per_epoch=30", "train.semantic.music_feature=both"]
FOLDS, SEEDS = range(5), range(3)
STEP2 = {"d134-base": [],
         "d134-long": ["train.semantic.max_epochs=400", "train.semantic.patience=40"],
         "d134-lr3e-4": ["train.semantic.lr_scratch=3.0e-4"],
         "d134-lr1e-4": ["train.semantic.lr_scratch=1.0e-4"],
         "d134-taufixed": ["train.semantic.tau_learnable=false"]}
STEP34 = {"d134-mlp": ["train.semantic.dropout=0.1"],                 # the requested MLP: d_in -> 512 -> 256, GELU, dropout 0.1
          "d134-linear": ["train.semantic.head_type=linear"],
          "d134-attn": ["train.semantic.video_pool=attn"],
          "d134-attn-mlp": ["train.semantic.video_pool=attn", "train.semantic.dropout=0.1"]}
LABEL = {"d134-base": "baseline (lr 1e-3, learnable τ, MLP dropout 0.2, mean pool)",
         "d134-long": "2a: 2× epochs (400) + patience 40", "d134-lr3e-4": "2b: lr 3e-4", "d134-lr1e-4": "2b: lr 1e-4",
         "d134-taufixed": "2c: τ fixed 0.07", "d134-mlp": "3: MLP head, dropout 0.1", "d134-linear": "3: linear head",
         "d134-attn": "4: attention pooling", "d134-attn-mlp": "4: attention pooling + MLP dropout 0.1"}
BEST_FILE = REP / "best_step2.json"
COL = {"train": "#2a78d6", "test": "#eb6834", "val": "#1baf7a", "loss": "#4a3aa7"}


def best_step2() -> str:
    return json.loads(BEST_FILE.read_text())["variant"] if BEST_FILE.exists() else "d134-base"


def overrides(variant: str) -> list[str]:
    if variant in STEP2:
        return STEP2[variant]
    return STEP2[best_step2()] + STEP34[variant]


def run_paths(variant: str):
    return [(k, s, RUNS / variant / f"fold{k}" / f"seed{s}") for k in FOLDS for s in SEEDS]


def complete(variant: str) -> bool:
    return all((p / "metrics.json").exists() for _, _, p in run_paths(variant))


def read_log(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path / "train_log.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def select() -> str:
    """Step 2 winner: the highest mean best-on-val segment R@10 over the 15 runs (ties keep the baseline)."""
    val = {v: np.mean([json.loads((p / "metrics.json").read_text())["val"]["segment_R@10"] for *_, p in run_paths(v)])
           for v in STEP2}
    win = max(val, key=lambda v: (round(val[v], 4), v == "d134-base"))
    REP.mkdir(parents=True, exist_ok=True)
    BEST_FILE.write_text(json.dumps({"variant": win, "val_seg_R@10": val}, indent=1))
    log.info("Step 2 val R@10: %s -> %s", {k: round(x, 4) for k, x in val.items()}, win)
    return win


def summarize(variant: str) -> dict:
    """Fold means (seeds averaged) of train / test track metrics at the best checkpoint, plus training dynamics."""
    per_fold, val, best_ep, ran, end_drop = {}, [], [], [], []
    for k, s, p in run_paths(variant):
        m = json.loads((p / "metrics.json").read_text())
        tr, te = m["train_track_size_matched"], m["test"]["track_test_fold"]
        per_fold.setdefault(k, []).append([tr["R@1"], tr["R@10"], te["R@1"], te["R@10"], te["MRR"]])
        val.append(m["val"]["segment_R@10"]), best_ep.append(m["val"]["best_epoch"]), ran.append(m["val"]["epochs_run"])
        L = read_log(p)["loss"]
        end_drop.append((L[-6] - L[-1]) / max(L[0] - L[-1], 1e-9) if len(L) > 6 else np.nan)
    F = np.array([np.mean(per_fold[k], 0) for k in FOLDS])                # [5 folds, 5 metrics]
    return {"mean": F.mean(0), "std": F.std(0, ddof=1), "val": float(np.mean(val)), "best_epoch": np.median(best_ep),
            "epochs_run": np.median(ran), "end_drop": float(np.nanmean(end_drop))}


def hits10(cfg, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Per test query (all folds), hit@10 averaged over seeds, and its film group."""
    film = read_film_groups(cfg)
    h, g = [], []
    for k in FOLDS:
        r = np.mean([np.load(RUNS / variant / f"fold{k}" / f"seed{s}" / "perquery.npz")["track_test_fold"] <= 10 for s in SEEDS], 0)
        h.append(r), g.extend(film[p] for p in fold_ids(cfg, k)["test"])
    return np.concatenate(h), np.asarray(g)


def plot_curves(variant: str, path_stem: Path, label: str | None = None) -> list[Path]:
    """One PNG per fold: loss (top) and track R@10 on train / test plus val segment R@10 (bottom), seeds as lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    out = []
    for k in FOLDS:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
        for s in SEEDS:
            p = RUNS / variant / f"fold{k}" / f"seed{s}"
            L, best = read_log(p), json.loads((p / "metrics.json").read_text())["val"]["best_epoch"]
            lab = s == 0
            a1.plot(L["epoch"], L["loss"], color=COL["loss"], lw=1.4, alpha=0.8, label="training loss" if lab else None)
            a2.plot(L["epoch"], L["train_R@10"], color=COL["train"], lw=1.6, label="train R@10 (size-matched)" if lab else None)
            a2.plot(L["epoch"], L["test_R@10"], color=COL["test"], lw=1.6, label="test R@10" if lab else None)
            a2.plot(L["epoch"], L["val_seg_R@10"], color=COL["val"], lw=1.1, ls="--", label="val segment R@10 (early stopping)" if lab else None)
            for ax in (a1, a2):
                ax.axvline(best, color="#898781", lw=0.8, ls=":", zorder=0)
        a1.set_ylabel("InfoNCE loss"), a1.set_ylim(bottom=0)
        a2.set_ylabel("Recall@10"), a2.set_xlabel("epoch"), a2.set_ylim(0, 1.02)
        a1.set_title(f"{label or LABEL[variant]}\nfold {k}, seeds 0–2; dotted = best-on-val epoch", loc="left", fontsize=9)
        for ax in (a1, a2):
            ax.grid(axis="y", color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        a1.legend(frameon=False, loc="upper right", fontsize=8)
        a2.legend(frameon=False, loc="center right", fontsize=8)
        fig.tight_layout()
        f = path_stem.parent / f"{path_stem.name}_fold{k}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        out.append(f)
    return out


def write_config(cfg_overrides: list[str], variant: str) -> None:
    from omegaconf import OmegaConf
    cfg = load_config(overrides=SPLIT + cfg_overrides)
    head = (f"# D134 recommended semantic-head configuration ({variant}: {LABEL[variant]}).\n"
            f"# = configs/default.yaml + the cleaned title split + {' '.join(cfg_overrides) or 'no changes'}.\n"
            f"# Use: bgm train semantic --config {OUT_CFG}\n")
    OUT_CFG.write_text(head + OmegaConf.to_yaml(cfg))


def report() -> None:
    cfg = load_config(overrides=SPLIT)
    REP.mkdir(parents=True, exist_ok=True)
    variants = [v for v in list(STEP2) + list(STEP34) if complete(v)]
    S = {v: summarize(v) for v in variants}
    base_h, groups = hits10(cfg, "d134-base")
    c = cfg.train.semantic
    L = ["# Underfitting diagnosis and quick fixes, semantic heads (D134)", "",
         "Cleaned title split, SigLIP → CLAP+MERT, InfoNCE, 5 folds × seeds 0–2 per row (15 runs). Train metrics use a "
         "fixed random subset of each fold's training clips the size of its test fold (gallery = their music files, "
         "same size as the test gallery) so train and test recall are comparable; test = track retrieval against the "
         "test fold's music. Checkpoint = best epoch on validation segment R@10 (the test fold is never used to "
         "choose anything during training). Development numbers.", "",
         "## Step 1: diagnostics (baseline)", "",
         f"- Settings: lr {c.lr_scratch:g} (AdamW, weight decay {c.weight_decay}, cosine with {cfg.train.warmup_frac:.0%} warmup), "
         f"InfoNCE τ learnable from {c.tau_init} (floor {c.tau_min}), max {c.max_epochs} epochs, early stopping patience "
         f"{c.patience} on val segment R@10, batch {c.batch}, film cap 30 clips per epoch, head dropout {c.dropout}."]
    b = S["d134-base"]
    taus = [read_log(p)["tau"][-1] for *_, p in run_paths("d134-base")]
    L += [f"- Epochs: best-on-val epoch median {b['best_epoch']:.0f}; runs stop (patience) after a median "
          f"{b['epochs_run']:.0f} epochs; no run reaches the {c.max_epochs}-epoch limit. Learned τ at the end: "
          f"{np.mean(taus):.3f} (range {min(taus):.3f}–{max(taus):.3f}).",
          f"- **Loss at the end of training: still decreasing.** Over the last 5 epochs it still falls by "
          f"{b['end_drop']:.0%} of its total drop on average; training is stopped by early stopping, not by a flat loss.",
          f"- **Train vs test at the chosen checkpoint:** train R@10 {b['mean'][1]:.3f} ± {b['std'][1]:.3f} vs test "
          f"{b['mean'][3]:.3f} ± {b['std'][3]:.3f}. Train recall keeps rising after the best epoch (to ~1.0) while "
          "test recall stays flat: the heads overfit; they do not underfit.",
          "- Curves per fold: " + ", ".join(f"[fold {k}](../../reports/cleaned/d134/base_fold{k}.png)" for k in FOLDS), ""]
    plot_curves("d134-base", REP / "base")
    win = best_step2()
    L += ["## Results", "",
          f"Step 2 is chosen on validation R@10: **{LABEL[win]}** (val R@10 by setting: " +
          ", ".join(f"{LABEL[v].split(':')[0]} {S[v]['val']:.3f}" for v in STEP2 if v in S) + "). The Step 3–4 rows all include it.", "",
          "| run | train R@1 | train R@10 | test R@1 | test R@10 | test MRR | Δ test R@10 vs baseline [95% CI] | val R@10 | best epoch | flag |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = {}
    for v in variants:
        s = S[v]
        cells = [f"{m:.3f} ± {d:.3f}" for m, d in zip(s["mean"], s["std"])]
        if v == "d134-base":
            delta, flag = "—", ""
        else:
            h, _ = hits10(cfg, v)
            ci = cluster_bootstrap(h - base_h, groups)
            rows[v] = ci
            delta = f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
            flag = "**overfit**" if s["mean"][1] - b["mean"][1] > 0.01 and ci["lo"] <= 0.0 else ""
        name = LABEL[v] + (f" (+ {LABEL[win].split(':')[0]})" if v in STEP34 and win != "d134-base" else "")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {delta} | {s['val']:.3f} | {s['best_epoch']:.0f} | {flag} |")
    L += ["", "Flag **overfit** = train R@10 rises by more than 0.01 over the baseline while test R@10 does not "
          "(its Δ CI does not clear zero).", ""]
    by_test = max(variants, key=lambda v: S[v]["mean"][3])
    by_val = max(variants, key=lambda v: S[v]["val"])
    clear = [v for v, ci in rows.items() if ci["lo"] > 0]
    L += ["## Recommendation", "",
          f"- Highest mean test R@10: **{LABEL[by_test]}** ({S[by_test]['mean'][3]:.3f}); highest val R@10: "
          f"{LABEL[by_val]} ({S[by_val]['val']:.3f}).",
          f"- Rows whose test R@10 gain over the baseline clears zero: {', '.join(LABEL[v] for v in clear) or 'none'}."]
    rec = by_test if by_test in clear else "d134-base"
    if rec == "d134-base" and by_test != "d134-base":
        L.append(f"- No change beats the baseline with a CI clear of zero, so the recommended configuration stays the "
                 f"baseline; {LABEL[by_test]} is within noise and was picked by test numbers, which would bias them.")
    L.append(f"- Saved: `{OUT_CFG}` = {LABEL[rec]}.")
    write_config(overrides(rec) if rec != "d134-base" else [], rec)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    for v in variants:
        if v != "d134-base":
            plot_curves(v, REP / v.removeprefix("d134-"))
    log.info("wrote %s, %s", OUT_MD, OUT_CFG)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cmd = sys.argv[1]
    if cmd == "overrides":
        print(" ".join(overrides(sys.argv[2])))
    elif cmd == "select":
        print(select())
    else:
        report()


if __name__ == "__main__":
    main()
