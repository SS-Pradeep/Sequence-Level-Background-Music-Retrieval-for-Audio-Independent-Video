"""D142: losses and layers for the boundary head (variant 4 of D140), chosen on val, test reported once.

`python -m bgm.segmentation.improve train [variant ...]` trains 5 folds x 3 seeds per variant (train_head.SPECS);
`python -m bgm.segmentation.improve eval [variant ...]` scores them against emb_p (D140 variant 4) on the frozen
title-split sequences: val AP (the selection metric, mean over folds x seeds from the run dirs) and, on test, AP,
boundary F1, and the segment metrics (purity, coverage, mIoU, Pk, WindowDiff; evaluate.segment_metrics), each with a
95% CI and a paired difference vs emb_p (cluster bootstrap over film groups, 10,000 resamples). theta per fold on val
F1 +-1 s, as in D140. Writes results/segmentation_improve.{md,json}.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.config import load_config
from bgm.segmentation import evaluate as ev

log = logging.getLogger(__name__)

BASE = "emb_p"
NAMES = {"emb_p": "4 BaSSL + head (D140)", "focal": "L1 focal loss", "rank": "L2 BCE + ranking loss",
         "contrast": "F1 + contrast features", "gru": "A1 + BiGRU over boundaries", "combo": "C combination",
         "emb_p_dur": "4b + duration, motion (D140)"}


def val_ap(cfg, v: str) -> tuple[float, float]:
    xs = [json.loads((Path(cfg.paths.runs) / "segmentation" / v / f"fold{k}" / f"seed{s}" / "metrics.json").read_text())["val_ap"]
          for k in range(5) for s in range(3)]
    return float(np.mean(xs)), float(np.std(xs))


def run_eval(cfg, variants: list[str]) -> dict:
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold, tune_theta
    from bgm.segmentation.train_head import SegData
    variants = [BASE] + [v for v in variants if v != BASE]
    data = SegData(cfg, need_siglip=True)
    import torch
    bassl = BaSSL("cuda" if torch.cuda.is_available() else "cpu")
    heads = {v: v for v in variants}
    per = {v: {"counts": [], "seg": [], "w": [], "score": [], "label": [], "type": [], "seq": []} for v in variants}
    clusters, n_seq, thetas = [], 0, {}
    lab = pd.read_csv(Path(cfg.paths.runs) / "titlesplit" / "cut_changes" / "test_predictions.csv",
                      usecols=["pair", "t", "label", "group", "fold"])
    probe = []
    for k in range(5):
        val, test = Fold(data, bassl, k, "val", heads), Fold(data, bassl, k, "test", heads)
        clusters += ev.seq_clusters(test.films, f"f{k}")
        for v in variants:
            th = thetas[(v, k)] = tune_theta(val, v)
            for i in range(len(test.seqs)):
                c = test.cT[i]
                cuts = test.cuts(v, i, th, False)
                rec = dict(test.truth[i], preds=cuts)
                d = per[v]
                d["counts"].append(ev.seq_counts(rec))
                d["seg"].append(ev.segment_metrics(cuts, test.truth[i]["gt"], c.duration))
                d["w"].append(c.duration)
                d["score"].append(test.score[v][i])
                d["label"].append(c.labels)
                d["type"].append(np.array(c.types))
                d["seq"].append(np.full(len(c.btimes), n_seq + i))
        probe += probe_rows(test, lab[lab.fold == k], variants)
        n_seq += len(test.seqs)
        log.info("fold %d thetas: %s", k, {v: thetas[(v, k)] for v in variants})
    bs = ev.Bootstrap(clusters, 10000, 0)
    draws, res = {}, {"n_seq": n_seq, "n_clusters": len(bs.uniq), "variants": {}}
    for v in variants:
        d, m, dr = per[v], {}, {}
        sc, lab, seq = (np.concatenate(d[x]) for x in ("score", "label", "seq"))
        typ = np.concatenate(d["type"])
        for name, sel in (("AP", np.ones_like(lab, bool)), ("AP in film", typ != "cross")):
            dr[name] = bs.ap(sc[sel], lab[sel], seq[sel])
            m[name] = {"mean": ev.average_precision(sc[sel], lab[sel]), "ci": ev.ci(dr[name])}
        for key, name in (("all/0.5", "F1 ±0.5 s"), ("all/1", "F1 ±1 s"), ("in_film/1", "F1 ±1 s in film")):
            c = np.array([r[key] for r in d["counts"]], float)
            dr[name] = bs.f1(c)
            m[name] = {"mean": float(ev.f1(*c.sum(0))), "ci": ev.ci(dr[name])}
        w = np.array(d["w"])
        per_c = np.zeros(len(bs.uniq))
        np.add.at(per_c, bs.inv, w)
        wsum = bs.M @ per_c
        for key in ev.SEG_KEYS:
            x = np.array([r[key] for r in d["seg"]])
            num = np.zeros(len(bs.uniq))
            np.add.at(num, bs.inv, x * w)
            dr[key] = (bs.M @ num) / wsum
            m[key] = {"mean": float((x * w).sum() / w.sum()), "ci": ev.ci(dr[key])}
        pf = lambda a, b: 2 * a * b / (a + b)  # noqa: E731
        dr["F (purity, coverage)"] = pf(dr["purity"], dr["coverage"])
        m["F (purity, coverage)"] = {"mean": pf(m["purity"]["mean"], m["coverage"]["mean"]), "ci": ev.ci(dr["F (purity, coverage)"])}
        m["val AP"] = val_ap(cfg, v)
        res["variants"][v] = m
        draws[v] = dr
    for v in variants[1:]:
        res["variants"][v]["delta"] = {key: {"mean": res["variants"][v][key]["mean"] - res["variants"][BASE][key]["mean"],
                                             "ci": ev.ci(draws[v][key] - draws[BASE][key])} for key in draws[v]}
    res["thetas"] = {f"{v}/fold{k}": t for (v, k), t in thetas.items()}
    res["probe"] = probe_summary(pd.DataFrame(probe), variants)
    write(res, variants)
    return res


def probe_rows(test, lab: pd.DataFrame, variants: list[str], tol: float = 0.12) -> list[dict]:
    """One row per labelled in-clip cut matched to a TransNetV2 in-clip boundary: label, film group and each variant's
    p (plus BaSSL's original probability)."""
    by_pair = {p: g for p, g in lab.groupby("pair")}
    rows = []
    for i, q in enumerate(test.seqs):
        c = test.cT[i]
        start = np.concatenate([[0.0], c.joins])
        for b in np.flatnonzero(~c.is_join):
            pair = q[c.shot_part[b]]
            if pair not in by_pair:
                continue
            g = by_pair[pair]
            t = c.btimes[b] - start[c.shot_part[b]]
            j = np.abs(g.t.to_numpy() - t)
            if j.min() > tol:
                continue
            r = g.iloc[int(j.argmin())]
            rows.append({"label": int(r.label), "group": r.group, "tn_bassl": float(test.score["tn_bassl"][i][b]),
                         **{v: float(test.score[v][i][b]) for v in variants}})
    return rows


def probe_summary(df: pd.DataFrame, variants: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score
    bs = ev.Bootstrap(list(df.group), 10000, 1)
    idx = np.arange(len(df))
    y = df.label.to_numpy()
    out = {"n": int(len(df)), "n_pos": int(y.sum()), "base_rate": float(y.mean())}
    draws = {}
    for v in ["tn_bassl", *variants]:
        sc = df[v].to_numpy()
        draws[v] = bs.ap(sc, y, idx)
        out[v] = {"AP": {"mean": ev.average_precision(sc, y), "ci": ev.ci(draws[v])}, "AUROC": float(roc_auc_score(y, sc))}
    for v in variants[1:]:
        out[v]["delta_AP"] = {"mean": out[v]["AP"]["mean"] - out[BASE]["AP"]["mean"], "ci": ev.ci(draws[v] - draws[BASE])}
    return out


COLS = ("AP", "F1 ±1 s", "F1 ±1 s in film", "purity", "coverage", "F (purity, coverage)", "miou", "pk", "windowdiff")
LOWER = ("pk", "windowdiff")


def write(res: dict, variants: list[str]) -> None:
    V = res["variants"]
    f = lambda m: f"{m['mean']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"  # noqa: E731

    def d(m, key):
        lo, hi = m["ci"]
        good = (hi < 0) if key in LOWER else (lo > 0)
        bad = (lo > 0) if key in LOWER else (hi < 0)
        return f"{m['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" ▲" if good else " ▼" if bad else "")
    lines = ["# Boundary head: losses and layers (D142)", "",
             f"Selection: val AP (mean ± std over 5 folds × 3 seeds). Test: frozen title-split sequences, 5 folds pooled "
             f"({res['n_seq']} sequences, {res['n_clusters']} film clusters), 3 seeds' probabilities averaged, θ per fold on "
             "val; each method's own cuts, without the oracle joins. [95% CI], cluster bootstrap over film groups, 10,000 "
             "resamples. Pk and WindowDiff: lower is better.", "",
             "| variant | val AP | " + " | ".join(COLS) + " |", "| --- " * (len(COLS) + 2) + "|"]
    for v in variants:
        m = V[v]
        lines.append(f"| {NAMES.get(v, v)} | {m['val AP'][0]:.3f} ± {m['val AP'][1]:.3f} | " + " | ".join(f(m[c]) for c in COLS) + " |")
    lines += ["", f"Paired differences vs {NAMES[BASE]} (▲ better, ▼ worse, CI excludes 0):", "",
              "| variant | " + " | ".join(COLS) + " |", "| --- " * (len(COLS) + 1) + "|"]
    for v in variants[1:]:
        lines.append(f"| {NAMES.get(v, v)} | " + " | ".join(d(V[v]["delta"][c], c) for c in COLS) + " |")
    pr = res["probe"]
    lines += ["", f"Music-awareness probe: shot cuts inside test clips with the film's own music (no splices), D103 audio labels; "
              f"{pr['n']} cuts, {pr['n_pos']} with a music change (base rate {pr['base_rate']:.3f} = AP of a random score).", "",
              "| variant | AP [95% CI] | Δ AP vs 4 | AUROC |", "| --- | --- | --- | --- |"]
    for v in ["tn_bassl", *variants]:
        m = pr[v]
        lines.append(f"| {NAMES.get(v, '3 BaSSL original head') } | {f(m['AP'])} | {d(m['delta_AP'], 'AP') if 'delta_AP' in m else '—'} | {m['AUROC']:.3f} |")
    lines += ["", "θ per fold: " + ", ".join(f"{k} {t:.2f}" for k, t in res["thetas"].items()), ""]
    Path("results/segmentation_improve.md").write_text("\n".join(lines))
    Path("results/segmentation_improve.json").write_text(json.dumps(res, indent=1))
    log.info("\n%s", "\n".join(lines))


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stage, variants = argv[0], argv[1:]
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    if stage == "train":
        from bgm.segmentation.train_head import run_train
        run_train(cfg, variants=tuple(variants))
    else:
        run_eval(cfg, variants)


if __name__ == "__main__":
    main(sys.argv[1:])


# ---------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8c8b86", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TITLES = {"AP": "AP", "F1 ±1 s": "F1 ±1 s", "F1 ±1 s in film": "F1 ±1 s, same film", "purity": "purity",
          "coverage": "coverage", "F (purity, coverage)": "F (pur., cov.)", "miou": "mIoU", "pk": "Pk", "windowdiff": "WindowDiff"}


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})


def figure(res: dict, chosen: str | None = None) -> Path:
    """results/segmentation_improve.png: per metric, each variant's difference from the D140 head, 95% CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    V = res["variants"]
    vs = [v for v in V if v != BASE][::-1]
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.4), facecolor=SURF, sharey=True)
    for ax, key in zip(axes.flat, COLS):
        ax.set_facecolor(SURF)
        ax.axvline(0, color=INK2, lw=1)
        for i, v in enumerate(vs):
            m = V[v]["delta"][key]
            c = BLUE if v == chosen else MUTED
            ax.plot(m["ci"], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.plot(m["mean"], i, "o", ms=7, color=c, mec=SURF, mew=1.5)
        lim = max(0.02, max(max(abs(V[v]["delta"][key]["ci"][0]), abs(V[v]["delta"][key]["ci"][1])) for v in vs) * 1.15)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        better = "↓ better" if key in LOWER else "↑ better"
        ax.set_title(f"{TITLES[key]}  (base {V[BASE][key]['mean']:.3f}; {better})", loc="left", fontsize=8.5, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(vs)), [NAMES.get(v, v) for v in vs])
        for t, v in zip(ax.get_yticklabels(), vs):
            t.set_color(BLUE if v == chosen else INK)
            t.set_fontweight("bold" if v == chosen else "normal")
    fig.suptitle(f"Difference from the D140 head (BaSSL + new head), test, 95% CI; base value in brackets", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.005, f"{res['n_seq']} test sequences, {res['n_clusters']} film clusters, paired cluster bootstrap (10,000). "
             "Variants chosen on val AP; test reported once.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = Path("results/segmentation_improve.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out


def metrics_explainer() -> Path:
    """reports/segmentation/metrics_explained.png: the metrics on a toy 60 s sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _style(plt)
    T, true, pred = 60.0, [20.0, 45.0], [8.0, 20.6, 46.8, 52.0]
    m = ev.segment_metrics(pred, true, T)
    tp, fp, fn = ev.match_counts(pred, true, 1.0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    cue_cols = ["#cfe0f6", "#fbd9c9", "#c9eedf"]
    for row, cuts, name in ((1.0, true, "true music cues"), (0.0, pred, "predicted segments")):
        b = [0.0] + cuts + [T]
        for j, (a, e) in enumerate(zip(b[:-1], b[1:])):
            col = cue_cols[j] if row == 1 else "#ecebe7"
            ax.add_patch(Rectangle((a, row - 0.32), e - a, 0.64, fc=col, ec=SURF, lw=2))
            ax.text((a + e) / 2, row, f"{'cue ' + 'ABC'[j] if row == 1 else f'seg {j + 1}'}\n{e - a:.1f} s", ha="center", va="center", fontsize=8, color=INK)
        ax.text(-1, row, name, ha="right", va="center", fontsize=9, color=INK)
    for t in true:
        ax.axvspan(t - 1, t + 1, ymin=0.05, ymax=0.95, color=AQUA, alpha=0.12, lw=0)
    for p in pred:
        hit = any(abs(p - t) <= 1 for t in true)
        ax.annotate("hit" if hit else "false cut", (p, -0.4), (p, -0.75), ha="center", fontsize=8, color=AQUA if hit else ORANGE,
                    arrowprops=dict(arrowstyle="-", color=AQUA if hit else ORANGE))
    for t in true:
        if not any(abs(p - t) <= 1 for p in pred):
            ax.annotate("missed", (t, 1.4), (t, 1.7), ha="center", fontsize=8, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_xlim(0, T)
    ax.set_ylim(-0.95, 1.85)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time (s); shaded band = ±1 s around each true change")
    txt = (f"boundary F1 ±1 s = {f1:.2f}  ({tp} hit{'s' if tp != 1 else ''}, {fp} false cuts, {fn} missed)     purity = {m['purity']:.2f}     "
           f"coverage = {m['coverage']:.2f}     mIoU = {m['miou']:.2f}     Pk = {m['pk']:.2f}     WindowDiff = {m['windowdiff']:.2f}")
    ax.set_title("How the metrics read one sequence (toy example)", loc="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, 0.01, txt, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path("reports/segmentation/metrics_explained.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out
