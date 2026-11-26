"""All test results of the transition (cue-switch) models in one folder: outputs/test_results/ (PROGRESS.md D97).

Run: python -m bgm.analysis.test_results        (re-run after the demo renders finish to add their chart and videos)

Sources (nothing is re-evaluated): the saved evaluations runs/full/cue_splice/eval/ (4,246 pairs, D95) and
runs/filmsplit/cue_splice/eval/ (1,248 pairs, D94), scored with bgm.cue.report's own Table / boot_f1, so every number
equals results/cue_switch_full_splice.md and results/cue_switch_splice.md; the probe sections are copied from those
reports; the rendered test films from outputs/cue_splice_demo/ (D96-D97), hard-linked (no copies).
Metric: boundary F1 at +-1 s between the music switches Viterbi places inside a film and the true music changes,
pooled over sequences (seed-averaged counts), 95% cluster-bootstrap CIs over film groups (10,000 resamples), paired.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from bgm.config import load_config
from bgm.cue.report import LABEL, MODES, Table, boot_f1, f1_of, load_evals

log = logging.getLogger(__name__)
OUT = Path("outputs/test_results")
DATASETS = {
    "full": ("Full dataset, 4,246 pairs", ["paths.manifests=data/manifests_full", "splits.folds_file=folds_film.json", "paths.runs=runs/full",
                                            "cue.exp=cue_splice"], "results/cue_switch_full_splice.md"),
    "1248": ("1,248-pair dataset", ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", "cue.exp=cue_splice"],
             "results/cue_switch_splice.md"),
}
SHORT = {"full": "Full data (4,246)", "1248": "1,248 pairs"}
SETS = {"blocked": "mixed films (blocked)", "same_film": "one film, random clip order", "timeline": "one film, real film order"}
SHOW = ("R0_tuned", "R1", "R2", "R3", "R4", "R3s", "R4s")           # the systems in the tables and charts
NAME = {"R0_tuned": "Fixed rules (current default)", "R1": "Scene-cut rule", "R2": "Cosine rule", "R3": "Learned R3",
        "R4": "Learned R4", "R3s": "R3s, + splices (best)", "R4s": "R4s, + splices", "oracle": "Perfect predictor"}
BEST, DEFAULT = "R3s", "R0_tuned"
INK, INK2, MUTED, GRID, SURF, HI, GREY = "#0b0b0b", "#52514e", "#c3c2b7", "#e8e7e3", "#fcfcfb", "#2a78d6", "#8a8983"


def tables(key: str) -> dict:
    title, over, report = DATASETS[key]
    T = Table(load_evals(load_config("configs/default.yaml", over)))
    res = {"title": title, "report": report, "systems": {}, "folds": {}, "seqs": []}
    for md in MODES:
        cl = T.clusters(md)
        base = T.counts(DEFAULT, md, "in_film/1")
        for s in SHOW + ("oracle",):
            c = T.counts(s, md, "in_film/1")
            tp, fp, fn = c.sum(0)
            res["systems"][s, md] = {"f1": boot_f1(c, cl), "delta": None if s == DEFAULT else boot_f1(c, cl, base),
                                     "precision": tp / max(tp + fp, 1e-9), "recall": tp / max(tp + fn, 1e-9),
                                     "switches_per_min": float(np.nanmean(T.per_case(s, md, "switches_per_min")))}
        res["systems"]["R3s-R3", md] = {"delta": boot_f1(T.counts("R3s", md, "in_film/1"), cl, T.counts("R3", md, "in_film/1"))}
        counts = {s: T.counts(s, md, "in_film/1") for s in SHOW}
        for i, (fold, idx, clus, case) in enumerate(T.cases[md]):
            row = {"test_set": md, "fold": fold, "idx": idx, "cluster": clus, "films": "|".join(sorted(set(case["films"]))),
                   "duration_s": round(case["duration"], 1), "true_changes_in_film": int(round(counts[DEFAULT][i, 0] + counts[DEFAULT][i, 2]))}
            for s in SHOW:
                tp, fp, fn = counts[s][i]
                row[f"{s}_tp"], row[f"{s}_fp"], row[f"{s}_fn"] = round(tp, 3), round(fp, 3), round(fn, 3)
                row[f"{s}_f1"] = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else ""
            res["seqs"].append(row)
    for e in T.evals:
        Tf = Table([e])
        res["folds"][e["fold"]] = {s: f1_of(Tf.counts(s, "blocked", "in_film/1")) for s in SHOW}
    return res


def probe_md(report: str) -> str:
    text = Path(report).read_text()
    i = text.find("## Clip joins or music changes?")
    return text[i:] if i >= 0 else ""


def probe_deltas(report: str) -> dict:
    """The D94 paired probe rows: {'R4s − R4': [(mean, lo, hi) x 3], ...}."""
    out = {}
    for line in Path(report).read_text().splitlines():
        m = re.match(r"\| (R[34]s − R[34]) \|(.*)\|$", line)
        if m:
            out[m.group(1)] = [tuple(float(x) for x in re.findall(r"[-+]?\d+\.\d+", cell)) for cell in m.group(2).split("|")]
    return out


# ---------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURF)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def chart_systems(R: dict, path: Path) -> None:
    """Small multiples (one per test set): pooled F1 with 95% CI per system; the best system highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, facecolor=SURF)
    ys = np.arange(len(SHOW))[::-1]
    for ax, md in zip(axes, MODES):
        _style(ax)
        for y, s in zip(ys, SHOW):
            d = R["systems"][s, md]["f1"]
            col = HI if s == BEST else GREY
            ax.plot([d["lo"], d["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.scatter([d["mean"]], [y], s=46, color=col, zorder=3, edgecolors=SURF, linewidths=1.5)
            ax.text(d["hi"] + 0.006, y, f"{d['mean']:.3f}", va="center", fontsize=8, color=INK if s == BEST else INK2)
        dflt = R["systems"][DEFAULT, md]["f1"]["mean"]
        ax.axvline(dflt, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
        ax.set_title(SETS[md], loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("boundary F1 ±1 s, music changes inside a film", fontsize=8.5, color=INK2)
        lo = min(R["systems"][s, md]["f1"]["lo"] for s in SHOW)
        hi = max(R["systems"][s, md]["f1"]["hi"] for s in SHOW)
        ax.set_xlim(lo - 0.01, hi + 0.04)
    axes[0].set_yticks(ys, [NAME[s] for s in SHOW], fontsize=8.5, color=INK)
    fig.suptitle(f"{R['title']}: F1 per system with 95% CI (dashed = current default)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.17, wspace=0.12)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def chart_deltas(RS: dict, path: Path) -> None:
    """Forest plot: paired F1 gain over the current default, per dataset x test set, for the learned systems."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(k, md) for k in RS for md in MODES]
    systems = ("R3s", "R4s", "R3", "R4")
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) * len(systems) / 2.2 + 1.2), facecolor=SURF)
    _style(ax)
    y, ticks, labels = 0.0, [], []
    for k, md in rows[::-1]:
        for s in systems[::-1]:
            d = RS[k]["systems"][s, md]["delta"]
            col = HI if s == BEST else GREY
            ax.plot([d["lo"], d["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.scatter([d["mean"]], [y], s=40 if s == BEST else 26, color=col, zorder=3, edgecolors=SURF, linewidths=1.2)
            sig = d["lo"] > 0
            ax.text(d["hi"] + 0.003, y, f"{d['mean']:+.3f}{' ✓' if sig else ''}", va="center", fontsize=7.5, color=INK if s == BEST else INK2)
            ticks.append(y)
            labels.append(f"{NAME[s]}")
            y += 1
        ax.text(-0.52, y - 0.35, f"{SHORT[k]} · {SETS[md]}", fontsize=8.5, color=INK, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        y += 1.1
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ticks, labels, fontsize=7.5, color=INK2)
    ax.set_xlabel("F1 gain over the current default (fixed rules), 95% CI · ✓ = CI above 0", fontsize=8.5, color=INK2)
    ax.set_title("Learned transition models vs the current default, every test set", loc="left", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.36, right=0.93, top=0.95, bottom=0.06)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def chart_folds(RS: dict, path: Path) -> None:
    """Dumbbells: per fold, the default vs R3s (blocked test sequences)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(RS), figsize=(11, 2.9), facecolor=SURF)
    for ax, (k, R) in zip(axes, RS.items()):
        _style(ax)
        folds = sorted(R["folds"])
        for y, f in enumerate(folds[::-1]):
            a, b = R["folds"][f][DEFAULT], R["folds"][f][BEST]
            ax.plot([a, b], [y, y], color=MUTED, lw=2, zorder=1)
            ax.scatter([a], [y], s=44, color=GREY, zorder=3, edgecolors=SURF, linewidths=1.2, label="Fixed rules (default)" if y == 0 else None)
            ax.scatter([b], [y], s=50, color=HI, zorder=3, edgecolors=SURF, linewidths=1.2, label="R3s (best)" if y == 0 else None)
            ax.text(max(a, b) + 0.006, y, f"{b - a:+.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(folds)), [f"fold {f}" for f in folds[::-1]], fontsize=8.5, color=INK)
        ax.set_title(f"{R['title']} · mixed films", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("boundary F1 ±1 s", fontsize=8.5, color=INK2)
        xs = [v for f in folds for v in (R["folds"][f][DEFAULT], R["folds"][f][BEST])]
        ax.set_xlim(min(xs) - 0.02, max(xs) + 0.05)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.18, wspace=0.25)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def chart_sequences(R: dict, path: Path) -> None:
    """Histogram of the per-sequence F1 change (R3s − default), blocked test sequences with a true change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.array([r[f"{BEST}_f1"] - r[f"{DEFAULT}_f1"] for r in R["seqs"]
                  if r["test_set"] == "blocked" and r[f"{BEST}_f1"] != "" and r[f"{DEFAULT}_f1"] != ""], float)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=SURF)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    bins = np.linspace(-0.6, 0.6, 25)
    ax.hist(np.clip(d, -0.6, 0.6), bins=bins, color=HI, rwidth=0.88)
    ax.axvline(0, color=INK2, lw=1)
    up, eq, dn = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
    ax.set_title(f"{R['title']} · each test sequence (mixed films, n = {len(d)})\nR3s better on {up}, equal on {eq}, worse on {dn}",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("F1 change of one sequence, R3s − fixed rules (clipped at ±0.6)", fontsize=8.5, color=INK2)
    ax.set_ylabel("sequences", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.18)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def chart_probes(P: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tests = ("length held fixed (test 3)", "change vs same-music join (test 2)")
    rows = [(k, comp, t, P[k][comp][i]) for k in P for comp in ("R4s − R4", "R3s − R3") if comp in P[k] for i, t in enumerate(tests)]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.1), facecolor=SURF)
    _style(ax)
    ticks, labels = [], []
    for y, (k, comp, t, (m, lo, hi)) in enumerate(rows[::-1]):
        col = HI if lo > 0 else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.scatter([m], [y], s=40, color=col, zorder=3, edgecolors=SURF, linewidths=1.2)
        ax.text(hi + 0.003, y, f"{m:+.3f}{' ✓' if lo > 0 else ''}", va="center", fontsize=8, color=INK if lo > 0 else INK2)
        ticks.append(y)
        labels.append(f"{SHORT[k]} · {comp} · {t}")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ticks, labels, fontsize=8, color=INK)
    ax.set_xlabel("AUROC change from the splices, 95% CI · blue / ✓ = CI above 0", fontsize=8.5, color=INK2)
    xs = [v for _, _, _, (m, lo, hi) in rows for v in (lo, hi)]
    ax.set_xlim(min(xs) - 0.01, max(xs) + 0.03)
    fig.suptitle("Did the splices teach more than clip length? Probe tests (AUROC; 0.5 = guessing)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.47, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def chart_demo(rows: list[dict], path: Path) -> None:
    """Rendered held-out films: per-film F1, D75 vs R3s, sorted by the change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["scores"]["cue"]["f1_1"] - r["scores"]["d75"]["f1_1"])
    if len(rows) > 40:                                   # too many for one dumbbell row each: the spread of the change
        d = np.array([r["scores"]["cue"]["f1_1"] - r["scores"]["d75"]["f1_1"] for r in rows])
        tot = {k: np.sum([r["scores"][k]["tp_fp_fn_1"] for r in rows], 0) for k in ("d75", "cue")}
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURF)
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.grid(axis="x", visible=False)
        ax.hist(np.clip(d, -0.8, 0.8), bins=np.linspace(-0.8, 0.8, 17), color=HI, rwidth=0.88)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{len(rows)} rendered held-out test videos: R3s better on {(d > 1e-9).sum()}, equal on {(np.abs(d) <= 1e-9).sum()}, "
                     f"worse on {(d < -1e-9).sum()}\ntrue changes found: D75 {tot['d75'][0]:.0f}, R3s {tot['cue'][0]:.0f}; false switches: "
                     f"D75 {tot['d75'][1]:.0f}, R3s {tot['cue'][1]:.0f}", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("F1 change of one video, R3s − D75", fontsize=8.5, color=INK2)
        ax.set_ylabel("videos", fontsize=8.5, color=INK2)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.17)
        fig.savefig(path, dpi=150, facecolor=SURF)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8, 0.16 * len(rows) + 1.4), facecolor=SURF)
    _style(ax)
    for y, r in enumerate(rows):
        a, b = r["scores"]["d75"]["f1_1"], r["scores"]["cue"]["f1_1"]
        ax.plot([a, b], [y, y], color=MUTED, lw=1.5, zorder=1)
        ax.scatter([a], [y], s=16, color=GREY, zorder=3, label="D75 (fixed rules)" if y == 0 else None)
        ax.scatter([b], [y], s=18, color=HI, zorder=3, label="R3s" if y == 0 else None)
    ax.set_yticks([])
    ax.set_xlabel("F1 ±1 s against the true music changes of the rendered video", fontsize=8.5, color=INK2)
    d = np.array([r["scores"]["cue"]["f1_1"] - r["scores"]["d75"]["f1_1"] for r in rows])
    ax.set_title(f"{len(rows)} rendered held-out test videos: R3s better on {(d > 0).sum()}, equal on {(d == 0).sum()}, "
                 f"worse on {(d < 0).sum()}", loc="left", fontsize=9.5, color=INK, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.subplots_adjust(left=0.04, right=0.98, top=1 - 0.75 / (0.16 * len(rows) + 1.4), bottom=0.08)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- folder
def link_videos(rows: list[dict]) -> int:
    """Hard links (no extra disk) of the rendered test videos and their timelines into OUT/videos/."""
    src = Path("outputs/cue_splice_demo")
    n = 0
    for r in rows:
        for sub, ext in (("d75", ".mp4"), ("r3s", ".mp4"), ("original", ".mp4"), ("timelines", ".png")):
            a, b = src / sub / f"{r['name']}{ext}", OUT / "videos" / sub / f"{r['name']}{ext}"
            if a.exists() and not b.exists():
                b.parent.mkdir(parents=True, exist_ok=True)
                os.link(a, b)
                n += 1
    return n


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for sub in ("tables", "charts", "videos"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    RS = {k: tables(k) for k in DATASETS}
    fmt = lambda d, sg=False: f"{d['mean']:{'+' if sg else ''}.3f} [{d['lo']:{'+' if sg else ''}.3f}, {d['hi']:{'+' if sg else ''}.3f}]"  # noqa: E731
    L = ["# Test results: learned transition models (cue switch)", "",
         "Boundary F1 ±1 s: the music switches placed inside a film vs the true music changes, pooled over the frozen test sequences "
         "(5 folds × 3 seeds, seed-averaged counts). [95% CI]: cluster bootstrap over film groups, 10,000 resamples; Δ: paired, "
         "same sequences and lattices. Same numbers as the full reports named per section.", ""]
    for k, R in RS.items():
        L += [f"## {R['title']} (full report: {R['report']})", ""]
        for md in MODES:
            n = sum(1 for r in R["seqs"] if r["test_set"] == md)
            L += [f"### Test set: {SETS[md]} ({n} sequences)", "", "| system | F1 [95% CI] | Δ vs current default [95% CI] | precision | recall | switches / min |",
                  "| --- | --- | --- | --- | --- | --- |"]
            for s in SHOW + ("oracle",):
                v = R["systems"][s, md]
                dl = "—" if v["delta"] is None else fmt(v["delta"], True) + (" ✓" if v["delta"]["lo"] > 0 else "")
                nm = f"**{NAME[s]}**" if s == BEST else NAME[s]
                L.append(f"| {nm} | {fmt(v['f1'])} | {dl} | {v['precision']:.3f} | {v['recall']:.3f} | {v['switches_per_min']:.2f} |")
            L += ["", f"R3s − R3 (the splices' own effect): {fmt(R['systems']['R3s-R3', md]['delta'], True)}", ""]
        L += [f"Per fold (mixed films): " + "; ".join(f"fold {f}: default {v[DEFAULT]:.3f} → R3s {v[BEST]:.3f}" for f, v in sorted(R["folds"].items())), ""]
        write_csv(OUT / "tables" / f"per_sequence_{k}.csv", R["seqs"])
        write_csv(OUT / "tables" / f"systems_{k}.csv", [
            {"dataset": k, "test_set": md, "system": s, "name": NAME[s], "f1": round(v["f1"]["mean"], 4), "f1_lo": round(v["f1"]["lo"], 4),
             "f1_hi": round(v["f1"]["hi"], 4), "delta_vs_default": "" if v["delta"] is None else round(v["delta"]["mean"], 4),
             "delta_lo": "" if v["delta"] is None else round(v["delta"]["lo"], 4), "delta_hi": "" if v["delta"] is None else round(v["delta"]["hi"], 4),
             "precision": round(v["precision"], 4), "recall": round(v["recall"], 4), "switches_per_min": round(v["switches_per_min"], 3)}
            for md in MODES for s in SHOW + ("oracle",) for v in [R["systems"][s, md]]])
        write_csv(OUT / "tables" / f"per_fold_{k}.csv", [{"fold": f, **{s: round(x, 4) for s, x in v.items()}} for f, v in sorted(R["folds"].items())])
        chart_systems(R, OUT / "charts" / f"f1_by_system_{k}.png")
        chart_sequences(R, OUT / "charts" / f"per_sequence_change_{k}.png")
    (OUT / "tables" / "summary.md").write_text("\n".join(L) + "\n")
    probes = {k: probe_deltas(DATASETS[k][2]) for k in DATASETS}
    (OUT / "tables" / "probes.md").write_text("\n\n".join(f"# {DATASETS[k][0]}\n\n{probe_md(DATASETS[k][2])}" for k in DATASETS))
    chart_deltas(RS, OUT / "charts" / "gain_over_default_all_test_sets.png")
    chart_folds(RS, OUT / "charts" / "per_fold.png")
    chart_probes(probes, OUT / "charts" / "probes.png")
    demo = Path("outputs/cue_splice_demo/_work/demo.json")
    rows = json.loads(demo.read_text())["rows"] if demo.exists() else []
    n_links = 0
    if rows:
        chart_demo(rows, OUT / "charts" / "rendered_test_videos.png")
        write_csv(OUT / "tables" / "rendered_test_videos.csv", [
            {"video": r["name"], "fold": r["fold"], "test_set": r["mode"], "film": r["film"], "duration_s": round(r["duration_s"], 1),
             "true_changes": len(r["true_changes"]), "d75_f1": round(r["scores"]["d75"]["f1_1"], 3), "r3s_f1": round(r["scores"]["cue"]["f1_1"], 3),
             "d75_switches_per_min": round(r["scores"]["d75"]["switches_per_min"], 2), "r3s_switches_per_min": round(r["scores"]["cue"]["switches_per_min"], 2)}
            for r in rows])
        n_links = link_videos(rows)
    write_readme(RS, rows)
    log.info("test results -> %s (%d video / timeline links added)", OUT, n_links)
    return OUT


def write_readme(RS: dict, rows: list[dict]) -> None:
    f, s = RS["full"]["systems"], RS["1248"]["systems"]
    L = ["# Test results: learned transition models", "",
         "Everything about how well the system decides **when the music should change**, in one place (PROGRESS.md D93–D97).", "",
         "## Headline", "",
         f"- Full dataset (4,246 pairs), mixed-film test (523 sequences, 1,430 min): **R3s {f['R3s', 'blocked']['f1']['mean']:.3f}** vs current default "
         f"{f['R0_tuned', 'blocked']['f1']['mean']:.3f}, gain **{f['R3s', 'blocked']['delta']['mean']:+.3f} [{f['R3s', 'blocked']['delta']['lo']:+.3f}, "
         f"{f['R3s', 'blocked']['delta']['hi']:+.3f}]**.",
         f"- Same film, random order: {f['R3s', 'same_film']['delta']['mean']:+.3f} [{f['R3s', 'same_film']['delta']['lo']:+.3f}, "
         f"{f['R3s', 'same_film']['delta']['hi']:+.3f}]; real film order: {f['R3s', 'timeline']['delta']['mean']:+.3f} "
         f"[{f['R3s', 'timeline']['delta']['lo']:+.3f}, {f['R3s', 'timeline']['delta']['hi']:+.3f}].",
         f"- 1,248-pair dataset, mixed films: {s['R3s', 'blocked']['delta']['mean']:+.3f} [{s['R3s', 'blocked']['delta']['lo']:+.3f}, "
         f"{s['R3s', 'blocked']['delta']['hi']:+.3f}].",
         f"- A perfect predictor would reach {f['oracle', 'blocked']['f1']['mean']:.3f}: the gain is real but small.", "",
         "## Folders", "",
         "| path | what |", "| --- | --- |",
         "| `tables/summary.md` | every system on every test set, both datasets: F1 [CI], gain over the default [CI], precision, recall, switches / min |",
         "| `tables/systems_<dataset>.csv` | the same as CSV |",
         "| `tables/per_sequence_<dataset>.csv` | every test sequence: films, length, true changes, hits / false switches / misses and F1 per system |",
         "| `tables/per_fold_<dataset>.csv` | F1 per fold (mixed films) |",
         "| `tables/probes.md` | the probe tests: did the model learn music changes or only clip length? |",
         "| `tables/rendered_test_videos.csv` | the rendered held-out test videos and their scores |",
         "| `charts/f1_by_system_<dataset>.png` | F1 with 95% CI per system, one panel per test set |",
         "| `charts/gain_over_default_all_test_sets.png` | gain over the current default for every learned model and test set |",
         "| `charts/per_fold.png` | default vs R3s in each fold |",
         "| `charts/per_sequence_change_<dataset>.png` | how the gain spreads over individual test sequences |",
         "| `charts/probes.png` | the probe-test gains from the same-music splices |",
         "| `charts/rendered_test_videos.png` | per rendered video, D75 vs R3s |",
         "| `videos/d75/`, `videos/r3s/`, `videos/original/` | held-out test films (1080p): old system, new model, the film's real music |",
         "| `videos/timelines/` | per video: true music changes vs where each system switched |", "",
         "## Systems", "",
         "- **Fixed rules (current default)**: Viterbi with fixed switch / jump penalties tuned on val (D75).",
         "- **Scene-cut rule / Cosine rule**: simple baselines (switch at scene cuts / at large visual changes).",
         "- **Learned R3 / R4**: the transition model predicts a change probability at each segment boundary (R3: the two neighbouring "
         "segments; R4: plus context) (D78–D81).",
         "- **R3s / R4s**: the same, trained with same-music splices so a clip join is not taken for a music change (D94–D95). "
         "R3s has the best validation score on both datasets, so it is the chosen model.", "",
         "## How it was tested", "",
         "Film-level 5-fold split (no film in both train and test); frozen test sequences; every setting chosen on validation films; "
         "5 folds × 3 seeds; 95% cluster-bootstrap CIs over film groups; paired comparisons on identical lattices; the saved "
         "evaluations reproduce the earlier runs exactly. Rendered videos: each from its own fold's models (leak-free)."]
    if rows:
        tot = {k: np.sum([r["scores"][k]["tp_fp_fn_1"] for r in rows], 0) for k in ("d75", "cue")}
        f1 = {k: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1e-9) for k, v in tot.items()}
        L += ["", f"Rendered test videos so far: {len(rows)}; pooled F1 D75 {f1['d75']:.3f} vs R3s {f1['cue']:.3f} "
              f"(hits {tot['d75'][0]:.0f} vs {tot['cue'][0]:.0f}). These are for listening; the test sets above are the evidence."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    print(main())
