"""Dataset statistics table and figure for the cleaned 1,011-pair dataset (PROGRESS.md D115).

python -m bgm.analysis.dataset_stats
    -> results/dataset_stats.md, results/dataset_stats.tex
    -> outputs/dataset_figure/stats.{png,pdf}

Counts come from the manifests only (no media is read):
- cleaned_dataset/manifests/{pairs,tracks,segments}.csv and folds_title.json (film groups, test folds)
- dataset_1248/pairs.csv and dataset_1248/trimmed/edits.json for the before / after and trimming counts

Film counting. A *film key* is `meta_source:meta_film` (one upload of one film by one source). The title split groups
film keys that share a track_id or a title link into *film groups* (bgm.data.films.film_groups); a film group is the
unit the folds split on. `meta_film` alone is not a film id: YouTube and SunNXT name films by small integers, so the
same string means different films in different sources.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.analysis.retrieval_figure import style

log = logging.getLogger(__name__)
MAN = Path("cleaned_dataset/manifests")
ORIG = Path("dataset_1248/pairs.csv")
EDITS = Path("dataset_1248/trimmed/edits.json")
OUT_FIG = Path("outputs/dataset_figure")
OUT_TAB = Path("results")
SOURCES = {"sunnxt": "SunNXT", "full_bgm_dataset_youtube": "YouTube A", "full_bgm_dataset_youtube2": "YouTube B",
           "full_bgm_dataset": "Archive films"}
COLS = {"SunNXT": "#4C78A8", "YouTube A": "#F58518", "YouTube B": "#54A24B", "Archive films": "#B279A2"}
SHORT_S = 5.0
WHOLE_TOL_S = 0.1                      # an edit within 0.1 s of both ends counts as "kept whole" (as in D115)


def load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Cleaned pairs with film key, film group, source label, n_segments and is_test; the folds file; segments."""
    pairs = pd.read_csv(MAN / "pairs.csv")
    folds = json.loads((MAN / "folds_title.json").read_text())
    segs = pd.read_csv(MAN / "segments.csv")
    pairs["source"] = pairs.meta_source.map(SOURCES)
    pairs["film_key"] = pairs.meta_source.astype(str) + ":" + pairs.meta_film.astype(str)
    pairs["film_group"] = pairs.pair_id.map(folds["film_group"])
    pairs["n_segments"] = pairs.pair_id.map(segs.groupby("pair_id").size()).fillna(0).astype(int)
    test = {p for f in folds["folds"].values() for p in f["test"]}
    pairs["is_test"] = pairs.pair_id.isin(test)
    assert pairs.film_group.notna().all() and pairs.source.notna().all()
    return pairs, folds, segs


def summary_row(g: pd.DataFrame, segs: pd.DataFrame) -> dict:
    """One table row for a set of pairs: counts, hours, clip-length stats (s), segment stats."""
    d = g.video_duration_s
    sl = segs[segs.pair_id.isin(g.pair_id)]
    sl = sl.end_s - sl.start_s
    return {"pairs": len(g), "hours": d.sum() / 3600, "tracks": g.track_id.nunique(), "film_keys": g.film_key.nunique(),
            "film_groups": g.film_group.nunique(), "test_pairs": int(g.is_test.sum()),
            "dur_mean": d.mean(), "dur_median": d.median(), "dur_min": d.min(), "dur_max": d.max(),
            "short": int((d < SHORT_S).sum()), "segments": len(sl), "seg_median": sl.median()}


def by_source(pairs: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """Rows per source plus 'All'. Film-group counts per source can overlap (a group may span sources)."""
    rows = {name: summary_row(pairs[pairs.source == name], segs) for name in SOURCES.values()}
    rows["All"] = summary_row(pairs, segs)
    return pd.DataFrame(rows).T


def trimming(pairs: pd.DataFrame) -> dict:
    """Before / after counts of the hand edit, from edits.json: excluded, kept whole, trimmed to one part, split (2+ parts)."""
    orig = pd.read_csv(ORIG)
    edits = json.loads(EDITS.read_text())
    o_dur = orig.set_index("pair_id").video_duration_s
    excluded = whole = trimmed = split = split_parts = 0
    for pid, e in edits.items():
        r = e.get("ranges") or []
        if e["action"] == "exclude" or not r:
            excluded += 1
        elif len(r) > 1:
            split, split_parts = split + 1, split_parts + len(r)
        elif r[0][0] <= WHOLE_TOL_S and abs(r[0][1] - o_dur[pid]) <= WHOLE_TOL_S:
            whole += 1
        else:
            trimmed += 1
    assert excluded + whole + trimmed + split == len(orig) == len(edits)
    return {"orig_pairs": len(orig), "orig_hours": o_dur.sum() / 3600, "orig_tracks": orig.track_id.nunique(),
            "excluded": excluded, "kept_whole": whole, "trimmed": trimmed, "split": split,
            "split_parts": split_parts, "orig_durations": o_dur.to_numpy()}


def folds_table(pairs: pd.DataFrame, folds: dict) -> pd.DataFrame:
    """Per fold: train / val / test pair counts and test film groups."""
    fg = pairs.set_index("pair_id").film_group
    rows = {}
    for k, f in folds["folds"].items():
        rows[f"fold {k}"] = {"train": len(f["train"]), "val": len(f["val"]), "test": len(f["test"]),
                             "test_film_groups": fg[f["test"]].nunique()}
    return pd.DataFrame(rows).T


def to_markdown(src: pd.DataFrame, tr: dict, ft: pd.DataFrame, pairs: pd.DataFrame) -> str:
    cpg = pairs.groupby("film_group").size()
    L = ["# Dataset statistics (cleaned dataset, D115)", "",
         "Generated by `python -m bgm.analysis.dataset_stats` from the manifests in `cleaned_dataset/manifests/`. "
         "Figure: `outputs/dataset_figure/stats.png`.", "",
         "## Table 1: Pairs by source", "",
         "| Source | Pairs | Hours | Tracks | Film keys | Film groups | Test pairs | Clip s: mean / median / min / max | < 5 s | Segments (median s) |",
         "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for name, r in src.iterrows():
        b = "**" if name == "All" else ""
        L.append(f"| {b}{name}{b} | {r.pairs:,.0f} | {r.hours:.2f} | {r.tracks:,.0f} | {r.film_keys:.0f} | {r.film_groups:.0f} | "
                 f"{r.test_pairs:.0f} | {r.dur_mean:.1f} / {r.dur_median:.1f} / {r.dur_min:.2f} / {r.dur_max:.0f} | "
                 f"{r.short:.0f} | {r.segments:,.0f} ({r.seg_median:.1f}) |")
    L += ["",
          "- **Film key** = one upload (`meta_source:meta_film`). **Film group** = film keys merged when they share a "
          "track_id or a title link (`data/manifests/title_links.csv`); the folds split by film group. Per-source "
          "film-group counts add up to more than the total because some groups span sources.",
          "- Tracks = track_id groups (identical music under different file names share one).",
          "- Segments come from `segments.csv` (the segmenter's output used for training and retrieval).",
          "- Archive films include Konjum salangai (117 pairs, one film group), kept train-only in every fold.",
          "", "## Table 2: Hand edit (dataset_1248 → cleaned)", "",
          "| | Before | After |", "|---|---:|---:|",
          f"| Pairs | {tr['orig_pairs']:,} | {src.loc['All', 'pairs']:,.0f} |",
          f"| Hours | {tr['orig_hours']:.2f} | {src.loc['All', 'hours']:.2f} |",
          f"| Tracks | {tr['orig_tracks']:,} | {src.loc['All', 'tracks']:,.0f} |", "",
          f"Of the {tr['orig_pairs']:,} original pairs: **{tr['excluded']} excluded**, **{tr['kept_whole']} kept whole**, "
          f"**{tr['trimmed']} trimmed to one part**, **{tr['split']} split** into {tr['split_parts']} parts (2–5 each). "
          ""
          "Counted from the edits (`dataset_1248/trimmed/edits.json`, 'kept whole' = both ends within 0.1 s). The builder later "
          "dropped one part under 1 s (sunnxt_182_182_bgm_0000_p1, 0.84 s), so the parts give 1,011 pairs, not 1,012.",
          "", "## Table 3: Folds (title split, carried over from D102)", "",
          "| Fold | Train | Val | Test | Test film groups |", "|---|---:|---:|---:|---:|"]
    for name, r in ft.iterrows():
        L.append(f"| {name} | {r.train} | {r.val} | {r.test} | {r.test_film_groups} |")
    L.append(f"| total test | | | {ft.test.sum()} | {ft.test_film_groups.sum()} |")
    L += ["", "## Clips per film group", "",
          f"{len(cpg)} film groups: median {cpg.median():.0f} clips, mean {cpg.mean():.1f}, "
          f"{(cpg == 1).sum()} groups with a single clip, {(cpg >= 10).sum()} with 10 or more, largest "
          f"{cpg.max()} ({cpg.idxmax()}).", ""]
    return "\n".join(L)


def to_latex(src: pd.DataFrame) -> str:
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Source & Pairs & Hours & Tracks & Films & Test & Median s & $<$5\,s \\", r"\midrule"]
    for name, r in src.iterrows():
        if name == "All":
            L.append(r"\midrule")
        L.append(f"{name} & {r.pairs:,.0f} & {r.hours:.2f} & {r.tracks:,.0f} & {r.film_groups:.0f} & {r.test_pairs:.0f} & "
                 f"{r.dur_median:.1f} & {r.short:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return re.sub(r"(\d),(\d)", r"\1{,}\2", "\n".join(L))


def figure(pairs: pd.DataFrame, tr: dict, path: Path) -> None:
    """(a) clip-length histogram by source (log x), original 1,248 outlined; (b) clips per film group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.geomspace(0.8, 200, 34)
    stack = [pairs[pairs.source == s].video_duration_s.to_numpy() for s in SOURCES.values()]
    a.hist(stack, bins=bins, stacked=True, color=[COLS[s] for s in SOURCES.values()], label=list(SOURCES.values()),
           edgecolor="white", linewidth=0.4)
    a.hist(tr["orig_durations"], bins=bins, histtype="step", color="0.25", linewidth=1.1, linestyle="--",
           label=f"before edit ({tr['orig_pairs']:,})")
    a.axvline(SHORT_S, color="0.5", linewidth=0.8, linestyle=":")
    a.text(SHORT_S * 0.93, a.get_ylim()[1] * 0.95, "5 s", ha="right", va="top", fontsize=8, color="0.4")
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20, 50, 100]), a.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    d = pairs.video_duration_s
    a.set_xlabel(f"Clip length (s, log scale); median {d.median():.1f} s, total {d.sum() / 3600:.2f} h")
    a.set_ylabel("Pairs")
    a.set_title(f"(a) Clip length, {len(pairs):,} pairs", fontsize=10)
    a.legend(fontsize=7.5, frameon=False)

    cpg = pairs.groupby("film_group").size()
    top = pairs.groupby("film_group").source.agg(lambda s: s.value_counts().idxmax())
    edges = np.array([1, 2, 3, 4, 5, 6, 8, 11, 16, 21, 31, 51, 200])
    labels = ["1", "2", "3", "4", "5", "6–7", "8–10", "11–15", "16–20", "21–30", "31–50", ">50"]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for s in SOURCES.values():
        c = cpg[top == s]
        h = np.histogram(c, bins=edges)[0]
        b.bar(x, h, bottom=bottom, color=COLS[s], edgecolor="white", linewidth=0.4)
        bottom += h
    for xi, v in zip(x, bottom):
        if v:
            b.text(xi, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    b.set_xticks(x), b.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    b.set_xlabel(f"Clips per film group (median {cpg.median():.0f}; colour = main source)")
    b.set_ylabel("Film groups")
    b.set_title(f"(b) Clips per film, {len(cpg)} film groups", fontsize=10)
    b.annotate(f"Konjum salangai\n{cpg.max()} clips (train-only)", xy=(x[-1], bottom[-1]), xytext=(x[-1] - 2.6, max(bottom) * 0.55),
               fontsize=7, arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.6))
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pairs, folds, segs = load()
    src = by_source(pairs, segs)
    tr = trimming(pairs)
    ft = folds_table(pairs, folds)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    (OUT_TAB / "dataset_stats.md").write_text(to_markdown(src, tr, ft, pairs))
    (OUT_TAB / "dataset_stats.tex").write_text(to_latex(src))
    pairs.drop(columns=["video_path", "music_path"]).to_csv(OUT_FIG / "stats_pairs.csv", index=False)
    figure(pairs, tr, OUT_FIG / "stats")
    log.info("wrote results/dataset_stats.{md,tex}, outputs/dataset_figure/stats.{png,pdf}")


if __name__ == "__main__":
    main()
