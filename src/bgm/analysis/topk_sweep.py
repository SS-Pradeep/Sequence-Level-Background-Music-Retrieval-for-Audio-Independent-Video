"""Shortlist-size sweep of the final system, cleaned title split (PROGRESS.md D135).

python -m bgm.analysis.topk_sweep  -> results/cleaned/topk_sweep.md, reports/cleaned/topk/heatmap.png

Inputs (reports/topk_chain.sh, reports/topk_k20_chain.sh): fuse + tune-stitch of the 10-head ensembles for
K = retrieval.coarse_top_k in {1, 5, 10, 15, 20} (the shortlist is the union of the whole video's top K files and each
segment's top K, pipeline.coarse_shortlist) and J = retrieval.fine_top_j (windows per segment kept for Viterbi) in
{5, 10, 15, 20}; 5 folds x run seeds 0-2. K10 J20 is the shipped setting (runs/cleaned/pipeline_ens10).
Whole-clip recall@K comes from the saved ensemble track ranks (test-fold gallery): a lower bound on how often the
union shortlist holds the true song.
Tables: mean ± std over the 5 fold means (seeds averaged first). Δ vs K10 J20: per query averaged over seeds, 95%
cluster bootstrap (film groups for clips / segments, sequences for concatenations), 10,000 resamples. The setting is
chosen on the validation stitching objective (part track accuracy + switch F1 on val concatenations); test is
reported only. Development numbers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.config import load_config
from bgm.data.manifest import read_film_groups
from bgm.eval.metrics import cluster_bootstrap

log = logging.getLogger(__name__)

RUNS = Path("runs/cleaned")
ENS = RUNS / "semantic_ens" / "siglip-both-ens10"
REP = Path("reports/cleaned/topk")
OUT_MD = Path("results/cleaned/topk_sweep.md")
SPLIT = ["splits.folds_file=folds_title.json", "paths.manifests=cleaned_dataset/manifests"]
FOLDS, SEEDS = range(5), range(3)
KS, JS = (1, 5, 10, 15, 20, 30, 50), (5, 10, 15, 20)
BASE = (10, 20)
RECALL_KS = (1, 5, 10, 15, 20, 30, 50)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential blue 100 -> 700


def exp_dir(K: int, J: int) -> Path:
    return RUNS / "pipeline_ens10" if (K, J) == BASE else RUNS / "topk" / f"K{K}_J{J}"


def complete(K: int, J: int) -> bool:
    return all((exp_dir(K, J) / f"fold{k}" / f"seed{s}" / "stitch_metrics.json").exists() for k in FOLDS for s in SEEDS)


def fold_stats(vals: dict[int, list[float]]) -> tuple[float, float]:
    """{fold: [per-seed values]} -> (mean, std over the fold means)."""
    F = np.array([np.mean(vals[k]) for k in FOLDS])
    return float(F.mean()), float(F.std(ddof=1))


def summarize(K: int, J: int) -> dict[str, tuple[float, float]]:
    cols = {}
    for k in FOLDS:
        for s in SEEDS:
            d = exp_dir(K, J) / f"fold{k}" / f"seed{s}"
            m, st = json.loads((d / "metrics.json").read_text()), json.loads((d / "stitch_metrics.json").read_text())
            t, ft = st["test"], m["test"]["fused"]
            row = {"val_obj": st["val_objective"], "switch_f1": t["concat/viterbi/switch_f1"],
                   "part_acc": t["concat/viterbi/part_track_acc"], "sw_min": t["concat/viterbi/switches_per_min"],
                   "nat_track": t["natural/viterbi/track"], "track@1": ft["track@1"], "window@1": ft["window@1"],
                   "window@J": ft["window@J"], "fuse_s": m["seconds"]}
            for c, v in row.items():
                cols.setdefault(c, {}).setdefault(k, []).append(v)
    return {c: fold_stats(v) for c, v in cols.items()}


def shortlist_recall() -> dict[int, tuple[float, float]]:
    """Share of test clips whose true track group is in the coarse top K (test-fold gallery), per K."""
    out = {}
    for K in RECALL_KS:
        out[K] = fold_stats({k: [float(np.mean(np.load(ENS / f"fold{k}" / f"seed{s}" / "perquery.npz")["track_test_fold"] <= K))
                                 for s in SEEDS] for k in FOLDS})
    return out


def all_files(film: dict, base_w1: np.ndarray) -> dict:
    """No coarse stage (every gallery file's windows; fuse's fused_no_coarse on the shipped runs): track@1, window@1
    as fold stats, and Δ window@1 vs the shipped shortlist."""
    tr, w1, per = {}, {}, []
    for k in FOLDS:
        for s in SEEDS:
            d = exp_dir(*BASE) / f"fold{k}" / f"seed{s}"
            t = json.loads((d / "metrics.json").read_text())["test"]["fused_no_coarse"]
            tr.setdefault(k, []).append(t["track@1"]), w1.setdefault(k, []).append(t["window@1"])
        per.append(np.mean([np.load(exp_dir(*BASE) / f"fold{k}" / f"seed{s}" / "perquery.npz")["fused_no_coarse/window@1"]
                            for s in SEEDS], 0))
    return {"track@1": fold_stats(tr), "window@1": fold_stats(w1), "per_w1": np.concatenate(per) - base_w1}


def per_query(K: int, J: int, film: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-query values averaged over seeds, with their clusters, all folds concatenated."""
    out = {}
    for name, fname, key, ckey in (("switch_f1", "stitch_perquery.npz", "concat/viterbi/switch_f1", "concat/clusters"),
                                   ("nat_track", "stitch_perquery.npz", "natural/viterbi/track", "natural/pair"),
                                   ("window@1", "perquery.npz", "fused/window@1", "pair")):
        v, g = [], []
        for k in FOLDS:
            Z = [np.load(exp_dir(K, J) / f"fold{k}" / f"seed{s}" / fname) for s in SEEDS]
            v.append(np.mean([z[key] for z in Z], 0))
            c = Z[0][ckey]
            g.extend(film[p] for p in c) if ckey != "concat/clusters" else g.extend(c)
        out[name] = (np.concatenate(v), np.asarray(g))
    return out


def plot_heatmap(S: dict, cells: list[tuple[int, int]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    panels = (("val_obj", "Validation objective (selects the setting)"), ("switch_f1", "Test switch F1 (Viterbi)"),
              ("window@1", "Test window@1"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, (c, title) in zip(axes, panels):
        M = np.full((len(KS), len(JS)), np.nan)
        for K, J in cells:
            M[KS.index(K), JS.index(J)] = S[(K, J)][c][0]
        lo, hi = np.nanmin(M), np.nanmax(M)
        ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(KS)):
            for j in range(len(JS)):
                if np.isfinite(M[i, j]):
                    dark = (M[i, j] - lo) / max(hi - lo, 1e-12) > 0.55
                    bold = (KS[i], JS[j]) == BASE
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                            color="white" if dark else "#1f1e1b", fontweight="bold" if bold else "normal")
        ax.set_xticks(range(len(JS)), [str(j) for j in JS]), ax.set_yticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlabel("J = windows kept per segment"), ax.set_ylabel("K = files shortlisted")
        ax.set_title(title, loc="left", fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Shortlist sizes, cleaned title split, 5 folds × 3 ensembles (bold = shipped K10 J20)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(x: tuple[float, float]) -> str:
    return f"{x[0]:.3f} ± {x[1]:.3f}"


def report() -> None:
    cfg = load_config(overrides=SPLIT)
    film = read_film_groups(cfg)
    cells = [(K, J) for K in KS for J in JS if complete(K, J)]
    assert BASE in cells, "the shipped K10 J20 runs are missing"
    S = {c: summarize(*c) for c in cells}
    REC = shortlist_recall()
    base_q = per_query(*BASE, film)
    L = ["# Shortlist-size sweep of the final system (D135)", "",
         "Cleaned title split, 10-head SigLIP → CLAP+MERT ensembles, fuse + tune-stitch, 5 folds × run seeds 0–2 per "
         "cell. K = files the coarse stage keeps for the whole video and for each segment, shortlist = their union "
         "(shipped 10); J = best-synced windows per segment passed to "
         "Viterbi (shipped 20). Inference only: the heads are the saved ones. Mean ± std over the 5 fold means. "
         f"Cells run: {len(cells)} of {len(KS) * len(JS)}. Development numbers.", "",
         "## Whole-clip recall: is the true song in the whole video's top K?", "",
         "Coarse stage alone, test-fold gallery, whole clip as the query. The pipeline's shortlist also adds each "
         "segment's top K, so this is a lower bound on how often the true song is reachable.", "",
         "| K | " + " | ".join(f"R@{k}" for k in RECALL_KS) + " |", "| --- |" + " --- |" * len(RECALL_KS),
         "| recall | " + " | ".join(fmt(REC[k]) for k in RECALL_KS) + " |", "",
         "## Pipeline and stitching, every K × J", "",
         "| K | J | val objective | test switch F1 | Δ switch F1 [95% CI] | test part track acc | switches / min | "
         "natural track acc | Δ natural track acc [95% CI] | window@1 | Δ window@1 [95% CI] | window@J | fuse s / run |",
         "| --- |" * 1 + " --- |" * 12]
    deltas = {}
    for c in cells:
        s = S[c]
        if c == BASE:
            d = ["—"] * 3
        else:
            q = per_query(*c, film)
            d = []
            for name in ("switch_f1", "nat_track", "window@1"):
                ci = cluster_bootstrap(q[name][0] - base_q[name][0], base_q[name][1])
                deltas[(c, name)] = ci
                d.append(f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
        tag = " **(shipped)**" if c == BASE else ""
        L.append(f"| {c[0]}{tag} | {c[1]} | {s['val_obj'][0]:.3f} | {fmt(s['switch_f1'])} | {d[0]} | {fmt(s['part_acc'])} | "
                 f"{s['sw_min'][0]:.2f} | {fmt(s['nat_track'])} | {d[1]} | {fmt(s['window@1'])} | {d[2]} | "
                 f"{fmt(s['window@J'])} | {s['fuse_s'][0]:.0f} |")
    A = all_files(film, base_q["window@1"][0])
    ci = cluster_bootstrap(A["per_w1"], base_q["window@1"][1])
    L += ["", f"**All files (no coarse stage, K = whole gallery)**, from the shipped runs (segment level, before "
          f"Viterbi): track@1 {fmt(A['track@1'])}, window@1 {fmt(A['window@1'])}, Δ window@1 vs shipped "
          f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]. Stitching was not run without the coarse stage."]
    win = max(cells, key=lambda c: (round(S[c]["val_obj"][0], 4), c == BASE))
    by_test = max(cells, key=lambda c: S[c]["switch_f1"][0])
    L += ["", "Δ = cell − shipped K10 J20, per query averaged over the three ensembles; clusters: film groups (natural "
          "clips, segments) or sequences (test concatenations). window@J is not comparable across J (more windows, "
          "more chances).", "", "## Choice", "",
          f"- Chosen on validation: **K {win[0]}, J {win[1]}** (val objective {S[win]['val_obj'][0]:.3f} vs shipped "
          f"{S[BASE]['val_obj'][0]:.3f})."]
    if win == BASE:
        L.append("- The shipped setting stays: no other cell scores higher on validation.")
    else:
        sig = [n for n in ("switch_f1", "nat_track", "window@1") if deltas[(win, n)]["lo"] > 0 or deltas[(win, n)]["hi"] < 0]
        L.append(f"- Its test Δ vs shipped: switch F1 {deltas[(win, 'switch_f1')]['mean']:+.3f}, natural track acc "
                 f"{deltas[(win, 'nat_track')]['mean']:+.3f}, window@1 {deltas[(win, 'window@1')]['mean']:+.3f}; CI clear "
                 f"of zero for: {', '.join(sig) or 'none'}.")
    L.append(f"- Highest test switch F1 (for reference, not used to choose): K {by_test[0]}, J {by_test[1]} "
             f"({S[by_test]['switch_f1'][0]:.3f}).")
    L.append("- Heatmap: [reports/cleaned/topk/heatmap.png](../../reports/cleaned/topk/heatmap.png)")
    REP.mkdir(parents=True, exist_ok=True)
    plot_heatmap(S, cells, REP / "heatmap.png")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    log.info("wrote %s (%d cells; val choice K%d J%d)", OUT_MD, len(cells), *win)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report()


if __name__ == "__main__":
    main()
