"""Aggregate stitching/results/fold*.json -> main_table.{csv,md}, wilcoxon.md, cue_stats.md, CRF plot, timelines.

`.venv/bin/python -m stitching.report`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stitching.evaluate import ROWS

OUT = Path("stitching/results")
COLS = [("track_acc", "Track acc. (R@1 of path)", 3), ("switch_f1", "Switch F1 ±1 s", 3), ("TBIoU", "TBIoU", 3),
        ("switches_per_min", "Switches / min", 2), ("jumps_per_min", "Jumps / min", 2), ("dev_T2P", "Deviation T→P (s)", 2),
        ("switch_precision", "Switch precision ±1 s", 3), ("align_success", "Switch recall = alignment success ±1 s", 3),
        ("rate_err", "|switches/min − 3.10|", 2), ("entropy", "Entropy", 3), ("runtime_ms", "Runtime / video (ms)", 1)]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6"]


def _derive(r: dict) -> dict:
    """Switch precision from stored fields: tp = recall x n_true, n_pred / n_true = pred rate / true rate, so
    precision = recall x true rate / pred rate (NaN without predicted switches). rate_err = |pred rate - 3.10|."""
    r = dict(r)
    pr, tr, rec = r["switches_per_min"], r["true_switches_per_min"], r["align_success"]
    r["switch_precision"] = float(rec * tr / pr) if pr > 0 and np.isfinite(rec) else np.nan
    r["rate_err"] = abs(pr - 3.10)
    return r


def per_episode(m: dict) -> tuple[list[dict], float]:
    """Rows (seed-averaged for the CRF) and entropy."""
    if "seeds" not in m:
        return [_derive(r) for r in m["rows"]], m["entropy"]
    S = m["seeds"]
    rows = []
    for i in range(len(S[0]["rows"])):
        r = {k: float(np.nanmean([s["rows"][i][k] for s in S])) for k in S[0]["rows"][i] if k != "key"}
        r["key"] = S[0]["rows"][i]["key"]
        rows.append(r)
    # precision per seed, then averaged (not derived from seed-averaged rates)
    for i, r in enumerate(rows):
        d = [_derive(s["rows"][i]) for s in S]
        r["switch_precision"] = float(np.nanmean([x["switch_precision"] for x in d])) if any(np.isfinite(x["switch_precision"]) for x in d) else np.nan
        r["rate_err"] = float(np.mean([x["rate_err"] for x in d]))
    return rows, float(np.mean([s["entropy"] for s in S]))


def main() -> None:
    folds = [json.loads((OUT / f"fold{k}.json").read_text()) for k in range(5)]
    global ROWS
    if all((OUT / f"fold{k}_gold.json").exists() for k in range(5)):
        from stitching.gold_prior import GOLD_ROWS
        ROWS = list(ROWS) + GOLD_ROWS
        for k, f in enumerate(folds):
            g = json.loads((OUT / f"fold{k}_gold.json").read_text())
            f["methods"].update(g["methods"])
            f["gold_prior"] = g["gold_prior"]
    for suffix, mod, name in (("phase0", "stitching.phase0", "PHASE0_ROWS"), ("xf", "stitching.crf_crossfit", "XF_ROWS"),
                              ("phase2", "stitching.phase2", "PHASE2_ROWS"), ("semi", "stitching.extra_rows", "SEMI_ROWS")):
        if all((OUT / f"fold{k}_{suffix}.json").exists() for k in range(5)):
            import importlib
            ROWS = list(ROWS) + getattr(importlib.import_module(mod), name)
            for k, f in enumerate(folds):
                f["methods"].update(json.loads((OUT / f"fold{k}_{suffix}.json").read_text())["methods"])
    fold_means = {key: [] for key, _ in ROWS}
    pooled = {key: [] for key, _ in ROWS}
    for f in folds:
        for key, _ in ROWS:
            rows, ent = per_episode(f["methods"][key])
            m = {c: float(np.nanmean([r[c] for r in rows])) for c, _, _ in COLS if c not in ("entropy", "runtime_ms")}
            m["entropy"] = ent
            m["runtime_ms"] = 1000 * float(np.mean([r["runtime_s"] for r in rows]))
            fold_means[key].append(m)
            pooled[key] += rows
    # ---- main table
    csv = ["method," + ",".join(f"{c}_mean,{c}_std" for c, _, _ in COLS)]
    md = ["# Stitching ablation (title split, 5 folds; test = frozen blocked sequences, BaSSL C-head own cuts)", "",
          "Mean ± std over the 5 test folds (each fold = mean over its test sequences; CRF rows also averaged over 3 "
          "training seeds). Every method decodes the same shortlist (10-head ensemble, top-20 windows + continuations); "
          "every hyperparameter tuned on the fold's val sequences (objective = track acc. + switch F1).", "",
          "| # | Method | " + " | ".join(n for _, n, _ in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for i, (key, name) in enumerate(ROWS, 1):
        fm = fold_means[key]
        cells = []
        for c, _, d in COLS:
            v = np.array([x[c] for x in fm])
            cells.append(f"{np.nanmean(v):.{d}f} ± {np.nanstd(v):.{d}f}")
        md.append(f"| {i} | {name} | " + " | ".join(cells) + " |")
        csv.append(f"\"{name}\"," + ",".join(f"{np.nanmean([x[c] for x in fm]):.5f},{np.nanstd([x[c] for x in fm]):.5f}" for c, _, _ in COLS))
    sl = [f["shortlist"] for f in folds]
    true_rate = np.mean([np.mean([r["true_switches_per_min"] for r in f["methods"]["greedy"]["rows"]]) for f in folds])
    md += ["", f"- Shortlist (identical for every method): track R@10 {np.mean([s['track_R@10'] for s in sl]):.3f} ± "
           f"{np.std([s['track_R@10'] for s in sl]):.3f}, MRR {np.mean([s['track_MRR'] for s in sl]):.3f} ± {np.std([s['track_MRR'] for s in sl]):.3f} "
           "(rank of the true track among the shortlisted files, per test scene). R@10 / MRR do not depend on the stitcher, so the "
           "table reports the decoded path's track accuracy (its R@1) instead.",
           f"- True changes: {true_rate:.2f} / min. Alignment success = share of true changes with a predicted switch within ±1 s; "
           "deviation = median distance from a true change to the nearest predicted switch.",
           f"- Test sequences: {sum(len(f['methods']['greedy']['rows']) for f in folds)}; galleries ~{int(np.mean([f['gallery']['test'] for f in folds]))} files per test fold.",
           "- Hand-set row reproduces D143's combo/own R0_tuned exactly in every fold: " +
           ", ".join(f"f{f['fold']} {f['reference_check_handset_vs_D143']['params_equal']}/{f['reference_check_handset_vs_D143']['max_abs_diff_objective']:.0e}" for f in folds)]
    (OUT / "main_table.csv").write_text("\n".join(csv) + "\n")
    # ---- params
    md += ["", "## Tuned hyperparameters (per fold)", "", "| fold | τ | L (s) | ρ | smooth w | two-stage | hand-set λ_switch / λ_jump | HMM λ | CRF γ (prior, seeds) | CRF α / λ (ctx+prior, seed 0) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in folds:
        M = f["methods"]
        cp = M["crf_ctx_prior"]["seeds"]
        md.append(f"| {f['fold']} | {f['tau']:.4f} | {f['L_s']:.2f} | {f['rho']:.3f} | {M['smooth']['params']['w']} | "
                  f"{M['two_stage']['params']['mode']} {M['two_stage']['params']['param']:.3f} | "
                  f"{M['handset']['params']['penalties']['lam_switch']:g} / {M['handset']['params']['penalties']['lam_jump']:g} | "
                  f"{M['hmm_tuned']['params']['lam']:.3g} | {', '.join(str(s['params']['gamma']) for s in cp)} | "
                  f"{cp[0]['params']['alpha']:.2f} / {cp[0]['params']['lam']:.2f} |")
    # ---- gold / guard
    md += ["", "## CRF checks", "", "Gold window among the plain candidates (train sequences use the frozen heads' own training clips):", "",
           "| fold | train | val | test | gold sim percentile train / val / test |", "|---|---|---|---|---|"]
    for f in folds:
        g = f["gold_stats"]
        md.append(f"| {f['fold']} | {g['train']['gold_in_candidates']:.3f} | {g['val']['gold_in_candidates']:.3f} | {g['test']['gold_in_candidates']:.3f} | "
                  f"{g['train']['gold_sim_percentile']:.2f} / {g['val']['gold_sim_percentile']:.2f} / {g['test']['gold_sim_percentile']:.2f} |")
    md += ["", "Shortcut guard (val, seed 0): predicted vs real switch rate.", "",
           "| fold | variant | pred. switches / min | true / min | pred. switch / boundary | posterior P(switch) | true change / boundary | derived p_change |",
           "|---|---|---|---|---|---|---|---|"]
    for f in folds:
        for v, g in f["crf_val"].items():
            md.append(f"| {f['fold']} | {v} | {g['pred_switches_per_min']:.2f} | {g['true_switches_per_min']:.2f} | {g['pred_switch_rate_per_boundary']:.3f} | "
                      f"{g['mean_posterior_switch_prob']:.3f} | {g['true_change_rate_per_boundary']:.3f} | {g['derived_p_change_mean']:.3f} |")
    if "gold_prior" in folds[0]:
        md += ["", "## Prior counted from the training gold paths (rows 11-13)", "",
               "| fold | L (s, MLE) | ρ | continue | jump | switch | π_c / π_j / π_s (raw frequencies) |", "|---|---|---|---|---|---|---|"]
        for f in folds:
            g = f["gold_prior"]
            md.append(f"| {f['fold']} | {g['L_s']:.2f} | {g['rho']:.3f} | {g['n_continue']} | {g['n_jump']} | {g['n_switch']} | "
                      f"{g['pi_c']:.3f} / {g['pi_j']:.3f} / {g['pi_s']:.3f} |")
    (OUT / "main_table.md").write_text("\n".join(md) + "\n")
    xf = [OUT / f"fold{k}_xf.json" for k in range(5)]
    if all(p.exists() for p in xf):
        X = [json.loads(p.read_text()) for p in xf]
        md += ["", "## Phase 1 check: gold window among the plain candidates", "",
               "| fold | cross-fitted train | in-sample train (old) | val | test | cross-fitted / test |", "|---|---|---|---|---|---|"]
        for x in X:
            g = x["gold_stats"]
            md.append(f"| {x['fold']} | {g['xtrain']['gold_in_candidates']:.3f} | {g['train_insample']['gold_in_candidates']:.3f} | "
                      f"{g['val']['gold_in_candidates']:.3f} | {g['test']['gold_in_candidates']:.3f} | {x['xtrain_over_test_ratio']:.2f} |")
        if all("crf_ctx_prior_xf" in x.get("crf_val", {}) for x in X):
            plot_switch(X, "crf_ctx_prior_xf", "crf_switch_vs_siglip_xf.png", "Cross-fitted CRF (context + prior)")
        (OUT / "main_table.md").write_text("\n".join(md) + "\n")
    wilcoxon(pooled)
    cue_md(folds)
    plot_switch(folds)
    timelines(folds)
    print("\n".join(md))


def wilcoxon(pooled: dict) -> None:
    from scipy.stats import wilcoxon as wx
    keys = [k for k, _ in ROWS]
    names = dict(ROWS)
    L = ["# Paired Wilcoxon signed-rank tests over test sequences (5 folds pooled)", "",
         "Row 10 (CRF, context + derived prior, mean over 3 seeds) minus each baseline, per test sequence; two-sided; "
         "zero differences dropped (Wilcoxon's method).", "",
         "| vs | metric | n | mean Δ | median Δ | W | p |", "|---|---|---|---|---|---|---|"]
    a = pooled["crf_ctx_prior"]
    for b in ("greedy", "dtw", "handset", "hmm_tuned"):
        for m in ("objective", "track_acc", "switch_f1", "TBIoU"):
            x = np.array([r[m] for r in a]) - np.array([r[m] for r in pooled[b]])
            try:
                res = wx(x)
                W, p = float(res.statistic), float(res.pvalue)
            except ValueError:
                W, p = float("nan"), 1.0
            L.append(f"| {keys.index(b) + 1}. {names[b]} | {m} | {len(x)} | {x.mean():+.4f} | {np.median(x):+.4f} | {W:.1f} | {p:.2g} |")
    (OUT / "wilcoxon.md").write_text("\n".join(L) + "\n")


def cue_md(folds) -> None:
    cs = json.loads((OUT / "cue_stats.json").read_text())["folds"]
    L = ["# Cue statistics from the training films' BGM (per fold)", "",
         "| fold | L (s) | ρ | cues | files | music (min) | cues ≥ 4 s | reprises | x-corr threshold (p99 of different-film pairs) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, c in cs.items():
        L.append(f"| {k} | {c['L_s']:.2f} | {c['rho']:.3f} | {c['n_cues']} | {c['n_files']} | {c['music_minutes']:.0f} | {c['n_cues_ge4s']} | {c['n_reprises']} | {c['xcorr_threshold_p99_null']:.3f} |")
    Ls, rs = [c["L_s"] for c in cs.values()], [c["rho"] for c in cs.values()]
    L += ["", f"Mean over folds: L = {np.mean(Ls):.2f} ± {np.std(Ls):.2f} s, ρ = {np.mean(rs):.3f} ± {np.std(rs):.3f}.", "",
          "Caveat: every dataset file is one clip's BGM (~20 s median), and file ends count as cue ends, so L is bounded by the clip length; "
          "it is the cue length *within these clips*, not of whole film scores."]
    (OUT / "cue_stats.md").write_text("\n".join(L) + "\n")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_switch(folds, key: str = "crf_ctx_prior", fname: str = "crf_switch_vs_siglip.png", title: str = "CRF (context + prior)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [f["crf_val"][key] for f in folds]
    d = np.concatenate([x["siglip_dist"] for x in g])
    p = np.concatenate([x["marginals"] for x in g])
    y = np.concatenate([x["label"] for x in g])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    _style(ax)
    ax.scatter(d, p, s=9, color=SERIES[0], alpha=0.25, lw=0, label="boundary (posterior)")
    edges = np.quantile(d, np.linspace(0, 1, 11))
    c = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, 9)
    ax.plot(c, [p[idx == i].mean() for i in range(10)], color=SERIES[0], lw=2, marker="o", ms=5, label="CRF P(switch), decile mean")
    ax.plot(c, [y[idx == i].mean() for i in range(10)], color=SERIES[1], lw=2, marker="s", ms=5, label="true change rate, decile mean")
    ax.set_xlabel("SigLIP cosine distance between adjacent scenes", color=INK, fontsize=10)
    ax.set_ylabel("P(music switch at boundary)", color=INK, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title} switch probability vs scene change, val, 5 folds", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def timelines(folds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    f = folds[0]
    rows = f["methods"]["greedy"]["rows"]
    picks = sorted(range(len(rows)), key=lambda i: -rows[i]["n_scenes"])[:3]
    data = f["test_meta"]
    names = {"greedy": "Greedy", "dtw": "Prétet-style DTW", "hmm_tuned": "HMM (λ tuned)", "crf_ctx_prior": "CRF (ctx + prior)"}
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.0 * len(picks) + 0.4), dpi=150)
    for ax, i in zip(np.atleast_1d(axes), picks):
        segs = np.array(data[str(i)]["segs"]) if data else None
        truth = data[str(i)]["truth_files"] if data else None
        rowsets = [("truth", truth)] + [(m, f["timelines"][m][i]) for m in names]
        allf = [x for _, r in rowsets for x in r]
        uniq, cnt = np.unique(allf, return_counts=True)
        top = list(uniq[np.argsort(-cnt)][:len(SERIES)])
        col = lambda m: SERIES[top.index(m)] if m in top else "#b9b8b0"  # noqa: E731
        for r, (m, files) in enumerate(rowsets):
            for t, fid in enumerate(files):
                a, b = segs[t]
                ax.add_patch(Rectangle((a, -r - 0.4), b - a, 0.8, color=col(fid), lw=0))
                ax.plot([a, a], [-r - 0.4, -r + 0.4], color="white", lw=1.5)
        for t in data[str(i)]["true_sw"]:
            ax.axvline(t, color=INK, lw=0.8, ls="--")
        ax.set_yticks([-r for r in range(len(rowsets))], ["Truth"] + list(names.values()), fontsize=8, color=INK)
        ax.set_xlim(0, segs[-1, 1])
        ax.set_ylim(-len(rowsets) + 0.4, 0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"fold 0 test sequence {i}: {len(segs)} scenes, colour = music file (grey = files outside the 7 most used), dashed = true change",
                     fontsize=8, color=INK, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "timelines_fold0.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
