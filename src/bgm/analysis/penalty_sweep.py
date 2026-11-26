"""Switch-penalty sensitivity of the Viterbi stitching, plus the paper figure that pairs it with the worked example.

python -m bgm.analysis.penalty_sweep sweep   -> outputs/penalty_sweep/sweep.json   (GPU; 5 folds x 3 ensembles)
python -m bgm.analysis.penalty_sweep plot    -> outputs/penalty_sweep/penalty_effect.{pdf,png}

Sweep: the final system (title split, pipeline_ens10, seeds 0-2 = three disjoint 10-head ensembles) on the same
val / test concatenations as `bgm tune-stitch`. For each run, lam_switch takes every value of LAM_S while lam_jump and
mu stay at that run's val-tuned values (stitch.json). This is a sensitivity analysis: nothing is selected on test; the
operating point is still the val-tuned penalty. At the tuned penalty the test means are asserted to equal the run's
saved stitch_metrics.json. Greedy (best window per segment) is the reference.

Figure (single column): (a) the worked example (D125) decoded with lam_switch 0 / tuned / high, (b) test switch F1 and
switches per minute against lam_switch, mean +/- std over the 15 runs, val curve dashed.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
OUT = Path("outputs/penalty_sweep")
ROOT = Path("runs/titlesplit/pipeline_ens10")
LAM_S = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 1000.0]
KEYS = ("switch_f1", "part_track_acc", "switches_per_min")


def path_metrics(store, vq, res, chosen) -> dict:
    """The tune-stitch metrics (evaluate_path) that do not need a render."""
    from bgm.retrieval.stitch_eval import part_track_accuracy, switch_f1, switch_times
    sw, _ = switch_times(chosen, res)
    return {"switch_f1": switch_f1(sw, [t for _, t, _ in vq.meta["parts"][1:]]),
            "part_track_acc": part_track_accuracy(store, vq, res, chosen), "switches_per_min": len(sw) / (vq.duration / 60)}


def sweep_run(cfg, store, fold: int, seed: int) -> dict:
    import torch
    from bgm.eval.store import fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.retrieval.stitch_eval import chosen_candidates, decode_lattice, greedy_lattice, read_concat_sets
    from bgm.retrieval.tune_stitch import prepare_concats

    d = ROOT / f"fold{fold}" / f"seed{seed}"
    fusion = json.loads((d / "fusion.json").read_text())
    stitch = json.loads((d / "stitch.json").read_text())
    saved = json.loads((d / "stitch_metrics.json").read_text())["test"]
    tuned, beam, tol = stitch["penalties"], stitch["beam"], stitch["tol_s"]
    f = fold_ids(cfg, fold)
    sets = read_concat_sets(cfg)[str(fold)]
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    out = {"fold": fold, "seed": seed, "tuned": tuned}
    for split in ("val", "test"):
        index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f[split])))
        concats = prepare_concats(store, index, models, sets[split], cfg, fusion)
        decode = lambda pen: {k: float(np.mean([path_metrics(store, vq, res, chosen_candidates(index, lat, decode_lattice(lat, pen, beam, tol)))[k]  # noqa: E731
                                                  for vq, res, lat in concats])) for k in KEYS}
        out[split] = {"sweep": [decode({**tuned, "lam_switch": ls}) for ls in LAM_S], "tuned": decode(tuned),
                      "greedy": {k: float(np.mean([path_metrics(store, vq, res, chosen_candidates(index, lat, greedy_lattice(lat)))[k]
                                                   for vq, res, lat in concats])) for k in KEYS}}
    # same per-sequence numbers as bgm tune-stitch (stitch_perquery.npz); a near-tie in the decoder may flip on GPU
    # float noise in the scores (fold 2 seed 1: 1 of 50 sequences, one extra switch), so allow at most 2 of 50
    pq = np.load(d / "stitch_perquery.npz")
    for name, dec in (("viterbi", lambda lat: decode_lattice(lat, tuned, beam, tol)), ("greedy", greedy_lattice)):
        per = [path_metrics(store, vq, res, chosen_candidates(index, lat, dec(lat))) for vq, res, lat in concats]
        diff = sorted({q for k in KEYS for q in np.flatnonzero(~np.isclose([x[k] for x in per], pq[f"concat/{name}/{k}"]))})
        assert len(diff) <= 2, (fold, seed, name, diff)
        out[f"n_seq_differing_from_saved_{name}"] = len(diff)
    log.info("fold %d seed %d tuned %s: test F1 %.3f, switches/min %.2f (%d / %d sequences differ from stitch_perquery.npz)",
             fold, seed, tuned, out["test"]["tuned"]["switch_f1"], out["test"]["tuned"]["switches_per_min"],
             out["n_seq_differing_from_saved_viterbi"], len(concats))
    del models
    torch.cuda.empty_cache()
    return out


def sweep() -> None:
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    done = {(r["fold"], r["seed"]) for r in runs}
    store = None
    for fold in range(5):
        for seed in range(3):
            if (fold, seed) in done:
                continue
            cfg = load_config(ROOT / f"fold{fold}" / f"seed{seed}" / "config.yaml")
            store = store or FeatureStore(cfg)
            runs.append(sweep_run(cfg, store, fold, seed))
            path.write_text(json.dumps({"lam_switch": LAM_S, "runs": runs}, indent=1))     # resumable


# ---------------------------------------------------------------- figure
def example_routes() -> dict:
    """Worked-example routes at lam_switch 0 / tuned / 4 (lam_jump, mu = that fold's tuned values)."""
    from bgm.analysis.viterbi_explainer import full_viterbi
    d = json.loads(Path("outputs/viterbi_explainer/lattice.json").read_text())
    L = [{k: np.array([c[i] for c in l["c"]]) for i, k in enumerate(("mid", "off", "dur", "key", "tempo", "z"))} for l in d["layers"]]
    lab = {f["code"]: f["label"] for f in d["files"]}
    nb, fx = len(L) - 1, d["fixed"]
    rows = []
    for ls, name in ((0.0, r"$\lambda_\mathrm{switch}=0$"), (fx["lam_switch"], rf"$\lambda_\mathrm{{switch}}={fx['lam_switch']:g}$ (tuned)"),
                     (4.0, r"$\lambda_\mathrm{switch}=4$")):
        pk = full_viterbi(L, np.full(nb, ls), np.full(nb, fx["lam_jump"]), fx["mu"], d["tol_s"])
        rows.append((name, [(lab[L[i]["mid"][j]], int(L[i]["mid"][j])) for i, j in enumerate(pk)]))
    return {"segments": d["segments"], "clips": d["case"]["clips"], "true": d["case"]["true_change_times_s"],
            "duration": d["case"]["duration_s"], "rows": rows}


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from bgm.analysis.method_figures import COL_W, style
    from bgm.analysis.worked_example import OTHER, TRUE_COLORS

    sw = json.loads((OUT / "sweep.json").read_text())
    assert len(sw["runs"]) == 15, f"{len(sw['runs'])} of 15 runs in sweep.json"
    ex = example_routes()
    style()
    fig = plt.figure(figsize=(COL_W, 3.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.5, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    T, segs = ex["duration"], ex["segments"]
    col = {f"S{k}": c for k, c in enumerate(TRUE_COLORS, 1)}
    rows = [("true music", [(f"S{s['true_clip']}", -s["true_clip"]) for s in segs])] + ex["rows"]
    for r, (name, route) in enumerate(rows):
        y = -r
        ax.text(-1.2, y, name, ha="right", va="center", fontsize=6.6)
        i = 0
        while i < len(route):                          # merge segments that play one file contiguously in the route
            j = i
            while j + 1 < len(route) and route[j + 1][1] == route[i][1]:
                j += 1
            a, b = segs[i]["a"], segs[j]["b"]
            lab = route[i][0]
            ax.add_patch(Rectangle((a + 0.25, y - 0.34), b - a - 0.5, 0.68, fc=col.get(lab, OTHER), ec="none"))
            if b - a > 2.6:
                ax.text((a + b) / 2, y, lab, ha="center", va="center", fontsize=6, fontweight="bold",
                        color="white" if lab in col else "#333")
            if r and i:                                 # a change: right (green) or wrong (red)
                ok = any(abs(a - t) <= 1 for t in ex["true"])
                ax.plot(a, y - 0.47, marker="^", ms=3.2, color="#1d8a5a" if ok else "#c2412c", clip_on=False)
            i = j + 1
        if r:
            ch = [segs[i]["a"] for i in range(1, len(route)) if route[i][1] != route[i - 1][1]]
            tp = sum(any(abs(c - t) <= 1 for c in ch) for t in ex["true"])
            ax.text(T + 1.2, y, f"{len(ch)} changes\n{tp}/3 true", ha="left", va="center", fontsize=5.6, color="#444", linespacing=1.0)
    for t in ex["true"]:
        ax.axvline(t, color="#222", lw=0.6, ls=(0, (2.5, 1.5)), zorder=0)
    ax.set_xlim(0, T)
    ax.set_ylim(-len(rows) + 0.45, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60 s"])
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("(a) one held-out video, three switch penalties", fontsize=7.2, pad=3, loc="left", x=-0.27)

    lam = np.array(sw["lam_switch"])
    xs = np.arange(len(lam))                           # categorical: the grid is uneven and ends at "never"
    ticks = [f"{v:g}" if v < 1000 else "∞" for v in lam]
    tuned_x = [int(np.argmin(abs(lam - r["tuned"]["lam_switch"]))) for r in sw["runs"]]
    for k, (key, title, ylim) in enumerate((("switch_f1", "(b) switch F1 (±1 s)", (0.5, 0.8)),
                                            ("switches_per_min", "(c) switches / min", (0, 7.6)))):
        ax = fig.add_subplot(gs[1, k])
        for split, ls, c in (("val", (0, (3, 1.5)), "#8a8f99"), ("test", "-", "#1f5fd1")):
            Y = np.array([[p[key] for p in r[split]["sweep"]] for r in sw["runs"]])
            m, s = Y.mean(0), Y.std(0)
            if split == "test":
                ax.fill_between(xs, m - s, m + s, color=c, alpha=0.18, lw=0)
            ax.plot(xs, m, color=c, lw=1.2, ls=ls, label=split)
        g = np.mean([r["test"]["greedy"][key] for r in sw["runs"]])
        ax.axhline(g, color="#c2412c", lw=0.8, ls=(0, (1, 1.2)))
        if key == "switch_f1":
            ax.text(5.5, g - 0.008, "greedy", ha="center", va="top", fontsize=5.8, color="#c2412c")
        else:
            ax.text(len(lam) - 1, g + 0.18, "greedy", ha="right", va="bottom", fontsize=5.8, color="#c2412c")
        for x in sorted(set(tuned_x)):                 # where val tuning put the 15 runs
            n = tuned_x.count(x)
            ax.plot(x, ylim[0], marker="v", ms=3.2, color="#222", clip_on=False, zorder=5)
            ax.text(x, ylim[0] + 0.05 * (ylim[1] - ylim[0]), f"{n}", ha="center", va="bottom", fontsize=5.2, color="#222")
        tk = [0, 2, 4, 6, 8, 10, 11]
        ax.set_xticks(xs[tk], [ticks[t] for t in tk])
        ax.set_xlim(0, len(lam) - 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\lambda_\mathrm{switch}$", labelpad=0.5, fontsize=6.8)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.tick_params(length=2, pad=1, labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.8, borderaxespad=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "penalty_effect.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "penalty_effect.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s/penalty_effect.{pdf,png}", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"sweep": sweep, "plot": plot}[sys.argv[1]]()
