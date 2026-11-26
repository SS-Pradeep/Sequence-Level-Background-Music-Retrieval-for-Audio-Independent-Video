"""D140: bidirectional segment retrieval (V2M and M2V) and training-free re-scoring against hubness.

Retrieval only, nothing is generated or retrained: the shipped 10-head ensembles (5 folds x 3 disjoint ensembles,
title split) score scenes against music windows with the same cosine in both directions; only the ranked axis changes.

V2M (scene -> windows) is the segment_all protocol of tab:segret (baseline must reproduce its per-query ranks
exactly). M2V: query = a test segment's aligned window at its own duration, gallery = the fold's test scenes, one
correct scene; scenes of other videos with the window's track are left out (the same-song mask).

Re-scorings (bgm.rerank.normalize), each tuned per fold on that fold's val split (bgm.rerank.tune), never on test:
    train-bank (inductive, headline)   inverted softmax (T, beta), QB-Norm (dynamic IS, k = 1; T, beta)
    test-gallery (transductive)         dual softmax (T), mutual-NN bonus (m, eta): use the other test scenes' scores

Run: python -m bgm.rerank.run_all   (after python -m bgm.rerank.scores)
  -> results/bidir_retrieval.{md,csv,tex}, runs/<dataset>/bidir/perquery_ranks.npz, runs/<dataset>/bidir/fold_report/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.eval.metrics import cluster_bootstrap
from bgm.rerank.evaluate import KS, m2v_ranks, summary, v2m_ranks
from bgm.rerank.hubness import hub_stats
from bgm.rerank.scores import ENS_RUNS, N_FOLDS
from bgm.rerank.tune import GRIDS, choose, m2v_scores, masked, rescore

log = logging.getLogger(__name__)
B = 10000
OUT = Path("results/bidir_retrieval")
SYSTEMS = {
    "v2m": (("base", "—", "baseline: ensemble cosine"),
            ("is", "train-bank", "inverted softmax"),
            ("qb", "train-bank", "QB-Norm (dynamic inverted softmax, k = 1)"),
            ("dsl", "test-gallery", "dual softmax"),
            ("mnn", "test-gallery", "mutual-NN bonus")),
    "m2v": (("base", "—", "baseline: ensemble cosine"),
            ("dsl", "test-gallery", "dual softmax")),
}
PARAM_NAMES = {"base": "", "dsl": "T", "is": "T, β", "qb": "T, β", "mnn": "m, η"}


def _cfg(dataset: str):
    from bgm.analysis.cleaned_compare import CLEAN, ORIG, make_cfg
    return make_cfg({"original": ORIG, "cleaned": CLEAN}[dataset])


def collect(dataset: str) -> dict:
    """Ranks per (direction, method) -> {fold: [ranks per ensemble]}, chosen parameters, hubness, clusters."""
    cfg = _cfg(dataset)
    root = Path(cfg.paths.runs)
    film = json.loads((Path(cfg.paths.manifests) / "folds_title.json").read_text())["film_group"]
    res = {"ranks": {}, "params": {}, "curves": {}, "hub": {}, "clusters": [], "repro": [], "n_gallery": []}
    for k in range(N_FOLDS):
        z = np.load(root / "bidir" / f"scores_fold{k}.npz")
        val = [masked(z, "val", e) for e in ENS_RUNS]
        test = [masked(z, "test", e) for e in ENS_RUNS]
        res["clusters"].append(np.array([film[p] for p in z["pairs_test"]]))
        res["n_gallery"].append(np.isfinite(test[0].v2m).sum(1))
        for e, x in enumerate(test):            # baseline must reproduce tab:segret's saved per-query ranks
            ref = np.load(root / "semantic_ens" / "siglip-both-ens10" / f"fold{k}" / f"seed{e}" / "perquery.npz")["segment_all"]
            res["repro"].append(float((v2m_ranks(x.v2m) == ref).mean()))
        for d, systems in SYSTEMS.items():
            for method, _, _ in systems:
                p, curve = choose(method, val, d)
                res["params"].setdefault((d, method), []).append(p)
                res["curves"].setdefault((d, method), []).append(curve)
                for e, x in enumerate(test):
                    S = rescore(method, p, x) if d == "v2m" else m2v_scores(method, p, x)
                    r = v2m_ranks(S) if d == "v2m" else m2v_ranks(S)
                    res["ranks"].setdefault((d, method), {}).setdefault(k, []).append(r)
                    res["hub"].setdefault((d, method), []).append(hub_stats(S if d == "v2m" else S.T))
        log.info("%s fold %d: %d test scenes, params %s", dataset, k, len(z["pairs_test"]),
                 {f"{d}/{m}": res["params"][(d, m)][-1] for d, s in SYSTEMS.items() for m, _, _ in s})
    res["clusters"] = np.concatenate(res["clusters"])
    res["n_gallery"] = np.concatenate(res["n_gallery"])
    np.savez(root / "bidir" / "perquery_ranks.npz", clusters=res["clusters"],
             **{f"{d}/{m}/fold{k}/e{e}": r for (d, m), f in res["ranks"].items() for k, rs in f.items() for e, r in enumerate(rs)})
    return res


def _pooled(ranks: dict) -> np.ndarray:
    """[n_runs_per_fold, Q_total] ranks, folds concatenated in order."""
    return np.stack([np.concatenate([ranks[k][e] for k in range(N_FOLDS)]) for e in range(len(ENS_RUNS))]).astype(float)


def rows_for(dataset: str, res: dict) -> list[dict]:
    rows = []
    cl = res["clusters"]
    for d, systems in SYSTEMS.items():
        base = _pooled(res["ranks"][(d, "base")])
        for method, norm, label in systems:
            R = _pooled(res["ranks"][(d, method)])
            runs = [summary(r) for f in res["ranks"][(d, method)].values() for r in f]
            hub = pd.DataFrame(res["hub"][(d, method)]).mean()
            row = {"dataset": dataset, "direction": d.upper(), "method": label, "key": method, "normalisation": norm,
                   "params_per_fold": "; ".join(", ".join(f"{v:g}" for v in p) for p in res["params"][(d, method)]),
                   "n_queries": R.shape[1], "n_runs": len(runs), "N10_skew": hub["N10_skew"], "N10_max": hub["N10_max"]}
            for m in [f"R@{k}" for k in KS] + ["MedR", "MRR"]:
                v = np.array([x[m] for x in runs])
                row[f"{m}_mean"], row[f"{m}_std"] = float(v.mean()), float(v.std(ddof=1))
            for m, f in [(f"R@{k}", lambda r, k=k: (r <= k).astype(float)) for k in KS] + [("MRR", lambda r: 1.0 / r)]:
                v, vb = f(R).mean(0), f(base).mean(0)
                ci = cluster_bootstrap(v, cl, B)
                row[f"{m}_lo"], row[f"{m}_hi"] = ci["lo"], ci["hi"]
                if method != "base":
                    dd = cluster_bootstrap(v - vb, cl, B)
                    row[f"d{m}"], row[f"d{m}_lo"], row[f"d{m}_hi"] = dd["mean"], dd["lo"], dd["hi"]
            rows.append(row)
    return rows


# ---------------------------------------------------------------- writing
def _d(r, m: str, star: str = "") -> str:
    if pd.isna(r.get(f"d{m}", np.nan)):
        return "—"
    sig = star if (r[f"d{m}_lo"] > 0 or r[f"d{m}_hi"] < 0) else ""
    return f"{r[f'd{m}']:+.3f} [{r[f'd{m}_lo']:+.3f}, {r[f'd{m}_hi']:+.3f}]{sig}"


def _ms(r, m: str, dec: int = 3) -> str:
    return f"{r[f'{m}_mean']:.{dec}f} ± {r[f'{m}_std']:.{dec}f}"


def markdown(df: pd.DataFrame, meta: dict) -> str:
    L = ["# Bidirectional segment retrieval and dual-softmax / bank re-ranking (D140)", "",
         "Generated by `python -m bgm.rerank.run_all` (src/bgm/rerank/). **Retrieval only, nothing is generated or retrained**: the "
         "shipped 10-head ensembles (title split, 5 folds × 3 disjoint ensembles = 15 runs) score scenes against music windows "
         "with the same cosine in both directions. Parameters tuned per fold on that fold's val split only (val segment MRR, "
         f"mean over the 3 ensembles). Mean ± std over the 15 runs; [CI] and Δ: 95% film-group cluster bootstrap ({B:,} "
         "resamples) on per-query values averaged over a fold's 3 ensembles; Δ is paired against the baseline; * = CI excludes 0.", "",
         "**Normalisation set.** `train-bank` = statistics from the fold's training scenes only (inductive, deployable: the "
         "headline rows). `test-gallery` = uses the other test scenes' scores (transductive: needs the whole test set at once; "
         "reported, not deployable as is).", "",
         "**N10** = k-occurrence: how many queries have a gallery item in their top 10. Skew > 0 = a few hubs fill many lists.", ""]
    for ds, g in df.groupby("dataset", sort=False):
        m = meta[ds]
        L += [f"## {ds.capitalize()} dataset", "",
              f"Baseline reproduction of tab:segret (segment_all per-query ranks, 15 runs): {m['repro']:.1%} identical. "
              f"{m['n_q']:,} test segments, {m['n_films']} film groups; gallery per query {m['gal_mean']:.0f} windows on average "
              f"({m['gal_min']}–{m['gal_max']}).", ""]
        for d, title in (("V2M", "Video → Music (query = scene, gallery = music windows)"),
                         ("M2V", "Music → Video (query = music window, gallery = the fold's scenes; new)")):
            h = g[g.direction == d]
            L += [f"### {title}", "",
                  "| method | normalisation | params per fold | R@1 | R@3 | R@5 | R@10 [95% CI] | MedR ↓ | MRR | N10 skew | max N10 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in h.iterrows():
                par = f"{PARAM_NAMES[r['key']]} = {r['params_per_fold']}" if r["key"] != "base" else "—"
                L.append(f"| {r['method']} | {r['normalisation']} | {par} | {_ms(r, 'R@1')} | {_ms(r, 'R@3')} | {_ms(r, 'R@5')} | "
                         f"{_ms(r, 'R@10')} [{r['R@10_lo']:.3f}, {r['R@10_hi']:.3f}] | {_ms(r, 'MedR', 1)} | {_ms(r, 'MRR')} | "
                         f"{r['N10_skew']:.2f} | {r['N10_max']:.1f} |")
            L += ["", "Paired differences vs the baseline (95% film-group CI):", "",
                  "| method | normalisation | ΔR@1 | ΔR@5 | ΔR@10 | ΔMRR |", "|---|---|---|---|---|---|"]
            for _, r in h[h.key != "base"].iterrows():
                L.append(f"| {r['method']} | {r['normalisation']} | " + " | ".join(_d(r, x, "*") for x in ("R@1", "R@5", "R@10", "MRR")) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def latex(df: pd.DataFrame) -> str:
    def tex_d(r, m):
        return _d(r, m, "$^\\ast$").replace("—", "--")

    L = ["% D140 tables (generated by python -m bgm.rerank.run_all). Needs \\usepackage{booktabs}.", ""]
    for d, lab, what in (("V2M", "tab:bidir_v2m", "Video$\\to$Music segment retrieval with training-free re-scoring"),
                         ("M2V", "tab:bidir_m2v", "Music$\\to$Video segment retrieval (query = music window, gallery = the fold's scenes)")):
        L += ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
              f"\\caption{{{what} (10-head ensemble, no retraining; 5 folds $\\times$ 3 ensembles; parameters tuned on validation; "
              "train-bank = inductive, test-gallery = transductive; $\\Delta$ vs baseline with 95\\% film-group bootstrap CI, "
              "$^\\ast$ = CI excludes 0).}", f"\\label{{{lab}}}", "\\begin{tabular}{llccccccc}", "\\toprule",
              "Data & Scoring & Norm. & R@1 & R@5 & R@10 & MedR & $\\Delta$R@10 & N$_{10}$ skew \\\\", "\\midrule"]
        first = True
        for ds, g in df[df.direction == d].groupby("dataset", sort=False):
            if not first:
                L.append("\\midrule")
            first = False
            for _, r in g.iterrows():
                L.append(f"{ds} & {r['method']} & {r['normalisation'].replace('—', '--')} & "
                         f"{r['R@1_mean']:.3f} $\\pm$ {r['R@1_std']:.3f} & {r['R@5_mean']:.3f} $\\pm$ {r['R@5_std']:.3f} & "
                         f"{r['R@10_mean']:.3f} $\\pm$ {r['R@10_std']:.3f} & {r['MedR_mean']:.1f} $\\pm$ {r['MedR_std']:.1f} & "
                         f"{tex_d(r, 'R@10')} & {r['N10_skew']:.2f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def main() -> Path:
    from bgm.runs import run_dir, write_run
    rows, meta = [], {}
    for ds in ("original", "cleaned"):
        res = collect(ds)
        rows += rows_for(ds, res)
        g = res["n_gallery"]
        meta[ds] = {"repro": float(np.mean(res["repro"])), "n_q": len(res["clusters"]), "n_films": len(np.unique(res["clusters"])),
                    "gal_mean": float(g.mean()), "gal_min": int(g.min()), "gal_max": int(g.max())}
        cfg = _cfg(ds)
        write_run(run_dir(cfg, "bidir", "report"), cfg, {"meta": meta[ds], "grids": {k: [list(p) for p in v] for k, v in GRIDS.items()},
                  "rows": [r for r in rows if r["dataset"] == ds]}, 0)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    OUT.with_suffix(".md").write_text(markdown(df, meta))
    OUT.with_suffix(".tex").write_text(latex(df))
    log.info("wrote %s.{md,csv,tex}", OUT)
    return OUT.with_suffix(".md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(main())
