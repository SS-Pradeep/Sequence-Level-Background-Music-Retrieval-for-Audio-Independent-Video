"""Best two re-rankers fed to the final decoder on the Table 3 sequences (D148, part 2).

`.venv/bin/python -m exp_supp.rerank.hmm` -> runs/exp_supp/rerank/hmm/fold<k>.json, results.csv,
                                           exp_supp/rerank/rerank_hmm_table.{md,tex}

Final system = "HMM, prior from gold paths, α (= λ) tuned" (stitching/results/fold<k>_gold.json, read-only), on the
Table 3 lattices (stitching/cache/fold<k>.pkl: 112 test sequences, original 1,248-pair data with the title folds the
cleaned split carries over; the cleaned dataset has no Table 3 lattices). Its saved per-sequence metrics are reproduced
first (asserted). Methods chosen BEFORE looking at this test: the two largest mean val gains of the segment-protocol
ensemble in the retrieval study (k-reciprocal, sequence context).

Re-ranking inside a scene: gallery = the scene's K_t lattice candidates, shortlist = the top 20 by sim; the re-ranker
re-orders the shortlist and the 20 sorted sim values are re-assigned in the new order (the emission scale and every
candidate outside the shortlist are unchanged). Sequence context: neighbours = scenes t ± n of the same sequence
(sequences join several clips, so neighbours may carry other music); file evidence from Episode.dense (all windows
of the coarse-shortlisted files) or the neighbour's candidates. Re-ranker parameters and λ are chosen jointly on the
val sequences (val_objective = track acc. + switch F1, first grid entry wins ties); test decoded once.
"""
from __future__ import annotations

import copy
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from exp_supp.common import RUNS, fmt_ci, log_run, merge_clusters, paired_ci
from exp_supp.rerank import methods as M
from exp_supp.rerank import clean_t3
from exp_supp.rerank.lattice import OUT as LAT, OUT_CLEAN as LAT_CLEAN
from stitching.base import Cands
from stitching.data import load_fold as load_fold_orig
from stitching.hmm_derived import HMMDerived
from stitching.metrics import evaluate_path, fold_entropy, val_objective
from stitching.report import _derive

log = logging.getLogger("exp_supp.rerank.hmm")
# data set: 'orig' = the Table 3 lattices (original 1,248 pairs, 112 sequences); 'clean' = the same decoder rebuilt on
# the cleaned 1,011 pairs by exp_supp.rerank.clean_t3 (92 sequences, D148b)
DATA = {
    "orig": {"load": load_fold_orig, "lat": LAT, "gold": Path("stitching/results"), "out": RUNS / "rerank" / "hmm",
             "table": "exp_supp/rerank/rerank_hmm_table", "group": "rerank_hmm", "lattices": "stitching/cache/fold{k}.pkl (Table 3)",
             "desc": "Table 3 lattices (stitching/cache; original 1,248-pair data with the title folds that cleaned_dataset carries "
                     "over unchanged)", "tag": "D148"},
    "clean": {"load": clean_t3.load_fold, "lat": LAT_CLEAN, "gold": clean_t3.GOLD, "out": clean_t3.ROOT / "hmm",
              "table": "exp_supp/rerank/rerank_hmm_table_cleaned", "group": "rerank_hmm_cleaned",
              "lattices": "runs/exp_supp/rerank/clean_t3/cache/fold{k}.pkl (cleaned)",
              "desc": "cleaned dataset (1,011 pairs): the Table 3 decoder chain rebuilt on cleaned_dataset by exp_supp.rerank.clean_t3 "
                      "(C boundary head retrained on the cleaned train clips, theta on cleaned val, lattices from the cleaned "
                      "10-seed siglip-both emission, gold-path prior fitted on cleaned training sequences, λ on cleaned val)",
              "tag": "D148b"},
}
METRICS = ["track_acc", "switch_f1", "TBIoU", "switches_per_min", "jumps_per_min", "align_success"]
LAMS = [float(x) for x in 2.0 ** np.linspace(-2, 2, 9)]
METHODS = {
    "k-reciprocal": [{"k1": a, "k2": b, "lam": c} for a in (5, 10, 20) for b in (1, 3, 6) for c in (0.1, 0.3, 0.5, 0.7, 0.9)],
    "Sequence context": [{"w": w, "n": n} for n in (1, 2, 99) for w in (0.25, 0.5, 1, 2)],
}


def permute_shortlist(sim: np.ndarray, R: np.ndarray, sl: np.ndarray) -> np.ndarray:
    """Re-assign the shortlist's sorted sim values in the order of (R, sim), descending. Outside sl: unchanged."""
    out = sim.copy()
    idx = np.where(sl)[0]
    new_order = idx[np.lexsort((sim[idx], R[idx]))[::-1]]
    out[new_order] = np.sort(sim[idx])[::-1]
    return out


def file_best_z(ep, t: int, f: int) -> float | None:
    c = ep.cands[t]
    mu, sd = c.sim.mean(), c.sim.std() + 1e-8
    vals = []
    if f in ep.dense:
        row = ep.dense[f][1][t]
        row = row[np.isfinite(row)]
        if len(row):
            vals.append(row.max())
    sel = c.mid == f
    if sel.any():
        vals.append(c.sim[sel].max())
    return (max(vals) - mu) / sd if vals else None


def rerank_episode(ep, lat_scenes, method: str, p: dict):
    ep2 = copy.copy(ep)
    cands = []
    for t, c in enumerate(ep.cands):
        K = len(c.sim)
        valid = np.ones(K, bool)
        sl = M.shortlist(c.sim[None], valid)[0]
        if method == "k-reciprocal":
            b = M.Block(c.sim[None], valid, lat_scenes[t]["G"], lat_scenes[t]["B"], c.mid)
            R = M.k_reciprocal(b, **p)[0]
        else:
            Z = (c.sim - c.sim.mean()) / (c.sim.std() + 1e-8)
            nb = [u for u in range(max(0, t - p["n"]), min(ep.T, t + p["n"] + 1)) if u != t]
            ctx = np.zeros(K)
            for f in np.unique(c.mid):
                v = [x for x in (file_best_z(ep, u, int(f)) for u in nb) if x is not None]
                if v:
                    ctx[c.mid == f] = np.mean(v)
            R = Z + p["w"] * ctx
        sim2 = permute_shortlist(c.sim, R, sl)
        cands.append(Cands(c.mid, c.off, sim2, c.n_top, c.emb_in, c.emb_out, c.rms_in, c.rms_out, c.ph_in, c.ph_out))
    ep2.cands = cands
    return ep2


def fold_run(k: int, data: str = "orig") -> dict:
    D = DATA[data]
    t0 = time.time()
    F = D["load"](k)
    va, te = F["episodes"]["val"], F["episodes"]["test"]
    with open(D["lat"] / f"fold{k}.pkl", "rb") as fh:
        lat = pickle.load(fh)["episodes"]
    final = json.loads((D["gold"] / f"fold{k}_gold.json").read_text())["methods"]["hmm_gold_tuned"]
    fp = final["params"]
    hmm = HMMDerived(fp["tau"], fp["L"], fp["rho"], fp["lam"])
    rep = [evaluate_path(e, hmm.decode(e)) for e in te]
    diff = max(abs(a[m] - b[m]) for a, b in zip(rep, final["rows"]) for m in METRICS if np.isfinite(a[m]))
    assert diff < 1e-9, f"fold {k}: final row not reproduced (max diff {diff})"
    rec = {"fold": k, "final_rows": final["rows"], "final_params": fp, "reproduction_max_diff": diff,
           "films": {e.key: sorted(set(e.films)) for e in te}, "methods": {}}
    for name, grid in METHODS.items():
        best, arg = -np.inf, None
        for p in grid:
            vr = [rerank_episode(e, lat["val"][i], name, p) for i, e in enumerate(va)]
            for lam in LAMS:
                obj = val_objective(HMMDerived(fp["tau"], fp["L"], fp["rho"], lam), vr)
                if obj > best + 1e-12:
                    best, arg = obj, (p, lam)
        p, lam = arg
        st = HMMDerived(fp["tau"], fp["L"], fp["rho"], lam)
        rows, paths, rt = [], [], []
        for i, e in enumerate(te):
            t1 = time.perf_counter()
            er = rerank_episode(e, lat["test"][i], name, p)
            path = st.timed(er)
            rt.append(time.perf_counter() - t1)
            paths.append(path)
            rows.append(dict(evaluate_path(e, path), key=e.key, runtime_s=rt[-1]))
        ent = fold_entropy(te, paths, F["gallery_test"])
        mean = {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS} | {"ms_per_video": 1000 * float(np.mean(rt))}
        log_run(D["group"], name, k, 0, "test", {"rerank": p, "lam": lam, "tau": fp["tau"], "L": fp["L"], "rho": fp["rho"],
                                                  "grid": grid, "lam_grid": LAMS, "lattices": D["lattices"].format(k=k)},
                mean | {"entropy": ent, "val_objective": best})
        rec["methods"][name] = {"params": {"rerank": p, "lam": lam, "val_objective": best}, "rows": rows, "entropy": ent}
        log.warning("fold %d %s: %s λ %.3g | test %s [%.0f s]", k, name, p, lam, {a: round(b, 3) for a, b in mean.items()}, time.time() - t0)
    D["out"].mkdir(parents=True, exist_ok=True)
    (D["out"] / f"fold{k}.json").write_text(json.dumps(rec, default=float))
    return rec


def table(recs: list[dict], data: str = "orig") -> None:
    D = DATA[data]
    keys = [r["key"] for rec in recs for r in rec["final_rows"]]
    films = {k: v for rec in recs for k, v in rec["films"].items()}
    clusters = merge_clusters([films[k] for k in keys])
    Fr = [r for rec in recs for r in rec["final_rows"]]

    def fold_stats(rows_of, m):
        per = [np.nanmean([_derive(r)[m] for r in rows_of(rec)]) for rec in recs]
        return float(np.mean(per)), float(np.std(per))
    systems = [("HMM, gold prior, α tuned (final)", lambda rec: rec["final_rows"])] + \
              [(f"+ {n} re-ranking (top-20)", (lambda n: lambda rec: rec["methods"][n]["rows"])(n)) for n in METHODS]
    ms_final = 1000 * np.mean([r["runtime_s"] for r in Fr])
    L = [f"# Re-ranked shortlists through the final decoder (Table 3 protocol, {D['tag']})", "",
         f"{len(keys)} test sequences, 5 title folds, {D['desc']}. The final row is re-decoded "
         "with its saved parameters and reproduces its saved values exactly. The two re-rankers were chosen before this test, by "
         "the largest mean val gain of the segment-protocol ensemble in the retrieval study. Each scene's top-20 candidates are "
         "re-ordered (sorted sim values re-assigned in the new order); re-ranker parameters and λ chosen jointly on val sequences. "
         "Mean ± std over folds (std for track acc. and switch F1, as Table 3). Δ = re-ranked − final per sequence, 95% cluster "
         f"bootstrap over film groups ({len(set(clusters))} clusters), 10,000 resamples; * = CI excludes 0.", "",
         "| System | Track acc. | Switch F1 ±1 s | TBIoU | Switches/min | Jumps/min | Align. succ. | ms/video |",
         "|---|---|---|---|---|---|---|---|"]
    for nm, get in systems:
        a, b = fold_stats(get, "track_acc"), fold_stats(get, "switch_f1")
        ms = ms_final if nm.startswith("HMM") else 1000 * np.mean([r["runtime_s"] for rec in recs for r in get(rec)])
        L.append(f"| {nm} | {a[0]:.3f} ± {a[1]:.3f} | {b[0]:.3f} ± {b[1]:.3f} | "
                 + " | ".join(f"{fold_stats(get, m)[0]:.{2 if 'min' in m else 3}f}" for m in METRICS[2:]) + f" | {ms:.1f} |")
    L += ["", "| Δ vs final | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    deltas = {}
    for nm, get in systems[1:]:
        H = [r for rec in recs for r in get(rec)]
        assert [r["key"] for r in H] == keys
        deltas[nm] = {m: paired_ci(np.array([h[m] - f[m] for h, f in zip(H, Fr)], float), clusters) for m in METRICS}
        L.append(f"| {nm} | " + " | ".join(fmt_ci(deltas[nm][m], 2 if "min" in m else 3) for m in METRICS) + " |")
    L += ["", "## Picks on val", "", "| fold | " + " | ".join(f"{n} (params, λ)" for n in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for rec in recs:
        L.append(f"| {rec['fold']} | " + " | ".join(
            f"{rec['methods'][n]['params']['rerank']}, {rec['methods'][n]['params']['lam']:.3g}" for n in METHODS) + " |")
    Path(D["table"] + ".md").write_text("\n".join(L) + "\n")

    t = lambda s: s.replace("−", "-")  # noqa: E731
    sig = lambda r: r["lo"] > 0 or r["hi"] < 0  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
           r"\caption{Re-ranked top-20 shortlists through the final decoder (HMM, gold-path prior, $\alpha$ tuned) on the",
           (r"%d test sequences (%s). " % (len(keys), "Tab.~3, original data" if data == "orig" else "cleaned dataset, 1{,}011 pairs"))
           + r"$\Delta$ vs.\ the final system, bold = 95\% film-level cluster-bootstrap CI excludes 0.}",
           r"\label{tab:supp_rerank_hmm%s}" % ("" if data == "orig" else "_cleaned"), r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
           r"System & Acc. & F1 & TBIoU & Sw./min & J./min & Align. \\", r"\midrule"]
    for nm, get in systems:
        tex.append(nm.replace("α", r"$\alpha$") + " & " + " & ".join(
            f"{fold_stats(get, m)[0]:.{2 if 'min' in m else 3}f}" for m in METRICS) + r" \\")
    tex.append(r"\midrule")
    for nm, d in deltas.items():
        tex.append(r"$\Delta$ " + nm.replace("+ ", "").replace(" re-ranking (top-20)", "") + " & " + " & ".join(
            t((r"$\mathbf{%+.*f}$" if sig(d[m]) else r"$%+.*f$") % (2 if "min" in m else 3, d[m]["mean"])) for m in METRICS) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(D["table"] + ".tex").write_text("\n".join(tex) + "\n")
    print("\n".join(L))


def main() -> None:
    import sys
    data = "clean" if "--clean" in sys.argv else "orig"
    recs = [fold_run(k, data) for k in range(5)]
    table(recs, data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    main()
