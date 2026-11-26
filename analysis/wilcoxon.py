"""Paired Wilcoxon signed-rank tests: is system A better than system B on the same test items? (D112)

Two comparisons, both on stored runs (nothing is retrained):

1. **The cue-switch predictor** (D78–D81): R4 (learned music-change predictor + context) against the same decoder
   with fixed penalties and no predictor (R0 re-tuned on val, and R0 with the stored D75 penalties = the shipped
   system). Per blocked test sequence of runs/filmsplit/cue (the only split the predictor was run on; 1,248 pairs).
2. **The final system** (D75, title split, runs/titlesplit/eval_suite, D111): Viterbi stitching against greedy
   (no transition model) and random music; render beat sync against random music + the same sync.

Per item, a system's value is averaged over its seeds first (CLAUDE.md rule 11); then the test runs on the paired
per-item differences. Two versions:
  * per item (sequence / video / concatenation): the standard test;
  * per cluster (film group, or a sequence's union-find film cluster): differences averaged within a cluster first,
    because items from one film are not independent. This is the conservative one.
p-values are two-sided, Holm-adjusted within each comparison. Effect size: matched-pairs rank-biserial
r = (R+ − R−) / (R+ + R−), in [−1, 1]; positive = A higher.

Run: python -m bgm.analysis.wilcoxon   -> results/wilcoxon.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

log = logging.getLogger(__name__)
OUT = Path("results/wilcoxon.md")
CUE_ROOT = Path("runs/filmsplit/cue/eval")


# ---------------------------------------------------------------- pure statistics
def paired_wilcoxon(a: np.ndarray, b: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """Two-sided Wilcoxon signed-rank test of a − b over paired items ([n] each; NaN pairs dropped).

    With `clusters` ([n]), differences are first averaged within each cluster and the test runs over clusters.
    Zero differences are dropped (Wilcoxon's convention). Returns n, n_nonzero, mean / median difference, wins
    (a > b) and losses, W (scipy's statistic: the smaller rank sum), p, and the rank-biserial r."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if clusters is not None:
        cl = np.asarray(clusters)[keep]
        uniq, inv = np.unique(cl, return_inverse=True)
        d = np.bincount(inv, weights=d, minlength=len(uniq)) / np.bincount(inv, minlength=len(uniq))
    nz = d[np.abs(d) > 1e-12]
    out = {"n": int(len(d)), "n_nonzero": int(len(nz)), "mean": float(d.mean()) if len(d) else np.nan,
           "median": float(np.median(d)) if len(d) else np.nan, "wins": int((nz > 0).sum()), "losses": int((nz < 0).sum())}
    if len(nz) < 1:
        return {**out, "W": np.nan, "p": 1.0, "r": 0.0}
    rk = rankdata(np.abs(nz))
    rp, rm = rk[nz > 0].sum(), rk[nz < 0].sum()
    res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return {**out, "W": float(res.statistic), "p": float(res.pvalue), "r": float((rp - rm) / (rp + rm))}


def holm(p: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, same order as `p`."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj.tolist()


def f1_counts(c: np.ndarray) -> np.ndarray:
    """[n, 3] (tp, fp, fn) -> [n] F1 = 2tp / (2tp + fp + fn); NaN where the denominator is 0."""
    c = np.asarray(c, float)
    den = 2 * c[:, 0] + c[:, 1] + c[:, 2]
    with np.errstate(all="ignore"):
        return np.where(den > 0, 2 * c[:, 0] / den, np.nan)


# ---------------------------------------------------------------- 1. cue-switch predictor
def cue_items() -> tuple[dict, np.ndarray]:
    """Per blocked test sequence (folds pooled): {system: {metric: [n]}} seed-averaged, and the clusters [n]."""
    from bgm.cue.report import Table
    evals = [json.loads((CUE_ROOT / f"fold{k}" / "eval.json").read_text()) for k in range(5)]
    T = Table(evals)
    out = {}
    for s in ("R4", "R0_tuned", "R0_stored", "R3"):
        n_true = T.per_case(s, "blocked", "in_film/n_true")
        f_in = f1_counts(T.counts(s, "blocked", "in_film/1"))
        f_in[n_true == 0] = np.nan                       # no in-film music change: F1 undefined for that sequence
        out[s] = {"in-film switch F1 ±1 s": f_in,
                  "all-boundary switch F1 ±1 s": f1_counts(T.counts(s, "blocked", "all/1")),
                  "part track accuracy": T.per_case(s, "blocked", "part_track_acc"),
                  "seam loudness jump, median LU": T.per_case(s, "blocked", "seam_jump_LU_median")}
    return out, T.clusters("blocked")


# ---------------------------------------------------------------- 2. final system (eval suite)
def suite_items() -> tuple[dict, dict]:
    """{key: [n] seed-averaged, folds pooled} for the eval-suite keys used here, and {kind: clusters}."""
    from bgm.analysis.eval_suite_report import Pool, _film, _load
    P = Pool(_load(), _film())
    keys = {}
    for s in ("viterbi", "greedy", "random"):
        for m in ("F1@2.0", "TBIoU", "DTW", "seam_LU"):
            keys[f"trans/{s}/{m}"] = "concat"
    for s in ("viterbi", "greedy", "random", "original", "viterbi_sync", "random_sync"):
        for m in ("LB", "LB_shift", "twLB", "twLB_shift", "in_LB", "in_LB_shift", "in_twLB", "in_twLB_shift", "IB", "KL"):
            keys[f"align/{s}/{m}"] = "video"
    vals = {k: P.pooled(k, kind)[0] for k, kind in keys.items()}
    cl = {"video": P.pooled("align/viterbi/IB", "video")[1], "concat": P.pooled("trans/viterbi/TBIoU", "concat")[1]}
    for s in ("viterbi", "greedy", "random", "original", "viterbi_sync", "random_sync"):
        for p in ("", "in_"):
            vals[f"align/{s}/{p}LBx"] = vals[f"align/{s}/{p}LB"] - vals[f"align/{s}/{p}LB_shift"]
            vals[f"align/{s}/{p}twLBx"] = vals[f"align/{s}/{p}twLB"] - vals[f"align/{s}/{p}twLB_shift"]
    return vals, cl


# ---------------------------------------------------------------- report
def _row(name: str, higher: bool, a, b, cl) -> dict:
    per = paired_wilcoxon(a, b)
    clu = paired_wilcoxon(a, b, cl) if cl is not None else None
    return {"name": name, "higher": higher, "A": float(np.nanmean(a[np.isfinite(b)])), "B": float(np.nanmean(b[np.isfinite(a)])),
            "item": per, "cluster": clu}


def _table(title: str, la: str, lb: str, rows: list[dict], unit: str, cunit: str) -> list[str]:
    for key in ("item", "cluster"):
        ps = [r[key]["p"] for r in rows if r[key] is not None]
        adj = holm(ps)
        for r, q in zip([r for r in rows if r[key] is not None], adj):
            r[key]["p_holm"] = q
    L = [f"#### {title}", "",
         f"A = {la}; B = {lb}. Δ = A − B per {unit}; wins / losses = {unit}s where A is higher / lower (ties dropped).", "",
         f"| metric | better | A | B | mean Δ | median Δ | wins / losses | r | p ({unit}) | p Holm | p ({cunit}, n) | p Holm | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        it, cu = r["item"], r["cluster"]
        good = (it["mean"] > 0) == r["higher"]
        sig_i = it["p_holm"] < 0.05
        sig_c = sig_i if cu is None else cu["p_holm"] < 0.05     # concatenations are already independent units
        if sig_i and sig_c:
            verdict = ("**A better**" if good else "**A worse**")
        elif sig_i:
            verdict = ("A better" if good else "A worse") + f" per {unit} only"
        else:
            verdict = "no difference"
        cu_s = (f"{cu['p']:.2g} ({cu['n']})", f"{cu['p_holm']:.2g}") if cu is not None else ("—", "—")
        L.append(f"| {r['name']} | {'higher' if r['higher'] else 'lower'} | {r['A']:.3f} | {r['B']:.3f} | {it['mean']:+.3f} | "
                 f"{it['median']:+.3f} | {it['wins']} / {it['losses']} | {it['r']:+.2f} | {it['p']:.2g} | {it['p_holm']:.2g} | "
                 f"{cu_s[0]} | {cu_s[1]} | {verdict} |")
    return L + [""]


def write_report() -> Path:
    L = ["# Paired Wilcoxon signed-rank tests (D112)", "",
         "Two-sided Wilcoxon signed-rank test on per-item differences A − B (both systems on the same test items; "
         "each item's value averaged over its seeds first). Zero differences dropped. **p Holm**: Holm-adjusted "
         "within each table. **Per cluster**: differences averaged within each film group first (sequences: their "
         "union-find film cluster), then tested over clusters, because items from one film are not independent; "
         "a result is called **A better / worse** only when both versions agree (p Holm < 0.05). "
         "r = matched-pairs rank-biserial effect size (+1: A higher on every item). Code: src/bgm/analysis/wilcoxon.py.", ""]

    # 1. cue predictor
    C, cl = cue_items()
    L += ["## 1. With vs without the learned cue-switch predictor (D78–D81)", "",
          "Blocked test sequences of runs/filmsplit/cue: 1,248 pairs, **film split** (the only split the predictor was "
          "evaluated on; the D75 emission is identical for every system, so only the switch decisions differ). "
          "R4 = learned predictor with context, 3 seeds; R0 = the same Viterbi decoder with fixed switch / jump "
          "penalties and no predictor. In-film switch F1 skips sequences without an in-film music change. F1 here is the "
          "mean of per-sequence F1 (the unit the test needs); results/cue_switch.md reports F1 pooled over sequences "
          "(R4 0.507), so the levels differ slightly.", ""]
    for b, lb in (("R0_tuned", "R0, fixed penalties re-tuned on val (the fair ablation)"),
                  ("R0_stored", "R0, the stored D75 penalties (the shipped system, `cue.enabled: false`)")):
        rows = [_row(m, m != "seam loudness jump, median LU", C["R4"][m], C[b][m], cl) for m in C["R4"]]
        L += _table(f"R4 vs {b}", "R4, learned predictor + context", lb, rows, "sequence", "cluster")

    # 2. final system
    V, CL = suite_items()
    L += ["## 2. Final system (D75) vs ablations and controls (title split, D111)", "",
          "runs/titlesplit/eval_suite: 982 test videos (898 with cuts for beat alignment), 250 test concatenations "
          "(clustered as units, so the cluster test equals the item test for transitions). LB / tw-LB = excess over "
          "shifted beats (D111).", ""]

    def suite(a, b, la, lb, metrics):
        rows = []
        for name, key, higher, kind in metrics:
            x, y = V[key.format(a)], V[key.format(b)]
            rows.append(_row(name, higher, x, y, None if kind == "concat" else CL[kind]))
        unit = "concatenation" if all(k == "concat" for *_, k in metrics) else "video"
        return _table(f"{la} vs {lb}", la, lb, rows, unit, "film group")

    TR = [("switch F1 ±2.0 s", "trans/{}/F1@2.0", True, "concat"), ("TBIoU", "trans/{}/TBIoU", True, "concat"),
          ("DTW to original CLAP sequence", "trans/{}/DTW", False, "concat"),
          ("seam loudness jump, LU", "trans/{}/seam_LU", False, "concat")]
    AL = [("LB excess (all cuts)", "align/{}/LBx", True, "video"), ("tw-LB excess (all cuts)", "align/{}/twLBx", True, "video"),
          ("IB (ImageBind video–music)", "align/{}/IB", True, "video"), ("KL to original (PANNs)", "align/{}/KL", False, "video")]
    SY = [("LB excess (within-piece cuts)", "align/{}/in_LBx", True, "video"),
          ("tw-LB excess (within-piece cuts)", "align/{}/in_twLBx", True, "video")]
    L += ["### Transitions (per test concatenation; each is its own cluster, as in D111)", ""]
    L += suite("viterbi", "greedy", "final (Viterbi)", "greedy (no transition model)", TR)
    L += suite("viterbi", "random", "final (Viterbi)", "random music", TR)
    L += ["### Alignment and content match (per test video)", ""]
    L += suite("viterbi", "greedy", "final (Viterbi)", "greedy (no transition model)", AL)
    L += suite("viterbi", "random", "final (Viterbi)", "random music", AL)
    L += suite("viterbi", "original", "final (Viterbi)", "original music", AL[:3])
    L += suite("viterbi_sync", "random_sync", "final + beat sync", "random music + the same beat sync", SY)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    log.info("wrote %s", OUT)
    return OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    write_report()
