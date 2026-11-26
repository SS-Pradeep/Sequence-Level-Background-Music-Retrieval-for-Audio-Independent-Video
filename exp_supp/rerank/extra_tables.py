"""Extra tables of the re-ranker study (D148c): exp_supp/rerank/extra_tables.{md,tex}.

`.venv/bin/python -m exp_supp.rerank.extra_tables`

From saved results only (runs/exp_supp/rerank/eval, ksweep, clean_t3/hmm); nothing re-tuned here.
  T1 headroom: what an oracle re-ordering of the top-K could reach (R@1 = R@10 = R@K) vs the best re-ranker
  T2 rank movement of the hit inside the shortlist (10-seed ensemble, test queries pooled over folds)
  T3 per-fold R@10 (10-seed ensemble)
  T4 shortlist size K in {10, 20, 50, 100} (10-seed ensemble; parameters re-chosen on val per K)
  T5 decoder per fold (cleaned 1,011 pairs, 92 sequences)
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from exp_supp.common import fmt_ci, paired_ci
from exp_supp.rerank.clean_t3 import ROOT as CT3
from exp_supp.rerank.hmm import METHODS as HMM_METHODS, METRICS
from exp_supp.rerank.ksweep import KS, OUT as KSW
from exp_supp.rerank.run import N_FOLDS, OUT
from exp_supp.rerank.table import REF, load_all, names
from stitching.report import _derive

PN = {"seg": "Segment", "trk": "Track"}
KN = {"single": "single head", "ens": "10-seed ens."}


def ranks(rec, kind, name, proto):
    """[n_seeds, n_queries] test ranks of one fold, or None."""
    d = rec[kind].get(name, {}).get(proto)
    return None if d is None else np.stack([d[s]["ranks"] for s in sorted(d)])


def pooled(recs, kind, name, proto):
    rs = [ranks(r, kind, name, proto) for r in recs]
    return None if any(x is None for x in rs) else np.concatenate(rs, 1)       # [seeds, all test queries]


def r_at(R, k):
    return float((R <= k).mean())


def t1_headroom(recs, nm) -> list[str]:
    L = ["## T1. Headroom of re-ranking the top 20", "",
         "Oracle = the hit inside the shortlist moved to rank 1 (an ideal re-ranker): oracle R@1 = oracle R@10 = base R@20. "
         "'Best' = the re-ranker with the highest test R@10 in that column (picked after the fact, so optimistic). "
         "Captured = (best − base) / (oracle R@10 − base). Single head: means over 50 runs. R@k here is pooled over all test "
         "queries, so it can differ from the main table's mean over folds in the third decimal (ensemble 0.124 / 0.305 vs 0.125 / 0.306).", "",
         "| protocol | base R@1 | base R@10 | base R@20 = oracle R@10 | hit outside top 20 | MedR | best re-ranker | best R@10 | captured |",
         "|---|---|---|---|---|---|---|---|---|"]
    for proto in ("seg", "trk"):
        for kind in ("single", "ens"):
            B = pooled(recs, kind, REF, proto)
            b10, b20 = r_at(B, 10), r_at(B, 20)
            best = max(((n, r_at(pooled(recs, kind, n, proto), 10)) for n in nm if n != REF and pooled(recs, kind, n, proto) is not None),
                       key=lambda x: x[1])
            med = float(np.mean([np.median(x) for x in B]))
            L.append(f"| {PN[proto]}, {KN[kind]} | {r_at(B, 1):.3f} | {b10:.3f} | {b20:.3f} | {1 - b20:.3f} | {med:.1f} | {best[0]} | "
                     f"{best[1]:.3f} | {100 * (best[1] - b10) / (b20 - b10):.0f}% |")
    return L + [""]


def t2_movement(recs, nm) -> list[str]:
    L = ["## T2. Where the hit moves inside the shortlist (10-seed ensemble, test queries pooled over folds)", "",
         "Only queries whose hit is in the base top 20 can move. ↑ / ↓ = rank improved / worsened; into / out of top 10 = "
         "crossing rank 10; to / from rank 1. Net top-10 = into − out.", ""]
    for proto in ("seg", "trk"):
        B = pooled(recs, "ens", REF, proto)[0]
        n_in = int((B <= 20).sum())
        L += [f"### {PN[proto]} ({len(B)} test queries, {n_in} with the hit in the base top 20)", "",
              "| re-ranker | ↑ | ↓ | = | into top 10 | out of top 10 | net top 10 | to rank 1 | from rank 1 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in nm:
            if n == REF:
                continue
            X = pooled(recs, "ens", n, proto)
            if X is None:
                continue
            X = X[0]
            m = B <= 20
            up, dn = int((X[m] < B[m]).sum()), int((X[m] > B[m]).sum())
            into, out = int(((B > 10) & (X <= 10)).sum()), int(((B <= 10) & (X > 10)).sum())
            to1, fr1 = int(((B > 1) & (X == 1)).sum()), int(((B == 1) & (X > 1)).sum())
            L.append(f"| {n} | {up} | {dn} | {int(m.sum()) - up - dn} | {into} | {out} | {into - out:+d} | {to1} | {fr1} |")
        L.append("")
    return L


def t3_folds(recs, nm) -> list[str]:
    L = ["## T3. R@10 per fold (10-seed ensemble)", "",
         "Folds better / worse = folds where the method's R@10 is above / below no re-ranking.", ""]
    for proto in ("seg", "trk"):
        L += [f"### {PN[proto]}", "", "| re-ranker | " + " | ".join(f"fold {k}" for k in range(N_FOLDS)) + " | mean | folds better / worse |",
              "|---|" + "---|" * (N_FOLDS + 2)]
        base = [r_at(ranks(r, "ens", REF, proto), 10) for r in recs]
        for n in nm:
            if ranks(recs[0], "ens", n, proto) is None:
                continue
            v = [r_at(ranks(r, "ens", n, proto), 10) for r in recs]
            bw = "—" if n == REF else f"{sum(a > b + 1e-12 for a, b in zip(v, base))} / {sum(a < b - 1e-12 for a, b in zip(v, base))}"
            L.append(f"| {n} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f} | {bw} |")
        L.append("")
    return L


def load_ksweep() -> list[dict]:
    out = []
    for k in range(N_FOLDS):
        with open(KSW / f"fold{k}.pkl", "rb") as fh:
            out.append(pickle.load(fh))
    return out


def t4_ksweep(recs, nm, ks) -> tuple[list[str], dict]:
    L = ["## T4. Shortlist size (10-seed ensemble; every parameter re-chosen on val for each K)", "",
         "Δ R@10 vs no re-ranking (which does not depend on K), paired per test query, 95% film-group cluster bootstrap; "
         "* = CI excludes 0. Oracle R@10 = share of queries whose hit is in the base top K (for K ≥ 10). K = 20 = the main "
         "table (reproduced exactly). K = 10 cannot change R@10 by construction (re-ordering the top 10 keeps the same 10 items) "
         "and is listed as a check. LambdaMART was trained for K = 20 only and is not swept.", ""]
    tex = {}
    for proto in ("seg", "trk"):
        cl = np.concatenate([r[f"{proto}_cluster"] for r in recs])
        B = pooled(recs, "ens", REF, proto)[0]
        L += [f"### {PN[proto]} (no re-ranking R@10 = {r_at(B, 10):.3f})", "",
              "| re-ranker | " + " | ".join(f"K = {K}" for K in KS) + " |", "|---|" + "---|" * len(KS),
              "| oracle R@10 | " + " | ".join(f"{r_at(B, K):.3f}" for K in KS) + " |"]
        tex[proto] = {}
        for n in nm:
            if n == REF or proto not in ks[0]["K"][KS[0]].get(n, {}):
                continue
            cells, tc = [], []
            for K in KS:
                X = np.concatenate([f["K"][K][n][proto]["ranks"] for f in ks])
                ci = paired_ci((X <= 10).astype(float) - (B <= 10), cl)
                cells.append(fmt_ci(ci))
                tc.append(ci)
            L.append(f"| {n} | " + " | ".join(cells) + " |")
            tex[proto][n] = tc
        L.append("")
    return L, tex


def t5_decoder() -> list[str]:
    recs = [json.loads((CT3 / "hmm" / f"fold{k}.json").read_text()) for k in range(N_FOLDS)]
    systems = [("Final HMM", lambda r: r["final_rows"])] + [(f"+ {n}", (lambda n: lambda r: r["methods"][n]["rows"])(n)) for n in HMM_METHODS]
    L = ["## T5. Decoder per fold (cleaned 1,011 pairs; HMM, gold-path prior, α tuned; 92 test sequences)", "",
         "| fold | seq. | system | " + " | ".join(METRICS) + " | λ (val) |", "|---|---|---|" + "---|" * (len(METRICS) + 1)]
    for r in recs:
        for nm, get in systems:
            rows = get(r)
            lam = r["final_params"]["lam"] if nm == "Final HMM" else r["methods"][nm[2:]]["params"]["lam"]
            vals = [np.nanmean([_derive(x)[m] for x in rows]) for m in METRICS]
            L.append(f"| {r['fold']} | {len(rows)} | {nm} | " + " | ".join(f"{v:.{2 if 'min' in m else 3}f}" for v, m in zip(vals, METRICS))
                     + f" | {lam:.3g} |")
    return L + [""]


def write_tex(recs, nm, kt) -> str:
    t = lambda s: s.replace("−", "-")  # noqa: E731
    rows = []
    for proto in ("seg", "trk"):
        for kind in ("single", "ens"):
            B = pooled(recs, kind, REF, proto)
            b10, b20 = r_at(B, 10), r_at(B, 20)
            best = max(((n, r_at(pooled(recs, kind, n, proto), 10)) for n in nm if n != REF and pooled(recs, kind, n, proto) is not None),
                       key=lambda x: x[1])
            rows.append(f"{PN[proto]}, {KN[kind]} & {r_at(B, 1):.3f} & {b10:.3f} & {b20:.3f} & {1 - b20:.3f} & "
                        f"{best[0].replace('&', chr(92) + '&')} & {best[1]:.3f} & {100 * (best[1] - b10) / (b20 - b10):.0f}\\% \\\\")
    a = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Headroom of re-ranking the top-20 shortlist. An oracle that moves the hit to rank 1 reaches R@10 = base R@20;",
         r"`best' is the strongest re-ranker on test (picked after the fact).}", r"\label{tab:supp_rerank_headroom}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r"Protocol & R@1 & R@10 & Oracle R@10 & Outside top-20 & Best re-ranker & R@10 & Captured \\", r"\midrule",
         *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    body = []
    for n in kt["trk"] | kt["seg"]:
        cells = []
        for proto in ("seg", "trk"):
            if n not in kt[proto]:
                cells += ["--"] * len(KS)
                continue
            for ci in kt[proto][n]:
                sig = ci["lo"] > 0 or ci["hi"] < 0
                cells.append(t((r"$\mathbf{%+.3f}$" if sig else r"$%+.3f$") % ci["mean"]))
        body.append(n.replace("α", r"$\alpha$") + " & " + " & ".join(cells) + r" \\")
    b = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Shortlist size: $\Delta$ R@10 vs.\ no re-ranking for the 10-seed ensemble when the re-ranker may re-order the",
         r"top $K$ candidates (parameters re-chosen on validation for each $K$). Bold = 95\% film-level cluster-bootstrap CI excludes 0.}",
         r"\label{tab:supp_rerank_k}", r"\begin{tabular}{@{}l" + "c" * (2 * len(KS)) + "@{}}", r"\toprule",
         r" & \multicolumn{%d}{c}{Segment} & \multicolumn{%d}{c}{Track} \\" % (len(KS), len(KS)),
         r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(KS), 2 + len(KS), 1 + 2 * len(KS)),
         "Re-ranker & " + " & ".join(f"$K{{=}}{K}$" for K in KS * 2) + r" \\", r"\midrule",
         *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(a + b) + "\n"


def main() -> None:
    recs = load_all()
    nm = names(recs)
    ks = load_ksweep()
    k4, kt = t4_ksweep(recs, nm, ks)
    L = ["# Re-ranker study: extra tables (D148c)", "",
         "Cleaned dataset (1,011 pairs), title split, 5 folds, hit rule track_id; same runs as exp_supp/rerank/rerank_table.md "
         "(T1–T3), the shortlist-size sweep exp_supp/rerank/ksweep.py (T4) and the cleaned decoder study "
         "exp_supp/rerank/rerank_hmm_table_cleaned.md (T5).", ""]
    L += t1_headroom(recs, nm) + t2_movement(recs, nm) + t3_folds(recs, nm) + k4 + t5_decoder()
    (OUT / "extra_tables.md").write_text("\n".join(L) + "\n")
    (OUT / "extra_tables.tex").write_text(write_tex(recs, nm, kt))
    print("\n".join(L))


if __name__ == "__main__":
    main()
