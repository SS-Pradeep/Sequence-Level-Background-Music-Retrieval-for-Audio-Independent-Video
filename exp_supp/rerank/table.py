"""Tables of the re-ranker study (D148): exp_supp/rerank/rerank_table.{md,tex}."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from exp_supp.common import fmt_ci, paired_ci
from exp_supp.rerank import methods as M
from exp_supp.rerank.run import EVAL, GRIDS, N_FOLDS, OUT

REF = "No re-ranking"
PROTO = {"seg": "Segment retrieval (segment_all: one hit, gallery = every test segment's aligned window, ≈ 400)",
         "trk": "Track retrieval (track_test_fold: gallery = the test fold's music files, ≈ 180; any file of the track is a hit)"}


def load_all() -> list[dict]:
    recs = []
    for k in range(N_FOLDS):
        with open(EVAL / f"fold{k}.pkl", "rb") as fh:
            r = pickle.load(fh)
        ltr = EVAL / f"fold{k}_ltr.pkl"
        if ltr.exists():
            with open(ltr, "rb") as fh:
                extra = pickle.load(fh)
            for kind in ("single", "ens"):
                r[kind] |= extra[kind]
        recs.append(r)
    return recs


def names(recs) -> list[str]:
    out = list(GRIDS)
    for n in recs[0]["single"]:
        if n not in out:
            out.append(n)
    return out


def pick_str(picks: list[dict]) -> str:
    if not picks or not picks[0]:
        return "—"
    c = Counter(tuple(sorted(p.items())) for p in picks).most_common(1)[0][0]
    return ", ".join(f"{a} {b}" for a, b in c if a != "how")


def gather(recs, kind: str, name: str, proto: str):
    """Per run summaries, per-query hit@10 / reciprocal rank (single: averaged over seeds), picks, ms/query."""
    runs, h10, rr, picks, ms = [], [], [], [], []
    for rec in recs:
        d = rec[kind].get(name, {}).get(proto)
        if d is None:
            return None
        seeds = sorted(d)
        R = np.stack([d[s]["ranks"] for s in seeds])
        runs += [M.summary(x) for x in R]
        h10.append((R <= 10).mean(0))
        rr.append((1 / R).mean(0))
        picks += [d[s]["pick"] for s in seeds]
        ms += [d[s]["ms_per_query"] for s in seeds]
    return {"runs": runs, "h10": np.concatenate(h10), "rr": np.concatenate(rr), "picks": picks, "ms": float(np.mean(ms))}


def write_table() -> str:
    recs = load_all()
    nm = names(recs)
    n_single = len(recs[0]["single"][REF]["seg"])
    L = ["# Re-rankers on the top-20 shortlist (supplementary, D148)", "",
         "Cleaned dataset (cleaned_dataset/, D115), title split (cleaned_dataset/manifests/folds_title.json), 5 folds. "
         "Retrieval heads fixed: the D116 siglip-both InfoNCE heads (runs/cleaned/semantic/siglip-both). Hit rule = "
         "track_id (the M5 / D145 protocols). Only the base top-20 shortlist of each query is re-ordered; everything "
         "below it keeps its base order, so a hit outside the top 20 keeps its rank. Ties after re-ranking are broken by "
         "the base score. Every parameter is chosen per run on the fold's val films (criterion R@10 + 0.1·MRR), then "
         "test is scored once. 'Other queries' for k-reciprocal, mutual kNN and dual softmax = the fold's training clips "
         "(track) / 1,500 training segments (segment), never val or test queries.", "",
         f"Single head = one seed, mean ± std over {N_FOLDS * n_single} runs (5 folds × {n_single} seeds); Ensemble = mean "
         "cosine of seeds 0–9 (the D116 ens10 rule), mean ± std over 5 folds. Δ = method − no re-ranking, paired per "
         "test query (single: per-query hit@10 / reciprocal rank averaged over seeds), 95% film-group cluster bootstrap, "
         "10,000 resamples; * = CI excludes 0. ms/query = re-ranking time only (numpy, CPU, base retrieval excluded).", ""]
    for proto in ("seg", "trk"):
        for kind, title in (("single", "single head"), ("ens", "10-seed ensemble")):
            L += [f"## {PROTO[proto]} — {title}", "",
                  "| re-ranker | R@1 | R@5 | R@10 | MRR | MedR | ms/query | Δ R@10 [95% CI] | Δ MRR [95% CI] | typical val pick |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            ref = gather(recs, kind, REF, proto)
            cl = np.concatenate([r[f"{proto}_cluster"] for r in recs])
            for n in nm:
                g = gather(recs, kind, n, proto)
                if g is None:
                    continue
                sm = lambda c: f"{np.mean([x[c] for x in g['runs']]):.3f} ± {np.std([x[c] for x in g['runs']], ddof=1):.3f}"  # noqa: E731
                med = f"{np.mean([x['MedR'] for x in g['runs']]):.1f}"
                if n == REF:
                    d1 = d2 = "—"
                else:
                    d1 = fmt_ci(paired_ci(g["h10"] - ref["h10"], cl))
                    d2 = fmt_ci(paired_ci(g["rr"] - ref["rr"], cl))
                L.append(f"| {n} | {sm('R1')} | {sm('R5')} | {sm('R10')} | {sm('MRR')} | {med} | {g['ms']:.2f} | {d1} | {d2} | "
                         f"{pick_str(g['picks'])} |")
            L.append("")
    md = "\n".join(L) + "\n"
    (OUT / "rerank_table.md").write_text(md)
    write_tex(recs, nm)
    print(md)
    return md


def write_tex(recs, nm) -> None:
    """Compact LaTeX: R@10 and MRR per protocol, single and ensemble, Δ R@10 with CI."""
    t = lambda s: s.replace("−", "-")  # noqa: E731
    rows = []
    for n in nm:
        cells = []
        for proto in ("seg", "trk"):
            for kind in ("single", "ens"):
                g = gather(recs, kind, n, proto)
                if g is None:
                    cells += ["--", "--"]
                    continue
                ref = gather(recs, kind, REF, proto)
                cl = np.concatenate([r[f"{proto}_cluster"] for r in recs])
                r10 = np.mean([x["R10"] for x in g["runs"]])
                if n == REF:
                    cells += [f"{r10:.3f}", "--"]
                else:
                    b = paired_ci(g["h10"] - ref["h10"], cl)
                    sig = b["lo"] > 0 or b["hi"] < 0
                    cells += [f"{r10:.3f}", t((r"$\mathbf{%+.3f}$" if sig else r"$%+.3f$") % b["mean"])]
        rows.append(n.replace("α", r"$\alpha$").replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Re-rankers on the top-20 shortlist (cleaned title split; heads fixed; parameters tuned on validation).",
           r"R@10 and paired $\Delta$ R@10 vs.\ no re-ranking; bold = 95\% film-level cluster-bootstrap CI excludes 0.}",
           r"\label{tab:supp_rerank}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r" & \multicolumn{4}{c}{Segment} & \multicolumn{4}{c}{Track} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & \multicolumn{2}{c}{Single} & \multicolumn{2}{c}{Ens.} & \multicolumn{2}{c}{Single} & \multicolumn{2}{c}{Ens.} \\",
           r"Re-ranker & R@10 & $\Delta$ & R@10 & $\Delta$ & R@10 & $\Delta$ & R@10 & $\Delta$ \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "rerank_table.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    write_table()
