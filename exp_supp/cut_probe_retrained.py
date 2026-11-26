"""Shot cuts inside continuous clips, fully on the cleaned dataset (D152).

Differs from exp_supp/cut_probe_cleaned.py (D147), which scored the 112 original test sequences with heads trained on
the 1,248-pair data. Here everything is cleaned:

  shots, BaSSL  cleaned_dataset (D148b)
  heads         runs/cleaned/segmentation/{emb_p, combo, emb_p_dur}, trained on the cleaned train clips, 5 folds x 3 seeds
  candidates    every TransNetV2 boundary inside a clip of the cleaned frozen test sequences
  labels        D147's cleaned labels (runs/exp_supp/cut_probe/labels_cleaned.csv; calibration on each fold's cleaned
                train clips), matched by cleaned pair id within 0.12 s

Test clips are only scored; no tuning. CIs: cluster bootstrap over film groups, 10,000 resamples.

`.venv/bin/python -m exp_supp.cut_probe_retrained {train|eval}`
    -> runs/exp_supp/cut_probe_retrained/, exp_supp/cut_probe_retrained.{md,tex}
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from exp_supp.common import RUNS, log_run
from exp_supp.cut_probe_cleaned import VARIANTS, match, summarize
from exp_supp.rerank.clean_t3 import seg_cfg

log = logging.getLogger("exp_supp.cut_probe_retrained")
OUT = RUNS / "cut_probe_retrained"
LABELS = RUNS / "cut_probe" / "labels_cleaned.csv"
NAMES = {"tn_bassl": "BaSSL, original head", "emb_p": "BaSSL, trained head", "combo": "BaSSL, trained head (ours)",
         "emb_p_dur": "Ours w/o contrast/BiGRU, + duration and motion"}


def stage_train() -> None:
    from bgm.segmentation.train_head import run_train
    run_train(seg_cfg(), variants=("emb_p", "emb_p_dur", "combo"))        # combo exists (D148b) and is skipped


def candidates() -> pd.DataFrame:
    """Every in-clip TransNetV2 boundary of the cleaned frozen test sequences, with each variant's score."""
    p = OUT / "candidates.pkl"
    if p.exists():
        return pd.read_pickle(p)
    import numpy as np
    import torch
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold
    from bgm.segmentation.train_head import SegData
    data = SegData(seg_cfg(), need_siglip=True)
    bassl = BaSSL("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for k in range(5):
        test = Fold(data, bassl, k, "test", {v: v for v in VARIANTS if v != "tn_bassl"})
        for i, q in enumerate(test.seqs):
            c = test.cT[i]
            start = np.concatenate([[0.0], c.joins])
            for b in np.flatnonzero(~c.is_join):
                rows.append({"fold": k, "seq": i, "b": int(b), "pair": q[c.shot_part[b]],
                             "t": float(c.btimes[b] - start[c.shot_part[b]]),
                             **{v: float(test.score[v][i][b]) for v in VARIANTS}})
        log.warning("fold %d: %d in-clip candidates so far", k, len(rows))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return df


def fmt(v: float) -> str:
    return f"{v:+.3f}".replace("-", "−")


def write(s: dict, n_cand: int, n_lab: int) -> None:
    L = ["# Shot cuts inside continuous clips, fully cleaned pipeline (D152)", "",
         f"{s['n']:,} matched cuts, {s['n_pos']} with a music change (base rate {s['base_rate']:.3f}), {s['n_clusters']} film "
         f"groups. {n_cand:,} TransNetV2 in-clip candidates; {n_lab:,} labelled cuts. Heads trained on cleaned train clips.", "",
         "| Segmenter | AP [95% CI] | AP − chance [95% CI] | AUROC [95% CI] |", "|---|---|---|---|",
         f"| Random | {s['base_rate']:.3f} | 0 | 0.500 |"]
    T = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Segmenter & AP [95\% CI] & AP $-$ chance [95\% CI] & AUROC [95\% CI] \\",
         r"\midrule", f"Random & {s['base_rate']:.3f} & 0 & 0.500 \\\\"]
    for v in VARIANTS:
        r = s[v]
        gap = r["AP"] - s["base_rate"]
        a, c, u = r["AP_ci"], r["AP_minus_chance_ci"], r["AUROC_ci"]
        L.append(f"| {NAMES[v]} | {r['AP']:.3f} [{a[0]:.3f}, {a[1]:.3f}] | {fmt(gap)} [{fmt(c[0])}, {fmt(c[1])}] | "
                 f"{r['AUROC']:.3f} [{u[0]:.3f}, {u[1]:.3f}] |")
        name = rf"\textbf{{{NAMES[v]}}}" if v == "combo" else NAMES[v]
        tx = lambda x: fmt(x).replace("−", "$-$")  # noqa: E731
        T.append(f"{name} & {r['AP']:.3f} [{a[0]:.3f}, {a[1]:.3f}] & {tx(gap)} [{tx(c[0])}, {tx(c[1])}] & "
                 f"{r['AUROC']:.3f} [{u[0]:.3f}, {u[1]:.3f}] \\\\")
    T += [r"\bottomrule", r"\end{tabular}", ""]
    (RUNS.parent.parent / "exp_supp" / "cut_probe_retrained.md").write_text("\n".join(L) + "\n")
    (RUNS.parent.parent / "exp_supp" / "cut_probe_retrained.tex").write_text("\n".join(T))


def stage_eval() -> None:
    cand = candidates()
    lab = pd.read_csv(LABELS)
    m = match(cand, lab, "pair", "t")
    s = summarize(m)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"cleaned_heads": s, "n_candidates": len(cand), "n_labels": len(lab),
                                                  "n_label_pos": int(lab.label.sum())}, default=float, indent=1))
    m.to_csv(OUT / "matched.csv", index=False)
    for v in VARIANTS:
        log_run("suppC_cut_probe_retrained", f"{v}/cleaned_heads", -1, 0, "test(all folds)",
                {"variant": v, "labels": "cleaned", "heads": "cleaned", "tol_s": 0.12,
                 "candidates": "TransNetV2 in-clip boundaries of the cleaned frozen test sequences"},
                {"n": s["n"], "n_pos": s["n_pos"], **s[v]})
    write(s, len(cand), len(lab))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval"])
    a = ap.parse_args()
    stage_train() if a.stage == "train" else stage_eval()
