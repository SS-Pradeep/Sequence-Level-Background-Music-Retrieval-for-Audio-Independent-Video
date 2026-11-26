"""Supplementary C: music changes at shot cuts INSIDE continuous test clips, on the cleaned dataset.

Paper source (Table 4 of results/segmentation_comparison.md, D142 probe in bgm.segmentation.improve): every
TransNetV2 boundary inside a test clip of the 112 frozen title-split test sequences, scored by the frozen segmentation
variants, labelled with the D103 audio labels (music change within +-1 s: calibrated MERT jump or a drop to silence).
That run used the 1,248-pair data. Here:

  candidates  identical (same sequences, same frozen heads; nothing retrained)
  labels      rebuilt on cleaned_dataset: per fold, the MERT-jump calibration is refit on the CLEANED training clips
              (bgm.cue.cuts.calibrate, same settings, seed = fold), then bgm.cue.cuts.label_cuts on every cleaned test
              clip (cuts >= 2 s from the cleaned clip's edges). A cleaned cut is mapped to its source clip by
              meta_trim_source_id + meta_trim_start_s and matched to a candidate within 0.12 s (the D142 tolerance).
              Cuts in excluded clips or trimmed-away spans drop out.
  sanity      the original labels through the same code must reproduce the paper's 3,467 cuts / 452 changes.

CIs: cluster bootstrap over film groups (cleaned film_group), 10,000 resamples. Tests on test folds only; no tuning.

`.venv/bin/python -m exp_supp.cut_probe_cleaned` -> exp_supp/cut_probe_cleaned.{md,tex}, runs/exp_supp/cut_probe/
"""
from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pandas as pd

from exp_supp.common import RUNS, log_run

log = logging.getLogger("exp_supp.cut_probe")
OUT = RUNS / "cut_probe"
VARIANTS = ["tn_bassl", "emb_p", "combo", "emb_p_dur"]
NAMES = {"tn_bassl": "TransNetV2 + BaSSL, original head", "emb_p": "TransNetV2 + BaSSL, trained head (4)",
         "combo": "TransNetV2 + BaSSL, trained head (ours, C)", "emb_p_dur": "4b + duration, motion"}
CLEAN = ["splits.folds_file=folds_title.json", "paths.manifests=cleaned_dataset/manifests",
         "paths.interim=cleaned_dataset/interim", "paths.features=cleaned_dataset/features",
         "paths.runs=runs/cleaned", "paths.results=results/cleaned"]
TOL = 0.12


def candidate_scores() -> pd.DataFrame:
    """Every in-clip TransNetV2 boundary of the frozen test sequences with each variant's score (as improve.probe_rows)."""
    p = OUT / "candidates.pkl"
    if p.exists():
        return pd.read_pickle(p)
    import torch
    from bgm.config import load_config
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold
    from bgm.segmentation.train_head import SegData
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    data = SegData(cfg, need_siglip=True)
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
        log.info("fold %d: %d in-clip candidates so far", k, len(rows))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return df


def cleaned_labels() -> tuple[pd.DataFrame, dict]:
    p = OUT / "labels_cleaned.csv"
    if p.exists():
        return pd.read_csv(p), json.loads((OUT / "calibration_cleaned.json").read_text())
    from bgm.config import load_config
    from bgm.cue.cuts import calibrate, label_cuts
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    cfg = load_config("configs/default.yaml", CLEAN)
    store = FeatureStore(cfg)
    pairs = manifest.read_pairs(cfg).set_index("pair_id")
    fg = manifest.read_folds(cfg)["film_group"]
    c = cfg.cue.cuts
    out, info = [], {}
    for k in range(5):
        f = fold_ids(cfg, k)
        cal, info[k] = calibrate(store, f["train"], c.half_s, c.n_joins, k)
        df = label_cuts(store, list(f["test"]), cal, c)
        df["fold"] = k
        df["group"] = df.pair.map(fg)
        df["source"] = df.pair.map(pairs["meta_trim_source_id"]).fillna(df.pair)
        df["t_source"] = df.t + df.pair.map(pairs["meta_trim_start_s"]).fillna(0.0).astype(float)
        out.append(df)
        log.info("fold %d cleaned labels: %d cuts, %d changes; calibration %s", k, len(df), int(df.label.sum()), info[k])
    lab = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lab.to_csv(p, index=False)
    (OUT / "calibration_cleaned.json").write_text(json.dumps(info, default=float, indent=1))
    return lab, info


def match(cand: pd.DataFrame, lab: pd.DataFrame, key_pair: str, key_t: str) -> pd.DataFrame:
    """Each candidate matched to its clip's nearest labelled cut within TOL (as improve.probe_rows)."""
    by = {p: g for p, g in lab.groupby(key_pair)}
    rows = []
    for r in cand.itertuples():
        g = by.get(r.pair)
        if g is None:
            continue
        g = g[g.fold == r.fold]
        if not len(g):
            continue
        j = np.abs(g[key_t].to_numpy() - r.t)
        if j.min() > TOL:
            continue
        m = g.iloc[int(j.argmin())]
        rows.append({"label": int(m.label), "group": m.group, "fold": r.fold, **{v: getattr(r, v) for v in VARIANTS}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_boot: int = 10000, seed: int = 1) -> dict:
    from sklearn.metrics import roc_auc_score
    from bgm.segmentation import evaluate as ev
    y = df.label.to_numpy()
    uniq, inv = np.unique(df.group.to_numpy().astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv] for _ in range(n_boot)])
    base_b = (W * y).sum(1) / W.sum(1)
    out = {"n": int(len(df)), "n_pos": int(y.sum()), "base_rate": float(y.mean()), "n_clusters": int(len(uniq))}
    for v in VARIANTS:
        s = df[v].to_numpy()
        o = np.argsort(-s, kind="stable")
        ys, Ws = y[o], W[:, o]
        tp, n = np.cumsum(Ws * ys, 1), np.cumsum(Ws, 1)
        ap_b = ((tp / np.maximum(n, 1e-9)) * Ws * ys).sum(1) / np.maximum((Ws * ys).sum(1), 1e-9)
        # AUROC per resample: weighted Mann-Whitney (ties count half)
        order = np.argsort(s, kind="stable")
        _, start = np.unique(s[order], return_index=True)
        au_b = []
        for w in W:
            wp, wn = w * y, w * (1 - y)
            wps, wns = wp[order], wn[order]
            neg_cum = np.r_[0, np.cumsum(np.add.reduceat(wns, start))][:-1]
            neg_tie = np.add.reduceat(wns, start)
            pos_grp = np.add.reduceat(wps, start)
            au_b.append(float((pos_grp * (neg_cum + 0.5 * neg_tie)).sum() / max(wp.sum() * wn.sum(), 1e-9)))
        au_b = np.array(au_b)
        out[v] = {"AP": ev.average_precision(s, y), "AP_ci": ev.ci(ap_b), "AP_minus_chance_ci": ev.ci(ap_b - base_b),
                  "AUROC": float(roc_auc_score(y, s)), "AUROC_ci": ev.ci(au_b)}
    return out


def main() -> None:
    cand = candidate_scores()
    old = pd.read_csv("runs/titlesplit/cut_changes/test_predictions.csv", usecols=["pair", "t", "label", "group", "fold"])
    m_old = match(cand, old, "pair", "t")
    s_old = summarize(m_old)
    log.info("sanity (original labels): %d cuts, %d changes; AUROC %s", s_old["n"], s_old["n_pos"],
             {v: round(s_old[v]["AUROC"], 3) for v in VARIANTS})
    assert (s_old["n"], s_old["n_pos"]) == (3467, 452), "original probe not reproduced"
    lab, cal = cleaned_labels()
    m_cl = match(cand, lab, "source", "t_source")
    s_cl = summarize(m_cl)
    res = {"original": s_old, "cleaned": s_cl, "n_cleaned_label_rows": int(len(lab)), "n_cleaned_label_pos": int(lab.label.sum()),
           "calibration": cal}
    (OUT / "summary.json").write_text(json.dumps(res, default=float, indent=1))
    for name, s in (("original_labels", s_old), ("cleaned", s_cl)):
        for v in VARIANTS:
            log_run("suppC_cut_probe", f"{v}/{name}", -1, 0, "test(all folds)", {"variant": v, "labels": name, "tol_s": TOL,
                    "candidates": "TransNetV2 in-clip boundaries of the 112 frozen title-split test sequences"},
                    {"n": s["n"], "n_pos": s["n_pos"], **s[v]})
    write(res)


def write(res: dict) -> None:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    o, c = res["original"], res["cleaned"]
    L = ["# Supplementary C: music changes at shot cuts inside continuous test clips (cleaned data)", "",
         "Candidates: every TransNetV2 boundary inside a test clip of the 112 frozen title-split test sequences, scored by the",
         "frozen segmentation variants (no retraining). Labels: D103 rule (music change within ±1 s: calibrated MERT jump or a",
         "drop to silence), rebuilt on cleaned_dataset with the calibration refit on each fold's cleaned training clips; cuts in",
         "excluded clips or trimmed-away spans drop out. 95% CI: cluster bootstrap over film groups, 10,000 resamples.", "",
         f"Cleaned: {c['n']:,} cuts, {c['n_pos']} with a music change (base rate {c['base_rate']:.3f} = AP of a random score), "
         f"{c['n_clusters']} film groups. Original data (paper): {o['n']:,} cuts, {o['n_pos']} changes (reproduced exactly).", "",
         "| Segmenter | AP [95% CI] | AP − chance [95% CI] | AUROC [95% CI] | AUROC, original data |", "|---|---|---|---|---|",
         f"| Random score | {c['base_rate']:.3f} | 0 | 0.500 | 0.500 |"]
    for v in VARIANTS:
        L.append(f"| {NAMES[v]} | {f(c[v]['AP'])} {ci(c[v]['AP_ci'])} | {c[v]['AP'] - c['base_rate']:+.3f} {ci(c[v]['AP_minus_chance_ci'])} | "
                 f"{f(c[v]['AUROC'])} {ci(c[v]['AUROC_ci'])} | {f(o[v]['AUROC'])} |")
    (OUT.parent.parent.parent / "exp_supp" / "cut_probe_cleaned.md").write_text("\n".join(L) + "\n")
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Music changes at shot cuts inside continuous test clips (cleaned data): " + f"{c['n']:,}".replace(",", "{,}")
           + f" cuts, {c['n_pos']} with a music change. Frozen segmenters; 95\\% film-level cluster-bootstrap CIs.}}",
           r"\label{tab:supp_incut}", r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Segmenter & AP & AUROC \\", r"\midrule",
           f"Random & {c['base_rate']:.3f} & 0.500 \\\\"]
    short = {"tn_bassl": "BaSSL, original head", "emb_p": "BaSSL, trained head", "combo": "BaSSL, trained head (ours)",
             "emb_p_dur": "+ duration, motion"}
    for v in VARIANTS:
        tex.append(f"{short[v]} & {f(c[v]['AP'])} {ci(c[v]['AP_ci'])} & {f(c[v]['AUROC'])} {ci(c[v]['AUROC_ci'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT.parent.parent.parent / "exp_supp" / "cut_probe_cleaned.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
