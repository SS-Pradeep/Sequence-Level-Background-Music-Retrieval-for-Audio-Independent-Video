"""Evaluate the segmentation variants on the frozen title-split sequences and write the tables (D140).

Variants (boundary candidates / score / cuts):
  psd_siglip       PySceneDetect shots + SigLIP merge (delta 0.85, 20 s): current baseline. Score for AP: 1 - cosine
                   of the SigLIP means of the two adjacent shots (a proxy; the merge itself is greedy)
  tn_siglip        TransNetV2 shots + the same merge
  tn_bassl         TransNetV2 + BaSSL's own MovieNet scene-boundary probability, threshold tuned on val
  tn_bassl_head    TransNetV2 + frozen BaSSL + the new head (emb_p), 3 seeds' probabilities averaged, theta on val
  tn_bassl_head_dur  ablation: the head with log shot duration and motion (emb_p_dur)
  uniform5         5 s windows (floor)
Thresholds are tuned per fold on that fold's val sequences for F1 at +-1 s (all changes); the merge variants keep
delta = 0.85. Test sequences are scored once. Outputs: results/segmentation.{md,csv,tex,json}; per-boundary
probabilities and segments (clip joins forced, so no segment crosses a join) under data/interim/segmentation/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from bgm.data.segments import span_embedding
from bgm.segmentation import evaluate as ev
from bgm.segmentation.segment import cuts_from_probs, segments_of, siglip_merge_cuts, uniform_cuts
from bgm.segmentation.sequences import Chain, edge_region
from bgm.segmentation.train_head import SegData, features, forward, load_head

log = logging.getLogger(__name__)

VARIANTS = ("psd_siglip", "tn_siglip", "tn_bassl", "tn_bassl_head", "tn_bassl_head_dur", "uniform5")
LABEL = {"psd_siglip": "1. PySceneDetect + SigLIP merge (current)", "tn_siglip": "2. TransNetV2 + SigLIP merge",
         "tn_bassl": "3. TransNetV2 + BaSSL original head", "tn_bassl_head": "4. TransNetV2 + BaSSL + new head (proposed)",
         "tn_bassl_head_dur": "4b. ... + log duration and motion (ablation)", "uniform5": "5. Uniform 5 s windows"}
HEAD = {"tn_bassl_head": "emb_p", "tn_bassl_head_dur": "emb_p_dur"}
THETAS = np.round(np.arange(0.02, 0.99, 0.01), 3)
SEEDS = (0, 1, 2)


def siglip_chain(data: SegData, parts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    embs, cents, off = [], [], 0.0
    for p in parts:
        e = data.siglip[p]
        embs.append(e)
        cents.append(np.arange(len(e)) + 0.5 + off)
        off += float(data.clips["transnetv2"][p].shots[-1, 1])
    return np.concatenate(embs), np.concatenate(cents)


def shot_cosine_score(ch: Chain, emb: np.ndarray, cent: np.ndarray) -> np.ndarray:
    v = np.stack([span_embedding(emb, a, b, cent) for a, b in ch.shots])
    return 1 - (v[:-1] * v[1:]).sum(1)


def truth(ch: Chain) -> dict:
    lab, typ = ch.labels, np.array(ch.types)
    j = ch.is_join
    lo, hi = edge_region(ch)
    S = len(ch.shots)
    s = np.arange(S - 1)
    edge = (s < 8) | (s + 8 > S - 1)
    return {"gt": ch.btimes[(lab == 1)].tolist(), "gt_film": ch.btimes[(lab == 1) & (typ == "film")].tolist(),
            "cross": ch.btimes[j & (typ == "cross")].tolist(), "gt_edge": ch.btimes[(lab == 1) & edge].tolist(),
            "edge_lo": lo, "edge_hi": hi}


class Fold:
    """All scores of one fold's frozen val or test sequences."""

    def __init__(self, data: SegData, bassl, k: int, split: str, heads: dict | None = None):
        self.k, self.split = k, split
        self.seqs = data.frozen(k, split)
        self.cT = [data.chain(q, "transnetv2") for q in self.seqs]
        self.cP = [data.chain(q, "pyscenedetect") for q in self.seqs]
        self.ctx = bassl.context_many([c.feats for c in self.cT])
        self.sig = [siglip_chain(data, q) for q in self.seqs]
        self.films = [[data.fg[p] for p in q] for q in self.seqs]
        self.truth = [truth(c) for c in self.cT]
        self.score: dict[str, list[np.ndarray]] = {
            "psd_siglip": [shot_cosine_score(c, *g) for c, g in zip(self.cP, self.sig)],
            "tn_siglip": [shot_cosine_score(c, *g) for c, g in zip(self.cT, self.sig)],
            "tn_bassl": [x["p"] for x in self.ctx]}
        self.seed_p: dict[str, list[list[np.ndarray]]] = {}
        for v, hv in (HEAD if heads is None else heads).items():
            per = []
            for s in SEEDS:
                head, scaler = load_head(data.cfg, hv, k, s, bassl.device)
                with torch.no_grad():
                    per.append([torch.sigmoid(forward(head, [torch.from_numpy(features(c, x, hv, scaler)).to(bassl.device)])).cpu().numpy()
                                if len(c.btimes) else np.zeros(0) for c, x in zip(self.cT, self.ctx)])
            self.seed_p[v] = per
            self.score[v] = [np.mean([per[s][i] for s in range(len(SEEDS))], 0) for i in range(len(self.seqs))]

    def chains(self, v: str) -> list[Chain]:
        return self.cP if v == "psd_siglip" else self.cT

    def cuts(self, v: str, i: int, theta: float | None, forced: bool, p: np.ndarray | None = None) -> list[float]:
        c = self.chains(v)[i]
        f = list(c.joins) if forced else []
        if v in ("psd_siglip", "tn_siglip"):
            return siglip_merge_cuts(c.btimes, c.duration, *self.sig[i], forced=f)
        if v == "uniform5":
            return uniform_cuts(c.duration, forced=f)
        p = self.score[v][i] if p is None else p
        return cuts_from_probs(c.btimes, p, theta, c.duration, forced=f)

    def records(self, v: str, theta: float | None, p: list | None = None) -> list[dict]:
        return [dict(self.truth[i], preds=self.cuts(v, i, theta, False, None if p is None else p[i])) for i in range(len(self.seqs))]


def pooled_f1(recs: list[dict], key: str = "all/1") -> float:
    c = np.sum([ev.seq_counts(r)[key] for r in recs], 0)
    return float(ev.f1(*c))


def tune_theta(folds_val: Fold, v: str, p: list | None = None) -> float:
    best = max(THETAS, key=lambda t: (pooled_f1(folds_val.records(v, float(t), p)), -abs(t - 0.5)))
    return float(best)


def run_eval(cfg) -> None:
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.run_all import out_dir
    data = SegData(cfg, need_siglip=True)
    bassl = BaSSL("cuda" if torch.cuda.is_available() else "cpu")
    baseline_check(data)
    rows = []          # one per (variant, test sequence)
    cand = {v: {"score": [], "label": [], "type": [], "edge": [], "seq": []} for v in VARIANTS if v != "uniform5"}
    seg_stats = {v: [] for v in VARIANTS}
    clusters, n_seq = [], 0
    thetas, seed_metrics = {}, {v: [] for v in HEAD}
    dump_b = {v: {} for v in VARIANTS if v != "uniform5"}
    dump_s = {v: {} for v in VARIANTS}
    for k in range(5):
        val, test = Fold(data, bassl, k, "val"), Fold(data, bassl, k, "test")
        for v in ("tn_bassl", *HEAD):
            thetas[(v, k)] = tune_theta(val, v)
        for v in HEAD:          # per-seed numbers: each seed's own theta
            for s in SEEDS:
                tv = tune_theta(val, v, val.seed_p[v][s])
                recs = test.records(v, tv, test.seed_p[v][s])
                seed_metrics[v].append({"fold": k, "seed": s, "theta": tv, "recs": [ev.seq_counts(r) for r in recs],
                                        "scores": test.seed_p[v][s]})
        clusters += ev.seq_clusters(test.films, f"f{k}")
        for v in VARIANTS:
            th = thetas.get((v, k))
            for i, q in enumerate(test.seqs):
                c = test.chains(v)[i]
                rec = dict(test.truth[i], preds=test.cuts(v, i, th, False))
                rows.append({"variant": v, "seq": n_seq + i, "counts": ev.seq_counts(rec), "n_pred": len(rec["preds"]),
                             "dur": c.duration})
                forced = test.cuts(v, i, th, True)
                sc = test.score.get(v, [None] * len(test.seqs))[i]
                segs = segments_of(forced, c.duration, c.btimes if sc is not None else None, sc)
                assert_no_cross(segs, c.joins)
                seg_stats[v].append([s[1] - s[0] for s in segs])
                dump_s[v].setdefault(str(k), {"test": [], "val": []})["test"].append({"parts": q, "segments": segs})
                if v in cand:
                    for key, val_ in (("score", sc), ("label", c.labels), ("type", c.types), ("edge", (np.arange(len(c.btimes)) < 8) | (np.arange(len(c.btimes)) + 8 > len(c.shots) - 1))):
                        cand[v][key].append(np.asarray(val_))
                    cand[v]["seq"].append(np.full(len(c.btimes), n_seq + i))
                    dump_b[v].setdefault(str(k), {"test": [], "val": []})["test"].append(
                        {"parts": q, "btimes": np.round(c.btimes, 4).tolist(), "p": np.round(sc, 5).tolist(),
                         "label": c.labels.tolist(), "type": c.types, "theta": th})
            for i, q in enumerate(val.seqs):           # val segments / probabilities too (for later tuning)
                c = val.chains(v)[i]
                th = thetas.get((v, k))
                sc = val.score.get(v, [None] * len(val.seqs))[i]
                segs = segments_of(val.cuts(v, i, th, True), c.duration, c.btimes if sc is not None else None, sc)
                dump_s[v].setdefault(str(k), {"test": [], "val": []})["val"].append({"parts": q, "segments": segs})
                if v in dump_b:
                    dump_b[v][str(k)]["val"].append({"parts": q, "btimes": np.round(c.btimes, 4).tolist(), "p": np.round(sc, 5).tolist(),
                                                     "label": c.labels.tolist(), "type": c.types, "theta": th})
        n_seq += len(test.seqs)
        log.info("fold %d: thetas %s", k, {v: thetas[(v, k)] for v in ("tn_bassl", *HEAD)})
    d = out_dir(cfg)
    for v in dump_s:
        (d / f"segments_{v}.json").write_text(json.dumps(dump_s[v]))
    for v in dump_b:
        (d / f"boundaries_{v}.json").write_text(json.dumps(dump_b[v]))
    results = summarise(rows, cand, seg_stats, clusters, thetas, seed_metrics)
    write_tables(cfg, results)


def assert_no_cross(segs: list, joins) -> None:
    for a, b, _ in segs:
        assert not any(a + 1e-6 < j < b - 1e-6 for j in joins), (a, b, "segment crosses a clip join")
        assert b - a >= 2.0 - 1e-6, (a, b, "segment under 2 s")
        assert b - a <= 20.0 + 1e-6, (a, b, "segment over 20 s")


def baseline_check(data: SegData) -> None:
    """Variant 1 per clip (PySceneDetect shots.csv, SigLIP, the default settings) must reproduce segments.csv."""
    from bgm.data import manifest
    from bgm.data.segments import make_segments
    shots, segs = manifest.read_shots(data.cfg), manifest.read_segments(data.cfg)
    s = data.cfg.segments
    n, same = 0, 0
    for p, g in segs.groupby("pair_id"):
        dur = float(g.end_s.max())
        mine = make_segments(manifest.cut_times(shots, p), dur, data.siglip[p], s.merge_cosine, s.max_len_s, s.min_shot_s)
        n += len(mine)
        same += int(len(mine) == len(g) and np.allclose([m[1] for m in mine], g.end_s.to_numpy(), atol=1e-3))
    log.info("baseline check: %d segments recomputed vs %d in segments.csv; %d / %d clips identical", n, len(segs), same, segs.pair_id.nunique())
    assert n == len(segs) and same == segs.pair_id.nunique(), "variant 1 does not reproduce segments.csv"


def summarise(rows, cand, seg_stats, clusters, thetas, seed_metrics) -> dict:
    bs = ev.Bootstrap(clusters, 10000, 0)
    keys = [f"{g}/{t:g}" for g in ev.GROUPS for t in ev.TOLS]
    res = {"n_seq": len(clusters), "n_clusters": len(bs.uniq), "variants": {}}
    draws = {}
    for v in VARIANTS:
        rv = sorted([r for r in rows if r["variant"] == v], key=lambda r: r["seq"])
        m, dv = {}, {}
        for key in keys:
            c = np.array([r["counts"][key] for r in rv], float)
            b = bs.f1(c)
            m[f"F1 {key}"] = {"mean": float(ev.f1(*c.sum(0))), "ci": ev.ci(b)}
            dv[f"F1 {key}"] = b
            if key == "all/1":
                tp, fp, fn = c.sum(0)
                m["precision all/1"], m["recall all/1"] = float(tp / max(tp + fp, 1)), float(tp / max(tp + fn, 1))
        if v in cand:
            sc = np.concatenate(cand[v]["score"]); lab = np.concatenate(cand[v]["label"])
            typ = np.concatenate(cand[v]["type"]); edg = np.concatenate(cand[v]["edge"]).astype(bool)
            seq = np.concatenate(cand[v]["seq"])
            for g, sel in (("all", np.ones_like(lab, bool)), ("in_film", typ != "cross"), ("edge", edg)):
                b = bs.ap(sc[sel], lab[sel], seq[sel])
                m[f"AP {g}"] = {"mean": ev.average_precision(sc[sel], lab[sel]), "ci": ev.ci(b), "n_pos": int(lab[sel].sum()), "n": int(sel.sum())}
                dv[f"AP {g}"] = b
        L = np.concatenate([np.asarray(x) for x in seg_stats[v]])
        tot = sum(r["dur"] for r in rv) / 60
        m["segments"] = {"per_min": float(len(L) / tot), "mean_s": float(L.mean()), "median_s": float(np.median(L)),
                         "min_s": float(L.min()), "max_s": float(L.max())}
        m["pred_cuts_per_min"] = float(sum(r["n_pred"] for r in rv) / tot)
        res["variants"][v] = m
        draws[v] = dv
    for v in VARIANTS:
        if v == "psd_siglip":
            continue
        res["variants"][v]["delta_vs_1"] = {}
        for key, b in draws[v].items():
            if key in draws["psd_siglip"]:
                dd = b - draws["psd_siglip"][key]
                res["variants"][v]["delta_vs_1"][key] = {"mean": res["variants"][v][key]["mean"] - res["variants"]["psd_siglip"][key]["mean"],
                                                         "ci": ev.ci(dd)}
    d4 = draws["tn_bassl_head"]
    res["variants"]["tn_bassl_head"]["delta_vs_3"] = {key: {"mean": res["variants"]["tn_bassl_head"][key]["mean"] - res["variants"]["tn_bassl"][key]["mean"],
                                                            "ci": ev.ci(d4[key] - draws["tn_bassl"][key])}
                                                      for key in d4 if key in draws["tn_bassl"]}
    res["thetas"] = {f"{v}/fold{k}": t for (v, k), t in thetas.items()}
    res["per_seed"] = {}
    for v, lst in seed_metrics.items():
        per = {}
        for s in SEEDS:
            ent = [e for e in lst if e["seed"] == s]
            c = np.sum([np.sum([r["all/1"] for r in e["recs"]], 0) for e in ent], 0)
            cf = np.sum([np.sum([r["in_film/1"] for r in e["recs"]], 0) for e in ent], 0)
            per[s] = {"F1 all/1": float(ev.f1(*c)), "F1 in_film/1": float(ev.f1(*cf))}
        sc = np.concatenate([np.concatenate(e["scores"]) for e in sorted(lst, key=lambda e: (e["fold"], e["seed"])) if e["seed"] == 0])
        res["per_seed"][v] = {k2: {"mean": float(np.mean([per[s][k2] for s in SEEDS])), "std": float(np.std([per[s][k2] for s in SEEDS]))}
                              for k2 in ("F1 all/1", "F1 in_film/1")}
        aps = []
        lab = np.concatenate(cand[v]["label"])
        for s in SEEDS:
            sc = np.concatenate([np.concatenate(e["scores"]) for e in sorted(lst, key=lambda e: e["fold"]) if e["seed"] == s])
            aps.append(ev.average_precision(sc, lab))
        res["per_seed"][v]["AP all"] = {"mean": float(np.mean(aps)), "std": float(np.std(aps))}
    return res


def _fmt(m: dict, d: int = 3) -> str:
    return f"{m['mean']:.{d}f} [{m['ci'][0]:.{d}f}, {m['ci'][1]:.{d}f}]"


def _dfmt(m: dict | None) -> str:
    if m is None:
        return "—"
    sig = "" if m["ci"][0] <= 0 <= m["ci"][1] else " *"
    return f"{m['mean']:+.3f} [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]{sig}"


def write_tables(cfg, res: dict) -> None:
    import pandas as pd
    out = Path("results")
    (out / "segmentation.json").write_text(json.dumps(res, indent=1))
    V = res["variants"]
    na = {"mean": float("nan"), "ci": (float("nan"), float("nan"))}
    lines = [f"# Step 1 segmentation: music-change boundaries (D140)", "",
             f"Test: frozen title-split blocked sequences, 5 folds pooled ({res['n_seq']} sequences, {res['n_clusters']} film clusters). "
             "Boundary metrics use each method's own cuts without the oracle clip joins; [95% CI], cluster bootstrap over film "
             "groups, 10,000 resamples. Δ vs 1 paired; * = CI excludes 0.", "",
             "| variant | AP (all) | AP (in film) | F1 ±0.5 s | F1 ±1 s | F1 ±1 s in film | F1 ±1 s edge | cuts / min | segments / min | mean / median s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    csv = []
    for v in VARIANTS:
        m = V[v]
        ap = m.get("AP all", None)
        apf = m.get("AP in_film", None)
        s = m["segments"]
        lines.append(f"| {LABEL[v]} | {_fmt(ap) if ap else 'n/a'} | {_fmt(apf) if apf else 'n/a'} | {_fmt(m['F1 all/0.5'])} | "
                     f"{_fmt(m['F1 all/1'])} | {_fmt(m['F1 in_film/1'])} | {_fmt(m['F1 edge/1'])} | {m['pred_cuts_per_min']:.1f} | "
                     f"{s['per_min']:.1f} | {s['mean_s']:.1f} / {s['median_s']:.1f} |")
        csv.append({"variant": v, **{f"{k} mean": (m[k]["mean"] if k in m else np.nan) for k in ("AP all", "AP in_film", "AP edge", "F1 all/0.5", "F1 all/1", "F1 in_film/0.5", "F1 in_film/1", "F1 edge/1")},
                    **{f"{k} lo": (m[k]["ci"][0] if k in m else np.nan) for k in ("AP all", "F1 all/1", "F1 in_film/1")},
                    **{f"{k} hi": (m[k]["ci"][1] if k in m else np.nan) for k in ("AP all", "F1 all/1", "F1 in_film/1")},
                    "cuts_per_min": m["pred_cuts_per_min"], "segments_per_min": s["per_min"], "seg_mean_s": s["mean_s"], "seg_median_s": s["median_s"]})
    lines += ["", "Paired differences vs variant 1:", "", "| variant | Δ AP (all) | Δ AP (in film) | Δ F1 ±0.5 s | Δ F1 ±1 s | Δ F1 ±1 s in film |", "| --- | --- | --- | --- | --- | --- |"]
    for v in VARIANTS[1:]:
        d = V[v]["delta_vs_1"]
        lines.append(f"| {LABEL[v]} | {_dfmt(d.get('AP all'))} | {_dfmt(d.get('AP in_film'))} | {_dfmt(d.get('F1 all/0.5'))} | {_dfmt(d.get('F1 all/1'))} | {_dfmt(d.get('F1 in_film/1'))} |")
    d3 = V["tn_bassl_head"]["delta_vs_3"]
    lines += ["", f"4 − 3 (what training the head adds): Δ AP {_dfmt(d3['AP all'])}; Δ F1 ±1 s {_dfmt(d3['F1 all/1'])}; in film {_dfmt(d3['F1 in_film/1'])}.", "",
              "Edge boundaries (BaSSL window clamped at a sequence edge):", "", "| variant | AP edge (n pos / n) | F1 ±1 s edge |", "| --- | --- | --- |"]
    for v in VARIANTS:
        m = V[v]
        a = m.get("AP edge")
        lines.append(f"| {LABEL[v]} | {(_fmt(a) + f' ({a['n_pos']} / {a['n']})') if a else 'n/a'} | {_fmt(m['F1 edge/1'])} |")
    lines += ["", "Per-seed (each seed's own θ; seed-averaged probabilities are the table rows): " + "; ".join(
        f"{v}: AP {r['AP all']['mean']:.3f} ± {r['AP all']['std']:.3f}, F1 ±1 s {r['F1 all/1']['mean']:.3f} ± {r['F1 all/1']['std']:.3f}"
        for v, r in res["per_seed"].items()), "",
        "θ per fold (val F1 ±1 s): " + ", ".join(f"{k} {t:.2f}" for k, t in res["thetas"].items()), ""]
    (out / "segmentation.md").write_text("\n".join(lines))
    pd.DataFrame(csv).to_csv(out / "segmentation.csv", index=False)
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Variant & AP & F1@0.5\,s & F1@1\,s & F1@1\,s in film & $\Delta$F1@1\,s vs 1 & seg/min \\", r"\midrule"]
    for v in VARIANTS:
        m = V[v]
        ap = m.get("AP all")
        d = m.get("delta_vs_1", {}).get("F1 all/1")
        c = lambda x: f"{x['mean']:.3f} {{\\scriptsize[{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]}}"  # noqa: E731
        tex.append(f"{LABEL[v].replace('&', 'and')} & {c(ap) if ap else 'n/a'} & {c(m['F1 all/0.5'])} & {c(m['F1 all/1'])} & {c(m['F1 in_film/1'])} & "
                   f"{('$' + f'{d['mean']:+.3f}' + '$ {\\scriptsize[' + f'{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}' + ']}') if d else '---'} & {m['segments']['per_min']:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "segmentation.tex").write_text("\n".join(tex) + "\n")
    log.info("\n%s", "\n".join(lines))


# ---------------------------------------------------------------- figure
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8c8b86", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def figure(cfg, fold: int = 0, seq: int | None = None) -> Path:
    """results/segmentation.png: (a) F1 +-1 s, (b) AP, with 95% CIs; (c) one test sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bgm.segmentation.run_all import out_dir
    res = json.loads(Path("results/segmentation.json").read_text())
    V = res["variants"]
    order = list(VARIANTS)[::-1]
    short = {"psd_siglip": "1  PySceneDetect + SigLIP merge (current)", "tn_siglip": "2  TransNetV2 + SigLIP merge",
             "tn_bassl": "3  TransNetV2 + BaSSL, original head", "tn_bassl_head": "4  TransNetV2 + BaSSL + new head",
             "tn_bassl_head_dur": "4b  + duration and motion (ablation)", "uniform5": "5  Uniform 5 s windows"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11, 7.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.25, 1], hspace=0.42, wspace=0.08)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    y = np.arange(len(order))
    col = lambda v: BLUE if v == "tn_bassl_head" else (INK if v == "psd_siglip" else MUTED)  # noqa: E731
    for ax in (axA, axB):
        ax.set_facecolor(SURF)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
    for i, v in enumerate(order):
        for key, dy, mk in (("F1 all/1", 0.14, "o"), ("F1 in_film/1", -0.14, "D")):
            m = V[v][key]
            axA.plot(m["ci"], [i + dy] * 2, color=col(v), lw=2, solid_capstyle="round")
            axA.plot(m["mean"], i + dy, mk, ms=7 if mk == "o" else 6, color=col(v), mec=SURF, mew=1.5,
                     mfc=col(v) if mk == "o" else SURF if False else col(v))
        axA.text(V[v]["F1 all/1"]["ci"][1] + 0.012, i + 0.14, f"{V[v]['F1 all/1']['mean']:.2f}", va="center", fontsize=8, color=INK2)
        if "AP all" in V[v]:
            m = V[v]["AP all"]
            axB.plot(m["ci"], [i, i], color=col(v), lw=2, solid_capstyle="round")
            axB.plot(m["mean"], i, "o", ms=7, color=col(v), mec=SURF, mew=1.5)
            axB.text(m["ci"][1] + 0.012, i, f"{m['mean']:.2f}", va="center", fontsize=8, color=INK2)
        else:
            axB.text(0.02, i, "no score (n/a)", va="center", fontsize=8, color=MUTED, transform=axB.get_yaxis_transform())
    axA.set_yticks(y, [short[v] for v in order])
    for t, v in zip(axA.get_yticklabels(), order):
        t.set_color(BLUE if v == "tn_bassl_head" else INK)
        t.set_fontweight("bold" if v == "tn_bassl_head" else "normal")
    axB.set_yticks(y, [""] * len(y))
    axA.set_xlim(0, 1)
    axB.set_xlim(0, 1)
    axA.set_xlabel("boundary F1 at ±1 s vs true music changes")
    axB.set_xlabel("average precision (boundary score)")
    axA.plot([], [], "o", color=INK2, label="all changes")
    axA.plot([], [], "D", color=INK2, label="same-film changes only")
    axA.legend(loc="lower right", frameon=False, fontsize=8)
    axA.set_title("a  Where does the music change?  F1 ±1 s, 95% CI", loc="left", fontsize=10, fontweight="bold")
    axB.set_title("b  Ranking quality, AP, 95% CI", loc="left", fontsize=10, fontweight="bold")

    # (c) one test sequence
    b3 = json.loads((out_dir(cfg) / "boundaries_tn_bassl.json").read_text())[str(fold)]["test"]
    b4 = json.loads((out_dir(cfg) / "boundaries_tn_bassl_head.json").read_text())[str(fold)]["test"]
    s1 = json.loads((out_dir(cfg) / "segments_psd_siglip.json").read_text())
    if seq is None:   # the sequence with the most same-film joins, for a readable example
        seq = int(np.argmax([sum(1 for t, l in zip(r["type"], r["label"]) if t == "film") for r in b4]))
    r3, r4 = b3[seq], b4[seq]
    data = SegData(cfg, need_siglip=True)
    ch = data.chain(r4["parts"], "transnetv2")
    chP = data.chain(r4["parts"], "pyscenedetect")
    emb, cent = siglip_chain(data, r4["parts"])
    cut1 = siglip_merge_cuts(chP.btimes, chP.duration, emb, cent)
    cut4 = cuts_from_probs(np.array(r4["btimes"]), np.array(r4["p"]), r4["theta"], ch.duration)
    axC = fig.add_subplot(gs[1, :])
    axC.set_facecolor(SURF)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)
    bt = np.array(r4["btimes"])
    lab, typ = np.array(r4["label"]), np.array(r4["type"])
    for t, l, ty in zip(bt, lab, typ):
        if ty != "clip":
            axC.axvline(t, color=ORANGE if l == 1 else MUTED, lw=1.6 if l == 1 else 1.0, ls="-" if l == 1 else (0, (3, 2)), zorder=1)
    axC.vlines(bt, 0, np.array(r3["p"]), color=MUTED, lw=2, label="BaSSL original scene probability", zorder=2)
    axC.plot(bt, np.array(r4["p"]), "o", ms=6, color=BLUE, mec=SURF, mew=1.2, label="new head: p(music change)", zorder=3)
    axC.axhline(r4["theta"], color=BLUE, lw=0.8, ls=(0, (4, 3)))
    axC.text(ch.duration, r4["theta"] + 0.02, f"θ = {r4['theta']:.2f}", ha="right", fontsize=8, color=BLUE)
    for yy, cuts, c, name in ((-0.12, cut4, BLUE, "cuts, 4 (new head)"), (-0.24, cut1, INK, "cuts, 1 (current)")):
        axC.plot(cuts, [yy] * len(cuts), "|", ms=11, mew=2, color=c)
        axC.text(-1, yy, name, ha="right", va="center", fontsize=8, color=c)
    films = [data.fg[p].split(":")[-1][:18] for p in r4["parts"]]
    starts = [0.0] + list(ch.joins)
    ends = list(ch.joins) + [ch.duration]
    for a, e, f, p in zip(starts, ends, films, r4["parts"]):
        axC.text((a + e) / 2, 1.07, f, ha="center", fontsize=7, color=INK2, clip_on=False)
    axC.plot([], [], color=ORANGE, lw=1.6, label="true music change (clip join, new track)")
    axC.plot([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="clip join, same track (no change)")
    axC.set_ylim(-0.32, 1.0)
    axC.set_yticks([0, 0.5, 1])
    axC.set_xlim(0, ch.duration)
    axC.set_xlabel("time in the chained test sequence (s)")
    axC.set_ylabel("probability")
    axC.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=4, frameon=False, fontsize=8)
    axC.set_title(f"c  One test sequence (fold {fold}): {len(r4['parts'])} clips (film-group ids above), θ tuned on val", loc="left", fontsize=10, fontweight="bold", pad=16)
    fig.text(0.01, 0.005, f"Title split, 5 folds pooled: {res['n_seq']} test sequences of 60–180 s, {res['n_clusters']} film clusters; "
             "cluster bootstrap, 10,000 resamples. Cuts are each method's own, without the oracle clip joins.", fontsize=7.5, color=MUTED)
    out = Path("results/segmentation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    return out
