"""results/titlesplit/eval_suite.md from runs/titlesplit/eval_suite/ (D111; metric definitions in eval_suite.py).

Tables: mean ± std over the 15 runs (5 folds × 3 disjoint 10-head ensembles); ImageBind and exact chance: 5 folds.
CIs: per-query values averaged over a fold's 3 seeds, pooled over folds, 95% cluster bootstrap over film groups
(concatenations resampled as units), cfg.eval.bootstrap_resamples resamples.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.analysis.eval_suite import EXP, FOLDS, RESULTS, ROOT, SEEDS, chance_ndcg, diversity
from bgm.eval.metrics import chance_mrr, chance_recall, cluster_bootstrap

log = logging.getLogger(__name__)
B = 10000
KS = (1, 5, 10)


def _load() -> dict[int, list[dict]]:
    out = {}
    for k in FOLDS:
        out[k] = []
        for e in SEEDS:
            with np.load(ROOT / EXP / f"fold{k}" / f"seed{e}" / "perquery.npz") as z:
                out[k].append({n: z[n] for n in z.files})
    return out


def _film() -> dict[str, str]:
    return json.loads(Path("data/manifests/folds_title.json").read_text())["film_group"]


def ms(v, d=3) -> str:
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}" if len(v) > 1 else f"{np.nanmean(v):.{d}f}"


def ci(b: dict, d=3) -> str:
    return f"[{b['lo']:+.{d}f}, {b['hi']:+.{d}f}]" if b["lo"] < 0 or b["hi"] < 0 else f"[{b['lo']:.{d}f}, {b['hi']:.{d}f}]"


class Pool:
    """Per-query arrays averaged over seeds (seed 0 only for keys only seed 0 has), pooled over folds, with clusters."""

    def __init__(self, R: dict, film: dict):
        self.R, self.film = R, film

    def per_run(self, key, fn=np.nanmean) -> list[float]:
        return [float(fn(r[key])) for k in FOLDS for r in self.R[k] if key in r]

    def pooled(self, key, clusters: str) -> tuple[np.ndarray, np.ndarray]:
        vals, cl = [], []
        for k in FOLDS:
            rs = [r for r in self.R[k] if key in r]
            with np.errstate(all="ignore"):
                vals.append(np.nanmean(np.stack([r[key].astype(float) for r in rs]), 0))
            cl.append(self.clusters(k, clusters))
        return np.concatenate(vals), np.concatenate(cl)

    def clusters(self, k, kind) -> np.ndarray:
        r = self.R[k][0]
        if kind == "track":
            return np.array([self.film[p] for p in r["ret/track/cl"]])
        if kind == "segment":
            return np.array([self.film[s.split("#")[0]] for s in r["ret/segment/cl"]])
        if kind == "video":
            return np.array([self.film[p] for p in r["align/pair"]])
        return np.array([f"concat{k}_{i}" for i in range(len(r["trans/viterbi/F1@2.0"]))])

    def boot(self, key, kind, minus: str | None = None) -> dict:
        v, cl = self.pooled(key, kind)
        if minus is not None:
            v = v - self.pooled(minus, kind)[0]
        return cluster_bootstrap(v, cl, B)


# ---------------------------------------------------------------- 1. retrieval
def _rank_cells(P: Pool, prefix: str, side: str, with_ci: bool) -> list[str]:
    cells = []
    rk = f"{prefix}/{side}/rank"
    for k in KS:
        vals = [float((r[rk] <= k).mean()) for f in FOLDS for r in P.R[f] if rk in r]
        cells.append(ms(vals))
    cells.append(ms([float(np.median(r[rk])) for f in FOLDS for r in P.R[f] if rk in r], 1))
    cells.append(ms([float((1 / r[rk]).mean()) for f in FOLDS for r in P.R[f] if rk in r]))
    cells.append(ms(P.per_run(f"{prefix}/{side}/ndcg")))
    if with_ci:
        hit, cl = [], []
        for f in FOLDS:
            hit.append(np.mean([(r[rk] <= 10).astype(float) for r in P.R[f] if rk in r], 0))
            cl.append(P.clusters(f, side))
        b10 = cluster_bootstrap(np.concatenate(hit), np.concatenate(cl), B)
        cells[2] += f" {ci(b10)}"
        cells[5] += f" {ci(P.boot(f'{prefix}/{side}/ndcg', side))}"
    return cells


def _chance_cells(P: Pool, side: str) -> list[str]:
    rng = np.random.default_rng(0)
    rk, mr, nd, md = {k: [] for k in KS}, [], [], []
    for f in FOLDS:
        r = P.R[f][0]
        N, h = r[f"ret/{side}/N"], r[f"ret/{side}/h"]
        for k in KS:
            rk[k].append(np.mean([chance_recall(int(n), int(x), k) for n, x in zip(N, h)]))
        mr.append(np.mean([chance_mrr(int(n), int(x)) for n, x in zip(N, h)]))
        nd.append(np.mean([chance_ndcg(int(n), int(x)) for n, x in zip(N, h)]))
        sims = [np.median([1 + rng.choice(int(n), int(x), replace=False).min() for n, x in zip(N, h)]) for _ in range(200)]
        md.append(float(np.mean(sims)))
    return [ms(rk[k]) for k in KS] + [ms(md, 1), ms(mr), ms(nd)]


def section_retrieval(P: Pool) -> list[str]:
    L = ["## 1. Retrieval", "",
         "Queries: 982 test clips (track: whole clip vs each file of the test fold's gallery, 196 files on average; a hit "
         "is any file of the query's track group) and 2,552 test segments (segment: vs every test segment's aligned "
         "music window, other same-track videos left out; one relevant window). nDCG@10: binary gain, every file of "
         "the track group is relevant. Random: exact expectation (MedR by simulation). [95% CI] after R@10 and nDCG@10.", ""]
    for side, title in (("track", "Track retrieval (whole clip → music file)"), ("segment", "Segment retrieval (segment → music window)")):
        L += [f"**{title}**", "", "| system | runs | R@1 | R@5 | R@10 | MedR ↓ | MRR | nDCG@10 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L.append("| random (exact chance) | 5 | " + " | ".join(_chance_cells(P, side)) + " |")
        L.append("| ImageBind zero-shot | 5 | " + " | ".join(_rank_cells(P, "ret_ib", side, True)) + " |")
        L.append("| **final: 10-head ensemble** | 15 | " + " | ".join(_rank_cells(P, "ret", side, True)) + " |")
        L.append("")
    b = P.boot("ret/track/ndcg", "track", minus="ret_ib/track/ndcg")
    bs = P.boot("ret/segment/ndcg", "segment", minus="ret_ib/segment/ndcg")
    L += [f"Paired, final − ImageBind: track nDCG@10 {b['mean']:+.3f} {ci(b)}; segment nDCG@10 {bs['mean']:+.3f} {ci(bs)}.", ""]
    return L


# ---------------------------------------------------------------- 2. alignment
ALIGN_ROWS = (("original", "original music (reference)"), ("viterbi", "**final (Viterbi)**"),
              ("greedy", "greedy (top window per segment)"), ("random", "random window per segment"),
              ("viterbi_sync", "final + render beat sync (D35)"), ("random_sync", "random + render beat sync (control)"))


def _excess(P: Pool, s: str, key: str, base: str) -> tuple[list[float], dict]:
    """Per-run mean excess and its film-group CI for system s."""
    vals = [float(np.nanmean(r[f"align/{s}/{key}"] - r[f"align/{s}/{base}"])) for f in FOLDS for r in P.R[f]]
    return vals, P.boot(f"align/{s}/{key}", "video", minus=f"align/{s}/{base}")


def _paired_excess(P: Pool, s: str, o: str, key: str, base: str) -> dict:
    v1, cl = P.pooled(f"align/{s}/{key}", "video")
    v1 = v1 - P.pooled(f"align/{s}/{base}", "video")[0]
    v2 = P.pooled(f"align/{o}/{key}", "video")[0] - P.pooled(f"align/{o}/{base}", "video")[0]
    return cluster_bootstrap(v1 - v2, cl, B)


def section_alignment(P: Pool) -> list[str]:
    n_cut = sum(int((P.R[f][0]["align/viterbi/n_cuts"] > 0).sum()) for f in FOLDS)
    n_vid = sum(len(P.R[f][0]["align/pair"]) for f in FOLDS)
    L = ["## 2. Alignment (natural test videos)", "",
         f"Per test video ({n_vid:,}). **Beat alignment** on the {n_cut:,} videos with at least one cut more than 0.5 s from "
         "the ends. LB: share of cuts with a soundtrack beat within ±70 ms. tw-LB: mean exp(−d²/2σ²) over the cuts, d = "
         "distance to the nearest beat, σ = 70 ms (near misses earn partial credit). Excess: minus the same score with "
         "the soundtrack's beats circularly shifted (20 draws), i.e. alignment beyond chance; [95% CI]. *All cuts*: the "
         "same cuts for every system. *Within-piece cuts*: only cuts where the music carries on (more than 50 ms from any "
         "piece start; D35's convention); a system that starts new music at a cut is not scored there. Render beat sync "
         "(D35) puts every piece start on a beat by construction, so for the sync rows only the within-piece columns "
         "measure alignment.", "",
         "| system | LB, all cuts | LB excess | tw-LB, all cuts | tw-LB excess | within-piece cuts / video | LB, within-piece | LB excess | tw-LB excess |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s, label in ALIGN_ROWS:
        ex, b_ex = _excess(P, s, "LB", "LB_shift")
        tw, b_tw = _excess(P, s, "twLB", "twLB_shift")
        iex, b_iex = _excess(P, s, "in_LB", "in_LB_shift")
        itw, b_itw = _excess(P, s, "in_twLB", "in_twLB_shift")
        L.append(f"| {label} | {ms(P.per_run(f'align/{s}/LB'))} | {ms(ex)} {ci(b_ex)} | {ms(P.per_run(f'align/{s}/twLB'))} | "
                 f"{ms(tw)} {ci(b_tw)} | {ms(P.per_run(f'align/{s}/n_inner'), 2)} | {ms(P.per_run(f'align/{s}/in_LB'))} | "
                 f"{ms(iex)} {ci(b_iex)} | {ms(itw)} {ci(b_itw)} |")
    L += ["", "**Content alignment.** IB: ImageBind cosine between each segment's video and its music window (span means of "
          "the cached 1 Hz ImageBind embeddings), duration-weighted per video; higher = the music suits the picture more "
          "in ImageBind's joint space. KL: KL(original window ‖ chosen window) between the PANNs CNN14 AudioSet tag "
          "distributions (per-second sigmoid outputs, window mean, normalized to sum 1), duration-weighted; lower = sounds "
          "more like the music the editor used.", "",
          "| system | IB [95% CI] | KL ↓ [95% CI] |", "| --- | --- | --- |"]
    for s, label in ALIGN_ROWS:
        if s.endswith("_sync"):
            continue
        kl = "0 (by definition)" if s == "original" else f"{ms(P.per_run(f'align/{s}/KL'))} {ci(P.boot(f'align/{s}/KL', 'video'))}"
        L.append(f"| {label} | {ms(P.per_run(f'align/{s}/IB'))} {ci(P.boot(f'align/{s}/IB', 'video'))} | {kl} |")
    L += ["", "Paired differences (per video, seeds averaged, film-group CIs):", ""]
    lab = {k: v.strip("*") for k, v in ALIGN_ROWS}
    for s, o in (("viterbi", "random"), ("viterbi", "greedy"), ("viterbi", "original"), ("viterbi_sync", "viterbi"),
                 ("viterbi_sync", "random_sync")):
        parts = []
        for name, key, base in (("LB excess, all cuts", "LB", "LB_shift"), ("within-piece LB excess", "in_LB", "in_LB_shift"),
                                ("within-piece tw-LB excess", "in_twLB", "in_twLB_shift")):
            b = _paired_excess(P, s, o, key, base)
            parts.append(f"{name} {b['mean']:+.3f} {ci(b)}")
        if not s.endswith("_sync"):
            for key in ("IB", "KL"):
                if key == "KL" and o == "original":
                    continue
                b = P.boot(f"align/{s}/{key}", "video", minus=f"align/{o}/{key}")
                parts.append(f"{key} {b['mean']:+.3f} {ci(b)}")
        L.append(f"- {lab[s]} − {lab[o]}: " + "; ".join(parts) + ".")
    L.append("")
    return L


# ---------------------------------------------------------------- 3. transitions
TRANS_ROWS = (("original", "original music (reference)"), ("viterbi", "**final (Viterbi)**"), ("greedy", "greedy"),
              ("random", "random window per segment"), ("random_bounds", "random boundaries (Viterbi's switch count)"))
TRANS_COLS = (("F1@1.5", 3), ("F1@2.0", 3), ("dev_T2P", 2), ("dev_P2T", 2), ("TBIoU", 3), ("switches_per_min", 2),
              ("jumps_per_min", 2), ("seam_LU", 2), ("DTW", 3))


def section_transitions(P: Pool) -> list[str]:
    L = ["## 3. Transitions (test concatenations)", "",
         "50 test concatenations per fold (3–4 clips of different films joined end to end, `concat_sets_title.json`); "
         "the true music switches are the joins. F1: one-to-one matching of predicted file switches to joins within "
         "±1.5 / ±2.0 s. Deviation (MIREX style, s): T→P = median distance from each join to the nearest predicted switch, "
         "P→T = from each predicted switch to the nearest join. TBIoU: each true piece's best temporal IoU with a predicted "
         "piece, averaged. Smoothness: switches / min (truth: 2.5), jumps / min (same file, non-contiguous), median loudness "
         "jump at the rendered soundtrack's seams (LU, ±3 s; the original's seams are the joins, rendered the same way). DTW: cosine-distance DTW between the per-second CLAP "
         "embeddings of the produced and the original soundtrack, per path step (0 = the original music).", "",
         "| system | F1 @±1.5 s | F1 @±2.0 s | dev T→P (s) ↓ | dev P→T (s) ↓ | TBIoU | switches/min | jumps/min | seam jump (LU) ↓ | DTW ↓ |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s, label in TRANS_ROWS:
        cells = []
        for c, d in TRANS_COLS:
            key = f"trans/{s}/{c}"
            if key not in P.R[0][0] or (s == "original" and c == "DTW"):
                cells.append("—" if s != "original" else ("0" if c == "DTW" else "—"))
                continue
            vals = P.per_run(key, (lambda v: np.nanmean(np.where(np.isfinite(v), v, np.nan))) if c.startswith("dev") else np.nanmean)
            cells.append(ms(vals, d))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "Deviation: a concatenation's median, averaged over its run's concatenations (a concatenation without any "
          "predicted switch is left out of T→P). Switches can only fall on segment boundaries and the joins are segment "
          "boundaries, so a switch is either exactly on a join or a whole segment away; that is why F1 is the same at ±1.5 "
          "and ±2.0 s and why the deviations are mostly 0 or several seconds. Paired (per concatenation, seeds averaged, "
          "concatenations resampled):", ""]
    for a, b_, cols in (("viterbi", "greedy", ("F1@2.0", "TBIoU", "seam_LU", "DTW")),
                        ("viterbi", "random_bounds", ("F1@2.0", "TBIoU")),
                        ("viterbi", "random", ("F1@2.0", "TBIoU", "seam_LU", "DTW"))):
        parts = []
        for c in cols:
            b = P.boot(f"trans/{a}/{c}", "concat", minus=f"trans/{b_}/{c}")
            parts.append(f"{c} {b['mean']:+.3f} {ci(b)}")
        L.append(f"- Viterbi − {dict(TRANS_ROWS)[b_]}: " + "; ".join(parts) + ".")
    L.append("")
    return L


# ---------------------------------------------------------------- 4. diversity
def section_diversity(P: Pool) -> list[str]:
    L = ["## 4. Diversity", "",
         "Per run, over the picks of all test segments of the fold (2,552 segments in all; ~510 per fold) or the top-1 "
         "file of track retrieval (one per clip). Entropy: of the distribution of chosen files, divided by log(gallery "
         "size) (1 = every file equally often). Coverage: share of the fold's gallery files chosen at least once. Top-5 "
         "share: share of picks taken by the 5 most-chosen files (hubness; uniform ≈ 5/196 = 0.026).", "",
         "| picks | system | runs | entropy | coverage | top-5 share ↓ |", "| --- | --- | --- | --- | --- | --- |"]
    rows = [("segments", "original music (reference)", "div/seg/original"), ("segments", "**final (Viterbi)**", "div/seg/viterbi"),
            ("segments", "greedy", "div/seg/greedy"), ("segments", "random window per segment", "div/seg/random"),
            ("track top-1", "**final: 10-head ensemble**", "ret/track/top1"), ("track top-1", "ImageBind zero-shot", "ret_ib/track/top1")]
    for unit, label, key in rows:
        D = [diversity(list(r[key]), int(r["ret/track/N"][0])) for f in FOLDS for r in P.R[f] if key in r]
        L.append(f"| {unit} | {label} | {len(D)} | " + " | ".join(ms([d[m] for d in D]) for m in ("entropy", "coverage", "top5_share")) + " |")
    L.append("")
    return L


def write_report() -> Path:
    P = Pool(_load(), _film())
    L = ["# Evaluation suite: retrieval, alignment, transitions, diversity (D111)", "",
         "Generated by `python -m bgm.analysis.eval_suite report` (src/bgm/analysis/eval_suite.py, eval_suite_report.py) from "
         "runs/titlesplit/eval_suite/. Title split (folds_title.json, 982 test clips in 5 folds); the final system (D75) as "
         "benchmarked: 10-head SigLIP → CLAP+MERT ensemble, coarse-to-fine retrieval over the test fold's music, Viterbi with "
         "each run's val-tuned penalties. Rebuilt per run; its ranks and Viterbi accuracies equal the saved runs exactly. "
         "Nothing was trained or tuned for this report. Tables: mean ± std over 15 runs (5 folds × 3 disjoint 10-head "
         "ensembles); 95% CIs: per-query values averaged over seeds, cluster bootstrap over film groups, 10,000 resamples.", ""]
    L += section_retrieval(P) + section_alignment(P) + section_transitions(P) + section_diversity(P)
    RESULTS.write_text("\n".join(L) + "\n")
    log.info("wrote %s", RESULTS)
    return RESULTS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    write_report()
