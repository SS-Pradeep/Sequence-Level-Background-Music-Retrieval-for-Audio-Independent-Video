"""D126: segment-level retrieval for the paper's §5, from the saved per-query ranks (nothing is re-scored or re-trained).

Both datasets on the title split: original (runs/titlesplit, 1,248 pairs) and cleaned (runs/cleaned, 1,011 pairs, D116).
Systems: exact chance, ImageBind zero-shot, single SigLIP → CLAP+MERT heads (seeds 0-2) and the final 10-head ensemble.

1. **Main table.** Segment R@1/5/10, MedR, MRR: mean ± std over runs, and 95% CIs from a cluster bootstrap over film
   groups on the pooled queries (per-query hit averaged over a fold's runs; MedR: the median of every (run, query)
   rank, clusters resampled). Paired Δ between systems as in final_film.py.
2. **Gallery size.** Track retrieval as run (test-fold gallery vs every music file). Plus the exact expected R@k when
   each query's gallery is cut to n items at random: with r − 1 non-relevant items above the best relevant one among
   N − h non-relevant ones, keeping n − h of them at random leaves X ~ Hypergeom(N − h, r − 1, n − h) above it, and
   hit@k = P(X ≤ k − 1). Exact from the ranks; it can only shrink a gallery.
3. **Duration.** Segment R@10 by segment-duration quartile, track R@10 by clip-duration quartile (quartile edges from the
   pooled test queries), with the longest − shortest difference (joint cluster bootstrap).

Run: python -m bgm.analysis.segment_retrieval   -> results/segment_retrieval.md
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from bgm.analysis.cleaned_compare import CLEAN, ORIG, make_cfg
from bgm.data import manifest
from bgm.eval.final import _ci
from bgm.eval.final_film import ENS, rank_runs, zeroshot_runs
from bgm.eval.metrics import chance_recall, cluster_bootstrap
from bgm.eval.report_filmsplit import Split, _cell, _chance, _hits, _row

log = logging.getLogger(__name__)
B = 10000
SINGLE_SEEDS = (0, 1, 2)
DATASETS = (("original", ORIG, "1,248 pairs, 982 test clips"), ("cleaned", CLEAN, "1,011 pairs, 894 test clips"))
SEG_N = (25, 50, 100, 200, 350)
TRACK_N = (50, 100, 200, 500)


# ---------------------------------------------------------------- loading
def load_systems(root: Path) -> dict[str, dict]:
    """{system: {fold: [{section: ranks}]}}."""
    return {"ImageBind zero-shot": zeroshot_runs(root, "imagebind"),
            "single heads (seeds 0–2)": rank_runs(root, "semantic", "siglip-both", SINGLE_SEEDS),
            "**final: 10-head ensemble**": rank_runs(root, "semantic_ens", ENS)}


def query_durations(cfg, split: Split) -> dict[str, np.ndarray]:
    """Per-query durations (s) pooled over folds in the protocol's order: segment (end − start) and clip."""
    segs = manifest.read_segments(cfg)
    seg_d = {p: (g.end_s - g.start_s).to_numpy(float) for p, g in segs.groupby("pair_id")}
    pairs = manifest.read_pairs(cfg)
    clip_d = dict(zip(pairs.pair_id, pairs.video_duration_s.astype(float)))
    seg = np.concatenate([np.concatenate([seg_d[p] for p in split.folds[k]["test"]]) for k in sorted(split.folds)])
    clip = np.concatenate([[clip_d[p] for p in split.q[(k, "track_test_fold")]] for k in sorted(split.folds)])
    assert len(seg) == sum(len(split.q[(k, "segment_all")]) for k in split.folds)
    return {"segment_all": seg, "track_test_fold": clip}


def pooled(split: Split, runs: dict, sec: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Ranks per run index (seed position), each pooled over folds, and the matching film clusters."""
    folds = sorted(runs)
    n = min(len(runs[k]) for k in folds)
    ranks = [np.concatenate([runs[k][i][sec] for k in folds]) for i in range(n)]
    return ranks, np.concatenate([split.clusters(k, sec) for k in folds])


# ---------------------------------------------------------------- bootstrap helpers
def median_ci(ranks: list[np.ndarray], clusters: np.ndarray, seed: int = 0) -> dict:
    """Median of every (run, query) rank, CI from resampling film groups (all runs of a query move together)."""
    v = np.concatenate(ranks).astype(float)
    cl = np.tile(clusters, len(ranks))
    uniq, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(v, kind="stable")
    v, inv = v[order], inv[order]
    rng = np.random.default_rng(seed)
    meds = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        c = np.cumsum(w)
        meds[b] = v[np.searchsorted(c, c[-1] / 2)]
    return {"mean": float(np.median(v)), "lo": float(np.quantile(meds, 0.025)), "hi": float(np.quantile(meds, 0.975))}


def diff_ci(values: np.ndarray, clusters: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """mean(values[a]) − mean(values[b]) over queries, CI from resampling film groups jointly."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sa, na = (np.bincount(inv, w, len(uniq)) for w in (values * a, a.astype(float)))
    sb, nb = (np.bincount(inv, w, len(uniq)) for w in (values * b, b.astype(float)))
    d = np.random.default_rng(seed).integers(0, len(uniq), (B, len(uniq)))
    boot = sa[d].sum(1) / na[d].sum(1) - sb[d].sum(1) / nb[d].sum(1)
    return {"mean": float(values[a].mean() - values[b].mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975))}


def _abs(c: dict, d: int = 3) -> str:
    return f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


# ---------------------------------------------------------------- 1. main table
def section_main(split: Split, systems: dict) -> list[str]:
    sec = "segment_all"
    gal = np.concatenate([split.N[(k, sec)] for k in split.folds])
    nq = len(gal)
    nf = len(np.unique(np.concatenate([split.clusters(k, sec) for k in split.folds])))
    L = [f"Queries: {nq:,} test segments, {nf} film groups. Gallery per query: every test segment's aligned music window in the "
         f"fold, other same-track videos left out: {gal.mean():.0f} windows on average ({gal.min()}–{gal.max()}); one correct window.", "",
         "| system | runs | R@1 | R@5 | R@10 | MedR ↓ | MRR | R@10, other-video windows |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    ch, cho = _chance(split, sec, False), _chance(split, "segment_other", False)
    L.append(f"| random (exact chance) | — | {_cell(ch['R@1'])} | {_cell(ch['R@5'])} | {_cell(ch['R@10'])} | "
             f"{np.mean([(n + 1) / 2 for n in gal]):.1f} (expected) | — | {_cell(cho['R@10'])} |")
    for name, runs in systems.items():
        r, ro = _row(split, runs, sec, False), _row(split, runs, "segment_other", False)
        L.append(f"| {name} | {sum(len(v) for v in runs.values())} | " +
                 " | ".join(_cell(r[m]) for m in ("R@1", "R@5", "R@10", "MedR", "MRR")) + f" | {_cell(ro['R@10'])} |")
    L += ["", "Pooled values with 95% film-group CIs (per-query values averaged over a fold's runs; MedR: median over all "
          "(run, query) ranks):", "", "| system | R@1 | R@5 | R@10 | MedR ↓ | MRR | R@10, other-video |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, runs in systems.items():
        cells = []
        for k in (1, 5, 10):
            h, cl = _hits(split, runs, sec, k, False)
            cells.append(_abs(cluster_bootstrap(h, cl, B)))
        ranks, cl = pooled(split, runs, sec)
        cells.append(_abs(median_ci(ranks, cl), 1))
        rr = np.mean([1 / r for r in ranks], 0)
        cells.append(_abs(cluster_bootstrap(rr, cl, B)))
        h, cl = _hits(split, runs, "segment_other", 10, False)
        cells.append(_abs(cluster_bootstrap(h, cl, B)))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    ens, single, ib = (systems[n] for n in ("**final: 10-head ensemble**", "single heads (seeds 0–2)", "ImageBind zero-shot"))
    L += ["", "Paired differences (per-query hit averaged over runs; film-group CIs):", ""]
    for a, b, label in ((ens, ib, "final ensemble − ImageBind zero-shot"), (ens, single, "final ensemble − single heads"),
                        (single, ib, "single heads − ImageBind zero-shot")):
        parts = []
        for k in (1, 5, 10):
            ha, cl = _hits(split, a, sec, k, False)
            hb, _ = _hits(split, b, sec, k, False)
            parts.append(f"R@{k} {_ci(cluster_bootstrap(ha - hb, cl, B))}")
        L.append(f"- {label}: " + "; ".join(parts) + ".")
    L += ["", "Per fold, final ensemble (mean of its 3 ensembles; single-fold numbers): " + "; ".join(
        f"fold {k} R@1 {np.mean([(r[sec] <= 1).mean() for r in ens[k]]):.3f}, R@10 {np.mean([(r[sec] <= 10).mean() for r in ens[k]]):.3f}, "
        f"MedR {np.mean([np.median(r[sec]) for r in ens[k]]):.1f} ({len(ens[k][0][sec])} queries)" for k in sorted(ens)) + "."]
    return L + [""]


# ---------------------------------------------------------------- 2. gallery size
def subsampled_hit(ranks: np.ndarray, N: np.ndarray, h: np.ndarray, n: int, k: int) -> np.ndarray:
    """Exact P(hit@k) per query when its gallery is cut to n items at random (all h relevant items kept)."""
    out = np.full(len(ranks), np.nan)
    ok = n <= N
    M, K, draw = N[ok] - h[ok], ranks[ok] - 1, n - h[ok]
    out[ok] = hypergeom.cdf(k - 1, M, K, draw)
    return out


def section_gallery(split: Split, systems: dict, n_files: int) -> list[str]:
    L = ["**Track retrieval as run** (query = whole test clip; hit = any file of its track group):", "",
         "| gallery | files | system | R@1 | R@5 | R@10 [95% CI] | MedR ↓ |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for sec, label in (("track_test_fold", "test fold's music"), ("track_all", "every music file")):
        gal = np.concatenate([split.N[(k, sec)] for k in split.folds])
        size = f"{gal.mean():.0f} ({gal.min()}–{gal.max()})" if sec == "track_test_fold" else f"{n_files:,}"
        ch = _chance(split, sec, False)
        L.append(f"| {label} | {size} | random (exact chance) | {_cell(ch['R@1'])} | {_cell(ch['R@5'])} | {_cell(ch['R@10'])} | — |")
        for name, runs in systems.items():
            r = _row(split, runs, sec, False)
            h, cl = _hits(split, runs, sec, 10, False)
            c = cluster_bootstrap(h, cl, B)
            L.append(f"| {label} | {size} | {name} | {_cell(r['R@1'])} | {_cell(r['R@5'])} | {_cell(r['R@10'])} "
                     f"[{c['lo']:.3f}, {c['hi']:.3f}] | {_cell(r['MedR'])} |")
    L += ["", "**Expected R@10 with the gallery cut to n items at random** (exact, from the ranks; every query's relevant items kept; "
          "per-query value averaged over runs; film-group CIs). Track: cut from the every-file gallery. Segment: cut from the "
          "test-fold window gallery. A cut gallery is a random sample of the library, unlike the test fold's own music, which "
          "comes from the same few films.", ""]
    for sec, grid, label in (("track_all", TRACK_N + (n_files,), "Track, gallery size n (files)"),
                             ("segment_all", SEG_N, "Segment, gallery size n (windows)")):
        full = sec == "segment_all"
        gal = np.concatenate([split.N[(k, sec)] for k in split.folds])
        cols = list(grid) + (["full"] if full else [])
        L += [f"| {label} | " + " | ".join(str(c) if c != "full" else f"full ({gal.mean():.0f})" for c in cols) + " |",
              "| --- |" + " --- |" * len(cols)]
        hrel = np.concatenate([split.h[(k, sec)] for k in split.folds])
        chance = [np.mean([chance_recall(n, hh, 10) for hh in hrel]) if n != "full" else
                  np.mean([chance_recall(N, hh, 10) for N, hh in zip(gal, hrel)]) for n in cols]
        L.append("| random | " + " | ".join(f"{c:.3f}" for c in chance) + " |")
        for name, runs in systems.items():
            cells = []
            for n in cols:
                per_k = []
                for k in sorted(runs):
                    N, hh = split.N[(k, sec)], split.h[(k, sec)]
                    per_k.append(np.mean([(r[sec] <= 10).astype(float) if n == "full" else subsampled_hit(r[sec], N, hh, n, 10)
                                          for r in runs[k]], 0))
                v = np.concatenate(per_k)
                cl = np.concatenate([split.clusters(k, sec) for k in sorted(runs)])
                cells.append(_abs(cluster_bootstrap(v, cl, B)) if name.startswith("**") else f"{np.nanmean(v):.3f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    return L


# ---------------------------------------------------------------- 3. duration
def section_duration(split: Split, systems: dict, dur: dict) -> list[str]:
    L = []
    for sec, what, unit in (("segment_all", "Segment retrieval by segment duration", "segment"),
                            ("track_test_fold", "Track retrieval (test-fold gallery) by clip duration", "clip")):
        d = dur[sec]
        edges = np.quantile(d, [0.25, 0.5, 0.75])
        q = np.searchsorted(edges, d, side="right")
        gal = np.concatenate([split.N[(k, sec)] for k in split.folds])
        hrel = np.concatenate([split.h[(k, sec)] for k in split.folds])
        names = list(systems)
        L += [f"**{what}** (quartiles of the pooled test {unit}s; edges {edges[0]:.1f} / {edges[1]:.1f} / {edges[2]:.1f} s):", "",
              f"| {unit} duration | queries | median (s) | random R@10 | " + " | ".join(f"{n} R@10" for n in names) +
              " | ensemble − ImageBind |", "| --- |" + " --- |" * (3 + len(names) + 1)]
        hits = {}
        for n in names:
            hits[n], cl = _hits(split, systems[n], sec, 10, False)
        lo_hi = [f"< {edges[0]:.1f} s", f"{edges[0]:.1f}–{edges[1]:.1f} s", f"{edges[1]:.1f}–{edges[2]:.1f} s", f"≥ {edges[2]:.1f} s"]
        ens, ib = names[-1], names[0]
        for b in range(4):
            m = q == b
            ch = np.mean([chance_recall(N, hh, 10) for N, hh in zip(gal[m], hrel[m])])
            cells = [_abs(cluster_bootstrap(hits[n][m], cl[m], B)) if n in (ens, ib) else f"{hits[n][m].mean():.3f}" for n in names]
            delta = _ci(cluster_bootstrap(hits[ens][m] - hits[ib][m], cl[m], B))
            L.append(f"| Q{b + 1}: {lo_hi[b]} | {m.sum():,} | {np.median(d[m]):.1f} | {ch:.3f} | " + " | ".join(cells) + f" | {delta} |")
        L += ["", f"- Longest − shortest quartile (Q4 − Q1), R@10: " + "; ".join(
            f"{n.strip('*')} {_ci(diff_ci(hits[n], cl, q == 3, q == 0))}" for n in names) + ".",
              f"- Spearman ρ between {unit} duration and the ensemble's per-query hit@10: "
              f"{_spearman(d, hits[ens]):+.3f}.", ""]
    return L


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


# ---------------------------------------------------------------- report
def main() -> Path:
    L = ["# Segment-level retrieval (paper §5; D126)", "",
         "Generated by `python -m bgm.analysis.segment_retrieval` (src/bgm/analysis/segment_retrieval.py) from the saved per-query "
         "ranks of runs/titlesplit/ (original) and runs/cleaned/ (cleaned); nothing re-scored or re-trained. Title split "
         "(folds_title.json), Konjum salangai train-only. Runs: ImageBind 5 (one per fold), single heads 15, final ensemble 15 "
         f"(5 folds × 3 disjoint 10-head ensembles). Mean ± std over runs; 95% CIs: cluster bootstrap over film groups, {B:,} resamples.", ""]
    for name, over, desc in DATASETS:
        cfg = make_cfg(over)
        split = Split(cfg, "folds_title.json", "film")
        systems = load_systems(Path(cfg.paths.runs))
        n_files = manifest.read_pairs(cfg).music_id.nunique()
        log.info("%s: %d folds", name, len(split.folds))
        L += [f"## {name.capitalize()} dataset ({desc})", "", "### 1. Segment retrieval", ""]
        L += section_main(split, systems)
        L += ["### 2. Gallery size", ""] + section_gallery(split, systems, n_files)
        L += ["### 3. Duration", ""] + section_duration(split, systems, query_durations(cfg, split))
    out = Path("results/segment_retrieval.md")
    out.write_text("\n".join(L))
    log.info("wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
