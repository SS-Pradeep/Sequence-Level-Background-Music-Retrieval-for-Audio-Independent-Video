"""results/semantic_filmsplit.md: semantic heads and zero-shot baselines on the film-level split (PROGRESS.md D36),
side by side with the track_id split, and results/learning_curve.md (step 3).

* Film split: runs/filmsplit/{zeroshot,semantic/<variant>}; test folds hold 982 clips (Konjum salangai is train-only).
  Bootstrap clusters are film groups (every track group lies inside one; PROGRESS.md D39).
* Track_id split: runs/{zeroshot,semantic/<variant>}, all 1,248 test clips (as results/semantic_<variant>.md, track
  clusters) and the same restricted to the 982 non-Konjum clips (the film split's query set).
* Per-query hits are averaged over seeds before any comparison (SPEC §5.12). Gallery sizes are per query, averaged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.eval.metrics import chance_summary, cluster_bootstrap, recall_summary

SECTIONS = {"track_test_fold": "Track retrieval, test-fold gallery", "track_all": "Track retrieval, all-library gallery",
            "segment_all": "Segment retrieval, all windows", "segment_other": "Segment retrieval, other-video windows"}
METRICS = ("R@1", "R@5", "R@10", "MedR", "MRR")
VARIANTS = ("siglip-both", "siglip-clap", "siglip-mert", "clip-clap")


class Split:
    """One split's folds, per-section query lists (pair_id per query, in the protocol's order), gallery sizes N and
    relevant counts h per query, and bootstrap clusters."""

    def __init__(self, cfg: DictConfig, folds_file: str, cluster: str):
        doc = manifest.read_folds(_with_folds(cfg, folds_file))
        self.folds = {int(k): v for k, v in doc["folds"].items()}
        pairs = manifest.read_pairs(cfg)
        self.track = dict(zip(pairs.pair_id, pairs.track_id))
        music_of = dict(zip(pairs.pair_id, pairs.music_id))
        music_track = dict(zip(pairs.music_id, pairs.track_id))
        segs = manifest.read_segments(cfg)
        nseg = segs.groupby("pair_id").size().to_dict()
        film = doc.get("film_group")
        if film is None:
            from bgm.data.films import film_groups
            film = film_groups(pairs)
        self.film = film
        self.konjum = {p for p in pairs.pair_id if film[p] == "full_bgm_dataset:Konjum salangai"}
        self.q, self.N, self.h = {}, {}, {}
        all_music = list(dict.fromkeys(pairs.music_id))
        for k, f in self.folds.items():
            test = f["test"]
            gal_test = list(dict.fromkeys(music_of[p] for p in test))
            for sec, gal in (("track_test_fold", gal_test), ("track_all", all_music)):
                gt = np.array([music_track[m] for m in gal])
                self.q[(k, sec)] = list(test)
                self.N[(k, sec)] = np.full(len(test), len(gal))
                self.h[(k, sec)] = np.array([(gt == self.track[p]).sum() for p in test])
            seg_q = [p for p in test for _ in range(nseg[p])]
            tot = sum(nseg[p] for p in test)
            by_track = {}
            for p in test:
                by_track[self.track[p]] = by_track.get(self.track[p], 0) + nseg[p]
            n_all = np.array([tot - (by_track[self.track[p]] - nseg[p]) for p in seg_q])     # minus other same-track videos
            n_other = np.array([n - nseg[p] + 1 for n, p in zip(n_all, seg_q)])            # also minus own other windows
            for sec, n in (("segment_all", n_all), ("segment_other", n_other)):
                self.q[(k, sec)], self.N[(k, sec)], self.h[(k, sec)] = seg_q, n, np.ones(len(seg_q), int)
        self.cluster_of = self.film if cluster == "film" else self.track

    def clusters(self, k: int, sec: str) -> np.ndarray:
        return np.array([self.cluster_of[p] for p in self.q[(k, sec)]])

    def mask(self, k: int, sec: str, drop_konjum: bool) -> np.ndarray:
        return np.array([not (drop_konjum and p in self.konjum) for p in self.q[(k, sec)]])


def _runs(root: Path, variant: str, kind: str = "semantic") -> dict[int, list[dict]]:
    out = {}
    for p in sorted(root.glob(f"{kind}/{variant}/fold*/seed*/perquery.npz")):
        with np.load(p) as z:
            m = json.loads((p.parent / "metrics.json").read_text())
            out.setdefault(int(p.parent.parent.name[4:]), []).append({**{s: z[s] for s in SECTIONS}, "val": m["val"]})
    return out


def _zeroshot(root: Path, system: str = "imagebind") -> dict[int, dict]:
    out = {}
    for p in sorted(root.glob("zeroshot/fold*/perquery.npz")):
        with np.load(p) as z:
            out[int(p.parent.name[4:])] = {s: z[f"{s}/{system}"] for s in SECTIONS}
    return out


def _cell(vals) -> str:
    v = np.asarray(vals, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def _row(split: Split, runs: dict[int, list], sec: str, drop_konjum: bool) -> dict:
    """Per-run metrics (mean ± std over runs) on the kept queries; runs: {fold: [{sec: ranks}]}."""
    per_run = []
    for k, rs in runs.items():
        m = split.mask(k, sec, drop_konjum)
        for r in rs:
            assert len(r[sec]) == len(m), (k, sec, len(r[sec]), len(m))
            per_run.append(recall_summary(r[sec][m]))
    return {x: [r[x] for r in per_run] for x in METRICS}


def _hits(split: Split, runs: dict[int, list], sec: str, kk: int, drop_konjum: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-query hit at k averaged over the fold's runs, pooled over folds, and the matching clusters."""
    h, cl = [], []
    for k in sorted(runs):
        m = split.mask(k, sec, drop_konjum)
        h.append(np.mean([(r[sec] <= kk).astype(float) for r in runs[k]], 0)[m])
        cl.append(split.clusters(k, sec)[m])
    return np.concatenate(h), np.concatenate(cl)


def _chance(split: Split, sec: str, drop_konjum: bool) -> dict:
    per = [chance_summary(split.N[(k, sec)][m], split.h[(k, sec)][m]) for k in split.folds
           for m in [split.mask(k, sec, drop_konjum)]]
    return {x: [c[x] for c in per] for x in ("R@1", "R@5", "R@10")}


def _gallery(split: Split, sec: str, drop_konjum: bool) -> str:
    n = np.concatenate([split.N[(k, sec)][split.mask(k, sec, drop_konjum)] for k in split.folds])
    return f"{n.mean():.0f} ({n.min()}–{n.max()})"


def write_filmsplit_report(cfg: DictConfig, new_root: str = "runs/filmsplit", old_root: str = "runs") -> Path:
    film = Split(cfg, "folds_film.json", "film")
    old = Split(cfg, "folds.json", "track")
    B = cfg.eval.bootstrap_resamples
    new_runs = {v: _runs(Path(new_root), v) for v in VARIANTS}
    old_runs = {v: _runs(Path(old_root), v) for v in VARIANTS}
    zs_new, zs_old = _zeroshot(Path(new_root)), _zeroshot(Path(old_root))
    as_runs = lambda zs: {k: [v] for k, v in zs.items()}  # noqa: E731
    main = "siglip-both"
    n_runs = sum(len(v) for v in new_runs[main].values())
    L = ["# Semantic heads on the film-level split", "",
         "Film split (PROGRESS.md D36): folds grouped by film (405 film groups: `meta_source:meta_film`, merged when they "
         "share a track_id); Konjum salangai (266 clips) is in every fold's train set and never in val or test, so the test "
         f"folds hold the other 982 clips. Heads: {main} unless named, {n_runs} runs (5 folds × 3 seeds), per-film cap of "
         "30 clips per epoch. The track_id split is the earlier protocol (results/semantic_siglip-both.md).", "",
         "Cells: mean ± std over runs (zero-shot: over folds). Gallery: files or windows per query, mean (min–max). Δ: "
         "per-query hits averaged over seeds, paired, 95% cluster bootstrap over **film groups** for the film split and "
         "over **track groups** for the track_id split (as before).", ""]
    verdict = None
    for sec, title in SECTIONS.items():
        L += [f"## {title}", "", "| split | queries | system | gallery | R@1 | R@5 | R@10 | MedR | MRR |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        blocks = [("**film**", film, new_runs[main], zs_new, False, "982"),
                  ("track_id", old, old_runs[main], zs_old, False, "1,248"),
                  ("track_id", old, old_runs[main], zs_old, True, "982 (no Konjum)")]
        for name, sp, runs, zs, dropk, nq in blocks:
            if not runs:
                L.append(f"| {name} | {nq} | heads | — | not run yet | | | | |")
                continue
            gal = _gallery(sp, sec, dropk)
            r = _row(sp, runs, sec, dropk)
            L.append(f"| {name} | {nq} | **heads** | {gal} | " + " | ".join(_cell(r[x]) for x in METRICS) + " |")
            if zs:
                z = _row(sp, as_runs(zs), sec, dropk)
                L.append(f"| {name} | {nq} | ImageBind | {gal} | " + " | ".join(_cell(z[x]) for x in METRICS) + " |")
            c = _chance(sp, sec, dropk)
            L.append(f"| {name} | {nq} | random (exact) | {gal} | " + " | ".join(_cell(c[x]) for x in ("R@1", "R@5", "R@10")) + " | — | — |")
        L.append("")
        for name, sp, runs, zs, dropk, nq in blocks:
            common = sorted(set(runs) & set(zs))                 # pair only folds present on both sides
            if not common:
                continue
            for kk in (1, 10):
                a, cl = _hits(sp, {k: runs[k] for k in common}, sec, kk, dropk)
                b, _ = _hits(sp, {k: [zs[k]] for k in common}, sec, kk, dropk)
                bs = cluster_bootstrap(a - b, cl, B)
                L.append(f"- {name} split, {nq} queries: Δ R@{kk} heads − ImageBind **{bs['mean']:+.3f}** [{bs['lo']:+.3f}, "
                         f"{bs['hi']:+.3f}] ({bs['n_queries']} queries, {bs['n_clusters']} clusters)")
                if sp is film and sec == "track_test_fold" and kk == 10:
                    verdict = bs
        L.append("")
    # variants
    L += ["## Variants (track R@10, test-fold gallery)", "",
          "| variant | film split | track_id split, 1,248 | track_id split, 982 (no Konjum) | film − track_id (982), R@10 | Δ vs siglip-clap, film split [95% CI] |",
          "| --- | --- | --- | --- | --- | --- |"]
    sec = "track_test_fold"
    base_runs = new_runs["siglip-clap"]
    for v in VARIANTS:
        if not new_runs[v]:
            L.append(f"| {v} | not run yet | | | | |")
            continue
        n = _row(film, new_runs[v], sec, False)["R@10"]
        o = _row(old, old_runs[v], sec, False)["R@10"] if old_runs[v] else [np.nan]
        o9 = _row(old, old_runs[v], sec, True)["R@10"] if old_runs[v] else [np.nan]
        common = sorted(set(new_runs[v]) & set(base_runs))            # pair only folds present on both sides
        d = "—"
        if v != "siglip-clap" and common:
            a, cl = _hits(film, {k: new_runs[v][k] for k in common}, sec, 10, False)
            b, _ = _hits(film, {k: base_runs[k] for k in common}, sec, 10, False)
            bs = cluster_bootstrap(a - b, cl, B)
            d = f"{bs['mean']:+.3f} [{bs['lo']:+.3f}, {bs['hi']:+.3f}]" + ("" if len(common) == len(film.folds) else f" ({len(common)} folds)")
        runs_note = "" if sum(len(r) for r in new_runs[v].values()) == 15 else f" ({sum(len(r) for r in new_runs[v].values())} of 15 runs)"
        L.append(f"| {v} | {_cell(n)}{runs_note} | {_cell(o)} | {_cell(o9)} | {np.mean(n) - np.mean(o9):+.3f} | {d} |")
    L.append("")
    if verdict is not None:
        L += ["## H1 on the film split", "", f"Trained heads beat ImageBind zero-shot on test-fold track R@10: "
              f"**{'yes' if verdict['lo'] > 0 else 'not shown'}** (Δ {verdict['mean']:+.3f}, 95% CI [{verdict['lo']:+.3f}, {verdict['hi']:+.3f}], "
              "film-group clusters)."]
        val = [r["val"] for rs in new_runs[main].values() for r in rs]
        L += ["", f"Validation (early stopping, film split): segment R@10 {np.mean([v['segment_R@10'] for v in val]):.3f} ± "
              f"{np.std([v['segment_R@10'] for v in val], ddof=1):.3f}; best epoch median {np.median([v['best_epoch'] for v in val]):.0f}."]
    out = Path(cfg.paths.results) / "semantic_filmsplit.md"
    out.write_text("\n".join(L) + "\n")
    return out


# ---------------------------------------------------------------- published-method baselines (D43)
BASELINES = {"cfd": "CFD-style dual path (content + emotion)", "utcmvmr": "UT-CMVMR-style (rhythm + optical flow, unified target)",
             "mvpt": "MVPt-style temporal context"}


def write_baselines_filmsplit(cfg: DictConfig, root: str = "runs/filmsplit", ours: str = "siglip-both") -> Path:
    """results/baselines_filmsplit.md: our heads vs published methods reimplemented on our features vs zero-shot
    systems, all on the film-split folds (982 test queries). Δ vs our heads: per-query hits averaged over seeds, paired,
    95% cluster bootstrap over film groups, on the folds both sides have."""
    film = Split(cfg, "folds_film.json", "film")
    B = cfg.eval.bootstrap_resamples
    systems = [(f"**ours: semantic heads ({ours})**", _runs(Path(root), ours))]
    systems += [(f"{BASELINES[v]}, reimpl.", _runs(Path(root), v, "baselines")) for v in BASELINES]
    systems += [(f"{n} zero-shot", {k: [v] for k, v in _zeroshot(Path(root), z).items()})
                for n, z in (("ImageBind", "imagebind"), ("text bridge", "text_bridge"), ("mood attributes", "attribute"))]
    ref = systems[0][1]
    L = ["# Baselines on the film split", "",
         "All systems on the same film-split folds (PROGRESS.md D36; 982 test clips, Konjum salangai train-only) and, for trained "
         "systems, the same seeds (0, 1, 2), training data, positives, batch rule, per-film cap and early stopping as our heads. "
         "The three published methods are **reimplemented on our frozen features** (SigLIP, CLAP, MERT, mood logits, dynamics "
         "curves; src/bgm/train/baselines.py, D43): they test each method's idea on this data, and are not the authors' code or "
         "numbers. Cells: mean ± std over runs (zero-shot: over folds). Δ vs ours: paired, per-query hits averaged over seeds, "
         "95% cluster bootstrap over film groups.", ""]
    for sec, title in SECTIONS.items():
        L += [f"## {title} (gallery {_gallery(film, sec, False)})", "",
              "| system | runs | R@1 | R@5 | R@10 | MedR | MRR | Δ R@10 vs ours [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, runs in systems:
            if not runs:
                L.append(f"| {name} | 0 | not run yet | | | | | |")
                continue
            r = _row(film, runs, sec, False)
            d = "—"
            common = sorted(set(runs) & set(ref))
            if runs is not ref and common:
                a, cl = _hits(film, {k: runs[k] for k in common}, sec, 10, False)
                b, _ = _hits(film, {k: ref[k] for k in common}, sec, 10, False)
                bs = cluster_bootstrap(a - b, cl, B)
                d = f"{bs['mean']:+.3f} [{bs['lo']:+.3f}, {bs['hi']:+.3f}]" + ("" if len(common) == len(film.folds) else f" ({len(common)} folds)")
            L.append(f"| {name} | {sum(len(v) for v in runs.values())} | " + " | ".join(_cell(r[x]) for x in METRICS) + f" | {d} |")
        c = _chance(film, sec, False)
        L += [f"| random (exact) | — | " + " | ".join(_cell(c[x]) for x in ("R@1", "R@5", "R@10")) + " | — | — | |", ""]
    out = Path(cfg.paths.results) / "baselines_filmsplit.md"
    out.write_text("\n".join(L) + "\n")
    return out


# ---------------------------------------------------------------- pretraining (H4, step 2)
def write_pretraining_filmsplit(cfg: DictConfig, root: str = "runs/filmsplit", scratch: str = "siglip-both",
                                pretrained: str = "siglip-both-pt") -> Path:
    """results/pretraining_filmsplit.md: M8-pretrained + fine-tuned heads vs heads from scratch, both on the film split
    (5 folds x 3 seeds each; per-query hits averaged over seeds; paired cluster bootstrap over film groups)."""
    film = Split(cfg, "folds_film.json", "film")
    a_runs, b_runs = _runs(Path(root), pretrained), _runs(Path(root), scratch)
    B = cfg.eval.bootstrap_resamples
    L = ["# Unlabeled pretraining on the film split (H4)", "",
         f"Heads: SigLIP + CLAP+MERT. From scratch: runs/filmsplit/semantic/{scratch}; pretrained on the M8 pool, then "
         f"fine-tuned at lr_finetune: runs/filmsplit/semantic/{pretrained}. 5 folds × 3 seeds each, 982 test queries. Δ: "
         "per-query hits averaged over seeds, paired, 95% cluster bootstrap over film groups.", "",
         "| section | metric | from scratch | pretrained + fine-tuned | Δ [95% CI] |", "| --- | --- | --- | --- | --- |"]
    for sec in SECTIONS:
        for kk in (1, 10):
            a, cl = _hits(film, a_runs, sec, kk, False)
            b, _ = _hits(film, b_runs, sec, kk, False)
            bs = cluster_bootstrap(a - b, cl, B)
            L.append(f"| {SECTIONS[sec]} | R@{kk} | {b.mean():.3f} | {a.mean():.3f} | **{bs['mean']:+.3f}** [{bs['lo']:+.3f}, {bs['hi']:+.3f}] |")
    a, cl = _hits(film, a_runs, "track_test_fold", 10, False)
    b, _ = _hits(film, b_runs, "track_test_fold", 10, False)
    bs = cluster_bootstrap(a - b, cl, B)
    L += ["", f"**H4** (pretraining beats training on the pairs alone, track R@10, test-fold gallery): "
          f"**{'supported' if bs['lo'] > 0 else 'not shown'}** (Δ {bs['mean']:+.3f} [{bs['lo']:+.3f}, {bs['hi']:+.3f}])."]
    out = Path(cfg.paths.results) / "pretraining_filmsplit.md"
    out.write_text("\n".join(L) + "\n")
    return out


# ---------------------------------------------------------------- learning curve (step 3)
FRACTIONS = {0.25: "lc-f025", 0.5: "lc-f050", 0.75: "lc-f075", 1.0: "siglip-both"}


def _cluster_ci(values: np.ndarray, clusters: np.ndarray, stat, n: int, seed: int = 0) -> tuple[float, float, float]:
    """stat(values) with a 95% CI from resampling whole clusters (for statistics that are not means, e.g. the median rank)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = [stat(values[np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])]) for _ in range(n)]
    return float(stat(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_learning_curve(cfg: DictConfig, root: str = "runs/filmsplit", seeds=(0, 1, 2)) -> Path:
    """results/learning_curve.md + .png/.pdf: test track metrics (test-fold gallery) against the number of non-Konjum
    training films. Seed s trains on its own nested film subsets (subset_seed = s, D45); 100% = siglip-both. A seed counts
    for a fraction only when all 5 folds are done. R@k: per-query hits averaged over seeds; MedR: mean over seeds of the
    pooled median rank. CIs: cluster bootstrap over film groups (982 queries pooled over folds)."""
    film = Split(cfg, "folds_film.json", "film")
    train_only = set(manifest.read_folds(_with_folds(cfg, "folds_film.json"))["train_only"])
    B = cfg.eval.bootstrap_resamples
    sec = "track_test_fold"
    folds = sorted(film.folds)
    cl = np.concatenate([film.clusters(k, sec) for k in folds])
    rows, ranks = [], {}
    for frac, v in FRACTIONS.items():
        R = {}
        for s in seeds:
            paths = [Path(root) / "semantic" / v / f"fold{k}" / f"seed{s}" / "perquery.npz" for k in folds]
            if all(p.exists() for p in paths):
                R[s] = np.concatenate([np.load(p)[sec] for p in paths])
        if not R:
            continue
        M = np.stack([R[s] for s in sorted(R)], 1)                    # [Q, S] ranks
        ranks[frac] = (dict(R), M)
        n_films = [np.ceil(frac * len({film.film[p] for p in film.folds[k]["train"]} - train_only)) for k in folds]
        row = {"fraction": frac, "films": float(np.mean(n_films)), "seeds": sorted(R)}
        for name, kk in (("R@10", 10), ("R@1", 1)):
            bs = cluster_bootstrap((M <= kk).mean(1), cl, B)
            row[name] = (bs["mean"], bs["lo"], bs["hi"])
            row[f"{name}_seeds"] = [(M[:, j] <= kk).mean() for j in range(M.shape[1])]
        row["MedR"] = _cluster_ci(M, cl, lambda x: np.median(x, 0).mean(), 2000)
        row["MedR_seeds"] = list(np.median(M, 0))
        rows.append(row)
    n_seeds = sorted({len(r["seeds"]) for r in rows})
    L = ["# Learning curve: semantic heads on the film split", "",
         f"SigLIP + CLAP+MERT heads, seeds {', '.join(map(str, seeds))} (seeds per point: {n_seeds}), 5 folds pooled (982 test queries, "
         "test-fold gallery ≈ 196 files). Per fold and seed, nested random subsets of the training films other than Konjum "
         "salangai (25 / 50 / 75 / 100%; each seed draws its own subsets, so the spread includes which films are used); Konjum "
         "stays in, with the 30-clips-per-epoch cap; val and test are unchanged. 100% = runs/filmsplit/semantic/siglip-both. "
         "R@k: per-query hits averaged over seeds; MedR: mean over seeds of the median rank. 95% CIs: cluster bootstrap over "
         "film groups; ± is the std over seeds.", "",
         "![learning curve](learning_curve.png)", "",
         "| training films (mean over folds) | share | R@10 [95% CI] | R@1 [95% CI] | MedR [95% CI] |", "| --- | --- | --- | --- | --- |"]
    sd = lambda v: f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else ""  # noqa: E731
    for r in rows:
        f = lambda t, d=3: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"  # noqa: E731
        L.append(f"| {r['films']:.0f} | {r['fraction']:.0%} | {f(r['R@10'])}{sd(r['R@10_seeds'])} | {f(r['R@1'])}{sd(r['R@1_seeds'])} | "
                 f"{f(r['MedR'], 1)} |")
    L += [""]
    for a, b in ((0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (0.25, 0.5)):
        if a in ranks and b in ranks:
            common = sorted(set(ranks[a][0]) & set(ranks[b][0]))
            ra = np.stack([ranks[a][0][s] for s in common], 1)
            rb = np.stack([ranks[b][0][s] for s in common], 1)
            for kk in (10, 1):
                bs = cluster_bootstrap((rb <= kk).mean(1) - (ra <= kk).mean(1), cl, B)
                L.append(f"- Gain {a:.0%} → {b:.0%} of the films, R@{kk}: **{bs['mean']:+.3f}** [{bs['lo']:+.3f}, {bs['hi']:+.3f}] "
                         f"(seeds {common})")
            med = _cluster_ci(np.concatenate([ra, rb], 1), cl,
                              lambda x, n=len(common): np.median(x[:, n:], 0).mean() - np.median(x[:, :n], 0).mean(), 2000)
            L.append(f"- Change {a:.0%} → {b:.0%}, MedR: {med[0]:+.1f} [{med[1]:+.1f}, {med[2]:+.1f}] (negative = better)")
    if len(rows) >= 2:
        x = np.log2([r["films"] for r in rows])
        y = np.array([r["R@10"][0] for r in rows])
        slope = np.polyfit(x, y, 1)[0]
        L += ["", f"Least-squares slope of R@10 against log2(training films): {slope:+.3f} per doubling (descriptive; 4 points)."]
    if rows:
        _plot_learning_curve(rows, Path(cfg.paths.results) / "learning_curve.png", n_seeds)
    out = Path(cfg.paths.results) / "learning_curve.md"
    out.write_text("\n".join(L) + "\n")
    return out


def _with_folds(cfg: DictConfig, folds_file: str) -> DictConfig:
    c = copy.deepcopy(cfg)
    c.splits.folds_file = folds_file
    return c


def _plot_learning_curve(rows: list[dict], out: Path, n_seeds=(1,)) -> None:
    """Two panels on one x axis (recall, and median rank; never a dual y axis). Reference palette slots 1-2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1a1a19", "#6b6a63", "#e4e3dc"
    x = np.array([r["films"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    series = (("R@10", "#2a78d6", "o"), ("R@1", "#eb6834", "s"))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("training films other than Konjum salangai (mean over folds)", fontsize=8, color=ink)
        ax.set_xticks(x, [f"{v:.0f}\n{r['fraction']:.0%}" for v, r in zip(x, rows)])
    ax = axes[0]
    for name, col, mk in series:
        y = np.array([r[name] for r in rows])
        ax.fill_between(x, y[:, 1], y[:, 2], color=col, alpha=0.15, lw=0)
        ax.plot(x, y[:, 0], color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.5, label=f"track {name}")
        ax.annotate(f"{name} {y[-1, 0]:.3f}", (x[-1], y[-1, 0]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=ink)
    ax.set_ylim(0, None)
    ax.set_ylabel("test recall (test-fold gallery)", fontsize=8, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    y = np.array([r["MedR"] for r in rows])
    ax.fill_between(x, y[:, 1], y[:, 2], color="#2a78d6", alpha=0.15, lw=0)
    ax.plot(x, y[:, 0], color="#2a78d6", lw=2, marker="o", ms=6, mec="white", mew=1.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("median rank of the true track (lower is better)", fontsize=8, color=ink)
    seeds = f"{max(n_seeds)} seeds" if max(n_seeds) > 1 else "seed 0"
    fig.suptitle(f"Semantic heads, film split: test retrieval vs training films ({seeds}, 95% film-cluster CIs)", fontsize=9, color=ink)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")          # vector copy for the paper
    plt.close(fig)
