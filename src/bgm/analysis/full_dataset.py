"""Semantic heads on the full dataset (PROGRESS.md D49): 4,246 pairs, the extended film split
data/manifests_full/folds_film.json (the 1,248-pair film split with the new films added, `bgm folds --film --extend`).

Run: python -m bgm.analysis.full_dataset        -> results/full_dataset.md, reports/full/full_dataset.json

1. Paired, did the new data help? Queries: the base split's test pairs (data/manifests/folds_film.json) that kept
   their assignment (pairs moved by merged film groups are left out). Per fold, the gallery is those queries' music
   files (the base test-fold gallery minus moved pairs) or the fixed old library (all 1,248 base music files); segment
   retrieval uses the same queries' segments. The base runs (runs/filmsplit/semantic/siglip-both) and the full-data
   runs (runs/full/semantic/siglip-both) are re-scored from their checkpoints by the same code, so only the training
   data differs. Δ: per-query hits averaged over seeds, paired, 95% cluster bootstrap over film groups (D39).
2. Whole dataset: every test query of the extended split against its test-fold gallery (old + new music) and the
   whole library, from the runs' perquery.npz; ImageBind zero-shot (runs/full/zeroshot) and exact random for
   reference; all queries, old clips, new clips, and a film-balanced mean (each film group weighs the same).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.metrics import chance_summary, cluster_bootstrap, recall_summary
from bgm.eval.protocol import segment_retrieval, track_retrieval
from bgm.eval.report_filmsplit import SECTIONS, Split, _cell, _runs, _zeroshot
from bgm.eval.store import FeatureStore
from bgm.train.semantic import Scorer, SemanticInputs, SemanticModel

log = logging.getLogger(__name__)
BASE_FOLDS = Path("data/manifests/folds_film.json")
BASE_RUNS = Path("runs/filmsplit/semantic/siglip-both")
FULL_ROOT = Path("runs/full")
FULL_RUNS = FULL_ROOT / "semantic" / "siglip-both"
FULL_OVERRIDES = ["paths.manifests=data/manifests_full", "splits.folds_file=folds_film.json", "paths.runs=runs/full",
                  "paths.results=results/full", "paths.reports=reports/full", "train.semantic.music_feature=both",
                  "train.semantic.max_clips_per_film_per_epoch=30"]
PAIRED = ("track_test_fold", "track_old_library", "segment_all")
PAIRED_TITLES = {"track_test_fold": "Track retrieval, test-fold gallery", "track_old_library": "Track retrieval, old library (1,248 files)",
                 "segment_all": "Segment retrieval, all windows"}
METRICS = ("R@1", "R@5", "R@10", "MedR", "MRR")


def full_cfg() -> DictConfig:
    return load_config("configs/default.yaml", FULL_OVERRIDES)


class CachedScorer(Scorer):
    """Scorer that embeds each music window once (the window protocol re-embeds every window for every query)."""

    def window(self, q, a, b, windows):
        v = self._emb(("wv", q, round(a, 3), round(b, 3)), "video", lambda: self.inp.video(q, a, b))
        keys = [("mw", m, round(s, 3), round(d, 3)) for m, s, d in windows]
        todo = [(k, w) for k, w in zip(keys, windows) if k not in self._cache]
        if todo:
            E = self.model.embed("music", np.stack([self.inp.music_window(m, s, d) for _, (m, s, d) in todo]))
            self._cache.update({k: e for (k, _), e in zip(todo, E)})
        return np.stack([self._cache[k] for k in keys]) @ v


def load_scorer(run: Path, store: FeatureStore, device: str, cached: bool = True) -> Scorer:
    ck = torch.load(run / "ckpt.pt", map_location=device)
    rc = OmegaConf.load(run / "config.yaml")
    inp = SemanticInputs(store, rc.train.semantic.video_feature, rc.train.semantic.music_feature,
                         bool(rc.train.semantic.get('mert_layer_mix', False)))
    assert tuple(inp.dims) == (ck["dv"], ck["dm"]), (run, inp.dims, ck["dv"], ck["dm"])
    model = SemanticModel(ck["dv"], ck["dm"], rc, device)
    model.load(ck["state"])
    model.eval()
    return (CachedScorer if cached else Scorer)(model, inp)


def comparison_queries(cfg: DictConfig) -> tuple[dict[int, list[str]], dict]:
    """Per fold: the base split's test pairs that kept their assignment in the extended split."""
    base = json.loads(BASE_FOLDS.read_text())
    ext = manifest.read_folds(cfg)
    moved = set(ext.get("moved_from_base", {}))
    return {int(k): [p for p in f["test"] if p not in moved] for k, f in base["folds"].items()}, ext


def score_paired(cfg: DictConfig, store: FeatureStore, queries: dict[int, list[str]], seeds=(0, 1, 2), device: str = "cuda") -> dict:
    """{model: {sec: {fold: [ranks per seed]}}} for the base and full-data runs on the comparison queries."""
    base_pairs = json.loads(BASE_FOLDS.read_text())["film_group"]
    old_library = list(dict.fromkeys(store.music_of[p] for p in sorted(base_pairs)))
    out = {}
    for name, root in (("base", BASE_RUNS), ("full", FULL_RUNS)):
        for k, q in queries.items():
            gallery = list(dict.fromkeys(store.music_of[p] for p in q))
            for s in seeds:
                sc = load_scorer(root / f"fold{k}" / f"seed{s}", store, device)
                r = {"track_test_fold": track_retrieval(store, q, gallery, sc.track).ranks,
                     "track_old_library": track_retrieval(store, q, old_library, sc.track).ranks,
                     "segment_all": segment_retrieval(store, q, sc.window, "all").ranks}
                for sec, ranks in r.items():
                    out.setdefault(name, {}).setdefault(sec, {}).setdefault(k, []).append(np.asarray(ranks))
                log.info("%s fold %d seed %d: track R@10 %.3f, segment R@10 %.3f", name, k, s,
                         float((r["track_test_fold"] <= 10).mean()), float((r["segment_all"] <= 10).mean()))
    return out


def check_reproduces(store: FeatureStore, device: str = "cuda") -> bool:
    """Re-scoring a base run with its own test fold and gallery must give the ranks it saved when it trained."""
    base = json.loads(BASE_FOLDS.read_text())
    test = base["folds"]["0"]["test"]
    run = BASE_RUNS / "fold0" / "seed0"
    sc = load_scorer(run, store, device, cached=False)
    ranks = track_retrieval(store, test, list(dict.fromkeys(store.music_of[p] for p in test)), sc.track).ranks
    with np.load(run / "perquery.npz") as z:
        saved = z["track_test_fold"]
    same = bool(np.array_equal(np.asarray(ranks), saved))
    log.info("reproduction check (base fold 0 seed 0, track test-fold ranks): %s", "identical" if same else
             f"DIFFERENT in {int((np.asarray(ranks) != saved).sum())} of {len(saved)} queries")
    return same


def segment_queries(store: FeatureStore, ids: list[str]) -> list[str]:
    from bgm.eval.protocol import test_segments
    return [s[0] for s in test_segments(store, ids)]


def paired_tables(store: FeatureStore, queries: dict[int, list[str]], scores: dict, film: dict[str, str], n_boot: int) -> dict:
    res = {}
    for sec in PAIRED:
        q_of = {k: (q if sec.startswith("track") else segment_queries(store, q)) for k, q in queries.items()}
        cl = np.concatenate([[film[p] for p in q_of[k]] for k in sorted(queries)])
        rows = {}
        for name in ("base", "full"):
            per_run = [recall_summary(r) for k in sorted(queries) for r in scores[name][sec][k]]
            rows[name] = {m: [x[m] for x in per_run] for m in METRICS}
        deltas = {}
        for kk in (1, 5, 10):
            hits = {name: np.concatenate([np.mean([(r <= kk).astype(float) for r in scores[name][sec][k]], 0) for k in sorted(queries)])
                    for name in ("base", "full")}
            deltas[f"R@{kk}"] = cluster_bootstrap(hits["full"] - hits["base"], cl, n_boot)
        rr = {name: np.concatenate([np.mean([1.0 / r for r in scores[name][sec][k]], 0) for k in sorted(queries)]) for name in ("base", "full")}
        deltas["MRR"] = cluster_bootstrap(rr["full"] - rr["base"], cl, n_boot)
        res[sec] = {"rows": rows, "deltas": deltas, "n_queries": int(len(cl)), "n_clusters": int(len(set(cl)))}
    return res


def whole_dataset(cfg: DictConfig, old_ids: set[str], n_boot: int) -> dict:
    """Section 2 from the full-data runs' perquery.npz and the zero-shot run."""
    split = Split(cfg, "folds_film.json", "film")
    runs = _runs(FULL_ROOT, "siglip-both")
    zs = _zeroshot(FULL_ROOT)
    out = {}
    for sec in SECTIONS:
        out[sec] = {}
        for subset in ("all", "old", "new"):
            masks = {k: np.array([subset == "all" or ((p in old_ids) == (subset == "old")) for p in split.q[(k, sec)]]) for k in split.folds}
            heads = [recall_summary(r[sec][masks[k]]) for k in sorted(runs) for r in runs[k]]
            row = {"heads": {m: [x[m] for x in heads] for m in METRICS}, "n_queries": int(sum(m.sum() for m in masks.values())),
                   "gallery": [int(n) for k in split.folds for n in split.N[(k, sec)][masks[k]][:1]]}
            if zs:
                z = [recall_summary(zs[k][sec][masks[k]]) for k in sorted(zs)]
                row["imagebind"] = {m: [x[m] for x in z] for m in METRICS}
            ch = [chance_summary(split.N[(k, sec)][masks[k]], split.h[(k, sec)][masks[k]]) for k in split.folds]
            row["random"] = {m: [c[m] for c in ch] for m in ("R@1", "R@5", "R@10")}
            cl = np.concatenate([split.clusters(k, sec)[masks[k]] for k in sorted(runs)])
            h10 = np.concatenate([np.mean([(r[sec] <= 10).astype(float) for r in runs[k]], 0)[masks[k]] for k in sorted(runs)])
            # film-balanced R@10: mean over film groups of the group's mean hit
            uniq, inv = np.unique(cl, return_inverse=True)
            row["film_balanced_R@10"] = float(np.mean([h10[inv == i].mean() for i in range(len(uniq))]))
            row["n_films"] = int(len(uniq))
            if zs:
                z10 = np.concatenate([(zs[k][sec] <= 10).astype(float)[masks[k]] for k in sorted(runs)])
                row["delta_vs_imagebind_R@10"] = cluster_bootstrap(h10 - z10, cl, n_boot)
            out[sec][subset] = row
    return out


def _fmt_ci(d: dict) -> str:
    return f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def training_summary(cfg: DictConfig, ext: dict, old_ids: set[str]) -> list[dict]:
    rows = []
    for k, f in sorted(ext["folds"].items(), key=lambda x: int(x[0])):
        films = {s: {ext["film_group"][p] for p in f[s]} for s in ("train", "val", "test")}
        r = {"fold": int(k), **{s: len(f[s]) for s in ("train", "val", "test")},
             "test_old": sum(p in old_ids for p in f["test"]), "test_new": sum(p not in old_ids for p in f["test"]),
             "train_films": len(films["train"]), "test_films": len(films["test"])}
        ep = []
        for s in (0, 1, 2):
            m = FULL_RUNS / f"fold{k}" / f"seed{s}" / "metrics.json"
            if m.exists():
                ep.append(json.loads(m.read_text())["val"]["best_epoch"])
        r["best_epochs"] = ep
        rows.append(r)
    return rows


def write_report(cfg: DictConfig, paired: dict, whole: dict, train_rows: list[dict], moved: int, reproduced: bool) -> Path:
    L = ["# Semantic heads on the full dataset (4,246 pairs)", "",
         "Same method as the film split (PROGRESS.md D36): SigLIP + CLAP+MERT heads, 5 folds × 3 seeds, per-film cap of 30 "
         "clips per epoch, early stopping on validation segment R@10, Konjum salangai train-only. New in this run (D49): the "
         "2,998 Demucs-separated clips of 32 films (data/manifests_full). The film split is **extended**, not redrawn: the "
         f"1,248 old pairs keep their folds except {moved} moved by film groups that the new clips revealed (old SunNXT / "
         "YouTube clips of *Jilla* and *Pokkiri* under several source keys); each new film is tested in one fold.", "",
         "## 1. Did the new data help? (paired, same queries and galleries)", "",
         "Queries: the old film split's test clips that kept their fold (the moved ones left out). Both models are re-scored "
         "from their checkpoints by the same code" + (" (check: re-scoring an old run reproduces its saved ranks exactly)." if reproduced else
                                                       " (**check failed**: re-scoring an old run did not reproduce its saved ranks)."),
         "Cells: mean ± std over 15 runs. Δ: full − old, per-query hits averaged over seeds, 95% cluster bootstrap over film groups.", ""]
    for sec in PAIRED:
        p = paired[sec]
        L += [f"### {PAIRED_TITLES[sec]} ({p['n_queries']} queries, {p['n_clusters']} film groups)", "",
              "| trained on | R@1 | R@5 | R@10 | MedR | MRR |", "| --- | --- | --- | --- | --- | --- |"]
        for name, label in (("base", "old data (1,248 pairs)"), ("full", "**full data (4,246 pairs)**")):
            L.append(f"| {label} | " + " | ".join(_cell(p["rows"][name][m]) for m in METRICS) + " |")
        L += ["", "Δ full − old: " + "; ".join(f"{m} {_fmt_ci(p['deltas'][m])}" for m in ("R@1", "R@5", "R@10", "MRR")), ""]
    L += ["## 2. Whole dataset (extended film split, all test folds)", "",
          "Every test clip against its test-fold gallery (old + new music files) or the whole library. Old = the 1,248-pair "
          "set; new = the 32 added films. Film-balanced: each film group weighs the same (the new films have ~95 clips each). "
          "Δ vs ImageBind: R@10, paired, 95% cluster bootstrap over film groups.", ""]
    for sec, title in SECTIONS.items():
        L += [f"### {title}", "", "| queries | n | films | system | R@1 | R@5 | R@10 | MedR | MRR | film-balanced R@10 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for subset in ("all", "old", "new"):
            r = whole[sec][subset]
            L.append(f"| {subset} | {r['n_queries']} | {r['n_films']} | **heads** | " + " | ".join(_cell(r["heads"][m]) for m in METRICS)
                     + f" | {r['film_balanced_R@10']:.3f} |")
            if "imagebind" in r:
                L.append(f"| {subset} | | | ImageBind | " + " | ".join(_cell(r["imagebind"][m]) for m in METRICS) + " | |")
            L.append(f"| {subset} | | | random (exact) | " + " | ".join(_cell(r["random"][m]) for m in ("R@1", "R@5", "R@10")) + " | — | — | |")
        d = [f"{s} {_fmt_ci(whole[sec][s]['delta_vs_imagebind_R@10'])}" for s in ("all", "old", "new") if "delta_vs_imagebind_R@10" in whole[sec][s]]
        L += ["", ("Δ R@10 heads − ImageBind: " + "; ".join(d)) if d else "", ""]
    L += ["## Folds and training", "", "| fold | train | val | test (old + new) | train films | test films | best epochs (seeds 0-2) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in train_rows:
        L.append(f"| {r['fold']} | {r['train']} | {r['val']} | {r['test']} ({r['test_old']} + {r['test_new']}) | {r['train_films']} | "
                 f"{r['test_films']} | {', '.join(map(str, r['best_epochs']))} |")
    out = Path("results") / "full_dataset.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main(seeds=(0, 1, 2)) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = full_cfg()
    store = FeatureStore(cfg)
    reproduced = check_reproduces(store)
    queries, ext = comparison_queries(cfg)
    old_ids = set(json.loads(BASE_FOLDS.read_text())["film_group"])
    scores = score_paired(cfg, store, queries, seeds)
    paired = paired_tables(store, queries, scores, ext["film_group"], cfg.eval.bootstrap_resamples)
    whole = whole_dataset(cfg, old_ids, cfg.eval.bootstrap_resamples)
    rows = training_summary(cfg, ext, old_ids)
    rep = Path(cfg.paths.reports) / "full_dataset.json"
    rep.write_text(json.dumps({"reproduced": reproduced, "paired": paired, "whole": whole, "folds": rows,
                               "moved": len(ext.get("moved_from_base", {}))}, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
    log.info("wrote %s and %s", write_report(cfg, paired, whole, rows, len(ext.get("moved_from_base", {})), reproduced), rep)


if __name__ == "__main__":
    main(*(() if len(sys.argv) < 2 else (tuple(int(x) for x in sys.argv[1].split(",")),)))
