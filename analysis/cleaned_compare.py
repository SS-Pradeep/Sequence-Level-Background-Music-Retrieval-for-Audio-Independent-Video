"""D116: the final system retrained on cleaned_dataset/ (D115, the user's pair-trimmer edit) against the title-split
system trained on all 1,248 pairs (runs/titlesplit, D102).

1. **Headline** on the cleaned test folds (895 clips): chance, ImageBind zero-shot, single heads, the 10-head ensemble,
   pipeline accuracy and stitching, all from runs/cleaned/.
2. **Fair comparison, cleaned test:** the same queries and galleries (cleaned test clips, cleaned features), scored by
   heads trained on the uncleaned data (runs/titlesplit) and on the cleaned data (runs/cleaned). Folds are carried by
   source pair (D115), so an old fold-k head never trained on a source of a cleaned fold-k test clip: leak-free.
3. **Reverse, original test:** the cleaned-trained heads on the original 982 test clips (original features) against the
   saved ranks of the original runs. Cleaned fold-k train sources are a subset of the original fold-k train set.

Section 2 and 3 scores are computed here (`score_cross`) and written as runs under runs/cleaned/cross_*; the scoring
path is checked to reproduce a saved run's ranks exactly before it is used.

Run: python -m bgm.analysis.cleaned_compare   -> results/cleaned/compare.md
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.final import _ci
from bgm.eval.final_film import ENS, _retrieval_row, n_runs, rank_runs, section_pipeline_variants, zeroshot_runs
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import Split, _cell, _chance
from bgm.eval.report_pipeline_filmsplit import SPECS as PIPE_SPECS, load_runs, section_offset, section_stitch
from bgm.eval.store import FeatureStore, fold_ids
from bgm.person.evaluate import MeanScorer
from bgm.train.semantic import evaluate

log = logging.getLogger(__name__)
FOLDS_FILE = "folds_title.json"
CLEAN = {"paths.manifests": "cleaned_dataset/manifests", "paths.interim": "cleaned_dataset/interim",
         "paths.features": "cleaned_dataset/features", "paths.runs": "runs/cleaned", "paths.results": "results/cleaned"}
ORIG = {"paths.runs": "runs/titlesplit", "paths.results": "results/titlesplit"}
SINGLE_SEEDS, ENS_RUN_SEEDS, ENS_SIZE = (0, 1, 2), (0, 1, 2), 10
SECTIONS = ("track_test_fold", "track_all", "segment_all", "segment_other")


def make_cfg(over: dict) -> DictConfig:
    from bgm.config import load_config
    return load_config("configs/default.yaml", [f"splits.folds_file={FOLDS_FILE}", "eval.concat_sets_file=concat_sets_title.json",
                                                *(f"{k}={v}" for k, v in over.items())])


def _save(path: Path, per: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "perquery.npz", **per)


def score_cross(cfg: DictConfig, store: FeatureStore, heads_root: Path, out_root: Path, device: str = "cuda") -> None:
    """Score heads_root's siglip-both heads (seeds 0–2 single, 10-head ensembles e = 0–2) on cfg's test folds with
    cfg's features -> out_root/semantic/siglip-both/... and out_root/semantic_ens/siglip-both-ens10/... (perquery.npz)."""
    run = lambda k, s: heads_root / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}"
    for k in range(cfg.splits.n_folds):
        test = fold_ids(cfg, k)["test"]
        for s in SINGLE_SEEDS:
            out = out_root / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}"
            if not (out / "perquery.npz").exists():
                with torch.no_grad():
                    _save(out, evaluate(cfg, store, load_scorer(run(k, s), store, device), test)[1])
        for e in ENS_RUN_SEEDS:
            out = out_root / "semantic_ens" / ENS / f"fold{k}" / f"seed{e}"
            if not (out / "perquery.npz").exists():
                members = [load_scorer(run(k, m), store, device) for m in range(ENS_SIZE * e, ENS_SIZE * (e + 1))]
                with torch.no_grad():
                    _save(out, evaluate(cfg, store, MeanScorer(members), test)[1])
                del members
                torch.cuda.empty_cache()
        log.info("%s on %s: fold %d scored", heads_root, cfg.paths.manifests, k)


def check_reproduces(cfg: DictConfig, store: FeatureStore, heads_root: Path, device: str = "cuda") -> None:
    """The scoring path gives the saved ranks of heads_root's fold-0 seed-0 run, scored on its own data."""
    saved = heads_root / "semantic" / "siglip-both" / "fold0" / "seed0"
    with torch.no_grad():
        per = evaluate(cfg, store, load_scorer(saved, store, device), fold_ids(cfg, 0)["test"])[1]
    with np.load(saved / "perquery.npz") as z:
        for sec in SECTIONS:
            assert np.array_equal(per[sec], z[sec]), f"{saved}: {sec} ranks differ from the saved run"


def trimmed_pairs(clean: pd.DataFrame, orig: pd.DataFrame, tol_s: float = 0.1) -> set[str]:
    """Cleaned pairs that differ from their source clip: split parts, or one part that does not span the whole clip.
    The trimmer stores trim_* for every saved edit, including a single full-length part (= kept whole)."""
    dur = dict(zip(orig.pair_id, orig.video_duration_s.astype(float)))
    src = [s or p for p, s in zip(clean.pair_id, clean.meta_trim_source_id)]
    n_parts = pd.Series(src).value_counts()
    out = set()
    for p, s, a, b in zip(clean.pair_id, src, clean.meta_trim_start_s, clean.meta_trim_end_s):
        if n_parts[s] > 1 or (a != "" and (float(a) > tol_s or float(b) < dur[s] - tol_s)):
            out.add(p)
    return out


def _paired(split: Split, a: dict, b: dict, sec: str, k: int, B: int, keep=None) -> dict:
    """Paired Δ hit@k (a − b), per query averaged over seeds, film-group bootstrap; keep: pair_id -> bool."""
    ha, hb, cl, q = [], [], [], []
    for f in sorted(set(a) & set(b)):
        ha.append(np.mean([(r[sec] <= k).astype(float) for r in a[f]], 0))
        hb.append(np.mean([(r[sec] <= k).astype(float) for r in b[f]], 0))
        cl.append(split.clusters(f, sec))
        q += split.q[(f, sec)]
    d, cl = np.concatenate(ha) - np.concatenate(hb), np.concatenate(cl)
    m = np.array([keep(p) for p in q]) if keep else np.ones(len(d), bool)
    return cluster_bootstrap(d[m], cl[m], B)


def _rows(split: Split, systems: list[tuple[str, dict]]) -> list[str]:
    ch = {sec: _chance(split, sec, False) for sec in SECTIONS}
    L = ["| system | runs | track R@1 | track R@5 | track R@10 | track MedR | track MRR | track R@10, all files | segment R@1 | "
         "segment R@10 | segment R@10, other-video |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
         f"| random (exact chance) | — | {_cell(ch['track_test_fold']['R@1'])} | {_cell(ch['track_test_fold']['R@5'])} | "
         f"{_cell(ch['track_test_fold']['R@10'])} | — | — | {_cell(ch['track_all']['R@10'])} | {_cell(ch['segment_all']['R@1'])} | "
         f"{_cell(ch['segment_all']['R@10'])} | {_cell(ch['segment_other']['R@10'])} |"]
    return L + [_retrieval_row(split, name, runs) for name, runs in systems if runs]


def _delta_lines(split: Split, new: dict, old: dict, B: int, keep_sets: dict) -> list[str]:
    L = ["| queries | n | Δ track R@1 | Δ track R@10 | Δ track R@10, all files | Δ segment R@10 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, keep in keep_sets.items():
        t1 = _paired(split, new, old, "track_test_fold", 1, B, keep)
        t10 = _paired(split, new, old, "track_test_fold", 10, B, keep)
        ta = _paired(split, new, old, "track_all", 10, B, keep)
        s10 = _paired(split, new, old, "segment_all", 10, B, keep)
        L.append(f"| {name} | {t10['n_queries']} | {_ci(t1)} | {_ci(t10)} | {_ci(ta)} | {_ci(s10)} |")
    return L


def write_compare(device: str = "cuda") -> Path:
    cc, co = make_cfg(CLEAN), make_cfg(ORIG)
    B = int(cc.eval.bootstrap_resamples)
    root_c, root_o = Path(cc.paths.runs), Path(co.paths.runs)
    sc, so = Split(cc, FOLDS_FILE, "film"), Split(co, FOLDS_FILE, "film")
    store_c, store_o = FeatureStore(cc), FeatureStore(co)
    check_reproduces(cc, store_c, root_c, device)
    check_reproduces(co, store_o, root_o, device)
    score_cross(cc, store_c, root_o, root_c / "cross_old_on_clean", device)
    score_cross(co, store_o, root_c, root_c / "cross_clean_on_orig", device)

    pairs_c = manifest.read_pairs(cc)
    trimmed = trimmed_pairs(pairs_c, manifest.read_pairs(co))
    single_c, ens_c = rank_runs(root_c, "semantic", "siglip-both", SINGLE_SEEDS), rank_runs(root_c, "semantic_ens", ENS)
    zs_c = zeroshot_runs(root_c, "imagebind")
    film_c = manifest.read_folds(cc)["film_group"]

    L = ["# The final system retrained on the cleaned dataset (D116)", "",
         "Generated by `python -m bgm.analysis.cleaned_compare` (src/bgm/analysis/cleaned_compare.py). **Cleaned dataset** "
         f"(cleaned_dataset/, D115): the user's pair-trimmer edit of the 1,248 pairs, {len(pairs_c):,} pairs ({len(trimmed)} trimmed or "
         "split parts), title split carried over by source pair (each pair keeps its source's fold), Konjum salangai train-only. "
         "Same system and settings as results/titlesplit/final.md (D75 / D102): 10-head ensembles of SigLIP → CLAP+MERT heads, "
         "semantic-only coarse-to-fine retrieval, Viterbi stitching tuned on val. Mean ± std over runs; Δ: paired, per query "
         f"averaged over seeds, 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         "## 1. Headline: cleaned training, cleaned test", "",
         f"Queries: the {sum(len(sc.q[(k, 'track_test_fold')]) for k in sc.folds):,} cleaned test clips and "
         f"{sum(len(sc.q[(k, 'segment_all')]) for k in sc.folds):,} test segments; track gallery = the test fold's music files. "
         "**Not comparable with the original 0.295 track R@10**: the test set itself changed (section 2 is the fair comparison).", ""]
    L += _rows(sc, [("ImageBind zero-shot", zs_c), ("heads SigLIP → CLAP+MERT (seeds 0–2)", single_c),
                    ("**final: 10-head ensemble**", ens_c)])
    if single_c and zs_c:
        L += ["", f"- H1 single heads − ImageBind, track R@10: {_ci(_paired(sc, single_c, zs_c, 'track_test_fold', 10, B))}."]
    if ens_c and single_c:
        L += [f"- Ensemble − single heads, track R@10: {_ci(_paired(sc, ens_c, single_c, 'track_test_fold', 10, B))}; "
              f"segment R@10: {_ci(_paired(sc, ens_c, single_c, 'segment_all', 10, B))}."]
    L.append("")
    p4, _ = section_pipeline_variants(root_c, film_c, B)
    spec = replace(PIPE_SPECS["ens"], runs=root_c / "pipeline_ens10", concat_file=f"{cc.paths.manifests}/{cc.eval.concat_sets_file}")
    P = load_runs(spec.runs, "perquery.npz")
    s5, _ = section_offset(P, film_c, B)
    s6, _ = section_stitch(P, film_c, B, spec)
    p4[0], s5[0], s6[0] = "## 1b. Pipeline accuracy", "## 1c. Offset precision, final system", "## 1d. Stitching: Viterbi vs greedy, final system"
    L += p4 + s5 + s6

    old_on_c = {k: rank_runs(root_c / "cross_old_on_clean", kind, v, seeds)
                for k, (kind, v, seeds) in {"single": ("semantic", "siglip-both", None), "ens": ("semantic_ens", ENS, None)}.items()}
    L += ["## 2. Fair comparison on the cleaned test clips: trained on cleaned vs trained on all 1,248", "",
          "Same queries, galleries and features; only the training data differs (runs/titlesplit heads vs runs/cleaned heads, "
          "same seeds). Δ = cleaned-trained − original-trained. 'kept whole' = test clips the user left untouched; 'trimmed' = "
          "trimmed clips and split parts.", "", "Single heads (seeds 0–2):", ""]
    L += _rows(sc, [("trained on all 1,248", old_on_c["single"]), ("trained on cleaned", single_c)])
    keep = {"all": None, "kept whole": lambda p: p not in trimmed, "trimmed": lambda p: p in trimmed}
    L += [""] + _delta_lines(sc, single_c, old_on_c["single"], B, keep) + ["", "10-head ensembles (3 disjoint ensembles):", ""]
    L += _rows(sc, [("trained on all 1,248", old_on_c["ens"]), ("trained on cleaned", ens_c)])
    L += [""] + _delta_lines(sc, ens_c, old_on_c["ens"], B, keep) + [""]

    clean_on_o = {"single": rank_runs(root_c / "cross_clean_on_orig", "semantic", "siglip-both"),
                  "ens": rank_runs(root_c / "cross_clean_on_orig", "semantic_ens", ENS)}
    single_o, ens_o = rank_runs(root_o, "semantic", "siglip-both", SINGLE_SEEDS), rank_runs(root_o, "semantic_ens", ENS)
    pairs_o = manifest.read_pairs(co)
    excluded = set(pairs_o.pair_id) - {r or p for p, r in zip(pairs_c.pair_id, pairs_c.meta_trim_source_id)}
    keep_o = {"all": None, "excluded by the user": lambda p: p in excluded, "kept (whole or trimmed)": lambda p: p not in excluded}
    L += ["## 3. Reverse: on the original 982 test clips (uncleaned)", "",
          "The cleaned-trained heads scored on the original test folds and features, against the original runs' saved ranks. "
          "Shows whether training on cleaned data costs anything on the messy clips, including those the user excluded.", "",
          "Single heads (seeds 0–2):", ""]
    L += _rows(so, [("trained on all 1,248 (results/titlesplit)", single_o), ("trained on cleaned", clean_on_o["single"])])
    L += [""] + _delta_lines(so, clean_on_o["single"], single_o, B, keep_o) + ["", "10-head ensembles:", ""]
    L += _rows(so, [("trained on all 1,248 (results/titlesplit)", ens_o), ("trained on cleaned", clean_on_o["ens"])])
    L += [""] + _delta_lines(so, clean_on_o["ens"], ens_o, B, keep_o) + [""]

    out = Path(cc.paths.results) / "compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    log.info("wrote %s (ensemble runs %d, single %d)", out, n_runs(ens_c), n_runs(single_c))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    print(write_compare())
