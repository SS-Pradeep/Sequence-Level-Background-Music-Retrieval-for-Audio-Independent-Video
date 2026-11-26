"""Step 1 diagnosis: the current system (D75) on the "same setting, different music" subsets vs all test segments.

Run: python -m bgm.person.diagnose   -> reports/person_step1.md, runs/filmsplit/person/eval/P0/fold<k>.npz

P0 = D75: the 10-seed siglip-both ensemble (mean cosine), semantic-only fusion, each fold's runs/filmsplit/
pipeline_ens10 fusion.json. Before anything is reported, the recomputed per-query values must reproduce the saved ones:
the seed-0 member's retrieval ranks (runs/filmsplit/semantic/siglip-both/fold<k>/seed0/perquery.npz) and the
ensemble pipeline's per-segment hits (runs/filmsplit/pipeline_ens10/fold<k>/seed0/perquery.npz).
Subset vs complement: difference of means with a 95% film-group cluster bootstrap (D39).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.analysis.full_dataset import load_scorer
from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.store import FeatureStore, fold_ids
from bgm.person import subset
from bgm.person.evaluate import MeanScorer, assert_reproduces, pipeline_fold, retrieval_chance, retrieval_ranks, retrieval_scores, retrieval_truth
from bgm.retrieval.index import FoldModels, MusicIndex, pair_query

log = logging.getLogger(__name__)
ROOT = Path("runs/filmsplit")
EVAL = ROOT / "person" / "eval"
OUT = Path("reports/person_step1.md")
SEEDS10 = list(range(10))
B = 10000
OVERRIDES = ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", "pipeline.dyn_variant=null",
             "eval.concat_sets_file=concat_sets_film.json"]


def film_cfg(extra: list[str] | None = None):
    return load_config("configs/default.yaml", OVERRIDES + (extra or []))


def diff_ci(values: np.ndarray, inside: np.ndarray, clusters: np.ndarray, n: int = B, seed: int = 0) -> dict:
    """mean(values[inside]) - mean(values[~inside]) with a CI from resampling whole clusters (unpaired groups)."""
    values, inside = np.asarray(values, float), np.asarray(inside, bool)
    uniq, inv = np.unique(clusters, return_inverse=True)
    s_in = np.bincount(inv, weights=values * inside, minlength=len(uniq))
    n_in = np.bincount(inv, weights=inside.astype(float), minlength=len(uniq))
    s_out = np.bincount(inv, weights=values * ~inside, minlength=len(uniq))
    n_out = np.bincount(inv, weights=(~inside).astype(float), minlength=len(uniq))
    d = np.random.default_rng(seed).integers(0, len(uniq), (n, len(uniq)))
    a, b = n_in[d].sum(1), n_out[d].sum(1)
    ok = (a > 0) & (b > 0)
    r = s_in[d].sum(1)[ok] / a[ok] - s_out[d].sum(1)[ok] / b[ok]
    return {"mean": float(values[inside].mean() - values[~inside].mean()), "lo": float(np.quantile(r, 0.025)), "hi": float(np.quantile(r, 0.975))}


def p0_fold(cfg, store, k: int) -> dict:
    """Per-query arrays of P0 on fold k (cached in EVAL/P0/fold<k>.npz)."""
    out = EVAL / "P0" / f"fold{k}.npz"
    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    f = fold_ids(cfg, k)
    truth = retrieval_truth(store, f["test"])
    members = [load_scorer(ROOT / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}", store, "cuda") for s in SEEDS10]
    r0 = retrieval_ranks(retrieval_scores(store, members[0], f["test"]), truth)
    with np.load(ROOT / "semantic" / "siglip-both" / f"fold{k}" / "seed0" / "perquery.npz") as z:
        same = {sec: float((r0[sec] == z[sec]).mean()) for sec in r0}
    log.info("fold %d: seed-0 member reproduces saved ranks %s", k, same)
    assert min(same.values()) > 0.97, same
    ranks = retrieval_ranks(retrieval_scores(store, MeanScorer(members), f["test"]), truth)
    chance = retrieval_chance(truth)
    del members
    run = ROOT / "pipeline_ens10" / f"fold{k}" / "seed0"
    fusion = json.loads((run / "fusion.json").read_text())
    models = FoldModels.load(cfg, store, k, 0, "siglip-both", None, train_ids=f["train"])
    index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    queries = {}
    for p in f["test"]:
        vq = pair_query(store, p)
        vq.prepare(models, store.min_span)
        queries[p] = vq
    pipe = pipeline_fold(store, index, queries, fusion)
    with np.load(run / "perquery.npz") as z:
        flips = assert_reproduces(z, pipe, f"P0 fold {k}")
        assert np.allclose(z["offset/chance/1"], pipe["offset/chance/1"], equal_nan=True), k
    log.info("fold %d: pipeline reproduces the saved D75 per-segment hits (%d first-index tie flips)", k, flips)
    res = {**{f"ret/{s}": v for s, v in ranks.items()}, **{f"ret/chance/{s}": v for s, v in chance.items()},
           "ret/pair": truth["pair"], "ret/seg_pair": truth["seg_pair"], "ret/seg_idx": truth["seg_idx"],
           **{f"pipe/{key}": v for key, v in pipe.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg(["pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"])
    store = FeatureStore(cfg)
    film = manifest.read_folds(cfg)["film_group"]
    frozen = subset.write_or_check(cfg)
    subs = subset.read(cfg)
    R = [p0_fold(cfg, store, k) for k in range(cfg.splits.n_folds)]
    cat = lambda key: np.concatenate([r[key] for r in R])  # noqa: E731
    # per-segment arrays: pipeline order and retrieval order are both test_segments order
    seg_key = list(zip(cat("pipe/pair"), cat("pipe/seg_idx")))
    assert seg_key == list(zip(cat("ret/seg_pair"), cat("ret/seg_idx")))
    seg_film = np.array([film[p] for p, _ in seg_key])
    vid, vid_film = cat("ret/pair"), np.array([film[p] for p in cat("ret/pair")])
    rows = {  # name -> (per-query values, chance values or None, level)
        "track@1 (pipeline)": (cat("pipe/track@1"), cat("pipe/offset/chance/track"), "seg"),
        "track R@10 (pipeline, per segment)": (cat("pipe/track@10"), cat("pipe/chance/track@10"), "seg"),
        "window@1": (cat("pipe/window@1"), cat("pipe/offset/chance/window"), "seg"),
        "window@J (J = 20)": (cat("pipe/window@J"), None, "seg"),
        "right track and within ±1 s": (cat("pipe/offset/1"), cat("pipe/offset/chance/1"), "seg"),
        "right track and within ±5 s": (cat("pipe/offset/5"), cat("pipe/offset/chance/5"), "seg"),
        "segment R@10 (all test windows)": ((cat("ret/segment_all") <= 10).astype(float), cat("ret/chance/segment_all"), "seg"),
        "track R@10 (clip level, test-fold gallery)": ((cat("ret/track_test_fold") <= 10).astype(float), cat("ret/chance/track_test_fold"), "vid"),
    }
    n_seg = len(seg_key)
    L = ["# Step 1: the current system (D75) on \"same setting, different music\" segments", "",
         "1,248-pair film split, 5 test folds, test-fold galleries (195–199 files). P0 = D75: 10-seed siglip-both ensemble, semantic-only "
         "fusion, coarse-to-fine pipeline with each fold's tuned fusion.json. Recomputed values reproduce the saved D75 per-segment "
         "hits and the saved seed-0 retrieval ranks, except one first-index window@1 hit on fold 4 where two windows tie exactly (the "
         "tie-shared hit, 0.5, is identical; GPU batch arithmetic flips the order). The recomputed value is used.", "",
         "**Subsets** (docs/person_branch_plan.md; data/manifests/person_subsets_film.json). Pairs of test segments from different "
         "videos within a fold; setting = SigLIP span-mean cosine, music = mean of CLAP and MERT span-mean cosines; thresholds are "
         "the fold's 90th percentile (setting) and 30th percentile (music) of all cross-video pairs.", "",
         "- **spec** (as specified): the segment is in at least one pair with setting ≥ q90 and music ≤ q30.",
         "- **nn** (diagnostic): the segment's single nearest other-video neighbour by setting passes both thresholds.", "",
         "| fold | segments | setting q90 | music q30 | pairs passing | spec | nn |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, f in frozen["folds"].items():
        L.append(f"| {k} | {f['n_segments']} | {f['setting_threshold']:.3f} | {f['music_threshold']:.3f} | {f['pairs_passing']:,} of {f['pairs']:,} | "
                 f"{len(f['spec'])} | {len(f['nn'])} |")
    L.append(f"| all | {n_seg} | | | | **{len(subs['spec'])}** ({len(subs['spec']) / n_seg:.0%}) | **{len(subs['nn'])}** ({len(subs['nn']) / n_seg:.0%}) |")
    summary = {"n_segments": n_seg}
    for name in ("nn", "spec"):
        in_seg = np.array([key in subs[name] for key in seg_key])
        in_vid = np.isin(vid, [p for p, _ in subs[name]])
        summary[name] = {"segments": int(in_seg.sum()), "videos": int(in_vid.sum())}
        L += ["", f"## Subset {name}: {in_seg.sum()} of {n_seg} segments; {in_vid.sum()} of {len(vid)} videos have at least one", "",
              "Clip-level track R@10 uses the videos with at least one subset segment. Δ = subset − rest, 95% film-group cluster bootstrap "
              f"({B:,} resamples).", "", "| metric | all | subset | rest | Δ subset − rest | chance (all / subset) |", "| --- | --- | --- | --- | --- | --- |"]
        for m, (v, ch, lev) in rows.items():
            ins, cl = (in_seg, seg_film) if lev == "seg" else (in_vid, vid_film)
            d = diff_ci(v, ins, cl)
            chs = "—" if ch is None else f"{np.nanmean(ch):.3f} / {np.nanmean(ch[ins]):.3f}"
            L.append(f"| {m} | {v.mean():.3f} | {v[ins].mean():.3f} | {v[~ins].mean():.3f} | {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {chs} |")
            summary[name][m] = {"all": float(v.mean()), "subset": float(v[ins].mean()), "rest": float(v[~ins].mean()), "diff": d}
    allv = cluster_bootstrap(cat("pipe/window@1"), seg_film, B)
    L += ["", f"All segments, window@1 with its film-group CI: {allv['mean']:.3f} [{allv['lo']:.3f}, {allv['hi']:.3f}].", ""]
    OUT.write_text("\n".join(L) + "\n")
    (EVAL / "step1.json").write_text(json.dumps(summary, indent=1))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
