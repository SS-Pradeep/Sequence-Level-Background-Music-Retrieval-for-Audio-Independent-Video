"""Table 3's final decoder rebuilt on the cleaned dataset (1,011 pairs), for the re-ranker HMM study (D148b).

Same code as the original Table 3 chain, pointed at cleaned_dataset/ and runs/cleaned/ (nothing in src/ or stitching/
edited; their functions are called with cleaned configs):

  shots   `python -m bgm.segmentation.run_all shots  <CLEAN overrides>`  TransNetV2 + PySceneDetect shots of the cleaned
  bassl   `python -m bgm.segmentation.run_all bassl  <CLEAN overrides>`  frozen BaSSL shot features      clips
  train   D142 'C' boundary head (combo), folds 0-4 x seeds 0-2, early stopping on the cleaned frozen val sequences
          -> runs/cleaned/segmentation/combo/fold<k>/seed<s>/head.pt
  segments theta tuned per fold on the cleaned val sequences (tune_theta), C head's own cuts on the cleaned frozen
          val / test blocked sequences (cleaned_dataset/manifests/cue_sequences_title.json: 92 test sequences)
          -> cleaned_dataset/interim/segmentation/retrieval_segments.json
  lattice per fold: stitching.data.build_episode on the cleaned emission (runs/cleaned/pipeline_ens10 fusion, cleaned
          siglip-both seeds 0-9), 200 training sequences cut by the same head / theta
          -> runs/exp_supp/rerank/clean_t3/cache/fold<k>.pkl
  gold    stitching.gold_prior rows 'HMM, π from gold paths, λ = 1 / λ tuned' (L, rho from the training gold paths,
          τ = mean of the cleaned heads' τ) -> runs/exp_supp/rerank/clean_t3/gold/fold<k>_gold.json

`.venv/bin/python -m exp_supp.rerank.clean_t3 {train|segments|lattice --fold k|gold --fold k}`
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from exp_supp.common import RUNS

log = logging.getLogger("exp_supp.rerank.clean_t3")
CLEAN = ["paths.manifests=cleaned_dataset/manifests", "paths.interim=cleaned_dataset/interim",
         "paths.features=cleaned_dataset/features", "paths.runs=runs/cleaned", "paths.results=results/cleaned"]
ROOT = RUNS / "rerank" / "clean_t3"
CACHE = ROOT / "cache"
GOLD = ROOT / "gold"


def seg_cfg():
    from bgm.config import load_config
    return load_config("configs/default.yaml", ["splits.folds_file=folds_title.json", *CLEAN])


def cue_cfg():
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    cfg = load_config("configs/default.yaml", [*CUE_OVERRIDES, *CLEAN])
    assert cfg.paths.runs == "runs/cleaned" and cfg.paths.manifests == "cleaned_dataset/manifests"
    return cfg


def seg_file() -> Path:
    from bgm.segmentation.retrieval_eval import SEG_FILE
    from bgm.segmentation.run_all import out_dir
    return out_dir(seg_cfg()) / SEG_FILE


# ---------------------------------------------------------------- segmentation
def stage_train() -> None:
    from bgm.segmentation.train_head import run_train
    run_train(seg_cfg(), variants=("combo",))


def stage_segments(device: str) -> Path:
    """As bgm.segmentation.retrieval_eval.stage_segments, C head only, cleaned data."""
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold, tune_theta
    from bgm.segmentation.retrieval_eval import to_rel
    from bgm.segmentation.train_head import SegData
    cfg = seg_cfg()
    data = SegData(cfg, need_siglip=True)
    bassl = BaSSL(device)
    heads = {"combo": "combo"}
    res = {"thetas": {}, "folds": {}, "data": "cleaned_dataset (1,011 pairs)"}
    for k in range(5):
        F = {s: Fold(data, bassl, k, s, heads) for s in ("val", "test")}
        th = {"combo": tune_theta(F["val"], "combo")}
        res["thetas"][str(k)] = th
        res["folds"][str(k)] = {}
        for split, fo in F.items():
            recs = []
            for i, parts in enumerate(fo.seqs):
                c = fo.chains("combo")[i]
                starts = [0.0] + [float(x) for x in c.joins]
                recs.append({"parts": list(parts), "cuts": {"combo/own": to_rel(fo.cuts("combo", i, th["combo"], False), starts)}})
            res["folds"][str(k)][split] = recs
        log.warning("segments fold %d: theta %s, %d val / %d test sequences", k, th, len(F["val"].seqs), len(F["test"].seqs))
    p = seg_file()
    p.write_text(json.dumps(res))
    log.warning("-> %s", p)
    return p


# ---------------------------------------------------------------- lattices (stitching.data.build_fold, cleaned)
def stage_lattice(k: int, n_train: int, device: str) -> Path:
    import stitching.data as sd
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.segmentation.train_head import SegData
    out_p = CACHE / f"fold{k}.pkl"
    if out_p.exists():
        return out_p
    t0 = time.time()
    cfg, cfg_seg = cue_cfg(), seg_cfg()
    segfile = json.loads(seg_file().read_text())
    theta = float(segfile["thetas"][str(k)]["combo"])
    segk = segfile["folds"][str(k)]
    store = FeatureStore(cfg)
    sd._FG.update(manifest.read_folds(cfg)["film_group"])
    f = fold_ids(cfg, k)
    em = Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{k}" / "seed0"
    fusion = json.loads((em / "fusion.json").read_text())
    d75 = json.loads((em / "stitch.json").read_text())
    models = FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device=device, train_ids=f["train"])
    desc = sd.MusicDesc(store)
    out = {"fold": k, "theta": theta, "fusion": fusion, "d75": d75, "episodes": {}, "data": "cleaned_dataset"}
    for split in ("train", "val", "test"):
        index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f[split])))
        if split == "train":
            seqs = sd.train_sequences(cfg, SegData(cfg_seg), k, n_train)
            cuts = sd.train_cuts(cfg_seg, k, seqs, theta, device)
        else:
            seqs = [r["parts"] for r in segk[split]]
            cuts = [r["cuts"]["combo/own"] for r in segk[split]]
        eps = [sd.build_episode(f"{k}/{split}/{i}", store, index, models, fusion, cfg, desc, parts, rel, with_gold=True)
               for i, (parts, rel) in enumerate(zip(seqs, cuts))]
        out["episodes"][split] = eps
        out[f"gallery_{split}"] = len(index.ids)
        log.warning("fold %d %s: %d episodes, %d scenes, gallery %d [%.0f s]", k, split, len(eps), sum(e.T for e in eps),
                    len(index.ids), time.time() - t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as fh:
        pickle.dump(out, fh)
    log.warning("-> %s [%.0f s]", out_p, time.time() - t0)
    return out_p


def load_fold(k: int) -> dict:
    with open(CACHE / f"fold{k}.pkl", "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- final decoder row (stitching.gold_prior, cleaned)
def stage_gold(k: int) -> Path:
    from stitching.evaluate import ensemble_tau
    from stitching.gold_prior import GOLD_ROWS, gold_counts
    from stitching.hmm_derived import HMMDerived
    from stitching.metrics import evaluate_path, fold_entropy
    t0 = time.time()
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    tau = ensemble_tau(cue_cfg(), k)
    gc = gold_counts(tr)
    out = {"fold": k, "gold_prior": gc, "tau": tau, "methods": {}, "data": "cleaned_dataset"}
    for key, st in (("hmm_gold_l1", HMMDerived(tau, gc["L_s"], gc["rho"], 1.0)),
                    ("hmm_gold_tuned", HMMDerived(tau, gc["L_s"], gc["rho"], tune_lam=True))):
        params = st.fit(tr, va)
        paths = [st.timed(e) for e in te]
        rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
        out["methods"][key] = {"name": dict(GOLD_ROWS)[key], "params": params, "rows": rows,
                               "entropy": fold_entropy(te, paths, F["gallery_test"])}
        log.warning("fold %d %s | test acc %.3f F1 %.3f | %s", k, key, np.mean([r["track_acc"] for r in rows]),
                    np.mean([r["switch_f1"] for r in rows]), params)
    GOLD.mkdir(parents=True, exist_ok=True)
    p = GOLD / f"fold{k}_gold.json"
    p.write_text(json.dumps(out, default=float))
    log.warning("fold %d -> %s [%.0f s]", k, p, time.time() - t0)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "segments", "lattice", "gold"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.stage == "train":
        stage_train()
    elif a.stage == "segments":
        stage_segments(a.device)
    elif a.stage == "lattice":
        for k in ([a.fold] if a.fold is not None else range(5)):
            stage_lattice(k, a.n_train, a.device)
    else:
        for k in ([a.fold] if a.fold is not None else range(5)):
            stage_gold(k)
