"""Phase 1: cross-fitted training episodes for outer fold k.

For each inner fold i != k (= title fold i's test clips T_i, train-only clips excluded), 75 blocked sequences
(60-180 s, the cue sampler, per-item seeds [1, k, i, j]) drawn from T_i only, so one model can score a whole sequence:
  * segmentation: the C head of title fold i (trained without T_i), fold i's theta, own cuts -> out-of-fold
  * emission / shortlist: the nested heads of split {k, i} (10-seed ensemble, never saw T_k or T_i), gallery = the
    music of T_i (~195 files, like a test gallery), the same make_lattice / fusion as everywhere else
  * p_change: the nested {k, i} BaSSL cue model (stitching.cue_bassl)
4 x 75 = 300 episodes per outer fold (the same count as the in-sample training episodes).
`.venv/bin/python -m stitching.crossfit --fold k` -> stitching/cache/crossfit_fold<k>.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from stitching.data import CACHE, MusicDesc, _FG, build_episode, train_cuts

log = logging.getLogger(__name__)
N_PER_INNER = 75


def build(k: int, device: str) -> Path:
    from bgm.config import load_config
    from bgm.cue.sequences import by_film, clip_table, sample_sequence
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.segmentation.train_head import SegData
    from stitching import cue_bassl
    from stitching.nested import nested_of
    t0 = time.time()
    cfg_n = cue_bassl.nested_cfg()
    cfg_seg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    seg = json.loads(Path("data/interim/segmentation/retrieval_segments.json").read_text())
    store = FeatureStore(cfg_n)
    _FG.update(manifest.read_folds(cfg_seg)["film_group"])
    fusion = json.loads((Path("runs/titlesplit/pipeline_ens10") / f"fold{k}" / "seed0" / "fusion.json").read_text())
    desc = MusicDesc(store)
    data = SegData(cfg_seg)
    eps, meta = [], []
    for i in [j for j in range(5) if j != k]:
        n = nested_of(k, i)
        Ti = list(fold_ids(cfg_seg, i)["test"])
        nf = fold_ids(cfg_n, n)
        assert set(Ti) <= set(nf["test"]) and not set(Ti) & set(nf["train"] + nf["val"])
        films = by_film(clip_table(data.pairs, data.fg, Ti))
        seqs = [sample_sequence(films, np.random.default_rng([1, k, i, j]), float(cfg_n.cue.seq_min_s), float(cfg_n.cue.seq_max_s),
                                "blocked", int(cfg_n.cue.block_max_clips), frozenset()) for j in range(N_PER_INNER)]
        cuts = train_cuts(cfg_seg, i, seqs, float(seg["thetas"][str(i)]["combo"]), device)
        models = FoldModels.load(cfg_n, store, n, 0, cfg_n.pipeline.sem_variant, cfg_n.pipeline.dyn_variant, device=device, train_ids=nf["train"])
        index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in Ti)))
        part = [build_episode(f"{k}/xtrain/{i}/{j}", store, index, models, fusion, cfg_n, desc, q, rel, with_gold=True)
                for j, (q, rel) in enumerate(zip(seqs, cuts))]
        raws = [cue_bassl.episode_tokens(store, q, e.segs) for q, e in zip(seqs, part)]
        for e, p in zip(part, cue_bassl.p_change(cfg_n, n, raws)):
            e.p_change = np.asarray(p, float)
        eps += part
        meta.append({"inner": i, "nested": n, "n": len(part), "gallery": len(index.ids), "scenes": sum(e.T for e in part)})
        log.info("outer %d inner %d (nested %d): %d episodes, gallery %d [%.0f s]", k, i, n, len(part), len(index.ids), time.time() - t0)
    p = CACHE / f"crossfit_fold{k}.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"fold": k, "episodes": eps, "meta": meta}, fh)
    log.info("-> %s [%.0f s]", p, time.time() - t0)
    return p


def load(k: int) -> dict:
    with open(CACHE / f"crossfit_fold{k}.pkl", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build(a.fold, a.device)
