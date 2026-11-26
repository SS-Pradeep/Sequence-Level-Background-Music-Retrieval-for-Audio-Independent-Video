"""Fixed-length music window embeddings for the semi-Markov rows (user decision: 10 s windows, hop 0.5 s).

For every music file in any val / test episode's shortlist of fold k: windows [s, s + 10] every 0.5 s (the last one
flush with the end; one window [0, dur] for files shorter than 10 s), raw CLAP + MERT means with the pipeline's span
rule, embedded by the same frozen 10-head ensemble as the scenes (`models.sem.embed("music", ...)`), L2-normalised.
`.venv/bin/python -m stitching.win10 --fold k` -> stitching/cache/win10_fold<k>.pkl  {music_id: (starts [W], emb [W, D])}
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from stitching.data import CACHE, load_fold

log = logging.getLogger(__name__)
WIN_S, HOP_S = 10.0, 0.5


def build(k: int, device: str) -> Path:
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    cfg = load_config("configs/default.yaml", CUE_OVERRIDES)
    store = FeatureStore(cfg)
    f = fold_ids(cfg, k)
    models = FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device=device, train_ids=f["train"])
    F = load_fold(k)
    need = sorted({e.files[i] for s in ("val", "test") for e in F["episodes"][s] for i in e.dense})
    index = MusicIndex(store, models, need)
    out = {}
    for i, m in enumerate(need):
        dur = index.items[i].duration
        last = max(0.0, dur - WIN_S)
        st = np.arange(0.0, last + 1e-9, HOP_S)
        if last - st[-1] > 1e-6:
            st = np.r_[st, last]
        spans = np.array([apply_span_rule(s, min(s + WIN_S, dur), dur, store.min_span) for s in st])
        x = index._sem_raw(i, spans[:, 0], spans[:, 1])
        e = models.sem.embed("music", x)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        out[m] = (st, e.astype(np.float32))
    p = CACHE / f"win10_fold{k}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(out, fh)
    log.info("fold %d: %d files, %d windows -> %s", k, len(out), sum(len(v[0]) for v in out.values()), p)
    return p


def load(k: int) -> dict:
    with open(CACHE / f"win10_fold{k}.pkl", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build(a.fold, a.device)
