"""Row 2c's change predictor: R3 pair-only cue model WITHOUT the log-duration input, retrained on BaSSL segments
(user decision 2026-11-17; no PySceneDetect anywhere).

1. tokens: every clip is segmented by the D142 C head (contrast + BiGRU) of the title fold where the clip is TEST, at
   that fold's val-tuned theta, own cuts (train-only Konjum clips: fold 0's head); tokens = the existing
   `segment_features` (unit SigLIP mean, log duration, motion mean / std) -> stitching/cache/bassl_tokens.npz.
   Every clip's segmentation is out-of-fold.
2. models (the existing `train_cue`, recipe unchanged; variant 'pair_nodur' = (pair-only, blocked, drop log_dur) is
   registered at runtime in bgm.cue.train.VARIANTS, no file edited), seeds 0-2:
     normal  title folds 0-4  -> runs/titlesplit/cue_bassl/pair_nodur/fold<k>/seed<s>        (val = cue_sequences_title.json)
     nested  10 nested splits -> runs/titlesplit_nested/cue_bassl/pair_nodur/fold<n>/seed<s> (val = cue_sequences_title_nested.json,
             built by the existing `write_sets` on folds_title_nested.json)
3. p_change at every segment boundary of an episode = mean over the 3 seeds of sigmoid(logit), tokens from the
   episode's own BaSSL segments (`segment_features` on the concatenated SigLIP / motion, as bgm.segmentation.retrieval_eval).
   val / test episodes of outer fold k: the normal fold-k model. Cross-fitted training episodes from inner fold i:
   the nested {k, i} model.

`.venv/bin/python -m stitching.cue_bassl tokens|sets|train-normal|train-nested`
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
TOKENS = Path("stitching/cache/bassl_tokens.npz")
NESTED_SEQ = "cue_sequences_title_nested.json"
SEEDS = (0, 1, 2)


def register_variant():
    import bgm.cue.train as ct
    ct.VARIANTS.setdefault("pair_nodur", (False, "blocked", ("log_dur",)))


def normal_cfg():
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    return load_config("configs/default.yaml", CUE_OVERRIDES + ["cue.exp=cue_bassl"])


def nested_cfg():
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    from stitching.nested import OVERRIDES
    return load_config("configs/default.yaml", CUE_OVERRIDES + OVERRIDES + [f"cue.sequences_file={NESTED_SEQ}", "cue.exp=cue_bassl"])


def build_tokens(device: str = "cuda") -> Path:
    from bgm.config import load_config
    from bgm.cue.tokens import clip_segments, segment_features
    from bgm.eval.store import FeatureStore
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold
    from bgm.segmentation.retrieval_eval import segments_from_cuts
    from bgm.segmentation.train_head import SegData
    cfg_seg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    data = SegData(cfg_seg, need_siglip=True)
    store = FeatureStore(cfg_seg)
    th = json.loads(Path("data/interim/segmentation/retrieval_segments.json").read_text())["thetas"]
    folds = data.folds["folds"]
    test_of = {p: int(k) for k, f in folds.items() for p in f["test"]}
    allp = list(store.pairs.pair_id)
    bassl = BaSSL(device)
    out, segs_out = {}, {}
    for k in range(5):
        clips = [p for p in allp if test_of.get(p, 0) == k]
        data.frozen = lambda kk, split, c=clips: [[p] for p in c]          # noqa: E731
        fo = Fold(data, bassl, k, "x", {"combo": "combo"})
        for i, p in enumerate(clips):
            c = fo.chains("combo")[i]
            cuts = fo.cuts("combo", i, float(th[str(k)]["combo"]), False)
            segs = segments_from_cuts(cuts, float(c.duration))
            v = store.video(p)
            dyn = np.asarray(v["dyn_25hz"])
            out[p] = segment_features(v["sem_siglip_1hz"], dyn[:, 0], clip_segments(segs, len(dyn)))
            segs_out[p] = segs
        log.info("tokens fold %d: %d clips, %d segments", k, len(clips), sum(len(segs_out[p]) for p in clips))
    np.savez(TOKENS, **out)
    Path("stitching/cache/bassl_clip_segments.json").write_text(json.dumps(segs_out))
    log.info("-> %s (%d clips, %.2f segments / clip)", TOKENS, len(out), np.mean([len(x) for x in out.values()]))
    return TOKENS


def load_tokens() -> dict:
    with np.load(TOKENS) as z:
        return {k: z[k] for k in z.files}


def write_nested_sets():
    from bgm.cue.sequences import write_sets
    write_sets(nested_cfg())
    log.info("-> data/manifests/%s", NESTED_SEQ)


def train(which: str, folds: list[int]):
    from bgm.cue.train import FoldData, train_cue
    from bgm.eval.store import FeatureStore
    register_variant()
    cfg = normal_cfg() if which == "normal" else nested_cfg()
    store = FeatureStore(cfg)
    tokens = load_tokens()
    for k in folds:
        data = FoldData(cfg, k, store, tokens, ("log_dur",))
        for s in SEEDS:
            train_cue(cfg, k, s, "pair_nodur", data)


def episode_tokens(store, parts: list[str], segs) -> np.ndarray:
    from bgm.cue.tokens import segment_features
    from bgm.retrieval.index import concat_query
    from bgm.segmentation.retrieval_eval import sequence_motion
    vq = concat_query(store, parts)
    lm = sequence_motion([np.asarray(store.video(p)["dyn_25hz"])[:, 0] for p in parts])
    return segment_features(vq.sem_1hz, lm, [tuple(x) for x in segs])


def p_change(cfg, fold: int, raw_list: list[np.ndarray]) -> list[np.ndarray]:
    from bgm.cue.model import predict
    from bgm.cue.train import load_cue
    register_variant()
    ps = []
    for s in SEEDS:
        model, scaler = load_cue(cfg, "pair_nodur", fold, s)
        z = predict(model, [scaler(r) for r in raw_list])
        ps.append([1 / (1 + np.exp(-zi)) for zi in z])
    return [np.mean([ps[s][i] for s in range(len(SEEDS))], 0) for i in range(len(raw_list))]


def attach_p(k: int, eps, split: str) -> None:
    """Val / test episodes of outer fold k (normal fold-k model). Sets ep.p_change [T - 1]."""
    from bgm.eval.store import FeatureStore
    cfg = normal_cfg()
    store = FeatureStore(cfg)
    seg = json.loads(Path("data/interim/segmentation/retrieval_segments.json").read_text())["folds"][str(k)][split]
    raws = [episode_tokens(store, seg[int(e.key.split("/")[-1])]["parts"], e.segs) for e in eps]
    for e, r, p in zip(eps, raws, p_change(cfg, k, raws)):
        assert len(r) == e.T and len(p) == e.T - 1
        e.p_change = np.asarray(p, float)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("tokens", "sets", "train-normal", "train-nested"))
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    if a.stage == "tokens":
        build_tokens()
    elif a.stage == "sets":
        write_nested_sets()
    else:
        which = "normal" if a.stage == "train-normal" else "nested"
        folds = [int(x) for x in a.folds.split(",")] if a.folds else list(range(5 if which == "normal" else 10))
        train(which, folds)
