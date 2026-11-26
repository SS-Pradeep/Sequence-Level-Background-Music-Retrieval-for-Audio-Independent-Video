"""R2, the cosine rule: p_i = sigmoid(a * (1 - cos(v_i, v_{i+1})) + b), v = a segment's span-mean SigLIP (the token's
first 1152 dims, unit norm). a, b: logistic regression on boundaries of training sequences drawn from the fold's train
clips (blocked mode), class-balanced, the same weighting as the learned models' pos_weight = neg / pos, so all
predictors put p = 0.5 at the same evidence balance.
"""
from __future__ import annotations

import json
import logging

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from bgm.cue.tokens import N_SCALAR
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)


def cosine_feature(raw_tokens: np.ndarray) -> np.ndarray:
    """[N, D] raw tokens -> [N - 1] 1 - cos between neighbouring segments' SigLIP means."""
    v = raw_tokens[:, :-N_SCALAR].astype(np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (v[:-1] * v[1:]).sum(1)


def fit_cosine(data, n_seqs: int) -> dict:
    """data: train.FoldData. Returns {'a', 'b', 'n', 'pos'}."""
    from bgm.cue.sequences import boundary_labels
    from bgm.cue.tokens import sequence_tokens
    xs, ys = [], []
    for i in range(n_seqs):
        parts = data.draw(np.random.default_rng([data.fold, 2, i]), "blocked")
        raw, seg_part = sequence_tokens(data.tokens, parts)
        y, _ = boundary_labels(seg_part, parts, data.store.track, data.fg)
        xs.append(cosine_feature(raw)), ys.append(y)
    x, y = np.concatenate(xs)[:, None], np.concatenate(ys)
    lr = LogisticRegression(C=1e4, class_weight="balanced").fit(x, y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": int(len(y)), "pos": int(y.sum())}


def cosine_logits(params: dict, raw_tokens: np.ndarray) -> np.ndarray:
    return params["a"] * cosine_feature(raw_tokens) + params["b"]


def run_fit_cosine(cfg: DictConfig, folds) -> dict:
    from bgm.cue.tokens import build_tokens
    from bgm.cue.train import FoldData
    from bgm.eval.store import FeatureStore
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    out = {}
    for k in folds:
        p = fit_cosine(FoldData(cfg, k, store, tokens), cfg.cue.train.pos_weight_seqs)
        d = run_dir(cfg, f"{cfg.cue.exp}/cosine", k)
        (d / "params.json").write_text(json.dumps(p, indent=1))
        write_run(d, cfg, p, None)
        log.info("cue cosine fold %d: a %.2f b %.2f (%d boundaries, %d changes)", k, p["a"], p["b"], p["n"], p["pos"])
        out[k] = p
    return out


def load_cosine(cfg: DictConfig, fold: int) -> dict:
    from pathlib import Path
    return json.loads((Path(cfg.paths.runs) / cfg.cue.exp / "cosine" / f"fold{fold}" / "params.json").read_text())
