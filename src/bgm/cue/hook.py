"""The cue predictor on an external video's segments, and its Viterbi decode (render hook, `cue.enabled=true`; Step 5).

p_i is averaged over the models that apply (the fold's `cue.seeds`, or every fold's for an excerpt from no dataset
film), and so are the tuned transition parameters (alpha, lam_jump, mu) of those models, read from the cue evaluation
(runs/<paths.runs>/<cue.exp>/eval/fold<k>/eval.json). Tokens use the video's own 1 Hz SigLIP grid and RAW motion curve
(render/query.py featurize), with the same rule as the dataset tokens (cue/tokens.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bgm.cue.model import predict
from bgm.cue.stitch import decode_lattice_cue, prob_to_logit
from bgm.cue.tokens import segment_features
from bgm.cue.train import load_cue

NAMES = {"pair": "R3", "ctx": "R4", "ctx_cross": "R5", "ctx_nodur": "R4nodur", "pair_splice": "R3s", "ctx_splice": "R4s"}


def video_tokens(feat: dict) -> np.ndarray:
    dyn = np.asarray(feat["dyn_25hz"])
    return segment_features(np.asarray(feat["sem_siglip_1hz"], np.float32), dyn[:, 0], [tuple(map(float, s)) for s in feat["segments"]])


def eval_params(cfg, fold: int, system: str) -> dict:
    e = json.loads((Path(cfg.paths.runs) / cfg.cue.exp / "eval" / f"fold{fold}" / "eval.json").read_text())
    return e["systems"][system]["params"]


def average_params(ps: list[dict]) -> dict:
    return {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}


def cue_logits(cfg, raw_tokens: np.ndarray, variant: str, folds: list[int], seeds: list[int]) -> tuple[np.ndarray, dict]:
    """Mean change probability over the (fold, seed) models -> logits [N - 1], and their averaged tuned parameters."""
    ps, params = [], []
    for k in folds:
        for s in seeds:
            model, scaler = load_cue(cfg, variant, k, s)
            ps.append(1.0 / (1.0 + np.exp(-predict(model, [scaler(raw_tokens)])[0])))
            params.append(eval_params(cfg, k, f"{NAMES[variant]}_s{s}"))
    return prob_to_logit(np.mean(ps, 0)), average_params(params)


def cue_decoder(cfg, feat: dict, n_layers: int, fold: int, stitch: dict):
    """The cue predictor's logits computed once -> (decode(layers) -> picks, info). Lets a render decode the same
    lattice several times (D101 replay penalty) without re-running the predictor."""
    raw = video_tokens(feat)
    assert len(raw) == n_layers, (len(raw), n_layers)
    seeds = list(cfg.cue.seeds)
    z, p = cue_logits(cfg, raw, cfg.cue.variant, [fold], seeds)
    decode = lambda layers: decode_lattice_cue(layers, z, p["alpha"], p["lam_jump"], p["mu"], stitch["beam"], stitch["tol_s"])
    return decode, {"variant": cfg.cue.variant, "fold": fold, "seeds": seeds, "params": p, "p": (1.0 / (1.0 + np.exp(-z))).tolist()}


def cue_decode(cfg, feat: dict, layers: list, fold: int, stitch: dict) -> tuple[list[int], dict]:
    """Render hook: Viterbi picks with the cue predictor's per-boundary transitions (fold = the render run's fold)."""
    decode, info = cue_decoder(cfg, feat, len(layers), fold, stitch)
    return decode(layers), info
