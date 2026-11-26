"""Attribute calibration and score (SPEC §5.3). Fit on the train split only.

Per side and per word, raw logits are z-scored with that word's mean / std over the train split's 1 Hz samples,
then softmaxed over the words with one temperature per side. The temperatures come from `temps` on train pairs,
maximizing mean s_attr(true pairs) - mean s_attr(shuffled pairs). s_attr = 1 - JSD2 (JSD in bits).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from bgm.eval.store import FeatureStore


def s_attr(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """1 - Jensen-Shannon divergence in bits, over the last axis (broadcasts). Equals 1 - jensenshannon(p, q, base=2) ** 2."""
    p, q = np.clip(p, 1e-12, None), np.clip(q, 1e-12, None)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * np.log2(p / m)).sum(-1) + 0.5 * (q * np.log2(q / m)).sum(-1)
    return 1.0 - jsd


def softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class _Transform:
    """Callable logits -> calibrated per-sample probabilities, with a stable name for the store's cache."""

    def __init__(self, mu, sd, t, name):
        self.mu, self.sd, self.t, self.name = mu, sd, t, name

    def __call__(self, x):
        return softmax((x - self.mu) / self.sd, self.t).astype(np.float32)


@dataclass
class Calibration:
    key: str                      # attr_logits_1hz | bridge_logits_1hz
    mu: dict                      # side -> [W]
    sd: dict
    temp: dict                    # side -> float
    tag: str = ""

    def transform(self, side: str) -> _Transform:
        return _Transform(self.mu[side], self.sd[side], self.temp[side], f"{self.key}:{side}:{self.tag}:{self.temp[side]}")

    def to_dict(self) -> dict:
        return {"key": self.key, "temp": self.temp, "mu": {k: v.tolist() for k, v in self.mu.items()},
                "sd": {k: v.tolist() for k, v in self.sd.items()}}


def fit_calibration(store: FeatureStore, train_ids: list[str], key: str, temps, tag: str = "") -> Calibration:
    V = np.concatenate([store.video(p)[key] for p in train_ids])
    M = np.concatenate([store.music(m)[key] for m in dict.fromkeys(store.music_of[p] for p in train_ids)])
    mu = {"video": V.mean(0), "music": M.mean(0)}
    sd = {"video": V.std(0) + 1e-6, "music": M.std(0) + 1e-6}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_ids))
    tr = [store.track[p] for p in train_ids]
    fix = [i for i in range(len(perm)) if tr[perm[i]] == tr[i]]          # no same-track "shuffled" pairs
    for i in fix:
        j = (i + 1) % len(perm)
        perm[i], perm[j] = perm[j], perm[i]
    best, best_t = -np.inf, None
    for tv, tm in itertools.product(temps, temps):
        pv = np.stack([softmax((store.video(p)[key] - mu["video"]) / sd["video"], tv).mean(0) for p in train_ids])
        pm = np.stack([softmax((store.music(store.music_of[p])[key] - mu["music"]) / sd["music"], tm).mean(0) for p in train_ids])
        gap = float(s_attr(pv, pm).mean() - s_attr(pv, pm[perm]).mean())
        if gap > best:
            best, best_t = gap, {"video": float(tv), "music": float(tm)}
    return Calibration(key, mu, sd, best_t, tag)
