"""In-memory feature access for evaluation and training, with the SPEC §3.5 span rule.

1 Hz arrays: sample k is centred at k + 0.5 s. `span_mean(x, a, b)` averages the samples centred in [a, b) (the
nearest sample if none), using float64 cumulative sums so any window mean is O(1) (SPEC §5.9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.features.dynamics import M_CHANNELS, V_CHANNELS, normalize_dyn
from bgm.features.extract import feature_path, load_features


def apply_span_rule(a: float, b: float, duration_s: float, min_span_s: float) -> tuple[float, float]:
    """Widen [a, b) symmetrically to >= min_span_s, shifted to stay inside [0, duration_s] (whole file if shorter)."""
    if b - a >= min_span_s:
        return a, b
    if duration_s <= min_span_s:
        return 0.0, duration_s
    c = (a + b) / 2
    a, b = c - min_span_s / 2, c + min_span_s / 2
    if a < 0:
        a, b = 0.0, min_span_s
    if b > duration_s:
        a, b = duration_s - min_span_s, duration_s
    return a, b


@dataclass
class Series1Hz:
    """A [T1, D] 1 Hz array with float64 prefix sums for O(1) span means."""
    x: np.ndarray
    cs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.x = np.asarray(self.x, np.float32)
        self.cs = np.vstack([np.zeros((1,) + self.x.shape[1:]), np.cumsum(self.x, 0, dtype=np.float64)])

    def span_mean(self, a: float, b: float) -> np.ndarray:
        T = len(self.x)
        lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T))          # first k with k + 0.5 >= a
        hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T))          # first k with k + 0.5 >= b
        if hi <= lo:
            k = int(np.clip(np.floor((a + b) / 2), 0, T - 1))
            return self.x[k].astype(np.float64)
        return (self.cs[hi] - self.cs[lo]) / (hi - lo)

    def span_means(self, starts: np.ndarray, dur: float) -> np.ndarray:
        """[W, D] means over [s, s + dur) for many starts (no span rule applied here)."""
        return np.stack([self.span_mean(s, s + dur) for s in starts])


class FeatureStore:
    """Lazily loads per-pair video and per-music-file arrays; `dyn` arrays come normalized (load-time rule)."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.pairs = manifest.read_pairs(cfg).set_index("pair_id", drop=False)
        self.track = dict(zip(self.pairs.pair_id, self.pairs.track_id))
        self.music_of = dict(zip(self.pairs.pair_id, self.pairs.music_id))
        self.music_track = dict(zip(self.pairs.music_id, self.pairs.track_id))
        segs = manifest.read_segments(cfg)
        self.segments = {p: list(zip(g.start_s.to_numpy(float), g.end_s.to_numpy(float))) for p, g in segs.groupby("pair_id")}
        self._v, self._m, self._s = {}, {}, {}
        self.min_span = float(cfg.segments.min_music_span_s)

    def video(self, pair_id: str) -> dict:
        if pair_id not in self._v:
            d = load_features(feature_path(self.cfg, "video", pair_id))
            for sub in ("video_extra", "video_face"):                    # D67 temporal video features, D77 face features, if extracted
                extra = feature_path(self.cfg, sub, pair_id)
                if extra.exists():
                    d.update({k: v for k, v in load_features(extra).items() if k != "meta"})
            self._v[pair_id] = d
        return self._v[pair_id]

    def music(self, music_id: str) -> dict:
        if music_id not in self._m:
            self._m[music_id] = load_features(feature_path(self.cfg, "music", music_id))
        return self._m[music_id]

    def series(self, side: str, item_id: str, key: str, transform=None) -> Series1Hz:
        k = (side, item_id, key, getattr(transform, "name", getattr(transform, "__name__", None)) if transform else None)
        if k not in self._s:
            d = self.video(item_id) if side == "video" else self.music(item_id)
            x = d[key].astype(np.float32)
            self._s[k] = Series1Hz(transform(x) if transform else x)
        return self._s[k]

    def duration(self, side: str, item_id: str) -> float:
        d = self.video(item_id) if side == "video" else self.music(item_id)
        return float(d["meta"]["duration_s"])

    def dyn(self, side: str, item_id: str) -> np.ndarray:
        d = self.video(item_id) if side == "video" else self.music(item_id)
        if side == "video" and self.cfg.train.dynamics.get("video_source", "curves") == "vjepa":     # D70
            if ("vj", item_id) not in self._s:
                from bgm.features.video_temporal import dyn_path, to_25hz
                with np.load(dyn_path(self.cfg, item_id)) as z:
                    self._s[("vj", item_id)] = to_25hz(z["dyn_vjepa_12hz"], len(d["dyn_25hz"]))
            return self._s[("vj", item_id)]
        return normalize_dyn(d["dyn_25hz"], V_CHANNELS if side == "video" else M_CHANNELS)

    def span(self, side: str, item_id: str, a: float, b: float) -> tuple[float, float]:
        return apply_span_rule(a, b, self.duration(side, item_id), self.min_span)

    def pooled(self, side: str, item_id: str, key: str, a: float, b: float, transform=None) -> np.ndarray:
        """Span-rule-widened mean of a 1 Hz key over [a, b)."""
        a, b = self.span(side, item_id, a, b)
        return self.series(side, item_id, key, transform).span_mean(a, b)

    def whole(self, side: str, item_id: str, key: str, transform=None) -> np.ndarray:
        return self.pooled(side, item_id, key, 0.0, self.duration(side, item_id), transform)


def fold_ids(cfg: DictConfig, fold: int) -> dict[str, list[str]]:
    return manifest.read_folds(cfg)["folds"][str(fold)]


def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def pairs_frame(store: FeatureStore, ids: list[str]) -> pd.DataFrame:
    return store.pairs.loc[ids]
