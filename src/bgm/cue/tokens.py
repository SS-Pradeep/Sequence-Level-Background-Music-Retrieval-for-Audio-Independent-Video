"""Segment tokens for the cue-switch predictor (Step 2): [span-mean SigLIP, log duration, motion mean, motion std].

* SigLIP: mean of the segment's OWN 1 Hz samples (centred in [a, b), the nearest one if none), L2-normalised. The
  5 s music-span rule (§3.5) is not applied: it would reach into the neighbouring segment, possibly another clip.
* Motion: `log_motion` (log1p of the per-frame optical-flow magnitude, cut frames masked, D27) read RAW from the cached
  `dyn_25hz` [T25, 5] (channel 0). `FeatureStore.dyn` z-scores it per clip, which would mark clip joins in a
  concatenation (a leak), so the three scalar features are z-scored with TRAIN-fold statistics instead. Frame 0 of a
  video always has log_motion 0 (no previous frame for the flow); in a concatenation that would mark every clip start,
  so frame 0 of the video is left out of the motion statistics (the same rule for render inputs, whose frame 0 is the
  video's). Render computes the same raw curve for its input video, so training and inference match.

Dataset tokens are computed per clip on the clip's own timeline, with the same segment filter as `concat_query`
(segments starting before the 25 Hz duration, clipped to it), and cached once in data/features/<cue.tokens_file>.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
RATE = 25
N_SCALAR = 3                      # log duration, motion mean, motion std
TOKEN_VERSION = 2               # 2: frame 0 left out of the motion statistics


def clip_segments(segments: list[tuple[float, float]], n25: int) -> list[tuple[float, float]]:
    """concat_query's segment filter for one clip of n25 frames."""
    d = n25 / RATE
    return [(float(a), float(min(b, d))) for a, b in segments if a < d]


def segment_features(sem_1hz: np.ndarray, log_motion_25hz: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    """sem_1hz [T1, D] (sample k centred at k + 0.5 s), log_motion_25hz [T25], segments [(a, b)] in s
    -> raw tokens [N, D + 3] float32 (scalars NOT yet standardised)."""
    sem = np.asarray(sem_1hz, np.float32)
    lm = np.asarray(log_motion_25hz, np.float64)
    T1, T25 = len(sem), len(lm)
    out = np.zeros((len(segments), sem.shape[1] + N_SCALAR), np.float32)
    for i, (a, b) in enumerate(segments):
        lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T1))
        hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T1))
        v = sem[lo:hi].mean(0) if hi > lo else sem[int(np.clip(np.floor((a + b) / 2), 0, T1 - 1))]
        out[i, :sem.shape[1]] = v / (np.linalg.norm(v) + 1e-12)
        f0 = int(np.clip(round(a * RATE), min(1, T25 - 1), T25 - 1))      # frame 0: no flow (see above)
        f1 = int(np.clip(max(f0 + 1, round(b * RATE)), f0 + 1, T25))
        m = lm[f0:f1]
        out[i, -3:] = (np.log(max(b - a, 1e-3)), m.mean(), m.std())
    return out


def tokens_path(cfg) -> Path:
    return Path(cfg.paths.features) / cfg.cue.tokens_file


def _source_hash(store) -> str:
    v = store.video(store.pairs.pair_id.iloc[0])["meta"]["hashes"]
    return hashlib.sha256(json.dumps({"v": TOKEN_VERSION, "sem_video": v["sem_video"], "dyn_video": v["dyn_video"],
                                      "segments": dict(store.cfg.segments)}, sort_keys=True).encode()).hexdigest()[:12]


def build_tokens(store) -> dict[str, np.ndarray]:
    """{pair_id: raw tokens [N_p, 1155]} for every pair, cached (recomputed only when the source hashes change)."""
    path = tokens_path(store.cfg)
    h = _source_hash(store)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["__hash__"]) == h:
                return {k: z[k] for k in z.files if k != "__hash__"}
        log.info("cue tokens: source hash changed, recomputing %s", path)
    out = {}
    for p in store.pairs.pair_id:
        v = store.video(p)
        dyn = np.asarray(v["dyn_25hz"])
        out[p] = segment_features(v["sem_siglip_1hz"], dyn[:, 0], clip_segments(store.segments[p], len(dyn)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __hash__=np.array(h), **out)
    log.info("cue tokens: %d clips, %d segments -> %s (hash %s)", len(out), sum(len(x) for x in out.values()), path, h)
    return out


SCALARS = ("log_dur", "motion_mean", "motion_std")


class Scaler:
    """z-score of the 3 scalar features; fit on training-fold tokens. `drop`: scalar features set to 0 after scaling
    (feature ablations, e.g. ('log_dur',))."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, drop: tuple[str, ...] = ()):
        self.mu, self.sd, self.drop = np.asarray(mu, np.float32), np.asarray(sd, np.float32), tuple(drop)

    @classmethod
    def fit(cls, tokens: list[np.ndarray], drop: tuple[str, ...] = ()) -> "Scaler":
        x = np.concatenate([t[:, -N_SCALAR:] for t in tokens]).astype(np.float64)
        return cls(x.mean(0), np.maximum(x.std(0), 1e-6), drop)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = t.copy()
        t[:, -N_SCALAR:] = (t[:, -N_SCALAR:] - self.mu) / self.sd
        for name in self.drop:
            t[:, t.shape[1] - N_SCALAR + SCALARS.index(name)] = 0.0
        return t

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "drop": list(self.drop)}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(d["mu"], d["sd"], tuple(d.get("drop", ())))


def sequence_tokens(tokens: dict[str, np.ndarray], parts: list[str]) -> tuple[np.ndarray, list[int]]:
    """Raw tokens of a sequence [N, D] and the part index of each segment."""
    xs = [tokens[p] for p in parts]
    return np.concatenate(xs), [i for i, x in enumerate(xs) for _ in range(len(x))]
