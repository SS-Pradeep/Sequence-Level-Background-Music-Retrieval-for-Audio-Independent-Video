"""`bgm segment` (SPEC §3.5): merge shots into segments that tile each video exactly.

1. Merge consecutive shots greedily: absorb the next shot while the cosine between the running segment's
   mean SigLIP 1 Hz embedding and the shot's is >= merge_cosine and the length stays <= max_len_s.
2. Fold shots (units) shorter than min_shot_s into the more similar neighbour. A clip under min_shot_s is one segment.
3. Edge case (PROGRESS.md D13): a unit still longer than max_len_s (one long shot) is split into equal parts <= max_len_s.
A span's embedding is the mean of the 1 Hz vectors centred inside it (sample k is centred at k + 0.5 s), or the
nearest vector if none is.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest

log = logging.getLogger(__name__)


def span_embedding(emb_1hz: np.ndarray, a: float, b: float, centres: np.ndarray | None = None) -> np.ndarray:
    """Unit-norm mean of the 1 Hz vectors centred in [a, b), or the nearest one. centres [T1] s: the vectors' centre
    times (default k + 0.5; a chain of clips passes each clip's centres plus its offset)."""
    centres = np.arange(len(emb_1hz)) + 0.5 if centres is None else centres
    inside = (centres >= a) & (centres < b)
    e = emb_1hz[inside].mean(0) if inside.any() else emb_1hz[int(np.argmin(np.abs(centres - (a + b) / 2)))]
    e = e.astype(np.float64)
    return e / (np.linalg.norm(e) + 1e-12)


def make_segments(cut_times_s, duration_s: float, emb_1hz: np.ndarray, merge_cosine: float = 0.85,
                  max_len_s: float = 20.0, min_shot_s: float = 2.0, centres: np.ndarray | None = None) -> list[tuple[float, float, int]]:
    """[(start_s, end_s, n_shots)] tiling [0, duration_s] exactly. centres: see `span_embedding`."""
    bounds = [0.0] + sorted(float(c) for c in cut_times_s if 0.0 < c < duration_s) + [float(duration_s)]
    shots = [[a, b, 1] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if duration_s < min_shot_s or len(shots) <= 1:
        units = [[0.0, float(duration_s), len(shots)]]
    else:
        emb = lambda u: span_embedding(emb_1hz, u[0], u[1], centres)  # noqa: E731
        units = [list(shots[0])]
        for s in shots[1:]:
            cur = units[-1]
            if s[1] - cur[0] <= max_len_s and float(emb(cur) @ emb(s)) >= merge_cosine:
                cur[1], cur[2] = s[1], cur[2] + s[2]
            else:
                units.append(list(s))
        while len(units) > 1:
            short = [i for i, u in enumerate(units) if u[1] - u[0] < min_shot_s]
            if not short:
                break
            i = min(short, key=lambda i: units[i][1] - units[i][0])
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(units)]
            j = max(nbrs, key=lambda j: float(emb(units[i]) @ emb(units[j])))
            lo, hi = min(i, j), max(i, j)
            units[lo:hi + 1] = [[units[lo][0], units[hi][1], units[lo][2] + units[hi][2]]]
    out = []
    for a, b, n in units:
        k = max(1, math.ceil((b - a) / max_len_s - 1e-9))
        for j in range(k):
            out.append((a + (b - a) * j / k, b if j == k - 1 else a + (b - a) * (j + 1) / k, n if k == 1 else 1))
    return out


def run_segment(cfg: DictConfig) -> pd.DataFrame:
    from bgm.features.extract import feature_path, load_features
    pairs, shots = manifest.read_pairs(cfg), manifest.read_shots(cfg)
    s = cfg.segments
    rows = []
    for pid in pairs["pair_id"]:
        d = load_features(feature_path(cfg, "video", pid))
        assert "sem_siglip_1hz" in d, f"{pid}: run `bgm extract --what sem_video` first"
        segs = make_segments(manifest.cut_times(shots, pid), d["meta"]["duration_s"], d["sem_siglip_1hz"].astype(np.float32),
                             s.merge_cosine, s.max_len_s, s.min_shot_s)
        rows += [{"pair_id": pid, "seg_idx": k, "start_s": round(a, 4), "end_s": round(b, 4), "n_shots": n}
                 for k, (a, b, n) in enumerate(segs)]
    df = pd.DataFrame(rows)
    manifest.write_segments(df, cfg)
    L = df.end_s - df.start_s
    log.info("segments: %d over %d videos; per video median %.0f; length median %.1f s, min %.2f, max %.2f",
             len(df), df.pair_id.nunique(), df.groupby("pair_id").size().median(), L.median(), L.min(), L.max())
    return df
