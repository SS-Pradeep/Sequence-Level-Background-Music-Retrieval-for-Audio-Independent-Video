"""Step 1: "same setting, different music" test subsets (docs/person_branch_plan.md).

Within each test fold, every pair of segments from different videos is scored twice:
  setting  cosine of the segments' span-mean SigLIP vectors (span rule, unit-normalised);
  music    cosine of their paired music spans: [l2 CLAP span mean | l2 MERT span mean] / sqrt 2, i.e. the mean of the
           CLAP and MERT cosines.
Thresholds are fold-wise quantiles over all cross-video pairs: setting >= q90, music <= q30.
  spec  (as specified)  the segment is in at least one pair passing both thresholds;
  nn    (diagnostic)    its single nearest other-video neighbour by setting passes both thresholds.
Frozen to data/manifests/person_subsets_film.json (rebuilding must reproduce it).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.data import manifest
from bgm.eval.protocol import test_segments
from bgm.eval.store import FeatureStore, fold_ids, l2

log = logging.getLogger(__name__)
FILE = "person_subsets_film.json"
Q_SETTING, Q_MUSIC = 0.9, 0.3


def segment_vectors(store: FeatureStore, ids: list[str]) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Test segments (pair_id, k, a, b) of `ids`, their setting vectors [n, 1152] and music vectors [n, 2048]."""
    segs = test_segments(store, ids)
    S = np.stack([l2(store.pooled("video", p, "sem_siglip_1hz", a, b)) for p, _, a, b in segs])
    M = np.stack([np.concatenate([l2(store.pooled("music", store.music_of[p], key, a, b)) for key in ("sem_clap_1hz", "sem_mert_1hz")])
                  for p, _, a, b in segs]) / np.sqrt(2)
    return segs, S, M


def subsets(S: np.ndarray, M: np.ndarray, video: np.ndarray, q_setting: float = Q_SETTING, q_music: float = Q_MUSIC) -> dict:
    """S, M: [n, d] unit vectors; video: [n] video id of each segment -> {'spec': bool[n], 'nn': bool[n], thresholds}."""
    cs, cm = S @ S.T, M @ M.T
    other = video[:, None] != video[None]
    upper = np.triu(other, 1)
    ts, tm = float(np.quantile(cs[upper], q_setting)), float(np.quantile(cm[upper], q_music))
    both = other & (cs >= ts) & (cm <= tm)
    nn = np.where(other, cs, -np.inf).argmax(1)
    r = np.arange(len(S))
    nn_ok = other[r, nn] & (cs[r, nn] >= ts) & (cm[r, nn] <= tm)
    return {"spec": both.any(1), "nn": nn_ok, "setting_threshold": ts, "music_threshold": tm,
            "pairs_passing": int(both[upper].sum()), "pairs": int(upper.sum())}


def build(cfg) -> dict:
    store = FeatureStore(cfg)
    out = {"q_setting": Q_SETTING, "q_music": Q_MUSIC, "folds": {}}
    for k in range(cfg.splits.n_folds):
        segs, S, M = segment_vectors(store, fold_ids(cfg, k)["test"])
        sub = subsets(S, M, np.array([p for p, *_ in segs]))
        out["folds"][str(k)] = {"n_segments": len(segs), "setting_threshold": sub["setting_threshold"], "music_threshold": sub["music_threshold"],
                                "pairs": sub["pairs"], "pairs_passing": sub["pairs_passing"],
                                **{name: [[p, int(i)] for (p, i, _, _), m in zip(segs, sub[name]) if m] for name in ("spec", "nn")}}
        log.info("fold %d: %d segments, spec %d, nn %d", k, len(segs), sub["spec"].sum(), sub["nn"].sum())
    return out


def path(cfg) -> Path:
    return manifest.manifests_dir(cfg) / FILE


def write_or_check(cfg) -> dict:
    """Build the subsets; write them once, afterwards assert that a rebuild reproduces the frozen file."""
    sub, p = build(cfg), path(cfg)
    if p.exists():
        old = json.loads(p.read_text())
        assert all(old["folds"][k][n] == sub["folds"][k][n] for k in sub["folds"] for n in ("spec", "nn")), "subsets changed"
        return old
    p.write_text(json.dumps(sub, indent=1))
    log.info("wrote %s", p)
    return sub


def read(cfg) -> dict[str, set[tuple[str, int]]]:
    """{'spec' | 'nn': {(pair_id, segment index)}} over all folds."""
    d = json.loads(path(cfg).read_text())
    return {n: {(p, i) for f in d["folds"].values() for p, i in f[n]} for n in ("spec", "nn")}
