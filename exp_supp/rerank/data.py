"""Split, galleries and hit masks of the re-ranker study (cleaned title split; hit rule = track_id, as D145 / M5)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FOLDS = Path("cleaned_dataset/manifests/folds_title.json")


@lru_cache(1)
def load_split() -> dict:
    return json.loads(FOLDS.read_text())


@dataclass
class Gallery:
    """One fold / split. Segment protocol (segment_all): queries = segments, columns = segments' aligned windows.
    Track protocol (track_test_fold): queries = clips, columns = the split's music files."""
    seg_pid: np.ndarray        # [Nq] pair id of each segment
    seg_k: np.ndarray          # [Nq] segment index within its clip
    seg_file: np.ndarray       # [Nq] index into `files` of each segment's (= window's) music file
    seg_valid: np.ndarray      # [Nq, Nq] bool: column allowed for the row (other videos of the query's track left out)
    seg_cluster: np.ndarray    # [Nq] film group
    clip_ids: list
    files: list
    file_track: np.ndarray     # [Nf] track_id
    trk_hits: np.ndarray       # [Nc, Nf] bool: every file of the clip's track is a hit
    trk_cluster: np.ndarray    # [Nc] film group
    clip_file: np.ndarray      # [Nc] index of the clip's own music file
    key_pc: np.ndarray         # [Nf] estimated key pitch class (librosa), -1 if unknown
    key_mode: np.ndarray       # [Nf] 1 major / 0 minor
    tempo: np.ndarray          # [Nf] bpm


_G: dict = {}


def gallery(store, split: dict, fold: int, which: str) -> Gallery:
    key = (fold, which)
    if key in _G:
        return _G[key]
    from bgm.eval.protocol import test_segments
    ids = split["folds"][str(fold)][which]
    segs = test_segments(store, ids)
    files = list(dict.fromkeys(store.music_of[p] for p in ids))
    fpos = {m: i for i, m in enumerate(files)}
    pid = np.array([p for p, *_ in segs])
    tr = np.array([store.track[p] for p in pid])
    valid = ~((tr[:, None] == tr[None]) & (pid[:, None] != pid[None]))
    ftr = np.array([store.music_track[m] for m in files])
    ctr = np.array([store.track[p] for p in ids])
    film = split["film_group"]
    kp, km, tb = [], [], []
    for m in files:
        d = store.music(m)
        kp.append(int(d["key_pc"])), km.append(int(d["key_mode"])), tb.append(float(d["tempo_bpm"]))
    g = Gallery(pid, np.array([k for _, k, _, _ in segs]), np.array([fpos[store.music_of[p]] for p in pid]), valid,
                np.array([film[p] for p in pid]), list(ids), files, ftr, ctr[:, None] == ftr[None],
                np.array([film[p] for p in ids]), np.array([fpos[store.music_of[p]] for p in ids]),
                np.array(kp), np.array(km), np.array(tb))
    _G[key] = g
    return g
