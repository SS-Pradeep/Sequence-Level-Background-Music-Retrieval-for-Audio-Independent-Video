"""SPEC §3.2 step 3 (after M3): near-duplicate candidates by CLAP file-mean cosine, for the user to confirm.

Confirmed links go to data/manifests/confirmed_links.csv (music_id_a, music_id_b); `bgm audit` unions them into the
track groups with evidence `embedding-confirmed`. Then rerun `bgm folds` and freeze.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.features.extract import feature_path, load_features


def candidates(cfg: DictConfig, threshold: float = 0.90) -> pd.DataFrame:
    pairs = manifest.read_pairs(cfg)
    music = list(dict.fromkeys(pairs.music_id))
    track = dict(zip(pairs.music_id, pairs.track_id))
    E = []
    for m in music:
        e = load_features(feature_path(cfg, "music", m))["sem_clap_1hz"].astype(np.float64).mean(0)
        E.append(e / np.linalg.norm(e))
    E = np.array(E)
    S = E @ E.T
    i, j = np.where(np.triu(S, 1) >= threshold)
    df = pd.DataFrame({"music_id_a": [music[k] for k in i], "music_id_b": [music[k] for k in j], "clap_cos": S[i, j]})
    df = df[[track[a] != track[b] for a, b in zip(df.music_id_a, df.music_id_b)]]
    return df.sort_values("clap_cos", ascending=False).reset_index(drop=True)


def write_report(cfg: DictConfig, df: pd.DataFrame, threshold: float = 0.90) -> Path:
    path = Path(cfg.paths.reports) / "near_duplicates.md"
    lines = ["# Near-duplicate candidates (SPEC §3.2 step 3)", "",
             f"CLAP file-mean cosine >= {threshold}, not already in one track group: **{len(df)} pairs**.",
             "Confirmed links go to `data/manifests/confirmed_links.csv` (columns music_id_a, music_id_b).", "",
             "| # | music A | music B | CLAP cos |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {r.music_id_a} | {r.music_id_b} | {r.clap_cos:.3f} |" for k, r in enumerate(df.itertuples())]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_confirmed(cfg: DictConfig) -> pd.DataFrame:
    path = manifest.manifests_dir(cfg) / "confirmed_links.csv"
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["music_id_a", "music_id_b"])
