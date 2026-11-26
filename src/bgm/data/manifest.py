"""The only reader and writer of data/manifests/ (SPEC §4.1): pairs.csv, tracks.csv, folds.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

PAIRS_COLUMNS = ["pair_id", "video_path", "music_path", "music_source", "music_id", "video_duration_s",
                 "music_duration_s", "video_start_s", "audio_start_s", "fps", "width", "height", "video_has_audio",
                 "music_sr", "music_channels", "track_id"]
TRACKS_COLUMNS = ["track_id", "music_ids", "evidence"]
_STR_COLS = {"pair_id": str, "video_path": str, "music_path": str, "music_source": str, "music_id": str, "track_id": str}


def manifests_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.manifests)


def read_pairs(cfg: DictConfig) -> pd.DataFrame:
    """pairs.csv, one row per pair, ordered by pair_id. ID columns are strings; meta_* columns as stored."""
    df = pd.read_csv(manifests_dir(cfg) / "pairs.csv", dtype=_STR_COLS, keep_default_na=False,
                     na_values={c: [""] for c in ["video_start_s", "audio_start_s"]})
    return df.sort_values("pair_id").reset_index(drop=True)


def write_pairs(df: pd.DataFrame, cfg: DictConfig) -> Path:
    missing = [c for c in PAIRS_COLUMNS if c not in df.columns]
    assert not missing, f"pairs.csv is missing columns {missing}"
    meta = sorted(c for c in df.columns if c.startswith("meta_"))
    path = manifests_dir(cfg) / "pairs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("pair_id")[PAIRS_COLUMNS + meta].to_csv(path, index=False)
    return path


def read_tracks(cfg: DictConfig) -> pd.DataFrame:
    return pd.read_csv(manifests_dir(cfg) / "tracks.csv", dtype=str, keep_default_na=False)


def write_tracks(df: pd.DataFrame, cfg: DictConfig) -> Path:
    path = manifests_dir(cfg) / "tracks.csv"
    df[TRACKS_COLUMNS].sort_values("track_id").to_csv(path, index=False)
    return path


def manifest_hash(cfg: DictConfig) -> str:
    """sha256 over pairs.csv and tracks.csv; recorded in folds.json to freeze the folds against them."""
    h = hashlib.sha256()
    for name in ("pairs.csv", "tracks.csv"):
        h.update((manifests_dir(cfg) / name).read_bytes())
    return h.hexdigest()


def folds_file(cfg: DictConfig) -> str:
    """`splits.folds_file`: folds.json (track_id split, SPEC §3.3) or folds_film.json (film split, PROGRESS.md D36)."""
    return cfg.splits.get("folds_file", "folds.json")


def read_folds(cfg: DictConfig) -> dict:
    """{"folds": {"0": {"train": [...], "val": [...], "test": [...]}, ...}, "manifest_hash", "created"}; the film split
    adds "film_group" {pair_id: group}, "train_only" and "split". Asserts the manifests have not changed since the
    folds were made."""
    path = manifests_dir(cfg) / folds_file(cfg)
    folds = json.loads(path.read_text())
    assert folds["manifest_hash"] == manifest_hash(cfg), \
        f"pairs.csv / tracks.csv changed after {path.name} was written: rerun `bgm folds` and everything downstream"
    return folds


def read_film_groups(cfg: DictConfig) -> dict[str, str] | None:
    """{pair_id: film group} of the configured folds file, or None for the track_id split."""
    return read_folds(cfg).get("film_group")


def write_folds(folds: dict, cfg: DictConfig) -> Path:
    path = manifests_dir(cfg) / folds_file(cfg)
    path.write_text(json.dumps(folds, indent=1))
    return path


def write_shots(df: pd.DataFrame, cfg: DictConfig) -> Path:
    """shots.csv: pair_id, cut_times_s (space-separated seconds on the interim timeline)."""
    path = manifests_dir(cfg) / "shots.csv"
    df[["pair_id", "cut_times_s"]].sort_values("pair_id").to_csv(path, index=False)
    return path


def read_shots(cfg: DictConfig) -> pd.DataFrame:
    return pd.read_csv(manifests_dir(cfg) / "shots.csv", dtype=str, keep_default_na=False)


def cut_times(shots: pd.DataFrame, pair_id: str) -> list[float]:
    row = shots.loc[shots["pair_id"] == pair_id, "cut_times_s"]
    assert len(row) == 1, f"{pair_id} not in shots.csv"
    return [float(t) for t in row.iloc[0].split()]


def write_segments(df: pd.DataFrame, cfg: DictConfig) -> Path:
    """segments.csv: pair_id, seg_idx, start_s, end_s, n_shots."""
    path = manifests_dir(cfg) / "segments.csv"
    df[["pair_id", "seg_idx", "start_s", "end_s", "n_shots"]].sort_values(["pair_id", "seg_idx"]).to_csv(path, index=False)
    return path


def read_segments(cfg: DictConfig) -> pd.DataFrame:
    return pd.read_csv(manifests_dir(cfg) / "segments.csv", dtype={"pair_id": str})
