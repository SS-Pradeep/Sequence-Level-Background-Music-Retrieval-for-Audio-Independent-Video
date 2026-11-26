"""Run synthetic pairs through the built pipeline stages in a temp dir (used by tests and `bgm smoke --fake`)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bgm.config import load_config
from bgm.data import manifest
from bgm.data.audit import probe_row
from bgm.data.prep import interim_video_path, prep_pair
from bgm.data.shots import detect_cuts
from tests.fixtures.synth import SynthPair


def fixture_config(root: str | Path, overrides: tuple[str, ...] = ()):
    root = Path(root)
    return load_config("configs/default.yaml", [f"paths.{k}={root / k}" for k in
                                                ("manifests", "interim", "features", "reports", "runs", "results", "renders")]
                       + ["features.device=cpu", *overrides])


def build_manifest(cfg, pairs: list[SynthPair]) -> pd.DataFrame:
    rows = [probe_row({"pair_id": p.pair_id, "video_path": p.video_path, "music_path": p.music_path or ""}) for p in pairs]
    df = pd.DataFrame(rows)
    df["track_id"] = [f"trk_{i:04d}" for i in range(len(df))]
    df["meta_source"] = "synth"
    manifest.write_pairs(df, cfg)
    manifest.write_tracks(pd.DataFrame({"track_id": df.track_id, "music_ids": df.music_id, "evidence": "none"}), cfg)
    return df


def prep_and_shots(cfg, df: pd.DataFrame) -> None:
    for r in df.to_dict("records"):
        prep_pair(r, cfg)
    cuts = [" ".join(f"{t:.4f}" for t in detect_cuts(interim_video_path(cfg, p), cfg.shots.adaptive_threshold)) for p in df.pair_id]
    manifest.write_shots(pd.DataFrame({"pair_id": df.pair_id, "cut_times_s": cuts}), cfg)


def run_through_extract(root: str | Path, pairs: list[SynthPair], what: str = "sem_video,sem_music,attr,bridge,imagebind,dyn_video,dyn_music,key"):
    from bgm.features.extract import run_extract
    cfg = fixture_config(root)
    df = build_manifest(cfg, pairs)
    prep_and_shots(cfg, df)
    run_extract(cfg, what, fake=True, workers=4)
    return cfg, df
