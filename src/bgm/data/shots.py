"""`bgm shots` and `bgm peek` (SPEC §3.5): shot cuts on the interim timeline, and contact sheets."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.data.prep import interim_video_path

log = logging.getLogger(__name__)


def detect_cuts(video_path: str | Path, adaptive_threshold: float = 3.0) -> list[float]:
    """Cut times (s) = the time of the first frame of every shot after the first (PySceneDetect AdaptiveDetector)."""
    from scenedetect import AdaptiveDetector, detect
    scenes = detect(str(video_path), AdaptiveDetector(adaptive_threshold=adaptive_threshold))
    return [round(s[0].seconds, 4) for s in scenes[1:]]


def _detect(args):
    pair_id, path, thr = args
    return pair_id, detect_cuts(path, thr)


def run_shots(cfg: DictConfig, workers: int = 16) -> pd.DataFrame:
    assert cfg.shots.detector == "pyscenedetect", "only the pyscenedetect detector is built"
    pairs = manifest.read_pairs(cfg)
    jobs = [(p, str(interim_video_path(cfg, p)), cfg.shots.adaptive_threshold) for p in pairs["pair_id"]]
    missing = [p for p, path, _ in jobs if not Path(path).exists()]
    assert not missing, f"{len(missing)} interim videos missing (run `bgm prep`), e.g. {missing[:3]}"
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_detect, jobs, chunksize=8))
    df = pd.DataFrame({"pair_id": pairs["pair_id"], "cut_times_s": [" ".join(f"{t:.4f}" for t in res[p]) for p in pairs["pair_id"]]})
    manifest.write_shots(df, cfg)
    n = df["cut_times_s"].str.split().str.len().fillna(0)
    log.info("shots: %d videos, cuts per video median %.0f, max %d, %d with none", len(df), n.median(), n.max(), int((n == 0).sum()))
    return df


def contact_sheet(video_path: str | Path, cut_times: list[float], out_png: str | Path, thumb_w: int = 240, cols: int = 6) -> Path:
    """PNG grid of each shot's first frame, labelled with its start time."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    tiles = []
    for t in [0.0] + list(cut_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000 + 1)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        tile = cv2.resize(frame, (thumb_w, int(h * thumb_w / w)))
        cv2.putText(tile, f"{t:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    assert tiles, f"could not read {video_path}"
    th = tiles[0].shape[0]
    tiles = [cv2.resize(t, (thumb_w, th)) for t in tiles]
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rows[-1] += [np.zeros_like(tiles[0])] * (cols - len(rows[-1]))
    sheet = np.vstack([np.hstack(r) for r in rows])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png


def peek(cfg: DictConfig, pair_id: str) -> Path:
    shots = manifest.read_shots(cfg)
    cuts = manifest.cut_times(shots, pair_id)
    return contact_sheet(interim_video_path(cfg, pair_id), cuts, Path(cfg.paths.reports) / "peek" / f"{pair_id}.png")
