"""R1, the scene-cut heuristic (JenBridge-style): PySceneDetect ContentDetector, threshold 30, minimum scene 8 s;
every detected scene boundary is a switch point (user approval 2026-10-19, Q2).

The detector's decision depends only on the sequence of per-frame content scores and its FlashFilter (MERGE mode).
So per clip we cache, once, the scores exactly as `scenedetect.detect` computes them (same OpenCV stream, the same
auto-downscale and interpolation as SceneManager, `ContentDetector._calculate_frame_score`), plus the clip's first
and last downscaled frames. A joined sequence is then scored by replaying the real FlashFilter over the concatenated
score series, where the first frame of each later clip scores against the last frame of the clip before it (as one
continuous video would; a frame of another size is resized to the previous frame's size). The cache is
threshold-free and lives in its own directory, data/features/<cue.scene.cache>/; the existing shot boundaries
(shots.csv, `cut_times_s`) are not touched.

Timeline: clip j's frame f sits at t0_j + f / 25 with t0_j = sum of earlier clips' 25 Hz lengths (concat_query);
each clip's score series is cut or zero-padded to its 25 Hz length.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
FPS = 25.0


def _downscale(frame: np.ndarray, factor: float) -> np.ndarray:
    from scenedetect.scene_manager import Interpolation
    if factor > 1.0:
        frame = cv2.resize(frame, (max(1, round(frame.shape[1] / factor)), max(1, round(frame.shape[0] / factor))),
                           interpolation=Interpolation.LINEAR.value)
    return frame


def content_scores(video_path: str | Path) -> dict:
    """Per-frame ContentDetector scores of one video (score[0] = 0, as the detector's first call), and its first /
    last downscaled frames."""
    from scenedetect import ContentDetector, open_video
    from scenedetect.scene_manager import compute_downscale_factor
    video = open_video(str(video_path), backend="opencv")
    w, h = video.frame_size
    factor = compute_downscale_factor(max(w, h))
    det = ContentDetector()
    scores, first, last = [], None, None
    while True:
        frame = video.read()
        if frame is False:
            break
        small = _downscale(frame, factor)
        scores.append(det._calculate_frame_score(video.position, small))
        if first is None:
            first = small
        last = small
    return {"scores": np.array(scores, np.float32), "first": first, "last": last, "fps": float(video.frame_rate),
            "size": [w, h], "downscale": float(factor)}


def pair_score(left: np.ndarray, right: np.ndarray) -> float:
    """ContentDetector score of `right` following `left` (resized to left's size if needed)."""
    from scenedetect import ContentDetector
    from scenedetect.common import FrameTimecode
    if right.shape != left.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    det = ContentDetector()
    det._calculate_frame_score(FrameTimecode(0, fps=FPS), left)
    return float(det._calculate_frame_score(FrameTimecode(1, fps=FPS), right))


def replay_cuts(scores: np.ndarray, threshold: float, min_scene_s: float, fps: float = FPS) -> list[int]:
    """Cut frames (first frame of each new scene) that ContentDetector(threshold, min_scene_len=min_scene_s) emits for
    this score series: the detector's own FlashFilter, fed frame by frame."""
    from scenedetect.common import FrameTimecode
    from scenedetect.detector import FlashFilter
    ff = FlashFilter(FlashFilter.Mode.MERGE, length=float(min_scene_s))
    cuts = []
    for i, s in enumerate(scores):
        cuts += [int(t.frame_num) for t in ff.filter(timecode=FrameTimecode(i, fps=float(fps)), above_threshold=bool(s >= threshold))]
    return sorted(set(cuts))


def detect_cut_frames(video_path: str | Path, threshold: float, min_scene_s: float) -> list[int]:
    """Reference: scenedetect.detect on a file (used to verify the replay)."""
    from scenedetect import ContentDetector, detect
    scenes = detect(str(video_path), ContentDetector(threshold=threshold, min_scene_len=float(min_scene_s)))
    return [int(s[0].frame_num) for s in scenes[1:]]


# ---------------------------------------------------------------- cache
def cache_dir(cfg) -> Path:
    return Path(cfg.paths.features) / cfg.cue.scene.cache


def _one(args):
    pair_id, src, dst = args
    if Path(dst).exists():
        return pair_id, "cached"
    r = content_scores(src)
    meta = {"id": pair_id, "detector": "ContentDetector", "weights": "default (1, 1, 1, 0)", "fps": r["fps"], "size": r["size"],
            "downscale": r["downscale"], "n_frames": int(len(r["scores"])), "source": str(src)}
    tmp = Path(dst).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, scores=r["scores"], first=r["first"], last=r["last"], meta=np.array(json.dumps(meta)))
    tmp.rename(dst)
    return pair_id, "done"


def build_scene_cache(cfg, workers: int = 16) -> Path:
    from bgm.data import manifest
    from bgm.data.prep import interim_video_path
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(interim_video_path(cfg, p)), str(out / f"{p}.npz")) for p in manifest.read_pairs(cfg).pair_id]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, jobs, chunksize=4))
    log.info("scene cache: %d clips (%d new) -> %s", len(res), sum(r == "done" for _, r in res), out)
    return out


class SceneCache:
    def __init__(self, cfg):
        self.dir = cache_dir(cfg)
        self._c: dict[str, dict] = {}

    def get(self, p: str) -> dict:
        if p not in self._c:
            with np.load(self.dir / f"{p}.npz", allow_pickle=False) as z:
                self._c[p] = {"scores": z["scores"], "first": z["first"], "last": z["last"]}
        return self._c[p]


def sequence_scores(cache: SceneCache, parts: list[str], n25: list[int]) -> np.ndarray:
    """The score series of the joined sequence on the 25 Hz concat timeline."""
    out = []
    for j, (p, n) in enumerate(zip(parts, n25)):
        s = cache.get(p)["scores"].astype(np.float64)[:n]
        s = np.r_[s, np.zeros(n - len(s))]
        if j:
            s[0] = pair_score(cache.get(parts[j - 1])["last"], cache.get(p)["first"])
        out.append(s)
    return np.concatenate(out)


def scene_logits(cut_times: list[float], boundary_times: np.ndarray, snap_s: float, eps: float) -> np.ndarray:
    """Each scene cut marks the nearest segment boundary within snap_s; logit(1 - eps) there, logit(eps) elsewhere."""
    bt = np.asarray(boundary_times, float)
    mark = np.zeros(len(bt), bool)
    for c in cut_times:
        if len(bt):
            k = int(np.argmin(np.abs(bt - c)))
            if abs(bt[k] - c) <= snap_s:
                mark[k] = True
    hi = float(np.log1p(-eps) - np.log(eps))
    return np.where(mark, hi, -hi)
