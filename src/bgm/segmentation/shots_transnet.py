"""TransNetV2 shot detection on the silent 25 fps interim videos (third_party/transnetv2, PyTorch port).

Weights: third_party/ckpt/transnetv2-pytorch-weights.pth, converted from the official TF SavedModel with the official
remapping (convert_weights.py); max |torch - TF| = 4e-6 on random inputs (D140).

Prediction follows the official `predict_frames` / `predictions_to_scenes`: 48x27 RGB frames, 25 copies of the first
frame before and 25 + pad of the last after, 100-frame windows with stride 50 keeping frames [25, 75), per-frame
single-transition probability, scenes = runs of frames below the threshold (inclusive frame ranges).
Shots tile the clip: shot k spans [start_k / fps, start_{k+1} / fps), so transition frames go to the earlier shot.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party" / "transnetv2" / "inference-pytorch"
WEIGHTS = ROOT / "third_party" / "ckpt" / "transnetv2-pytorch-weights.pth"
FPS = 25


def load_model(device: str = "cuda") -> torch.nn.Module:
    sys.path.insert(0, str(REPO))
    from transnetv2_pytorch import TransNetV2
    sys.path.remove(str(REPO))
    m = TransNetV2()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval().to(device)


def read_frames_48x27(video_path: str | Path) -> np.ndarray:
    """[T, 27, 48, 3] uint8 RGB, all frames (ffmpeg scaling as in the official inference code)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "48x27", "pipe:"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, 27, 48, 3)


@torch.no_grad()
def predict_frames(model: torch.nn.Module, frames: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Per-frame transition probability [T] (official sliding window)."""
    T = len(frames)
    end_pad = 25 + (50 - T % 50 if T % 50 != 0 else 50)
    padded = np.concatenate([np.repeat(frames[:1], 25, 0), frames, np.repeat(frames[-1:], end_pad, 0)])
    wins = np.stack([padded[i:i + 100] for i in range(0, len(padded) - 50, 50)])
    probs = []
    for i in range(0, len(wins), 16):
        single, _ = model(torch.from_numpy(wins[i:i + 16]).to(device))
        probs.append(torch.sigmoid(single.float())[:, 25:75, 0].cpu().numpy())
    return np.concatenate(probs).reshape(-1)[:T]


def predictions_to_scenes(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """(official) [n, 2] inclusive frame ranges of scenes."""
    p = (pred > threshold).astype(np.uint8)
    scenes, t, t_prev, start = [], -1, 0, 0
    for i, t in enumerate(p):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(p) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def scenes_to_shots(scenes: np.ndarray, n_frames: int, fps: int = FPS) -> list[list[float]]:
    """[(start_frame, end_frame, start_s, end_s)] tiling [0, n_frames / fps]: shot k starts at scene k's first frame
    (the first shot at 0) and ends where shot k + 1 starts; end_frame is the scene's own last frame."""
    starts = [0] + [int(s) for s in scenes[1:, 0]]
    out = []
    for k, (sc, s0) in enumerate(zip(scenes, starts)):
        s1 = starts[k + 1] if k + 1 < len(starts) else n_frames
        out.append([s0, int(sc[1]), round(s0 / fps, 4), round(s1 / fps, 4)])
    return out


def shots_from_cuts(cut_times_s, duration_s: float, fps: int = FPS) -> list[list[float]]:
    """The same shot format from cut times (PySceneDetect shots.csv)."""
    b = [0.0] + sorted(float(c) for c in cut_times_s if 0 < c < duration_s) + [float(duration_s)]
    return [[int(round(a * fps)), max(int(round(e * fps)) - 1, int(round(a * fps))), round(a, 4), round(e, 4)]
            for a, e in zip(b[:-1], b[1:]) if e > a]


def run(paths: dict[str, Path], threshold: float = 0.5, device: str = "cuda", workers: int = 6) -> dict:
    """{pair_id: {'shots': [...], 'prob': [T] float16}} for {pair_id: interim video path}."""
    model = load_model(device)
    out = {}
    ids = list(paths)
    with ThreadPoolExecutor(workers) as ex:
        for n, (pid, frames) in enumerate(zip(ids, ex.map(lambda p: read_frames_48x27(paths[p]), ids))):
            prob = predict_frames(model, frames, device)
            out[pid] = {"shots": scenes_to_shots(predictions_to_scenes(prob, threshold), len(frames)),
                        "prob": prob.astype(np.float16)}
            if n % 200 == 0:
                log.info("transnetv2: %d / %d", n, len(ids))
    return out
