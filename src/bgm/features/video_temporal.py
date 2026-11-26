"""Temporal video features (PROGRESS.md D67): a frozen video encoder that sees motion across frames, at 1 Hz.

Run: python -m bgm.features.video_temporal        (all pairs of the configured manifests; cached, resumable)

For each second k of the silent 25 fps interim video (rule 8: no soundtrack), a clip of N_FRAMES frames taken every
STRIDE frames and centred on k + 0.5 s (indices clamped to the video) goes through V-JEPA 2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`, frozen, eval, no_grad, fp16). The mean of the encoder's output tokens is
L2-normalized: `sem_vjepa_1hz` [T1, 1024], T1 = the pair's SigLIP length. `sem_siglip_vjepa_1hz` [T1, 2176] is the
SigLIP and V-JEPA unit vectors concatenated and divided by sqrt(2). Files: data/features/video_extra/<pair_id>.npz;
`FeatureStore.video` merges them into the pair's feature dict.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.data.prep import interim_video_path
from bgm.features.extract import feature_path

log = logging.getLogger(__name__)
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES, STRIDE, SIZE, RATE = 16, 3, 256, 25
CONFIG = {"model": MODEL_ID, "n_frames": N_FRAMES, "stride": STRIDE, "size": SIZE, "pool": "mean_tokens", "v": 1}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def extra_path(cfg: DictConfig, pair_id: str) -> Path:
    return Path(cfg.paths.features) / "video_extra" / f"{pair_id}.npz"


def clip_indices(n_frames_video: int, t1: int, n: int = N_FRAMES, stride: int = STRIDE, rate: int = RATE) -> np.ndarray:
    """[t1, n] frame indices: for second k, n frames every `stride` frames centred on (k + 0.5) s, clamped to the video."""
    centre = (np.arange(t1) + 0.5) * rate
    off = (np.arange(n) - (n - 1) / 2) * stride
    return np.clip(np.round(centre[:, None] + off[None, :]).astype(int), 0, n_frames_video - 1)


def read_frames(path: Path, size: int = SIZE) -> np.ndarray:
    """[T, size, size, 3] uint8 RGB: short side resized to `size`, centre crop."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = size / min(h, w)
        fr = cv2.resize(fr, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = fr.shape[:2]
        y, x = (h - size) // 2, (w - size) // 2
        out.append(cv2.cvtColor(fr[y:y + size, x:x + size], cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out)


class VJepaEncoder:
    """encode_frames(clips [B, n, H, W, 3] uint8) -> [B, 1024] unit vectors. Frozen."""

    def __init__(self, device: str = "cuda"):
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.mean = torch.tensor(MEAN, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def encode_frames(self, clips: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(clips).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0      # [B, n, 3, H, W]
        x = ((x - self.mean) / self.std).half()
        h = self.model(pixel_values_videos=x).last_hidden_state.float().mean(1)
        return torch.nn.functional.normalize(h, dim=-1).cpu().numpy()


class FakeTemporalEncoder:
    """Deterministic stand-in for tests: a fixed random projection of each clip's mean colour and frame-to-frame change."""

    def __init__(self, dim: int = 32):
        self.W = np.random.default_rng(0).standard_normal((6, dim)).astype(np.float32)

    def encode_frames(self, clips: np.ndarray) -> np.ndarray:
        c = clips.astype(np.float32) / 255.0
        f = np.concatenate([c.mean((1, 2, 3)), np.abs(np.diff(c, axis=1)).mean((1, 2, 3))], 1) @ self.W
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)


def encode_video(enc, frames: np.ndarray, t1: int, batch: int = 16) -> np.ndarray:
    """[t1, D] float32 unit vectors, one per second."""
    idx = clip_indices(len(frames), t1)
    return np.concatenate([enc.encode_frames(frames[idx[i:i + batch]]) for i in range(0, t1, batch)])


# ---------------------------------------------------------------- dense series for the dynamics branch (D70)
DYN_CONFIG = {"model": MODEL_ID, "clip_frames": N_FRAMES, "hop_frames": N_FRAMES, "size": SIZE, "pool": "spatial_mean_per_tubelet", "v": 1}
DYN_HASH = hashlib.sha1(json.dumps(DYN_CONFIG, sort_keys=True).encode()).hexdigest()[:12]
TUBELET = 2                                   # V-JEPA 2 merges 2 frames per temporal token: 12.5 tokens per second


def dyn_path(cfg: DictConfig, pair_id: str) -> Path:
    return Path(cfg.paths.features) / "video_dyn" / f"{pair_id}.npz"


@torch.no_grad()
def encode_dense(enc: "VJepaEncoder", frames: np.ndarray, batch: int = 16) -> np.ndarray:
    """[T, H, W, 3] uint8 at 25 fps -> [ceil(T / 2), 1024] float32: consecutive 16-frame clips (the last one padded by
    repeating the final frame), each token = the spatial mean of one 2-frame tubelet, so sample j covers frames
    2j and 2j + 1 (time 2j / 25 s). Not normalized here."""
    T = len(frames)
    n = int(np.ceil(T / N_FRAMES))
    idx = np.clip(np.arange(n * N_FRAMES).reshape(n, N_FRAMES), 0, T - 1)
    out = []
    for i in range(0, n, batch):
        x = torch.from_numpy(frames[idx[i:i + batch]]).to(enc.device).permute(0, 1, 4, 2, 3).float() / 255.0
        h = enc.model(pixel_values_videos=((x - enc.mean) / enc.std).half()).last_hidden_state.float()
        out.append(h.view(len(h), N_FRAMES // TUBELET, -1, h.shape[-1]).mean(2).reshape(-1, h.shape[-1]).cpu().numpy())
    return np.concatenate(out)[: int(np.ceil(T / TUBELET))]


def to_25hz(x12: np.ndarray, n25: int) -> np.ndarray:
    """[T12, D] tubelet series -> [n25, D] at 25 Hz: each sample repeated for its two frames, z-scored per file and
    dimension over time (the file's static content is removed; what is left is change over time), fitted to n25."""
    x = np.repeat(x12.astype(np.float32), TUBELET, 0)
    x = x[:n25] if len(x) >= n25 else np.concatenate([x, np.repeat(x[-1:], n25 - len(x), 0)])
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def run_dyn(cfg: DictConfig, device: str = "cuda") -> int:
    pairs = manifest.read_pairs(cfg)
    enc, done = None, 0
    for n, p in enumerate(pairs.pair_id):
        out = dyn_path(cfg, p)
        if out.exists():
            continue
        enc = enc or VJepaEncoder(device)
        v = encode_dense(enc, read_frames(Path(interim_video_path(cfg, p))))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, dyn_vjepa_12hz=v.astype(np.float16), meta=np.array(json.dumps({"config_hash": DYN_HASH, **DYN_CONFIG})))
        tmp.replace(out)
        done += 1
        if done % 100 == 0:
            log.info("dense: %d / %d pairs", n + 1, len(pairs))
    log.info("video_dyn: %d files written", done)
    return done


def run(cfg: DictConfig, device: str = "cuda") -> int:
    pairs = manifest.read_pairs(cfg)
    enc, done = None, 0
    for n, p in enumerate(pairs.pair_id):
        out = extra_path(cfg, p)
        if out.exists():
            with np.load(out) as z:
                if json.loads(str(z["meta"])).get("config_hash") == CONFIG_HASH:
                    continue
        with np.load(feature_path(cfg, "video", p)) as z:
            sig = z["sem_siglip_1hz"].astype(np.float32)
        enc = enc or VJepaEncoder(device)
        v = encode_video(enc, read_frames(Path(interim_video_path(cfg, p))), len(sig)).astype(np.float32)
        both = np.concatenate([sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12), v], 1) / np.sqrt(2.0)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, sem_vjepa_1hz=v.astype(np.float16), sem_siglip_vjepa_1hz=both.astype(np.float16),
                 meta=np.array(json.dumps({"config_hash": CONFIG_HASH, **CONFIG})))
        tmp.replace(out)
        done += 1
        if done % 50 == 0:
            log.info("%d / %d pairs (%d new)", n + 1, len(pairs), done)
    log.info("video_extra: %d files written", done)
    return done


if __name__ == "__main__":
    from bgm.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    import sys
    (run_dyn if sys.argv[1:] == ["dense"] else run)(load_config("configs/default.yaml", []))
