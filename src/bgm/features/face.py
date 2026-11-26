"""Facial-expression features at 1 Hz (PROGRESS.md D77): how many faces are on screen and what they express.

Run: python -m bgm.features.face        (all pairs of the configured manifests; cached, resumable)

For each second k, the frame at k + 0.5 s of the silent 25 fps interim video (rule 8: no soundtrack):
  faces       OpenCV YuNet (data/models/face_detection_yunet_2023mar.onnx), score >= 0.6, side >= 24 px;
  expression  `trpakov/vit-face-expression` (ViT, FER-2013 classes), frozen, eval, no_grad, fp16, on each face crop
              (box widened by 20%).
`face_expr_1hz` [T1, 17] float32, T1 = the pair's SigLIP length:
  0-6   soft counts: the classifier's probabilities summed over the frame's faces (EXPRESSIONS order);
  7-13  hard counts: number of faces whose top expression is each class;
  14    number of faces;  15  share of the frame covered by faces;  16  1 when the frame has no face.
Files: data/features/video_face/<pair_id>.npz; `FeatureStore.video` merges them into the pair's feature dict.
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
DETECTOR = Path("data/models/face_detection_yunet_2023mar.onnx")
CLASSIFIER = "trpakov/vit-face-expression"
EXPRESSIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
MIN_SCORE, MIN_SIDE, MARGIN, DIM = 0.6, 24, 0.2, 17
CONFIG = {"detector": DETECTOR.name, "classifier": CLASSIFIER, "min_score": MIN_SCORE, "min_side": MIN_SIDE, "margin": MARGIN, "v": 1}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def face_path(cfg: DictConfig, pair_id: str) -> Path:
    return Path(cfg.paths.features) / "video_face" / f"{pair_id}.npz"


def frame_vector(probs: np.ndarray, areas: np.ndarray, frame_area: float) -> np.ndarray:
    """probs: [F, 7] expression probabilities of the frame's F faces; areas: [F] box areas in px -> [17] (see module doc)."""
    v = np.zeros(DIM, np.float32)
    if len(probs) == 0:
        v[16] = 1.0
        return v
    v[0:7] = probs.sum(0)
    v[7:14] = np.bincount(probs.argmax(1), minlength=7)
    v[14], v[15] = len(probs), float(areas.sum() / frame_area)
    return v


def read_second_frames(path: Path, t1: int, rate: int = 25) -> list[np.ndarray]:
    """BGR frames at k + 0.5 s for k = 0 .. t1 - 1 (the last readable frame is repeated if the video ends early)."""
    cap = cv2.VideoCapture(str(path))
    want = {int(round((k + 0.5) * rate)): k for k in range(t1)}
    out, i, last = [None] * t1, 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last = fr
        if i in want:
            out[want[i]] = fr
        i += 1
    cap.release()
    return [f if f is not None else last for f in out]


class FaceExpression:
    """detect(frame) -> boxes; classify(crops) -> [F, 7] probabilities in EXPRESSIONS order. Frozen."""

    def __init__(self, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = device != "cpu"                          # fp16 on GPU; fp32 on CPU
        self.det = cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320), MIN_SCORE, 0.3, 50)
        self.proc = AutoImageProcessor.from_pretrained(CLASSIFIER)
        self.model = AutoModelForImageClassification.from_pretrained(CLASSIFIER, dtype=torch.float16 if self.half else torch.float32).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        labels = [self.model.config.id2label[i].lower() for i in range(len(self.model.config.id2label))]
        self.order = [labels.index(e) for e in EXPRESSIONS]
        self.device = device

    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, f = self.det.detect(frame)
        if f is None:
            return np.zeros((0, 4), int)
        b = f[:, :4]
        return b[(b[:, 2] >= MIN_SIDE) & (b[:, 3] >= MIN_SIDE)].astype(int)

    @torch.no_grad()
    def classify(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 7), np.float32)
        x = self.proc(images=[cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops], return_tensors="pt")["pixel_values"].to(self.device)
        x = x.half() if self.half else x
        p = torch.softmax(self.model(pixel_values=x).logits.float(), -1).cpu().numpy()
        return p[:, self.order]


def crop(frame: np.ndarray, box: np.ndarray, margin: float = MARGIN) -> np.ndarray:
    x, y, w, h = box
    H, W = frame.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return frame[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


def encode_video(fe: FaceExpression, frames: list[np.ndarray]) -> np.ndarray:
    boxes = [fe.detect(f) for f in frames]
    crops = [crop(f, b) for f, bs in zip(frames, boxes) for b in bs]
    probs = np.concatenate([fe.classify(crops[i:i + 64]) for i in range(0, len(crops), 64)]) if crops else np.zeros((0, 7), np.float32)
    out, j = [], 0
    for f, bs in zip(frames, boxes):
        out.append(frame_vector(probs[j:j + len(bs)], (bs[:, 2] * bs[:, 3]).astype(float), float(f.shape[0] * f.shape[1])))
        j += len(bs)
    return np.stack(out)


def run(cfg: DictConfig, device: str | None = None, shard: int = 0, n_shards: int = 1) -> int:
    pairs = manifest.read_pairs(cfg)
    fe, done = None, 0
    for n, p in enumerate(pairs.pair_id):
        if n % n_shards != shard:
            continue
        out = face_path(cfg, p)
        if out.exists():
            continue
        with np.load(feature_path(cfg, "video", p)) as z:
            t1 = len(z["sem_siglip_1hz"])
        fe = fe or FaceExpression(device)
        v = encode_video(fe, read_second_frames(Path(interim_video_path(cfg, p)), t1))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, face_expr_1hz=v, meta=np.array(json.dumps({"config_hash": CONFIG_HASH, **CONFIG})))
        tmp.replace(out)
        done += 1
        if done % 100 == 0:
            log.info("%d / %d pairs", n + 1, len(pairs))
    log.info("video_face: %d files written", done)
    return done


if __name__ == "__main__":
    from bgm.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    import sys
    a = [int(x) for x in sys.argv[1:3]] or [0, 1]
    run(load_config("configs/default.yaml", []), shard=a[0], n_shards=a[1])
