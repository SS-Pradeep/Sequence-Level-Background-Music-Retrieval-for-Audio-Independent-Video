"""Step 2: person crops at 1 Hz, encoded with the same SigLIP image encoder as the frames (PROGRESS.md D87+).

Run: python -m bgm.person.features [shard n_shards]   (all pairs; cached and resumable; never overwrites a file)

For each second k, the frame SigLIP saw (`frames_1hz`: frame 25k + 12 of the silent 25 fps interim video, rule 8):
  faces     OpenCV YuNet with D77's settings (score >= 0.6, side >= 24 px), the 5 largest; crop = the box widened by
            30% on each side, then grown to a square inside the frame (SigLIP resizes to 384 x 384 without keeping
            aspect);
  fallback  only when the frame has no face: torchvision Keypoint R-CNN (COCO person class; weights already cached,
            nothing downloaded), score >= 0.7, box height >= 48 px, the 5 largest; crop = the box grown to a square;
  encoding  google/siglip-so400m-patch14-384 (frozen, eval, no_grad, fp16; `EncoderBank`), unit vectors, the
            second's vector is the mean of its crops' vectors (not re-normalised).
All settings are fixed in advance, not tuned.
Arrays (data/features/video_person/<pair_id>.npz; T1 = the pair's SigLIP length):
  person_siglip_1hz [T1, 1152] f16   0 where the second has no person
  person_kind_1hz   [T1] int8        0 none, 1 faces, 2 person boxes
  person_n_1hz      [T1] int8        number of crops
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
from bgm.features.extract import EncoderBank, feature_path, frames_1hz

log = logging.getLogger(__name__)
DETECTOR = Path("data/models/face_detection_yunet_2023mar.onnx")
FACE_SCORE, FACE_MIN_SIDE, FACE_MARGIN = 0.6, 24, 0.3
PERSON_SCORE, PERSON_MIN_H, MAX_CROPS = 0.7, 48, 5
CONFIG = {"detector": DETECTOR.name, "face_score": FACE_SCORE, "face_min_side": FACE_MIN_SIDE, "face_margin": FACE_MARGIN,
          "person_model": "keypointrcnn_resnet50_fpn/COCO_V1", "person_score": PERSON_SCORE, "person_min_h": PERSON_MIN_H,
          "max_crops": MAX_CROPS, "encoder": "google/siglip-so400m-patch14-384", "frames": "frames_1hz", "v": 1}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
NONE, FACE, BODY = 0, 1, 2


def person_path(cfg: DictConfig, pair_id: str) -> Path:
    return Path(cfg.paths.features) / "video_person" / f"{pair_id}.npz"


def square_box(x0: float, y0: float, x1: float, y1: float, W: int, H: int) -> tuple[int, int, int, int]:
    """Grow [x0, x1) x [y0, y1) to a square around its centre (side = the longer side, capped by the frame), shifted
    to stay inside a W x H frame. Returns integer (x0, y0, x1, y1)."""
    side = min(max(x1 - x0, y1 - y0), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = int(round(min(max(cx - side / 2, 0), W - side)))
    b = int(round(min(max(cy - side / 2, 0), H - side)))
    s = int(round(side))
    return a, b, min(a + s, W), min(b + s, H)


def face_crop_box(box: np.ndarray, W: int, H: int, margin: float = FACE_MARGIN) -> tuple[int, int, int, int]:
    """YuNet box (x, y, w, h) -> square crop box widened by `margin` of the box on each side."""
    x, y, w, h = map(float, box)
    return square_box(x - margin * w, y - margin * h, x + w + margin * w, y + h + margin * h, W, H)


def largest(boxes: np.ndarray, area: np.ndarray, k: int = MAX_CROPS) -> np.ndarray:
    return boxes[np.argsort(-area, kind="stable")[:k]]


class PersonCropper:
    """Frozen detectors. faces(rgb) -> [F, 4] crop boxes; bodies(list of rgb) -> list of [P, 4] crop boxes."""

    def __init__(self, device: str):
        self.det = cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320), FACE_SCORE, 0.3, 50)
        self.device, self._body = device, None

    def faces(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        self.det.setInputSize((W, H))
        _, f = self.det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if f is None:
            return np.zeros((0, 4), int)
        b = f[:, :4]
        b = b[(b[:, 2] >= FACE_MIN_SIDE) & (b[:, 3] >= FACE_MIN_SIDE)]
        b = largest(b, b[:, 2] * b[:, 3])
        return np.array([face_crop_box(x, W, H) for x in b], int).reshape(-1, 4)

    def _body_model(self):
        if self._body is None:
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
            m = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).to(self.device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self._body = m
        return self._body

    @torch.no_grad()
    def bodies(self, frames: list[np.ndarray], batch: int = 8) -> list[np.ndarray]:
        if not frames:
            return []
        m = self._body_model()
        out = []
        for i in range(0, len(frames), batch):
            x = [torch.from_numpy(np.array(f)).permute(2, 0, 1).to(self.device).float() / 255 for f in frames[i:i + batch]]
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                out += [{k: v.detach() for k, v in o.items()} for o in m(x)]
        res = []
        for f, o in zip(frames, out):
            H, W = f.shape[:2]
            b, s, lab = o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy()
            b = b[(lab == 1) & (s >= PERSON_SCORE) & (b[:, 3] - b[:, 1] >= PERSON_MIN_H)]
            b = largest(b, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
            res.append(np.array([square_box(*x, W, H) for x in b], int).reshape(-1, 4))
        return res


def person_arrays(cropper: PersonCropper, bank: EncoderBank, frames: np.ndarray) -> dict[str, np.ndarray]:
    """frames: [T1, H, W, 3] uint8 RGB -> the module's three arrays."""
    T1 = len(frames)
    boxes = [cropper.faces(f) for f in frames]
    kind = np.array([FACE if len(b) else NONE for b in boxes], np.int8)
    no_face = [k for k in range(T1) if not len(boxes[k])]
    for k, b in zip(no_face, cropper.bodies([frames[k] for k in no_face])):
        boxes[k] = b
        kind[k] = BODY if len(b) else NONE
    crops = [frames[k][y0:y1, x0:x1] for k in range(T1) for x0, y0, x1, y1 in boxes[k]]
    emb = np.zeros((T1, 1152), np.float32)
    if crops:
        E = bank.frames("siglip", crops)
        j = 0
        for k in range(T1):
            n = len(boxes[k])
            if n:
                emb[k] = E[j:j + n].mean(0)
                j += n
    return {"person_siglip_1hz": emb.astype(np.float16), "person_kind_1hz": kind,
            "person_n_1hz": np.array([len(b) for b in boxes], np.int8)}


def save(path: Path, arrays: dict) -> None:
    assert not path.exists(), f"{path} exists: person caches are never overwritten"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays, meta=np.array(json.dumps({"config_hash": CONFIG_HASH, **CONFIG})))
    tmp.replace(path)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        assert meta["config_hash"] == CONFIG_HASH, (path, meta["config_hash"], CONFIG_HASH)
        return {k: z[k] for k in z.files if k != "meta"}


def run(cfg: DictConfig, shard: int = 0, n_shards: int = 1) -> int:
    pairs = manifest.read_pairs(cfg)
    device = cfg.features.device if torch.cuda.is_available() else "cpu"
    cropper, bank, done = None, None, 0
    for n, p in enumerate(pairs.pair_id):
        if n % n_shards != shard or person_path(cfg, p).exists():
            continue
        with np.load(feature_path(cfg, "video", p)) as z:
            t1 = len(z["sem_siglip_1hz"])
        cropper, bank = cropper or PersonCropper(device), bank or EncoderBank(cfg, fake=False)
        save(person_path(cfg, p), person_arrays(cropper, bank, frames_1hz(Path(interim_video_path(cfg, p)), t1)))
        done += 1
        if done % 100 == 0:
            log.info("%d / %d pairs", n + 1, len(pairs))
    log.info("video_person: %d files written", done)
    return done


def coverage(cfg: DictConfig, out: Path = Path("runs/filmsplit/person/coverage.json")) -> dict:
    """Share of seconds with faces / person boxes only / nobody, and of segments with no person second."""
    from bgm.eval.store import FeatureStore
    from bgm.person.train import span_index
    store = FeatureStore(cfg)
    kinds, seg_none = [], []
    for p in store.pairs.pair_id:
        k = load(person_path(cfg, p))["person_kind_1hz"]
        kinds.append(k)
        for a, b in store.segments[p]:
            a, b = store.span("video", p, a, b)
            seg_none.append(not (k[span_index(len(k), a, b)] > 0).any())
    k = np.concatenate(kinds)
    c = {"pairs": len(kinds), "seconds": int(len(k)), "face": float((k == FACE).mean()), "body": float((k == BODY).mean()),
         "none": float((k == NONE).mean()), "segments_none": float(np.mean(seg_none)), "config_hash": CONFIG_HASH}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=1))
    log.info("coverage: %s", c)
    return c


if __name__ == "__main__":
    import sys
    from bgm.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_film.json"])
    if sys.argv[1:2] == ["coverage"]:
        coverage(cfg)
    else:
        a = [int(x) for x in sys.argv[1:3]] or [0, 1]
        run(cfg, shard=a[0], n_shards=a[1])
