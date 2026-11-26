"""Frozen BaSSL (Kakao Brain, ACCV 2022) over shot sequences (third_party/bassl, wrapped, not edited).

Checkpoint: the MovieNet-SSeg fine-tuned BaSSL released on ModelScope (damo/cv_resnet50-bert_video-scene-segmentation_
movienet, pytorch_model.pt): ResNet-50 shot encoder, 2-layer 768-d transformer CRN over 2 x 8 neighbour shots, and the
2-way scene-boundary head `head_sbd`. The GitHub repo itself only releases self-supervised checkpoints without a head.

Inference follows BaSSL's fine-tuning code (finetune_wrapper.py, dataset/sampler.py SBDSampler, eval transforms):
  * 3 keyframes per shot, each resized (short side 256), centre-cropped to 224, ImageNet-normalised; the ResNet
    features [2048] of the 3 keyframes are averaged into one shot feature;
  * the boundary after shot s sees shots s - 8 .. s + 8, indices clamped to the sequence (edge padding repeats the
    first / last shot); a [CLS] token = mean of the 17 shot features is prepended; the CRN output at the centre shot
    is the boundary embedding [768] and softmax(head_sbd)[1] BaSSL's own scene-boundary probability;
  * edge = True when the window was clamped (s < 8 or s > S - 9).
Shot features depend only on the shot, so they are cached per clip; the CRN runs on the chained sequence, so the
boundaries at clip joins see real context on both sides. Frozen: eval(), no_grad, fp16 on GPU.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party" / "bassl" / "bassl"
CKPT = ROOT / "third_party" / "ckpt" / "bassl_ms" / "pytorch_model.pt"
CONF = ROOT / "third_party" / "ckpt" / "bassl_ms" / "configuration.json"
N_KEYFRAMES = 3
NEIGHBOR = 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VERSION = 1


def config_hash(shots_file: Path) -> str:
    blob = json.dumps({"v": VERSION, "ckpt_bytes": CKPT.stat().st_size, "k": N_KEYFRAMES,
                       "shots": hashlib.sha1(Path(shots_file).read_bytes()).hexdigest()}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class BaSSL:
    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from model.crn.trn import TransformerCRN
        from model.shot_encoder.resnet import resnet50
        sys.path.remove(str(REPO))
        from transformers import BertConfig
        c = json.loads(CONF.read_text())["model"]
        p = dict(c["contextual_relation_network"]["params"]["trn"])
        p["neighbor_size"] = 2 * NEIGHBOR
        bc = BertConfig(**p)
        bc.input_dim, bc.neighbor_size, bc.pooling_method = 2048, 2 * NEIGHBOR, "center"
        self.net = torch.nn.ModuleDict({"shot_encoder": resnet50(pretrained=False, zero_init_residual=True, in_channel_dim=3),
                                        "crn": TransformerCRN(bc), "head_sbd": torch.nn.Linear(768, 2)})
        res = self.net.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
        assert not res.missing_keys and not res.unexpected_keys, res
        self.device = device
        self.half = device.startswith("cuda")
        self.net = self.net.eval().to(device)
        if self.half:
            self.net = self.net.half()

    @torch.no_grad()
    def encode_keyframes(self, frames: np.ndarray, batch: int = 96) -> np.ndarray:
        """frames [N, H, W, 3] uint8 RGB -> ResNet features [N, 2048] float32."""
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255
            h, w = x.shape[-2:]
            s = 256 / min(h, w)
            x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", antialias=True, align_corners=False)
            h, w = x.shape[-2:]
            t, l = (h - 224) // 2, (w - 224) // 2
            x = (x[..., t:t + 224, l:l + 224] - MEAN.to(x)) / STD.to(x)
            out.append(self.net["shot_encoder"](x.half() if self.half else x).float().cpu().numpy())
        return np.concatenate(out)

    def crn(self, w: torch.Tensor) -> torch.Tensor:
        """TransformerCRN.forward with centre pooling and no attention mask. Same output as BaSSL's all-ones mask
        (checked: max diff 0); BaSSL's own forward rebuilds that mask in fp32 on every call (its shape test compares
        the 4-D buffer's dim 1), which breaks fp16."""
        crn = self.net["crn"]
        h = crn.encoder(crn.shot_embedding(w), attention_mask=None).last_hidden_state
        return h[:, h.shape[1] // 2]

    @torch.no_grad()
    def context(self, shot_feats: np.ndarray, batch: int = 512) -> dict:
        """shot_feats [S, 2048] of one (chained) sequence -> boundary embeddings emb [S - 1, 768] f32, BaSSL scene
        probability p [S - 1] and edge [S - 1] bool, for the boundaries after shots 0 .. S - 2."""
        S = len(shot_feats)
        if S < 2:
            return {"emb": np.zeros((0, 768), np.float32), "p": np.zeros(0, np.float32), "edge": np.zeros(0, bool)}
        idx, edge = window_index(S)
        x = torch.from_numpy(np.asarray(shot_feats, np.float32)).to(self.device)
        x = x.half() if self.half else x
        embs, ps = [], []
        for i in range(0, len(idx), batch):
            pooled = self.crn(x[torch.from_numpy(idx[i:i + batch]).to(self.device)])
            embs.append(pooled.float())
            ps.append(torch.softmax(self.net["head_sbd"](pooled).float(), -1)[:, 1])
        return {"emb": torch.cat(embs).cpu().numpy(), "p": torch.cat(ps).cpu().numpy(), "edge": edge}


    @torch.no_grad()
    def context_many(self, seqs: list[np.ndarray], batch: int = 4096) -> list[dict]:
        """`context` for many sequences in one batched pass (same numbers)."""
        idx, edges, off, sizes = [], [], 0, []
        for f in seqs:
            S = len(f)
            if S >= 2:
                i, e = window_index(S)
                idx.append(i + off)
                edges.append(e)
            else:
                edges.append(np.zeros(0, bool))
            sizes.append(max(S - 1, 0))
            off += S
        x = torch.from_numpy(np.concatenate([np.asarray(f, np.float32) for f in seqs])).to(self.device)
        x = x.half() if self.half else x
        idx = np.concatenate(idx) if idx else np.zeros((0, 2 * NEIGHBOR + 1), int)
        embs, ps = [], []
        for i in range(0, len(idx), batch):
            pooled = self.crn(x[torch.from_numpy(idx[i:i + batch]).to(self.device)])
            embs.append(pooled.float())
            ps.append(torch.softmax(self.net["head_sbd"](pooled).float(), -1)[:, 1])
        emb = torch.cat(embs).cpu().numpy() if embs else np.zeros((0, 768), np.float32)
        p = torch.cat(ps).cpu().numpy() if ps else np.zeros(0, np.float32)
        out, k = [], 0
        for n, e in zip(sizes, edges):
            out.append({"emb": emb[k:k + n], "p": p[k:k + n], "edge": e})
            k += n
        return out


def window_index(S: int, n: int = NEIGHBOR) -> tuple[np.ndarray, np.ndarray]:
    """Shot indices [S - 1, 2n + 1] of each boundary's window (centre shot s = boundary after shot s), clamped to
    [0, S - 1] as BaSSL's SBDSampler does, and edge [S - 1] = the window was clamped."""
    s = np.arange(S - 1)
    idx = s[:, None] + np.arange(-n, n + 1)[None]
    edge = (s < n) | (s + n > S - 1)
    return np.clip(idx, 0, S - 1), edge


def keyframe_index(shot: list, k: int = N_KEYFRAMES) -> list[int]:
    """k frame indices spread evenly inside a shot's own frames [start_frame, end_frame]."""
    a, b = int(shot[0]), int(shot[1])
    return [int(round(a + (b - a) * (j + 0.5) / k)) for j in range(k)]


def read_frames(video_path: str | Path, idx: list[int]) -> np.ndarray:
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    idx = [min(i, len(vr) - 1) for i in idx]
    return vr.get_batch(idx).asnumpy()


def shot_features(model: BaSSL, video_path, shots: list) -> np.ndarray:
    """[S, 2048] float16: mean ResNet feature of each shot's keyframes."""
    idx = [i for s in shots for i in keyframe_index(s)]
    f = model.encode_keyframes(read_frames(video_path, idx))
    return f.reshape(len(shots), N_KEYFRAMES, -1).mean(1).astype(np.float16)


def run(paths: dict[str, Path], shots: dict[str, list], device: str = "cuda", workers: int = 6) -> dict[str, np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor
    model = BaSSL(device)
    ids = list(paths)
    out = {}

    def load(p):
        return read_frames(paths[p], [i for s in shots[p] for i in keyframe_index(s)])
    with ThreadPoolExecutor(workers) as ex:
        for n, (p, fr) in enumerate(zip(ids, ex.map(load, ids))):
            out[p] = model.encode_keyframes(fr).reshape(len(shots[p]), N_KEYFRAMES, -1).mean(1).astype(np.float16)
            if n % 200 == 0:
                log.info("bassl shot features: %d / %d", n, len(ids))
    return out
