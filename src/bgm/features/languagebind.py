"""LanguageBind video and audio encoders, frozen (D117; the LanguageBind score of the evaluation suite).

LanguageBind (Zhu et al., ICLR 2024; github.com/PKU-YuanGroup/LanguageBind, MIT licence) binds video and audio to
one 768-d space through language, so cos(video, audio) is a video-music fit score. The official code targets
transformers 4.3x (its CLIPAttention / PEFT wrappers do not run on our transformers 5), so the two vision towers
are re-implemented here module for module and the official checkpoints load into them with strict key matching:

* LanguageBind_Video_FT: CLIP ViT-L/14 with a temporal attention block before each layer's spatial attention
  (per-layer temporal embedding, own LayerNorm, no temporal MLP); 8 frames, 224 px; the CLS token of every frame
  goes through post_layernorm, then the 8 are averaged, then visual_projection.
* LanguageBind_Audio_FT: CLIP ViT-L/14 on a [3, 112, 1036] log-mel image (16 kHz Kaldi fbank, 112 bins, 10 ms
  hop; the 3 channels are 3 chunks of 1036 frames, or the same tiled chunk 3 times for short audio), position
  embedding already resized to the 8 x 74 patch grid in the checkpoint.

Inference-only differences from the official preprocessing, both for determinism: no random horizontal flip of
the frames, and for audio longer than 1036 frames the three chunks start at the first index of each third
(the official code draws them at random; its own commented-out 'fixed' variant is this one).

Frames come from the silent 25 fps interim video (CLAUDE.md rule 8); audio is asserted at 16 kHz (rule 7).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000
N_FRAMES = 8
IMG = 224
N_MEL, TARGET_LEN = 112, 1036
AUDIO_MEAN, AUDIO_STD = -4.2677393, 4.5689974
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
REPOS = {"video": "LanguageBind/LanguageBind_Video_FT", "audio": "LanguageBind/LanguageBind_Audio_FT"}
CONFIG = {"video": REPOS["video"], "audio": REPOS["audio"], "frames": N_FRAMES, "img": IMG, "sr": SR, "n_mel": N_MEL,
          "target_len": TARGET_LEN, "flip": False, "long_audio_chunks": "first index of each third", "dtype": "fp16"}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- the towers (official key names)
class _Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = (p(x).view(B, N, self.h, D // self.h).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        return self.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))


class _MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))                     # config hidden_act "gelu" (exact)


class _Layer(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int, eps: float, t: int):
        super().__init__()
        self.self_attn, self.mlp = _Attention(d, heads), _MLP(d, hidden)
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
        self.t = t
        if t > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, t, d))
            self.temporal_attn = _Attention(d, heads)
            self.temporal_layer_norm1 = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B * t, N, D] (frames of a clip contiguous)."""
        if self.t > 1:
            BT, N, D = x.shape
            y = x.view(BT // self.t, self.t, N, D).transpose(1, 2).reshape(-1, self.t, D)      # (b n) t d
            y = y + self.temporal_embedding
            res = y
            y = res + self.temporal_attn(self.temporal_layer_norm1(y))
            x = y.view(BT // self.t, N, self.t, D).transpose(1, 2).reshape(BT, N, D)
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class _Embeddings(nn.Module):
    def __init__(self, d: int, patch: int, n_pos: int):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(d))
        self.patch_embedding = nn.Conv2d(3, d, patch, patch, bias=False)
        self.position_embedding = nn.Embedding(n_pos, d)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(px).flatten(2).transpose(1, 2)
        x = torch.cat([self.class_embedding.expand(len(x), 1, -1).to(x.dtype), x], 1)
        return x + self.position_embedding.weight[None]


class _Encoder(nn.Module):
    def __init__(self, layers: list[_Layer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _VisionModel(nn.Module):
    def __init__(self, vc: dict, t: int, grid: tuple[int, int]):
        super().__init__()
        d, eps = vc["hidden_size"], vc["layer_norm_eps"]
        self.embeddings = _Embeddings(d, vc["patch_size"], grid[0] * grid[1] + 1)
        self.pre_layrnorm, self.post_layernorm = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
        self.encoder = _Encoder([_Layer(d, vc["num_attention_heads"], vc["intermediate_size"], eps, t)
                                 for _ in range(vc["num_hidden_layers"])])


class LanguageBindTower(nn.Module):
    """One LanguageBind modality tower + visual_projection. forward: [B * t, 3, H, W] -> [B, 768] unit vectors."""

    def __init__(self, vc: dict, projection_dim: int, t: int, grid: tuple[int, int]):
        super().__init__()
        self.t = t
        self.vision_model = _VisionModel(vc, t, grid)
        self.visual_projection = nn.Linear(vc["hidden_size"], projection_dim, bias=False)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        m = self.vision_model
        x = m.pre_layrnorm(m.embeddings(px))
        for layer in m.encoder.layers:
            x = layer(x)
        cls = m.post_layernorm(x[:, 0]).view(-1, self.t, x.shape[-1]).mean(1)
        return F.normalize(self.visual_projection(cls).float(), dim=-1)


def load_tower(modality: str, device: str = "cuda") -> LanguageBindTower:
    """The frozen official tower (eval, fp16 on GPU; CLAUDE.md rule 2). Text weights are skipped."""
    from huggingface_hub import hf_hub_download
    repo = REPOS[modality]
    cfg = json.loads(open(hf_hub_download(repo, "config.json")).read())
    vc = cfg["vision_config"]
    t = vc["num_frames"] if vc.get("add_time_attn") else 1
    assert (modality == "video") == (t == N_FRAMES), (modality, t)
    grid = (vc["image_size"] // vc["patch_size"],) * 2 if modality == "video" else (N_MEL // vc["patch_size"], TARGET_LEN // vc["patch_size"])
    tower = LanguageBindTower(vc, cfg["projection_dim"], t, grid)
    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if k.startswith(("vision_model.", "visual_projection."))
          and not k.endswith("position_ids")}
    tower.load_state_dict(sd, strict=True)
    tower.eval().requires_grad_(False)
    if device.startswith("cuda"):
        tower = tower.half()
    return tower.to(device)


# ---------------------------------------------------------------- preprocessing (pure)
def frame_indices(a_s: float, b_s: float, fps: float = 25.0, n: int = N_FRAMES, n_total: int | None = None) -> np.ndarray:
    """[n] frame indices spread evenly over [a_s, b_s) (the official np.linspace(0, len - 1, n) on the clip)."""
    f0 = int(round(a_s * fps))
    f1 = max(f0 + 1, int(round(b_s * fps)))
    if n_total is not None:
        f1 = min(f1, n_total)
        f0 = min(f0, f1 - 1)
    return np.linspace(f0, f1 - 1, n).astype(int)


def video_pixels(frames: np.ndarray) -> torch.Tensor:
    """frames: [T, H, W, 3] uint8 RGB -> [T, 3, 224, 224] float32: /255, CLIP normalize, short side to 224
    (bilinear), centre crop 224 (the official order)."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(CLIP_MEAN)[:, None, None]) / torch.tensor(CLIP_STD)[:, None, None]
    h, w = x.shape[-2:]
    nh, nw = (int(math.floor(h / w * IMG)), IMG) if w < h else (IMG, int(math.floor(w / h * IMG)))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = (nh - IMG) // 2, (nw - IMG) // 2
    return x[..., top:top + IMG, left:left + IMG]


def audio_mel(y: np.ndarray, sr: int) -> torch.Tensor:
    """y: mono float waveform at 16 kHz -> [3, 112, 1036] normalized Kaldi fbank image (see module doc)."""
    import torchaudio
    assert sr == SR, f"LanguageBind audio needs {SR} Hz, got {sr}"
    w = torch.from_numpy(np.asarray(y, np.float32))[None]
    w = w - w.mean()
    if w.shape[1] < 400:                                         # one 25 ms frame at least
        w = F.pad(w, (0, 400 - w.shape[1]))
    mel = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=SR, use_energy=False,
                                            window_type="hanning", num_mel_bins=N_MEL, dither=0.0,
                                            frame_length=25, frame_shift=10)               # [T, 112]
    return mel_chunks(mel)


def mel_chunks(mel: torch.Tensor) -> torch.Tensor:
    """[T, 112] -> [3, 112, 1036]: tiled if short, three chunks (first index of each third) if long; normalized."""
    T = mel.shape[0]
    if T > TARGET_LEN:
        ranges = np.array_split(np.arange(T - TARGET_LEN + 1), 3)
        starts = [int(r[0]) if len(r) else 0 for r in ranges]
        x = torch.stack([mel[s:s + TARGET_LEN] for s in starts])
    else:
        m = mel.repeat(TARGET_LEN // T + 1, 1)[:TARGET_LEN] if T < TARGET_LEN else mel
        x = torch.stack([m, m, m])
    return (x.transpose(1, 2) - AUDIO_MEAN) / (AUDIO_STD * 2)


def resample_48k_to_16k(y48: np.ndarray, sr: int) -> np.ndarray:
    import soxr
    assert sr == 48000, f"interim music must be 48 kHz, got {sr}"
    return soxr.resample(np.asarray(y48, np.float32), sr, SR).astype(np.float32)


# ---------------------------------------------------------------- scores (pure)
def tw_score(seg_scores: np.ndarray, durations: np.ndarray) -> float:
    """Time-weighted mean of per-segment scores: Σ d_k s_k / Σ d_k (segments of different length count by length)."""
    d = np.asarray(durations, float)
    return float(np.dot(np.asarray(seg_scores, float), d) / d.sum())


# ---------------------------------------------------------------- batched encoding
@torch.no_grad()
def encode_video_clips(tower: LanguageBindTower, clips: list[np.ndarray], batch: int = 16) -> np.ndarray:
    """clips: list of [8, H, W, 3] uint8 -> [n, 768] float32 unit vectors."""
    dev, dt = next(tower.parameters()).device, next(tower.parameters()).dtype
    out = []
    for i in range(0, len(clips), batch):
        px = torch.cat([video_pixels(c) for c in clips[i:i + batch]]).to(dev, dt)
        out.append(tower(px).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, tower.visual_projection.out_features), np.float32)


@torch.no_grad()
def encode_audio_clips(tower: LanguageBindTower, waves: list[np.ndarray], sr: int, batch: int = 32) -> np.ndarray:
    """waves: list of mono 16 kHz waveforms -> [n, 768] float32 unit vectors."""
    assert sr == SR, f"LanguageBind audio needs {SR} Hz, got {sr}"
    dev, dt = next(tower.parameters()).device, next(tower.parameters()).dtype
    out = []
    for i in range(0, len(waves), batch):
        px = torch.stack([audio_mel(w, sr) for w in waves[i:i + batch]]).to(dev, dt)
        out.append(tower(px).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, tower.visual_projection.out_features), np.float32)
