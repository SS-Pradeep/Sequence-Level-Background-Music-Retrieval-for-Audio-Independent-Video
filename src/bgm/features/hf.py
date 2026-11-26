"""Real frozen encoders behind the interfaces in `encoders.py`. All are frozen: eval(), torch.no_grad(),
fp16 on GPU (CLAUDE.md rule 2), and each asserts its native sample rate (rule 7).

Shapes: encode_frames [N, H, W, 3] uint8 -> [N, D]; encode_audio(windows, sr) with windows a list of 1-D float
arrays -> [N, D]; encode_text(list[str]) -> [N, D]. Outputs are float32 and L2-normalized.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def _pooled(out) -> torch.Tensor:
    return out if torch.is_tensor(out) else out.pooler_output


def _unit(x: torch.Tensor) -> np.ndarray:
    return F.normalize(x.float(), dim=-1).cpu().numpy()


class SiglipEncoder:
    """google/siglip-so400m-patch14-384. Stored attribute logits use `logit_scale.exp() * cos + logit_bias`."""

    def __init__(self, model_id: str, device: str = "cuda", batch: int = 64):
        from transformers import AutoModel, AutoProcessor
        self.device, self.batch = device, batch
        self.model = AutoModel.from_pretrained(model_id, dtype=_dtype(device)).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.dim = self.model.config.vision_config.hidden_size
        self.logit_scale = float(self.model.logit_scale.exp())
        self.logit_bias = float(self.model.logit_bias)

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(frames), self.batch):
            px = self.proc(images=list(frames[i:i + self.batch]), return_tensors="pt")["pixel_values"]
            out.append(_pooled(self.model.get_image_features(pixel_values=px.to(self.device, _dtype(self.device)))))
        return _unit(torch.cat(out))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        t = self.proc(text=texts, padding="max_length", return_tensors="pt").to(self.device)   # SigLIP was trained this way
        return _unit(_pooled(self.model.get_text_features(**t)))


class ClipEncoder:
    """openai/clip-vit-large-patch14 image embeddings (the SigLIP-vs-CLIP check)."""

    def __init__(self, model_id: str, device: str = "cuda", batch: int = 64):
        from transformers import CLIPModel, CLIPProcessor
        self.device, self.batch = device, batch
        self.model = CLIPModel.from_pretrained(model_id, dtype=_dtype(device)).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(frames), self.batch):
            px = self.proc(images=list(frames[i:i + self.batch]), return_tensors="pt")["pixel_values"]
            out.append(_pooled(self.model.get_image_features(pixel_values=px.to(self.device, _dtype(self.device)))))
        return _unit(torch.cat(out))


class ClapEncoder:
    """CLAP audio (48 kHz) and text. Stored attribute logits use `logit_scale_a.exp() * cos`."""

    sr = 48000

    def __init__(self, model_id: str, device: str = "cuda", batch: int = 32):
        from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel
        self.device, self.batch = device, batch
        self.model = ClapModel.from_pretrained(model_id, dtype=_dtype(device)).to(device).eval()
        self.fe = ClapFeatureExtractor.from_pretrained(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        assert self.fe.sampling_rate == self.sr
        self.dim = self.model.config.projection_dim
        self.logit_scale = float(self.model.logit_scale_a.exp())

    @torch.no_grad()
    def encode_audio(self, windows: list[np.ndarray], sr: int) -> np.ndarray:
        assert sr == self.sr, f"CLAP expects {self.sr} Hz audio, got {sr} Hz"
        out = []
        for i in range(0, len(windows), self.batch):
            x = self.fe([np.asarray(w, np.float32) for w in windows[i:i + self.batch]], sampling_rate=sr, return_tensors="pt")
            out.append(_pooled(self.model.get_audio_features(input_features=x["input_features"].to(self.device, _dtype(self.device)),
                                                             is_longer=x["is_longer"].to(self.device))))
        return _unit(torch.cat(out))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        t = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
        return _unit(_pooled(self.model.get_text_features(**t)))


class MertEncoder:
    """m-a-p/MERT-v1-330M at 24 kHz. `encode_track` returns [T1, 1024]: per-second mean of the 75 Hz frames, each
    of the 25 hidden states L2-normalized, averaged over layers, L2-normalized again (SPEC §5.1).

    Under transformers 5 the remote code ignores `output_hidden_states`, so forward hooks collect the same 25 states
    (each layer's input, plus the final layer-normed output; checked equal to last_hidden_state)."""

    sr = 24000
    frame_rate = 75.0

    def __init__(self, model_id: str, device: str = "cuda", chunk_s: float = 30.0, context_s: float = 5.0):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device, self.chunk_s, self.context_s = device, chunk_s, context_s
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=_dtype(device)).to(device).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        assert self.fe.sampling_rate == self.sr
        self._caps: list[torch.Tensor] = []
        pre = lambda mod, args, kwargs: self._caps.append(args[0] if args else kwargs["hidden_states"])  # noqa: E731
        for layer in self.model.encoder.layers:
            layer.register_forward_pre_hook(pre, with_kwargs=True)
        self.model.encoder.layer_norm.register_forward_hook(lambda mod, i, o: self._caps.append(o))
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def _layers(self, y: np.ndarray) -> torch.Tensor:
        """[25, frames, 1024] for one waveform chunk."""
        self._caps = []
        x = self.fe(y, sampling_rate=self.sr, return_tensors="pt")["input_values"].to(self.device, _dtype(self.device))
        self.model(input_values=x)
        assert len(self._caps) == 25, f"expected 25 hidden states, got {len(self._caps)}"
        return torch.stack([c[0] for c in self._caps]).float()

    @torch.no_grad()
    def encode_track(self, y: np.ndarray, sr: int, keep_layers: bool = False):
        assert sr == self.sr, f"MERT expects {self.sr} Hz audio, got {sr} Hz"
        T1 = int(np.floor(len(y) / sr + 1e-9))
        sums = torch.zeros(25, max(T1, 1), self.dim, device=self.device)
        counts = torch.zeros(max(T1, 1), device=self.device)
        chunk, ctx = int(self.chunk_s * sr), int(self.context_s * sr)
        for c0 in range(0, len(y), chunk):
            a, b = max(0, c0 - ctx), min(len(y), c0 + chunk + ctx)
            if b - a < sr // 10:
                continue
            H = self._layers(y[a:b])
            t = a / sr + (torch.arange(H.shape[1], device=self.device) + 0.5) / self.frame_rate
            keep = (t >= c0 / sr) & (t < min(c0 + chunk, len(y)) / sr)
            sec = t.floor().long()
            keep &= sec < T1
            sums.index_add_(1, sec[keep], H[:, keep])
            counts.index_add_(0, sec[keep], torch.ones(int(keep.sum()), device=self.device))
        per_layer = F.normalize(sums / counts.clamp_min(1)[None, :, None], dim=-1)[:, :T1]
        pooled = F.normalize(per_layer.mean(0), dim=-1).cpu().numpy()
        return (pooled, per_layer.cpu().numpy()) if keep_layers else pooled


class ImageBindEncoder:
    """ImageBind huge: vision frames and 2 s audio windows at 16 kHz (the zero-shot baseline only)."""

    sr = 16000

    def __init__(self, ckpt: str, device: str = "cuda", batch: int = 32):
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.M = ModalityType
        self.device, self.batch, self.dim = device, batch, 1024
        model = imagebind_model.imagebind_huge(pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = model.to(device).eval()
        if device.startswith("cuda"):
            self.model = self.model.half()
        from torchvision import transforms
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(frames), self.batch):
            x = torch.stack([self.tf(f) for f in frames[i:i + self.batch]]).to(self.device, _dtype(self.device))
            out.append(self.model({self.M.VISION: x})[self.M.VISION])
        return _unit(torch.cat(out))

    @torch.no_grad()
    def encode_audio(self, windows: list[np.ndarray], sr: int) -> np.ndarray:
        from imagebind.data import waveform2melspec
        assert sr == self.sr, f"ImageBind expects {self.sr} Hz audio, got {sr} Hz"
        out = []
        for i in range(0, len(windows), self.batch):
            mels = [(waveform2melspec(torch.from_numpy(np.asarray(w, np.float32))[None], sr, 128, 204) + 4.268) / 9.138
                    for w in windows[i:i + self.batch]]
            x = torch.stack(mels)[:, None].to(self.device, _dtype(self.device))   # [B, clips=1, 1, 128, 204]
            out.append(self.model({self.M.AUDIO: x})[self.M.AUDIO])
        return _unit(torch.cat(out))
