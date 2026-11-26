"""Cue-switch predictor (Step 2).

token_i = MLP(x_i); [CTX] + 2-layer bidirectional Transformer encoder over the segment sequence (pos: 'none' = no
positional encoding, the pair head supplies the order; 'index' = sinusoidal encoding of the segment index added to the
tokens, defined for any length);
boundary logit_i = MLP([h_i, h_{i+1}, |h_i - h_{i+1}|, h_i * h_{i+1}]). context=False is the pair-only variant: the
same head on the MLP tokens, no [CTX] and no Transformer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CueModel(nn.Module):
    def __init__(self, d_in: int, d: int = 256, heads: int = 4, layers: int = 2, ff: int = 512, dropout: float = 0.4,
                 context: bool = True, pos: str = "none"):
        super().__init__()
        assert pos in ("none", "index"), pos
        self.context, self.pos, self.d = context, pos, d
        self.tok = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        if context:
            self.ctx = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [B, N, d_in], valid [B, N] bool -> boundary logits [B, N - 1] (meaningful where both sides are valid)."""
        h = self.tok(x)
        if self.context:
            B = len(h)
            if self.pos == "index":
                h = h + sinusoid(h.shape[1], self.d, h.device)[None]
            h = torch.cat([self.ctx.expand(B, -1, -1), h], 1)
            pad = ~torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), valid], 1)
            h = self.norm(self.enc(h, src_key_padding_mask=pad))[:, 1:]
        a, b = h[:, :-1], h[:, 1:]
        return self.head(torch.cat([a, b, (a - b).abs(), a * b], -1)).squeeze(-1)


def sinusoid(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


def collate(xs: list[np.ndarray], ys: list[np.ndarray] | None = None, device: str = "cpu"):
    """Pad a batch of sequences: x [B, N, D], valid [B, N], y [B, N - 1], bvalid [B, N - 1]."""
    B, N, D = len(xs), max(len(x) for x in xs), xs[0].shape[1]
    x = np.zeros((B, N, D), np.float32)
    valid = np.zeros((B, N), bool)
    y = np.zeros((B, max(N - 1, 1)), np.float32)
    for i, t in enumerate(xs):
        x[i, :len(t)] = t
        valid[i, :len(t)] = True
        if ys is not None and len(t) > 1:
            y[i, :len(t) - 1] = ys[i]
    bvalid = valid[:, :-1] & valid[:, 1:] if N > 1 else np.zeros((B, 1), bool)
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(x), t(valid), t(y), t(bvalid)


@torch.no_grad()
def predict(model: CueModel, xs: list[np.ndarray], device: str = "cpu", batch: int = 64) -> list[np.ndarray]:
    """Boundary logits [N_s - 1] per sequence (eval mode, no dropout)."""
    model.eval()
    out = []
    for k in range(0, len(xs), batch):
        chunk = xs[k:k + batch]
        x, valid, _, _ = collate(chunk, device=device)
        z = model(x, valid).float().cpu().numpy() if x.shape[1] > 1 else np.zeros((len(chunk), 0))
        out += [z[i, :len(s) - 1].astype(np.float64) for i, s in enumerate(chunk)]
    return out
