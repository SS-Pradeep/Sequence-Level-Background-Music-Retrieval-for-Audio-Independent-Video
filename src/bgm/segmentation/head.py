"""The new boundary head on frozen BaSSL boundary embeddings (D140)."""
from __future__ import annotations

import torch
from torch import nn


class BoundaryHead(nn.Module):
    """[B, d] -> music-change logit [B]: LayerNorm -> Linear(d, 256) -> GELU -> Dropout(0.4) -> Linear(256, 1)."""

    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SequenceHead(nn.Module):
    """D142: per-boundary projection, a BiGRU over the boundaries of each sequence, then a per-boundary classifier on
    [projection, GRU state]. Input: a list of [B_i, d] tensors (one per sequence); output: logits [sum B_i]. The GRU sees
    the boundaries' order only (no times or durations)."""

    def __init__(self, d: int, hidden: int = 256, rnn: int = 128, dropout: float = 0.4):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, rnn, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden + 2 * rnn), nn.Dropout(dropout), nn.Linear(hidden + 2 * rnn, 1))

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
        lens = torch.tensor([len(x) for x in xs])
        keep = [i for i, n in enumerate(lens.tolist()) if n > 0]
        if not keep:
            return torch.zeros(0, device=self.out[-1].weight.device)
        h = self.proj(pad_sequence([xs[i] for i in keep], batch_first=True))
        packed = pack_padded_sequence(h, lens[keep], batch_first=True, enforce_sorted=False)
        g, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=h.shape[1])
        z = self.out(torch.cat([h, g], -1)).squeeze(-1)
        return torch.cat([z[j, :lens[i]] for j, i in enumerate(keep)])
