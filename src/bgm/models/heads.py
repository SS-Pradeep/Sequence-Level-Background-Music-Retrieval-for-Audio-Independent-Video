"""Trainable modules (SPEC §5.5). Pretrained encoders stay frozen; only these learn."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticHead(nn.Module):
    """[B, d_in] -> [B, d_out], unit norm. n_hidden = 1 is the SPEC §5.5 head; each extra hidden layer adds
    Linear(hidden, hidden) + GELU + Dropout before the output layer (depth ablation, PROGRESS.md D63).
    n_hidden = 0 is the linear head (head_type=linear, D134): LayerNorm + Linear(d_in, d_out).
    p_in > 0 drops input features in training (input_dropout, D135); no parameters, so checkpoints are unchanged.
    noise > 0 adds N(0, noise^2) in training after the input LayerNorm, where every feature has unit scale (input_noise,
    D138; the raw inputs are unit-norm, ~0.03 per dim, so noise on them would swamp the signal); no parameters."""
    def __init__(self, d_in: int, hidden: int = 512, d_out: int = 256, p: float = 0.2, n_hidden: int = 1, p_in: float = 0.0,
                 noise: float = 0.0):
        super().__init__()
        self.drop_in = nn.Dropout(p_in) if p_in > 0 else None
        self.noise = float(noise)
        if n_hidden == 0:
            self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out))
            return
        extra = [m for _ in range(n_hidden - 1) for m in (nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p))]
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(p), *extra, nn.Linear(hidden, d_out))

    def forward(self, x):
        if self.drop_in is not None:
            x = self.drop_in(x)
        if self.noise > 0 and self.training:
            x = self.net[0](x)
            return F.normalize(self.net[1:](x + self.noise * torch.randn_like(x)), dim=-1)
        return F.normalize(self.net(x), dim=-1)


class AttnPool(nn.Module):
    """video_pool=attn (D134): [B, T, d] 1 Hz tokens + [B, T] bool mask -> [B, d] unit norm, a softmax-weighted
    token mean. The scorer's last layer starts at zero, so training starts from the plain mean (the baseline input)."""
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def weights(self, x, mask):                          # [B, T], zero on padding
        return torch.softmax(self.score(x).squeeze(-1).masked_fill(~mask, float("-inf")), dim=1)

    def forward(self, x, mask):
        return F.normalize((self.weights(x, mask).unsqueeze(-1) * x).sum(1), dim=-1)


def pad_tokens(seqs, device) -> tuple[torch.Tensor, torch.Tensor]:
    """List of [n_i, d] arrays -> ([B, max n, d] zero-padded tokens, [B, max n] bool mask of real tokens)."""
    n = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), n, seqs[0].shape[1], device=device)
    mask = torch.zeros(len(seqs), n, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.from_numpy(s).to(device)
        mask[i, :len(s)] = True
    return X, mask


class StepNorm(nn.Module):
    """LayerNorm over channels at each time step: [B, ch, T] -> [B, ch, T]. Time-local, unlike the spec's
    GroupNorm(1, ch), whose statistics span the whole sequence and made context-cropped windows differ from
    whole-file encodings (PROGRESS.md D19). Same parameter count (2 x ch)."""
    def __init__(self, ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, dilation=dilation, padding=dilation * (k - 1) // 2)
        self.norm = StepNorm(ch)

    def forward(self, x):                                # [B, ch, T] -> [B, ch, T]
        return x + F.gelu(self.norm(self.conv(x)))


class DynamicsEncoder(nn.Module):
    """[B, c_in, T] -> [B, T, ch], unit norm per step. Receptive field 61 steps (2.44 s at 25 Hz),
    about 87K parameters. One instance for video curves, a separate one for music curves."""
    def __init__(self, c_in: int = 5, ch: int = 64, k: int = 5, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.blocks = nn.Sequential(*[ResBlock(ch, k, d) for d in dilations])
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        h = self.out(self.blocks(self.inp(x)))
        return F.normalize(h.transpose(1, 2), dim=-1)


class AttributeMap(nn.Module):
    """Optional: [B, M] calibrated log-probs -> [B, M] logits, initialized to the identity."""
    def __init__(self, m: int):
        super().__init__()
        self.lin = nn.Linear(m, m)
        nn.init.eye_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(x)
