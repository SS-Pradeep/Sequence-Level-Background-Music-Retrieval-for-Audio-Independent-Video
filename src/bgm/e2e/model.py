"""The end-to-end soundtrack model (D141): the run's 10 semantic heads, an emission scale and the transition parameters.

Emission (unary cost) of candidate c of segment i, as the shipped z-scored score with a learned scale a:
    u_i(c) = -a * (mean_m cos(hv_m(x_i), hm_m(x_c)) - mu_i) / sigma_i       (mu_i, sigma_i: the frozen z-statistics)
Transitions (crf.py): base_i = -logit_i of a small MLP on the video boundary features (the jointly trained change
predictor); A_i / B_i = the music exit / entry features times (w + U^T h_i), with h_i the same MLP's boundary embedding
(cross-modal agreement). Initialised so the first decode IS the shipped one: a = 1, logit = 0, h = 0, w = U = V = 0,
b_sw = lam_switch, b_jp = lam_jump, mu = mu of stitch.json.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from bgm.e2e.bounds import N_MUSIC_MAIN, VIDEO_NAMES

H = 4


class E2EModel(nn.Module):
    def __init__(self, sem_models: list, pen: dict, music: bool, edge: bool, dur: bool, train_heads: bool,
                 vstats: tuple[np.ndarray, np.ndarray], device: str):
        super().__init__()
        self.hv = nn.ModuleList([m.hv for m in sem_models])
        self.hm = nn.ModuleList([m.hm for m in sem_models])
        for p in list(self.hv.parameters()) + list(self.hm.parameters()):
            p.requires_grad_(train_heads)
        self.train_heads, self.music, self.edge, self.dur = train_heads, music, edge, dur
        self.n_music = N_MUSIC_MAIN + (2 if edge else 0)
        nv = len(VIDEO_NAMES) - (0 if dur else 2)
        self.register_buffer("v_mu", torch.tensor(vstats[0][:nv], dtype=torch.float32))
        self.register_buffer("v_sd", torch.tensor(vstats[1][:nv], dtype=torch.float32))
        self.log_a = nn.Parameter(torch.zeros(()))
        self.vmlp = nn.Sequential(nn.Linear(nv, 16), nn.GELU(), nn.Linear(16, 1 + H))
        nn.init.zeros_(self.vmlp[-1].weight), nn.init.zeros_(self.vmlp[-1].bias)
        self.b_sw = nn.Parameter(torch.tensor(float(pen["lam_switch"])))
        self.b_jp = nn.Parameter(torch.tensor(float(pen["lam_jump"])))
        mu = max(float(pen["mu"]), 1e-4)
        self.mu_raw = nn.Parameter(torch.tensor(math.log(math.expm1(mu))))       # softplus^-1
        self.w_out = nn.Parameter(torch.zeros(self.n_music))
        self.w_in = nn.Parameter(torch.zeros(self.n_music))
        self.U = nn.Parameter(torch.zeros(H, self.n_music))
        self.V = nn.Parameter(torch.zeros(H, self.n_music))
        self.device = device
        self.to(device)

    @property
    def mu(self) -> torch.Tensor:
        return nn.functional.softplus(self.mu_raw)

    def head_mode(self, train: bool) -> None:
        for h in list(self.hv) + list(self.hm):
            h.train(train and self.train_heads)

    def cosines(self, xv: torch.Tensor, seg_of: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        """Ensemble mean cosine of each candidate with its segment: xv [N, dv], seg_of [n] segment index, xm [n, dm]."""
        out = 0.0
        for hv, hm in zip(self.hv, self.hm):
            out = out + (hv(xv)[seg_of] * hm(xm)).sum(-1)
        return out / len(self.hv)

    def boundary(self, vfeat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[N - 1, 9] raw video boundary features -> (base costs [N - 1] = -logit, embeddings [N - 1, H])."""
        x = vfeat[:, : len(self.v_mu)]
        o = self.vmlp((x - self.v_mu) / self.v_sd)
        return -o[:, 0], o[:, 1:]

    def seam_weights(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per boundary exit / entry weights [N - 1, F]; zero when the music term is off."""
        if not self.music:
            z = torch.zeros(len(h), self.n_music, device=h.device)
            return z, z
        return self.w_out[None] + h @ self.U, self.w_in[None] + h @ self.V

    def describe(self) -> dict:
        return {"a": float(self.log_a.exp()), "b_sw": float(self.b_sw), "b_jp": float(self.b_jp), "mu": float(self.mu),
                "w_out": self.w_out.detach().cpu().tolist(), "w_in": self.w_in.detach().cpu().tolist()}
