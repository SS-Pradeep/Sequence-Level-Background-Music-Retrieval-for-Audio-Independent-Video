"""D124 losses: limits that tie them to info_nce or to a loop, and same-track masking."""
import math

import torch
import torch.nn.functional as F

from bgm.train.losses import dcl, hn_nce, info_nce, sigmoid_loss, vse_max


def _batch(B=8, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    zv = F.normalize(torch.randn(B, D, generator=g), dim=1)
    zm = F.normalize(torch.randn(B, D, generator=g), dim=1)
    return zv, zm


def test_hn_nce_beta0_equals_info_nce():
    zv, zm = _batch()
    t = torch.arange(8)
    t[3] = t[5]
    lt = torch.tensor(math.log(0.07))
    assert torch.allclose(hn_nce(zv, zm, t, lt, beta=0.0), info_nce(zv, zm, t, lt), atol=1e-5)


def test_dcl_loop():
    zv, zm = _batch()
    t = torch.arange(8)
    t[1] = t[2]
    lt = torch.tensor(math.log(0.1))
    s = zv @ zm.T / 0.1
    rows = []
    for S in (s, s.T):
        for i in range(8):
            negs = [S[i, j] for j in range(8) if j != i and t[j] != t[i]]
            rows.append(torch.logsumexp(torch.stack(negs), 0) - S[i, i])
    ref = 0.5 * (torch.stack(rows[:8]).mean() + torch.stack(rows[8:]).mean())
    assert torch.allclose(dcl(zv, zm, t, lt), ref, atol=1e-5)


def test_sigmoid_loop_and_mask():
    zv, zm = _batch()
    t = torch.arange(8)
    t[0] = t[7]
    lt, b = torch.tensor(math.log(0.1)), torch.tensor(-2.0)
    s = zv @ zm.T / 0.1 - 2.0
    tot = 0.0
    for i in range(8):
        for j in range(8):
            if i != j and t[i] == t[j]:
                continue
            tot += -F.logsigmoid(s[i, j] if i == j else -s[i, j])
    assert torch.allclose(sigmoid_loss(zv, zm, t, lt, b), tot / 8, atol=1e-5)


def test_vse_max_ignores_same_track():
    zv, zm = _batch()
    t = torch.arange(8)
    base = vse_max(zv, zm, t)
    zm2 = zm.clone()
    zm2[1] = zv[0]                      # a perfect confounder for row 0 ...
    t2 = t.clone()
    t2[1] = t2[0]                       # ... that shares row 0's track, so it is masked
    l_masked = vse_max(zv, zm2, t2)
    l_open = vse_max(zv, zm2, t)
    assert l_open > l_masked and torch.isfinite(base)
