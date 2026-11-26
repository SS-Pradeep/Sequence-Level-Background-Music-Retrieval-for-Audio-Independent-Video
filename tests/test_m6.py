"""M6: batch sampler (one segment per video), InfoNCE same-track masking, dynamics-loss gradient fix."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from bgm.models.heads import DynamicsEncoder, SemanticHead
from bgm.train.losses import dynamics_loss, info_nce
from bgm.train.semantic import epoch_batches


def test_each_video_once_per_epoch_and_never_twice_in_a_batch():
    ids = [f"p{i}" for i in range(300)]
    order = np.random.default_rng(0).permutation(300)
    batches = epoch_batches(ids, order, 128)
    flat = [p for b in batches for p in b]
    assert sorted(flat) == sorted(ids) and all(len(set(b)) == len(b) for b in batches) and [len(b) for b in batches] == [128, 128, 44]


def test_info_nce_masks_same_track_negatives():
    torch.manual_seed(0)
    zv = F.normalize(torch.randn(4, 8), dim=-1)
    zm = F.normalize(torch.randn(4, 8), dim=-1)
    log_tau = torch.tensor(math.log(0.07))
    tr = torch.tensor([0, 0, 1, 2])                       # items 0 and 1 share a track
    base = info_nce(zv, zm, tr, log_tau)
    zm2 = zm.clone()
    zm2[1] = zv[0]                                         # make item 1's music a perfect match for video 0 ...
    zv2 = zv.clone()
    zv2[1] = zm[0]
    # ... the masked (same-track) cross terms must not change the loss terms for items 2 and 3
    logits = zv @ zm.T / 0.07
    same = tr[:, None] == tr[None, :]
    masked = logits.masked_fill(same & ~torch.eye(4, dtype=torch.bool), float("-inf"))
    want = 0.5 * (F.cross_entropy(masked, torch.arange(4)) + F.cross_entropy(masked.T, torch.arange(4)))
    assert torch.allclose(base, want)
    assert torch.isfinite(info_nce(zv2, zm2, tr, log_tau))


def test_dynamics_loss_gradient_is_finite_with_masking():
    torch.manual_seed(0)
    B, T, D, K = 4, 20, 8, 3
    hv = F.normalize(torch.randn(B, T, D), dim=-1)
    hm = F.normalize(torch.randn(B, T, D), dim=-1)
    hs = F.normalize(torch.randn(B, K, T, D), dim=-1)
    valid = torch.tensor([[1, 1, 0]] * B, dtype=torch.bool)
    log_tau = torch.nn.Parameter(torch.tensor(math.log(0.1)))
    loss = dynamics_loss(hv, hm, hs, valid, torch.tensor([0, 0, 1, 2]), log_tau)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(log_tau.grad)
    # the spec's ordering (mask with -inf, then divide by tau) gives a NaN temperature gradient
    lt = torch.nn.Parameter(torch.tensor(math.log(0.1)))
    s = torch.einsum("btd,ctd->bc", hv, hm) / T
    s = s.masked_fill(torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool), float("-inf"))
    F.cross_entropy(s / lt.exp(), torch.arange(B)).backward()
    assert not torch.isfinite(lt.grad)


def test_module_shapes_and_sizes():
    h = SemanticHead(1152)
    assert h(torch.randn(3, 1152)).shape == (3, 256)
    enc = DynamicsEncoder()
    out = enc(torch.randn(2, 5, 200))
    assert out.shape == (2, 200, 64) and torch.allclose(out.norm(dim=-1), torch.ones(2, 200), atol=1e-5)
    n = sum(p.numel() for p in enc.parameters())
    assert 80_000 < n < 95_000, n                          # SPEC: about 87K


def test_mert_layer_mix_is_uniform_average_at_init():
    """1a (D53): at init the mixer reproduces the uniform layer average (sem_mert_1hz construction) up to the
    per-second renormalization, and the CLAP part passes through untouched."""
    import numpy as np
    import torch

    from bgm.train.semantic import MertLayerMix
    rng = np.random.default_rng(0)
    lay = rng.standard_normal((4, 25, 1024)).astype(np.float32)
    lay /= np.linalg.norm(lay, axis=2, keepdims=True)
    clap = rng.standard_normal((4, 512)).astype(np.float32)
    x = np.concatenate([clap, lay.reshape(4, -1)], 1)
    m = MertLayerMix(512)
    out = m(torch.from_numpy(x)).detach().numpy()
    assert out.shape == (4, 512 + 1024)
    assert np.allclose(out[:, :512], clap)
    want = lay.mean(1)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert np.allclose(out[:, 512:], want, atol=1e-5)
    assert np.allclose(np.linalg.norm(out[:, 512:], axis=1), 1.0, atol=1e-5)


def test_maxsim_matches_loop_with_padding():
    import numpy as np
    import torch

    from bgm.train.semantic_tokens import maxsim
    rng = np.random.default_rng(0)
    B, C, V, M, D = 3, 4, 5, 6, 8
    zv = torch.tensor(rng.standard_normal((B, V, D)), dtype=torch.float32)
    zm = torch.tensor(rng.standard_normal((C, M, D)), dtype=torch.float32)
    vlen, mlen = [5, 3, 1], [6, 2, 4, 1]
    vpad = torch.tensor([[t >= n for t in range(V)] for n in vlen])
    mpad = torch.tensor([[t >= n for t in range(M)] for n in mlen])
    s = maxsim(zv, vpad, zm, mpad).numpy()
    for b in range(B):
        for c in range(C):
            want = np.mean([max(float(zv[b, v] @ zm[c, m]) for m in range(mlen[c])) for v in range(vlen[b])])
            assert abs(s[b, c] - want) < 1e-5


def test_token_model_fits_parameter_budget():
    import torch
    from omegaconf import OmegaConf

    from bgm.train.semantic_tokens import TokenModel
    cfg = OmegaConf.load("configs/default.yaml")
    m = TokenModel(1152, 1536, cfg, "cpu")
    assert m.n_params <= 500_000


def test_linear_head_and_mlp_init_unchanged():
    """D134: head_type=linear is LayerNorm + one Linear; the mlp head's parameters (and their init) are untouched."""
    lin = SemanticHead(16, 8, 4, n_hidden=0)
    assert [k for k in lin.state_dict()] == ["net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias"]
    y = lin(torch.randn(5, 16))
    assert y.shape == (5, 4) and torch.allclose(y.norm(dim=-1), torch.ones(5), atol=1e-5)
    torch.manual_seed(0)
    a = SemanticHead(16, 8, 4).state_dict()
    torch.manual_seed(0)
    b = SemanticHead(16, 8, 4, n_hidden=1).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_attn_pool_starts_at_the_mean_and_ignores_padding():
    """D134: at init the attention pool equals the re-normalized token mean; padded tokens get zero weight."""
    from bgm.models.heads import AttnPool, pad_tokens
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((n, 12)).astype(np.float32) for n in (3, 7, 1)]
    X, mask = pad_tokens(seqs, "cpu")
    assert X.shape == (3, 7, 12) and mask.sum().item() == 11
    out = AttnPool(12)(X, mask).detach().numpy()
    want = np.stack([s.mean(0) / np.linalg.norm(s.mean(0)) for s in seqs])
    assert np.allclose(out, want, atol=1e-5)
    pool = AttnPool(12)
    torch.nn.init.normal_(pool.score[-1].weight)
    X2 = X.clone()
    X2[0, 3:] = 100.0                                      # garbage in the padding of sequence 0
    assert torch.allclose(pool(X, mask), pool(X2, mask), atol=1e-6)
    w = pool.weights(X, mask)
    assert torch.allclose(w.sum(1), torch.ones(3)) and (w[~mask] == 0).all()


def test_fixed_tau_is_not_trained():
    from omegaconf import OmegaConf

    from bgm.train.semantic import SemanticModel
    base = {"hidden": 8, "out_dim": 4, "dropout": 0.1, "tau_init": 0.07}
    m = SemanticModel(16, 12, OmegaConf.create({"train": {"semantic": base | {"tau_learnable": False}}}), "cpu")
    assert all(p is not m.log_tau for p in m.params()) and not m.log_tau.requires_grad
    m2 = SemanticModel(16, 12, OmegaConf.create({"train": {"semantic": base | {"video_pool": "attn", "head_type": "linear"}}}), "cpu")
    assert any(p is m2.log_tau for p in m2.params()) and m2.pool is not None
    toks = [np.random.default_rng(i).standard_normal((n, 16)).astype(np.float32) for i, n in enumerate((2, 5))]
    z = m2.video_forward(toks)
    assert z.shape == (2, 4)
    m2.eval()
    assert np.allclose(m2.embed("video", toks[1][None])[0], z[1].detach().numpy(), atol=1e-5)


def test_input_dropout_is_train_only_and_keeps_checkpoint_keys():
    """D135: input dropout changes training outputs only; the head's parameters and state-dict keys are unchanged."""
    torch.manual_seed(0)
    a = SemanticHead(16, 8, 4)
    torch.manual_seed(0)
    b = SemanticHead(16, 8, 4, p_in=0.5)
    assert list(a.state_dict()) == list(b.state_dict()) and all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    x = torch.randn(6, 16)
    a.eval(), b.eval()
    assert torch.allclose(a(x), b(x))
    b.train()
    torch.manual_seed(1)
    assert not torch.allclose(b(x), a(x))


def test_ema_update_moves_toward_the_live_weights():
    from bgm.train.semantic import clone_state, ema_update
    live = {"hv": {"w": torch.ones(3)}, "log_tau": torch.tensor(1.0)}
    ema = clone_state({"hv": {"w": torch.zeros(3)}, "log_tau": torch.tensor(0.0)})
    ema_update(ema, live, 0.9)
    assert torch.allclose(ema["hv"]["w"], torch.full((3,), 0.1)) and torch.isclose(ema["log_tau"], torch.tensor(0.1))
    ema_update(ema, live, 0.9)
    assert torch.allclose(ema["hv"]["w"], torch.full((3,), 0.19))
    assert torch.equal(live["hv"]["w"], torch.ones(3))


def test_input_noise_is_train_only_and_keeps_checkpoint_keys():
    """D138: Gaussian input noise (after the input LayerNorm) changes training outputs only; same parameters and keys."""
    torch.manual_seed(0)
    a = SemanticHead(16, 8, 4)
    torch.manual_seed(0)
    b = SemanticHead(16, 8, 4, noise=0.05)
    assert list(a.state_dict()) == list(b.state_dict()) and all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    x = torch.randn(6, 16)
    a.eval(), b.eval()
    assert torch.allclose(a(x), b(x))
    a.train(), b.train()
    for m in (a, b):
        m.net[3].p = 0.0                                   # isolate the noise from the head's own dropout
    assert torch.allclose(a(x), a(x)) and not torch.allclose(b(x), a(x))
