"""Loss-sweep registry: masking (decision 3), equivalences, and every term/weighting runs and backpropagates."""
import math

import pytest
import torch
import torch.nn.functional as F

from bgm.losses.registry import AUX_DEFAULT, MAIN, WEIGHTING, GradNorm, SweepLoss
from bgm.losses.score_aware_loss import ScoreAwareLoss, same_track_mask
from bgm.losses.tag_aware_loss import TagAwareLoss
from bgm.train.losses import info_nce

B, D_V, D_M = 6, 10, 12
D = B + 2                    # v_i = e_i; component D-1 is orthogonal to every v
SLICES = [list(range(0, 4)), list(range(4, 6)), list(range(6, 9))]


def _tags(g, n=B):
    t = torch.rand(n, 9, generator=g, dtype=torch.float64)
    for s in SLICES:
        t[:, s] /= t[:, s].sum(1, keepdim=True)
    return t


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.eye(B, D, dtype=torch.float64)
    m = torch.randn(B, D, generator=g, dtype=torch.float64) * 0.3
    m[:, -1] = 2.0                                     # room to keep |m_j| fixed while moving one component
    m = F.normalize(m, dim=-1)
    batch = {"v_raw": torch.randn(B, D_V, generator=g, dtype=torch.float64),
             "m_raw": torch.randn(B, D_M, generator=g, dtype=torch.float64),
             "track": torch.tensor([0, 0, 1, 2, 3, 4]),   # items 0 and 1 share a track
             "t_scene": _tags(g), "t_music": _tags(g), "genre": torch.tensor([0, 1, 0, 1, 2, 2]), "epoch": 0}
    return v, m, batch


def _perturb(m, i, j, delta):
    """Change s_ij = v_i . m_j = m_j[i] by delta, keeping |m_j| = 1 through the free last component."""
    m = m.clone()
    old = m[j, i].item()
    m[j, i] = old + delta
    m[j, -1] = torch.sqrt(m[j, -1] ** 2 + old ** 2 - m[j, i] ** 2)
    assert torch.allclose(m[j].norm(), torch.tensor(1.0, dtype=m.dtype))
    return m


def _ctx(log_tau):
    return {"log_tau": log_tau, "d": D, "dv": D_V, "dm": D_M, "group_slices": SLICES, "n_genres": 3}


# multipos is excluded on purpose: it treats same-track pairs as positives (tests/test_loss_bank_tagfree.py).
MASKED = [(main, {}) for main in MAIN if main != "multipos"] + [("soft", {"tag_teacher": 0.0})]


@pytest.mark.parametrize("main,aux", MASKED)
@pytest.mark.parametrize("epoch", [0, 20])
def test_same_track_pair_is_not_a_negative(main, aux, epoch):
    """Moving only s_01 / s_10 (a same-track pair) leaves the loss unchanged; moving s_02 (different tracks) does not."""
    torch.manual_seed(0)
    log_tau = torch.nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float64))
    loss = SweepLoss(main, aux, "fixed", _ctx(log_tau)).double()
    v, m, batch = _setup()
    batch["epoch"] = epoch
    f = lambda mm: loss(v, mm, batch)[0].item()
    base = f(m)
    for i, j in ((0, 1), (1, 0)):
        for delta in (-0.2, 0.2):
            assert f(_perturb(m, i, j, delta)) == pytest.approx(base, abs=1e-10), (main, i, j, delta)
    assert abs(f(_perturb(m, 0, 2, 0.6)) - base) > 1e-6, f"{main}: control perturbation had no effect"


@pytest.mark.parametrize("main,aux", MASKED)
def test_same_track_pair_gets_no_gradient(main, aux):
    """The same statement through autograd: d loss / d s_01 = 0 (s = v m^T with v = I, so s_01 = m[1, 0])."""
    log_tau = torch.nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float64))
    loss = SweepLoss(main, aux, "fixed", _ctx(log_tau)).double()
    v, m, batch = _setup()
    s = (v @ m.T).clone().requires_grad_(True)
    # rebuild unit m whose scores against v are s (rows of s.T fill the first B components); the last component keeps
    # |m| = 1 and is constant w.r.t. s at first order only, so test d/ds via the scores directly:
    m2 = torch.cat([s.T, torch.zeros(B, D - B - 1, dtype=s.dtype), torch.sqrt(1 - (s.T ** 2).sum(1, keepdim=True))], 1)
    loss(v, m2, batch)[0].backward()
    g = s.grad
    # d/ds_01 includes the path through m_1's last component, which only moves s with |m_1| (no v uses it): zero
    assert abs(g[0, 1].item()) < 1e-10 and abs(g[1, 0].item()) < 1e-10, (main, g[0, 1].item(), g[1, 0].item())
    if main != "hgm":                  # hgm: every negative hinge is inactive at these scores (d^2 > 1.5)
        assert abs(g[0, 2].item()) > 1e-8


def test_infonce_equals_trainer_loss():
    torch.manual_seed(1)
    v, m = F.normalize(torch.randn(16, 32), dim=-1), F.normalize(torch.randn(16, 32), dim=-1)
    track = torch.tensor([0, 0] + list(range(1, 15)))
    lt = torch.nn.Parameter(torch.tensor(math.log(0.07)))
    ours = SweepLoss("infonce", {}, "fixed", {"log_tau": lt, "d": 32, "dv": 4, "dm": 4})(v, m, {"track": track, "epoch": 0})[0]
    assert ours.item() == pytest.approx(info_nce(v, m, track, lt, 0.01).item(), abs=1e-6)


def test_tag_teacher_beta0_equals_score_aware():
    """The reported bug fixed: TagAwareLoss(beta=0) targets and soft loss equal ScoreAwareLoss's, with and without masking."""
    torch.manual_seed(2)
    base = torch.randn(64)
    m_raw = base + 0.3 * torch.randn(12, 64) * torch.linspace(0.5, 3, 12)[:, None]
    v, m = F.normalize(torch.randn(12, 16), dim=-1), F.normalize(torch.randn(12, 16), dim=-1)
    t = torch.rand(12, 9)
    sa = ScoreAwareLoss(lam_intra=0, lam_trans=0)
    ta = TagAwareLoss(SLICES, 16, beta=0.0, lam_tag=0.0, lam_intra=0, lam_trans=0)
    for track in (None, torch.tensor([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9])):
        same = same_track_mask(track) if track is not None else None
        q = sa.soft_targets(m_raw, same)
        q = q / q.sum(1, keepdim=True)
        q1, q2 = ta.targets(m_raw, t, t, same)
        assert torch.allclose(q1, q, atol=1e-6) and torch.allclose(q2, q, atol=1e-6)
        assert torch.allclose(q1.sum(1), torch.ones(12)) and torch.allclose(q2.sum(1), torch.ones(12))
        a = sa(v, m, torch.randn(12, 8), m_raw, track=track)[1]["soft"]
        b = ta(v, m, torch.randn(12, 8), m_raw, t, t, track=track)[1]["soft"]
        assert a.item() == pytest.approx(b.item(), abs=1e-6)


def _real_batch(Bn=32, d=16):
    g = torch.Generator().manual_seed(3)
    v_pre, m_pre = torch.randn(Bn, d, generator=g, requires_grad=True), torch.randn(Bn, d, generator=g, requires_grad=True)
    t = torch.rand(Bn, 9, generator=g)
    for s in SLICES:
        t[:, s] /= t[:, s].sum(1, keepdim=True)
    v, m = F.normalize(v_pre, dim=-1), F.normalize(m_pre, dim=-1)
    batch = {"v_raw": torch.randn(Bn, D_V, generator=g), "m_raw": torch.randn(Bn, D_M, generator=g),
             "v_pre": v_pre, "m_pre": m_pre, "track": torch.arange(Bn) // 2 * 0 + torch.arange(Bn),
             "t_scene": t, "t_music": t.flip(0), "genre": torch.arange(Bn) % 3, "epoch": 12,
             "v2": F.normalize(v_pre + 0.01, dim=-1), "m2": F.normalize(m_pre - 0.01, dim=-1),
             "v_all": v, "pairs": torch.tensor([[0, 1], [2, 3]]), "cont": torch.tensor([1.0, 0.0]),
             "v_view1": F.normalize(v_pre + 0.02, dim=-1), "v_view2": F.normalize(v_pre - 0.02, dim=-1),
             "m_view1": F.normalize(m_pre + 0.02, dim=-1), "m_view2": F.normalize(m_pre - 0.02, dim=-1)}
    return v, m, batch, (v_pre, m_pre)


@pytest.mark.parametrize("weighting", WEIGHTING)
@pytest.mark.parametrize("term", [k for k in AUX_DEFAULT if k != "tag_teacher"])
def test_every_aux_runs(term, weighting):
    Bn, d = 32, 16
    W_v, W_m = torch.nn.Parameter(torch.randn(d, 20) * 0.1), torch.nn.Parameter(torch.randn(d, 20) * 0.1)
    lt = torch.nn.Parameter(torch.tensor(math.log(0.07)))
    loss = SweepLoss("infonce", {term: AUX_DEFAULT[term]}, weighting,
                     {"log_tau": lt, "d": d, "dv": D_V, "dm": D_M, "group_slices": SLICES, "n_genres": 3,
                      "last_layers": (W_v, W_m)})
    v, m, batch, (v_pre, m_pre) = _real_batch(Bn, d)
    total, parts, mon = loss(v, m, batch)
    assert torch.isfinite(total) and all(torch.isfinite(x) for x in parts.values()) and torch.isfinite(mon)
    if weighting == "gradnorm":
        GradNorm(loss, [W_v, W_m, v_pre, m_pre]).step()
    total.backward()
    assert v_pre.grad is not None and torch.isfinite(v_pre.grad).all()
    loss.end_epoch(), loss.end_epoch()


def test_barlow_asserts_d_le_b():
    lt = torch.nn.Parameter(torch.tensor(math.log(0.07)))
    loss = SweepLoss("infonce", {"barlow": 0.005}, "fixed", {"log_tau": lt, "d": 64, "dv": D_V, "dm": D_M})
    v, m, batch, _ = _real_batch(32, 64)
    with pytest.raises(AssertionError):
        loss(v, m, batch)


def test_tag_teacher_only_with_soft_and_no_vicreg_barlow():
    lt = torch.nn.Parameter(torch.tensor(0.0))
    ctx = {"log_tau": lt, "d": 8, "dv": 4, "dm": 4, "group_slices": SLICES}
    with pytest.raises(AssertionError):
        SweepLoss("infonce", {"tag_teacher": 0.0}, "fixed", ctx)
    with pytest.raises(AssertionError):
        SweepLoss("infonce", {"vicreg": 0.04, "barlow": 0.005}, "fixed", ctx)


def test_asac_no_eps_fixed_tau_is_infonce():
    """soft with eps = 0 and the scale fixed at 1/0.07 is masked InfoNCE at tau = 0.07 (ASAC ablation sanity)."""
    torch.manual_seed(4)
    v, m = F.normalize(torch.randn(16, 32), dim=-1), F.normalize(torch.randn(16, 32), dim=-1)
    track = torch.tensor([0, 0] + list(range(1, 15)))
    lt = torch.nn.Parameter(torch.tensor(math.log(0.07)))
    ctx = {"log_tau": lt, "d": 32, "dv": 4, "dm": 4}
    loss = SweepLoss("soft", {}, "fixed", ctx, {"eps": 0.0, "learn_tau": False})
    batch = {"track": track, "epoch": 0, "m_raw": torch.randn(16, 8)}
    assert loss(v, m, batch)[0].item() == pytest.approx(info_nce(v, m, track, lt, 0.01).item(), abs=1e-5)
    assert not loss.main.fn.logit_scale.requires_grad and all(p is not loss.main.fn.logit_scale for p in loss.model_params())
