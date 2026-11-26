"""Tag-free loss bank (2026-11-20): MultiPos, SupConMusic, VVNCE."""
import math

import torch
import torch.nn.functional as F

from bgm.losses.registry import TAG_BASED, InfoNCE, MultiPos, SupConMusic, SweepLoss, VVNCE


def _batch(B=8, d=16, seed=0, tracks=None):
    g = torch.Generator().manual_seed(seed)
    n = lambda: F.normalize(torch.randn(B, d, generator=g), dim=-1)
    tr = torch.tensor(tracks if tracks is not None else list(range(B)))
    return n(), n(), {"track": tr, "m_view1": n(), "m_view2": n(), "v_view1": n(), "v_view2": n()}


def test_multipos_reduces_to_two_positive_nce_when_tracks_unique():
    v, m, b = _batch()
    lt = torch.tensor(math.log(0.07))
    got = MultiPos(lt)(v, m, b)
    g = torch.cat([m, b["m_view1"]])
    L = v @ g.T / 0.07
    idx = torch.arange(8)
    want_v = -(torch.logsumexp(torch.stack([L[idx, idx], L[idx, idx + 8]], 1), 1) - torch.logsumexp(L, 1)).mean()
    want_m = -(torch.cat([L[idx, idx], L[idx, idx + 8]]) - torch.logsumexp(L.T, 1)).mean()   # rows [m; m'], one positive each
    assert torch.allclose(got, 0.5 * (want_v + want_m), atol=1e-5)


def test_multipos_shared_track_counts_as_positive():
    v, m, b = _batch(tracks=[0, 0, 1, 2, 3, 4, 5, 6])
    lt = torch.tensor(math.log(0.07))
    m2 = m.clone(); m2[1] = v[0]                 # make video 0 match video 1's music perfectly
    b2 = dict(b); b2["m_view1"] = b["m_view1"].clone()
    assert MultiPos(lt)(v, m2, b2) < MultiPos(lt)(v, m, b) + 1.0
    assert torch.isfinite(MultiPos(lt)(v, m, b))


def test_supcon_music_perfect_views_beat_random():
    v, m, b = _batch()
    good = dict(b); good["m_view2"] = b["m_view1"]
    assert SupConMusic()(v, m, good) < SupConMusic()(v, m, b)


def test_vvnce_equals_infonce_on_views():
    v, m, b = _batch()
    lt = torch.tensor(math.log(0.07))
    assert torch.allclose(VVNCE(lt)(v, m, b), InfoNCE(lt)(b["v_view1"], b["v_view2"], b))


def test_new_terms_are_tag_free_and_build():
    assert not ({"supcon_music", "vv_nce"} & TAG_BASED)
    lt = torch.nn.Parameter(torch.tensor(math.log(0.07)))
    v, m, b = _batch()
    b |= {"v_raw": torch.randn(8, 5), "m_raw": torch.randn(8, 6)}
    loss = SweepLoss("multipos", {"supcon_music": 0.5, "vv_nce": 0.5, "intra": 0.5}, "uncert",
                     {"log_tau": lt, "d": 16, "dv": 5, "dm": 6})
    tot, parts, mon = loss(v, m, b)
    assert torch.isfinite(tot) and set(parts) == {"main", "supcon_music", "vv_nce", "intra"}
