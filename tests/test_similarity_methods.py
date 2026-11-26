"""Similarity functions (D145): max-sim, hubness corrections, rank helpers."""
import numpy as np
import torch

from bgm.eval.metrics import rank_of_first_hit
from bgm.sweep.similarity import (HUB, apply_hub, csls, frames, hub_variants, inv_softmax, maxsim, qb_norm,
                                  ranks_segment, ranks_track)


def _unit(*shape, seed=0):
    x = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_frames_rule():
    x = np.arange(10)[:, None]
    assert frames(x, 0.0, 3.0)[:, 0].tolist() == [0, 1, 2]
    assert frames(x, 2.6, 2.7)[:, 0].tolist() == [2]          # empty span -> middle row


def test_maxsim_single_frame_equals_cosine_and_masks_padding():
    q, g = _unit(3, 1, 8), _unit(4, 1, 8, seed=1)
    m = lambda n: torch.ones(n, 1, dtype=torch.bool)
    got = maxsim(torch.from_numpy(q), m(3), torch.from_numpy(g), m(4))
    assert np.allclose(got, q[:, 0] @ g[:, 0].T, atol=1e-6)
    gp = np.concatenate([g, np.ones((4, 2, 8), np.float32) * 9], 1)       # padding must be ignored
    gm = torch.zeros(4, 3, dtype=torch.bool); gm[:, 0] = True
    assert np.allclose(maxsim(torch.from_numpy(q), m(3), torch.from_numpy(gp), gm), got, atol=1e-6)


def test_maxsim_symmetric_definition():
    q, g = _unit(1, 3, 8), _unit(1, 5, 8, seed=2)
    S = q[0] @ g[0].T
    want = 0.5 * (S.max(1).mean() + S.max(0).mean())
    got = maxsim(torch.from_numpy(q), torch.ones(1, 3, dtype=torch.bool), torch.from_numpy(g), torch.ones(1, 5, dtype=torch.bool))
    assert np.isclose(got[0, 0], want, atol=1e-6)


def test_hubness_demotes_a_hub():
    rng = np.random.default_rng(0)
    B = rng.normal(0, 0.1, (200, 6)); B[:, 0] += 0.5          # item 0 is a hub for the bank
    S = rng.normal(0, 0.1, (50, 6)); S[:, 0] += 0.5
    before = (S.argmax(1) == 0).mean()
    assert before > 0.9
    for m, p in (("csls", 10), ("is", 8.0), ("qb", 8.0)):   # CSLS discounts by mean closeness only: weaker here
        assert (apply_hub(S, B, m, p).argmax(1) == 0).mean() < before, m
    assert (apply_hub(S, B, "is", 8.0).argmax(1) == 0).mean() < 0.5


def test_hub_variants_match_apply_hub():
    rng = np.random.default_rng(1)
    S, B = rng.normal(size=(20, 7)), rng.normal(size=(30, 7))
    hubs = [(m, p) for m, ps in HUB.items() for p in ps]
    for m, p, got in hub_variants(S, B, hubs):
        assert np.allclose(got, apply_hub(S, B, m, p)), (m, p)


def test_rank_helpers_match_rank_of_first_hit():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(12, 12))
    valid = rng.random((12, 12)) > 0.2
    np.fill_diagonal(valid, True)
    r = ranks_segment(S, valid)
    for i in range(12):
        h = np.zeros(12, bool); h[i] = True
        assert r[i] == rank_of_first_hit(S[i][valid[i]], h[valid[i]])
    hits = rng.random((12, 9)) > 0.7
    hits[np.arange(12), rng.integers(9, size=12)] = True
    rt = ranks_track(S[:, :9], hits)
    assert all(rt[i] == rank_of_first_hit(S[i, :9], hits[i]) for i in range(12))
