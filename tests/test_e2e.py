"""bgm.e2e: CRF algebra, equivalence with the shipped decoder at initialisation, boundary features."""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import torch

from bgm.e2e.bounds import (N_MUSIC_MAIN, bar_phase, grid_distance, music_frame_features, rank01,
                            video_boundary_features)
from bgm.e2e.crf import crf_nll, keytempo, log_partition, path_energy, structure, transition, viterbi_beam
from bgm.e2e.model import E2EModel
from bgm.models.heads import SemanticHead
from bgm.retrieval.stitch_eval import Layer, decode_lattice, transition_matrix

PEN = {"lam_switch": 1.0, "lam_jump": 0.5, "mu": 0.3}


def toy_lattice(seed=0, n_layers=4, n_top=3):
    """Layers over 3 files with contiguous continuations of every previous candidate (build_lattice's structure)."""
    rng = np.random.default_rng(seed)
    layers, dur = [], 2.0
    for i in range(n_layers):
        mid = list(rng.integers(0, 3, n_top))
        off = list(np.round(rng.uniform(0, 20, n_top) * 2) / 2)
        cont_of = None
        if i:
            P = layers[-1]
            cont_of = np.full(len(P.mid), -1)
            for j, (m, o) in enumerate(zip(P.mid, P.off)):
                mid.append(int(m)), off.append(float(o + dur))
                cont_of[j] = len(mid) - 1
        n = len(mid)
        layers.append(Layer(np.array(mid), np.array(off, float), np.full(n, dur), rng.integers(-1, 12, n),
                            rng.choice([0.0, 90.0, 120.0], n), rng.normal(0, 1, n), n_top, cont_of))
    return layers


def fixed_T(P, C):
    same, cont = structure(P.mid, P.off, P.dur, C.mid, C.off)
    kt = keytempo(P.key, P.tempo, C.key, C.tempo)
    z = torch.zeros(len(P.mid), dtype=torch.float64)
    zc = torch.zeros(len(C.mid), dtype=torch.float64)
    return transition(torch.from_numpy(same), torch.from_numpy(cont), torch.from_numpy(kt), torch.tensor(PEN["lam_switch"], dtype=torch.float64),
                      torch.tensor(PEN["lam_jump"], dtype=torch.float64), z, zc, torch.tensor(PEN["mu"], dtype=torch.float64))


def test_transition_equals_shipped_matrix():
    L = toy_lattice()
    for P, C in zip(L, L[1:]):
        np.testing.assert_allclose(fixed_T(P, C).numpy(), transition_matrix(P, C, PEN), atol=1e-12)


def test_log_partition_brute_force_and_nll():
    rng = np.random.default_rng(1)
    sizes = [3, 2, 4]
    u = [torch.tensor(rng.normal(size=n), requires_grad=True) for n in sizes]
    T = [torch.tensor(rng.normal(size=(a, b))) for a, b in zip(sizes, sizes[1:])]
    brute = torch.logsumexp(torch.stack([-path_energy(u, T, list(p)) for p in itertools.product(*map(range, sizes))]), 0)
    assert torch.allclose(log_partition(u, T), brute)
    nll = crf_nll(u, T, [0, 1, 2])
    assert nll.item() >= 0
    nll.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in u)


def test_viterbi_beam_equals_decode_lattice():
    for seed in range(5):
        L = toy_lattice(seed, n_layers=6)
        Ts = [fixed_T(P, C).numpy() for P, C in zip(L, L[1:])]
        for beam in (None, 4):
            ours = viterbi_beam([x.cost for x in L], [x.n_top for x in L], [x.cont_of for x in L],
                                lambda i, a, c: Ts[i - 1][np.ix_(a, c)], beam)
            assert ours == decode_lattice(L, PEN, beam)


def _fake_sem(dv=6, dm=5, seed=0):
    torch.manual_seed(seed)
    return SimpleNamespace(hv=SemanticHead(dv, 8, 4, 0.0), hm=SemanticHead(dm, 8, 4, 0.0))


def test_model_starts_at_the_shipped_decoder():
    vstats = (np.zeros(9), np.ones(9))
    m = E2EModel([_fake_sem(seed=s) for s in range(3)], PEN, music=True, edge=False, dur=True, train_heads=True,
                 vstats=vstats, device="cpu")
    base, h = m.boundary(torch.randn(5, 9))
    assert torch.allclose(base, torch.zeros(5)) and torch.allclose(h, torch.zeros(5, 4))
    wo, wi = m.seam_weights(h)
    assert torch.allclose(wo, torch.zeros_like(wo)) and torch.allclose(wi, torch.zeros_like(wi))
    assert abs(float(m.mu.detach()) - PEN["mu"]) < 1e-5 and float(m.log_a.detach().exp()) == 1.0
    assert m.n_music == N_MUSIC_MAIN
    edge = E2EModel([_fake_sem()], PEN, True, True, False, False, vstats, "cpu")
    assert edge.n_music == N_MUSIC_MAIN + 2 and len(edge.v_mu) == 7
    assert not any(p.requires_grad for p in edge.hv.parameters())


def test_music_boundary_features():
    beats = np.arange(1, 41) * 0.48                          # 125 BPM, every beat on the 25 Hz grid (12 frames)
    t = np.array([0.48, 0.72, 0.96])
    np.testing.assert_allclose(grid_distance(t, beats, 0.48), [0.0, 1.0, 0.0], atol=1e-9)
    onset = np.zeros(500)
    onset[np.round(beats[2::4] * 25).astype(int)] = 5.0       # strong beats at residue 2
    assert bar_phase(beats, onset) == 2
    r = rank01(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(r, [1.0, 0.0, 0.5])
    dyn = np.random.default_rng(0).normal(size=(500, 5))
    dyn[:, 0] = onset
    F = music_frame_features(beats, dyn, 20.0)
    assert F.shape == (500, 9) and F.min() >= 0 and F.max() <= 1
    np.testing.assert_allclose(F[0, 8], 0.0) and np.testing.assert_allclose(F[-1, 7], 0.004, atol=1e-6)
    assert F[int(round(beats[2] * 25)), 1] == 0.0             # a downbeat (residue 2)
    assert grid_distance(t, np.zeros(0), np.nan).tolist() == [0.5, 0.5, 0.5]


def test_video_boundary_features():
    sem = np.ones((10, 4))
    sem[5:] = -1.0
    dyn = np.zeros((250, 5))
    dyn[125:] = 1.0
    F = video_boundary_features(sem, dyn, np.array([5.0]), [(0.0, 5.0), (5.0, 10.0)])
    assert F.shape == (1, 9)
    np.testing.assert_allclose(F[0, 0], 2.0, atol=1e-6)        # opposite SigLIP vectors
    np.testing.assert_allclose(F[0, 1:6], 1.0)
    assert F[0, 6] == 1.0 and np.isclose(F[0, 7], np.log(5.0))


def test_marginal_nll():
    from bgm.e2e.crf import crf_marginal_nll
    rng = np.random.default_rng(2)
    sizes = [3, 2, 4]
    u = [torch.tensor(rng.normal(size=n)) for n in sizes]
    T = [torch.tensor(rng.normal(size=(a, b))) for a, b in zip(sizes, sizes[1:])]
    one = [torch.tensor([i == g for i in range(n)]) for n, g in zip(sizes, [0, 1, 2])]
    assert torch.allclose(crf_marginal_nll(u, T, one), crf_nll(u, T, [0, 1, 2]))
    masks = [torch.tensor([True, True, False]), torch.tensor([True, False]), torch.tensor([False, True, True, False])]
    allowed = [p for p in itertools.product(*map(range, sizes)) if all(m[i] for m, i in zip(masks, p))]
    pY = torch.stack([torch.exp(-path_energy(u, T, list(p))) for p in allowed]).sum() / torch.exp(log_partition(u, T))
    assert torch.allclose(crf_marginal_nll(u, T, masks), -torch.log(pY))
    everything = [torch.ones(n, dtype=torch.bool) for n in sizes]
    assert abs(float(crf_marginal_nll(u, T, everything))) < 1e-9
