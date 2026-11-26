"""Sanity tests (synthetic only; no dataset). `.venv/bin/python -m pytest -q stitching/tests`"""
import itertools

import numpy as np
import pytest
import torch

from stitching.base import Cands, Episode, viterbi_ragged, logz_ragged
from stitching.crf import CRF, CRFModel, Tensors
from stitching.hmm_derived import derived_transition
from stitching.dtw_single import monotonic_dtw


def brute(emit, trans):
    best, bp, scores = -np.inf, None, []
    for p in itertools.product(*[range(len(e)) for e in emit]):
        s = sum(emit[t][p[t]] for t in range(len(emit))) + sum(trans[t - 1][p[t - 1], p[t]] for t in range(1, len(emit)))
        scores.append(s)
        if s > best:
            best, bp = s, list(p)
    return bp, best, np.array(scores)


def test_viterbi_equals_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        emit = [rng.normal(size=3) for _ in range(4)]
        trans = [rng.normal(size=(3, 3)) for _ in range(3)]
        p, s = viterbi_ragged(emit, trans)
        bp, bs, _ = brute(emit, trans)
        assert p == bp and abs(s - bs) < 1e-9


def test_forward_logz_equals_bruteforce():
    from scipy.special import logsumexp
    rng = np.random.default_rng(1)
    for _ in range(20):
        emit = [rng.normal(size=3) for _ in range(4)]
        trans = [rng.normal(size=(3, 3)) for _ in range(3)]
        _, _, all_s = brute(emit, trans)
        assert abs(logz_ragged(emit, trans) - logsumexp(all_s)) < 1e-9


def test_crf_torch_forward_equals_bruteforce():
    """The CRF's own (torch) forward algorithm: NLL = logZ - score(gold)."""
    from scipy.special import logsumexp
    ep = synth_episode(np.random.default_rng(2), T=4, K=3)
    x = Tensors(ep, 10.0, 0.3, True, "cpu")
    m = CRFModel(0.07)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        emit, trans = m.scores(x)
        e = [v.numpy().astype(float) for v in emit]
        tr = [v.numpy().astype(float) for v in trans]
        _, _, all_s = brute(e, tr)
        g = ep.gold
        gs = sum(e[t][g[t]] for t in range(4)) + sum(tr[t - 1][g[t - 1], g[t]] for t in range(1, 4))
        assert abs(m.nll(x).item() * 4 - (logsumexp(all_s) - gs)) < 1e-3


def synth_episode(rng, T=4, K=3, files=3, dur=4.0):
    """Candidates over `files` files; scene t's candidates include every previous candidate's continuation."""
    cands, segs = [], np.array([[t * dur, (t + 1) * dur] for t in range(T)])
    for t in range(T):
        mid = rng.integers(files, size=K)
        off = rng.choice([0.0, 2.0, 10.0, 20.0], size=K)
        if t:
            mid[0], off[0] = cands[-1].mid[0], cands[-1].off[0] + dur         # a continuation
        # keep (mid, off) unique within a scene
        for j in range(1, K):
            while any((mid[j] == mid[i] and abs(off[j] - off[i]) < 1e-6) for i in range(j)):
                off[j] += 30.0
        cands.append(Cands(mid, off.astype(float), rng.normal(0.2, 0.05, K), K,
                           emb_in=rng.normal(size=(K, 8)).astype(np.float32), emb_out=rng.normal(size=(K, 8)).astype(np.float32),
                           rms_in=rng.normal(-20, 3, K), rms_out=rng.normal(-20, 3, K), ph_in=rng.uniform(size=K), ph_out=rng.uniform(size=K)))
    n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
    for c in cands:
        c.emb_in, c.emb_out = n(c.emb_in), n(c.emb_out)
    return Episode("synth", cands, segs, n(rng.normal(size=(T, 16))), n(rng.normal(size=(T, 16))), np.full((T, 4), 0.25),
                   [f"f{i}" for i in range(files)], np.array([f"t{i}" for i in range(files)], object), np.full(files, 200.0),
                   np.zeros(files, int), np.full(files, 120.0), gold=np.zeros(T, int))


def test_derived_rows_sum_to_one():
    rng = np.random.default_rng(3)
    n_full = 0
    for _ in range(50):
        ep = synth_episode(rng, T=3, K=6, files=2)
        for t in range(1, ep.T):
            A = derived_transition(ep, t, L=12.0, rho=0.3)
            s = np.exp(A).sum(1)
            assert np.allclose(s, 1.0, atol=1e-6), s
            from stitching.base import relation
            c, j, w = relation(ep, t)
            n_full += int((c.any(1) & j.any(1) & w.any(1)).sum())
    assert n_full > 0      # rows with all three categories present (no renormalisation) were exercised


def test_worked_example():
    eA, eB = [0.80, 0.60, 0.75], [0.70, 0.70, 0.50]
    emit = [np.array([a, b]) for a, b in zip(eA, eB)]
    for sw, path, score in ((-0.2, [0, 0, 0], 2.15), (-0.02, [0, 1, 0], 2.21)):
        trans = [np.array([[0.0, sw], [sw, 0.0]])] * 2
        p, s = viterbi_ragged(emit, trans)
        assert p == path and abs(s - score) < 1e-9, (p, s)


def test_crf_nll_decreases():
    rng = np.random.default_rng(4)
    eps = [synth_episode(rng, T=5, K=4) for _ in range(12)]
    for e in eps:          # gold = the continuation chain at index 0, with a higher similarity
        for c in e.cands:
            c.sim[0] += 0.05
        e.gold_win = [dict(mid=int(c.mid[0]), off=float(c.off[0]), sim=float(c.sim[0]), emb_in=c.emb_in[0], emb_out=c.emb_out[0],
                           rms_in=c.rms_in[0], rms_out=c.rms_out[0], ph_in=c.ph_in[0], ph_out=c.ph_out[0]) for c in e.cands]
    crf = CRF(0.07, 10.0, 0.3, True, device="cpu")
    info = crf.fit(eps[:8], eps[8:])
    h = info["history"]
    assert h[-1]["epoch"] >= 1
    assert min(x["val_nll"] for x in h) < h[0]["val_nll"]
    assert min(x["train_loss"] for x in h[1:]) < h[1]["train_loss"] + 1e-9


def test_monotonic_dtw():
    C = np.array([[0.1, 0.9, 0.9], [0.9, 0.9, 0.1], [0.9, 0.0, 0.9]])
    c, p = monotonic_dtw(C)
    assert abs(c - 1.0 / 3) < 1e-9 and list(p) in ([0, 0, 1], [0, 1, 1])     # two optimal paths, cost 1.0 / T
    for t in range(1, 3):
        assert p[t] >= p[t - 1]


def test_semi_viterbi_equals_bruteforce():
    """Segmental DP = best over every segmentation x file of (run scores - lam per switch + beta * vis)."""
    from stitching.semi_viterbi import SemiViterbi, run_tables
    rng = np.random.default_rng(5)
    for _ in range(10):
        ep = synth_episode(rng, T=4, K=3, files=3)
        grid = np.arange(0, 60.0, 0.5)
        ep.dense = {f: (grid, rng.normal(0.2, 0.05, (4, len(grid))).astype(np.float32)) for f in range(3)}
        st = SemiViterbi(lam=0.7, beta=1.3, m=0.3)
        C, B, S, files = run_tables(ep)
        R = 0.7 * C + 0.3 * B
        vis = np.r_[0.0, 1 - (ep.scene_emb[1:] * ep.scene_emb[:-1]).sum(1)]
        best = -np.inf
        for cuts in itertools.product([0, 1], repeat=3):
            bounds = [0] + [i + 1 for i, c in enumerate(cuts) if c] + [4]
            s = sum(R[:, a, b].max() for a, b in zip(bounds[:-1], bounds[1:]))
            s += sum(-0.7 + 1.3 * vis[a] for a in bounds[1:-1])
            best = max(best, s)
        p = st.decode(ep)
        # score of the decoded path under the same objective
        runs, a = [], 0
        for t in range(1, 5):
            if t == 4 or p.mid[t] != p.mid[t - 1] or abs(p.off[t] - (p.off[t - 1] + ep.scene_dur[t - 1])) > 1e-6:
                runs.append((a, t)); a = t
        got = sum(R[list(files).index(p.mid[a0]), a0, b0] for a0, b0 in runs) + sum(-0.7 + 1.3 * vis[a0] for a0, _ in runs[1:])
        assert abs(got - best) < 1e-6, (got, best)
