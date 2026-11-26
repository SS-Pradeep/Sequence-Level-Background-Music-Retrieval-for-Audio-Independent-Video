"""DP == brute force on small random lattices (every segmentation x every start candidate per cue)."""
import itertools

import numpy as np
import pytest

from exp_supp.hsmm import LogNormal, Tables, decode_tables
from stitching.base import TOL_S, Cands, Episode


def random_episode(rng, T=4, F=3, J=2):
    d = rng.choice([2.0, 3.0, 4.0], T)
    s0 = np.r_[0, np.cumsum(d)[:-1]]
    segs = np.stack([s0, s0 + d], 1)
    flen = np.full(F, 12.0)
    cands = []
    for t in range(T):
        mid = list(rng.integers(0, F, J))
        off = list(rng.choice(np.arange(0, 6, 0.5), J))
        if t:                                                    # continuations of the previous scene's candidates
            for mm, oo in zip(cands[-1].mid, cands[-1].off):
                o2 = oo + d[t - 1]
                if o2 + d[t] <= flen[mm] and not any(m2 == mm and abs(x - o2) <= TOL_S for m2, x in zip(mid, off)):
                    mid.append(mm), off.append(o2)
        cands.append(Cands(np.array(mid), np.array(off, float), rng.normal(0.3, 0.1, len(mid)), J))
    return Episode(key="x", cands=cands, segs=segs, scene_emb=np.zeros((T, 2)), scene_sig=np.zeros((T, 2)),
                   scene_mood=np.zeros((T, 2)), files=[f"f{i}" for i in range(F)], file_track=np.array(list("abc")[:F], object),
                   file_len=flen, file_key=rng.integers(-1, 12, F), file_tempo=rng.uniform(80, 160, F))


def brute(ep, tb, alpha, lam_jump, mu, gamma, m, dur):
    T = ep.T
    best = -np.inf
    for cut in itertools.product([0, 1], repeat=T - 1):
        bounds = [0] + [i + 1 for i, c in enumerate(cut) if c] + [T]
        cues = list(zip(bounds[:-1], bounds[1:]))
        for js in itertools.product(*(range(tb.Ks[a]) for a, _ in cues)):
            sc, prev = 0.0, None
            for (a, b), j in zip(cues, js):
                if not np.isfinite(tb.Ec[a, j, b]):
                    sc = -np.inf
                    break
                v = (1 - m) * tb.Ec[a, j, b] + m * tb.Et[a, j, b]
                v += gamma * dur.logpdf(tb.D[a, b]) + alpha * tb.inner[a, b]
                f = tb.fid[a, j]
                if prev is not None:
                    v += alpha * tb.logp[a] - (lam_jump if f == prev else mu * tb.delta[prev, f])
                sc += v
                prev = f
            best = max(best, sc)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    ep = random_episode(rng, T=int(rng.integers(2, 5)))
    dur = LogNormal(rng.uniform(3, 15, 50))
    tb = Tables(ep, tau=0.07, L=8.0)
    for params in [dict(alpha=1.0, lam_jump=1.0, mu=1.0, gamma=1.0, m=0.0),
                   dict(alpha=0.25, lam_jump=8.0, mu=0.0, gamma=0.0, m=0.5),
                   dict(alpha=2.0, lam_jump=0.0, mu=2.0, gamma=4.0, m=0.5)]:
        path, score = decode_tables(tb, ep, dur=dur, return_score=True, **params)
        assert score == pytest.approx(brute(ep, tb, dur=dur, **params), abs=1e-8)
        # the decoded windows are valid candidates and continue within a cue
        for t in range(ep.T):
            c = ep.cands[t]
            assert np.any((c.mid == path.mid[t]) & (np.abs(c.off - path.off[t]) <= TOL_S))
