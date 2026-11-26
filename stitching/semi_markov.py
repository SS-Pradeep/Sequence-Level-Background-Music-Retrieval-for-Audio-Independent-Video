"""
Semi-Markov (segmental) Viterbi for soundtrack assembly.

Why: in the ablation, two-stage (group scenes, then pick one track per group using pooled
evidence) beat every per-scene decoder, doubling track accuracy (0.083 vs 0.040).
Per-scene emissions are too weak; pooling over several scenes helps.
This decoder keeps the HMM formulation but makes its unit a SEGMENT (a run of scenes
sharing one music cue) instead of a single scene. It chooses the segmentation AND the
track per segment jointly and exactly, where two-stage fixes the segmentation first.

    score = sum over segments [ emission(segment, track)            (pooled over its scenes)
                                + lam_dur * log p(duration of segment)  (cue-length prior, L)
                                + transition(prev track -> track) ]     (switch / reprise)

Exact DP, O(T * max_len * A^2). A = number of distinct tracks in the video's shortlists.
"""
import math

import numpy as np


# ----------------------------------------------------------------- duration prior
class LogNormalDuration:
    """Cue-length prior in seconds. mean_s = L measured on training BGM (9.44 s)."""

    def __init__(self, mean_s=9.44, sigma=0.6):
        self.sigma = sigma
        self.mu = math.log(mean_s) - sigma**2 / 2

    def __call__(self, d):
        d = max(d, 1e-3)
        return -math.log(d * self.sigma * math.sqrt(2 * math.pi)) - (math.log(d) - self.mu) ** 2 / (
            2 * self.sigma**2
        )


# --------------------------------------------------------------- segment emissions
def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class SumMaxEmission:
    """emission(s, e, a) = scale * sum_{t=s..e} max_{w in track a} cos(v_t, w)."""

    def __init__(self, scene_emb, track_windows, scale):
        V = _norm(scene_emb)  # [T, D]
        self.S = np.stack([(V @ _norm(W).T).max(1) for W in track_windows], 1) * scale  # [T, A]
        self.cum = np.vstack([np.zeros((1, self.S.shape[1])), np.cumsum(self.S, 0)])

    def __call__(self, s, e):
        return self.cum[e + 1] - self.cum[s]  # [A]


class PooledEmission:
    """Two-stage style: emission(s, e, a) = scale * n * max_{w in track a} cos(mean(v_s..v_e), w).
    Multiplying by n (number of scenes) keeps segments of different length comparable
    with the per-scene HMM (a sum over scenes)."""

    def __init__(self, scene_emb, track_windows, scale):
        self.V = _norm(scene_emb)
        self.cumV = np.vstack([np.zeros((1, self.V.shape[1])), np.cumsum(self.V, 0)])
        self.Ws = [_norm(W) for W in track_windows]
        self.scale = scale
        self._cache = {}

    def __call__(self, s, e):
        key = (s, e)
        if key not in self._cache:
            q = _norm(self.cumV[e + 1] - self.cumV[s])
            n = e - s + 1
            self._cache[key] = np.array([float((W @ q).max()) for W in self.Ws]) * self.scale * n
        return self._cache[key]


# ---------------------------------------------------------------------- decoder
def semi_markov_decode(
    emission,          # callable (s, e) -> np.ndarray [A]
    scene_dur,         # [T] seconds
    n_tracks,          # A
    max_len=12,        # max scenes per segment
    dur_prior=None,    # callable seconds -> log-prob
    lam_dur=1.0,
    switch_logp=0.0,   # score added when the track changes between segments
    reprise_logp=-math.inf,  # score when the next segment uses the same track (a 'jump'); -inf forbids
):
    T, A = len(scene_dur), n_tracks
    dur_prior = dur_prior or LogNormalDuration()
    cum = np.concatenate([[0.0], np.cumsum(scene_dur)])
    M = np.full((A, A), switch_logp, dtype=float)
    np.fill_diagonal(M, reprise_logp)

    D = np.full((T, A), -np.inf)
    back_s = np.zeros((T, A), dtype=int)
    back_a = np.full((T, A), -1, dtype=int)
    for e in range(T):
        for s in range(max(0, e - max_len + 1), e + 1):
            seg = emission(s, e) + lam_dur * dur_prior(cum[e + 1] - cum[s])  # [A]
            if s == 0:
                cand, prev = seg, np.full(A, -1)
            else:
                trans = D[s - 1][:, None] + M          # [A_prev, A]
                prev = trans.argmax(0)
                cand = seg + trans.max(0)
            better = cand > D[e]
            D[e][better] = cand[better]
            back_s[e][better] = s
            back_a[e][better] = prev[better]

    # backtrack
    a = int(D[T - 1].argmax())
    best = float(D[T - 1, a])
    e, segments = T - 1, []
    while e >= 0:
        s, pa = int(back_s[e, a]), int(back_a[e, a])
        segments.append((s, e, a))
        e, a = s - 1, pa
    segments.reverse()
    return segments, best


def layout_segment(scene_emb, scene_dur, track_windows_emb, hop_s):
    """Inside one segment, lay the track's windows contiguously: choose the start window p
    that maximises sum_t cos(v_t, w_{p + offset_t}), offset_t from elapsed time / hop."""
    V, W = _norm(scene_emb), _norm(track_windows_emb)
    offs = np.floor(np.concatenate([[0.0], np.cumsum(scene_dur)[:-1]]) / hop_s).astype(int)
    best_p, best = 0, -np.inf
    for p in range(len(W)):
        idx = np.minimum(p + offs, len(W) - 1)
        sc = float((V * W[idx]).sum())
        if sc > best:
            best_p, best = p, sc
    return np.minimum(best_p + offs, len(W) - 1)  # window index per scene


# -------------------------------------------------------------------------- tests
def _brute_force(emission, scene_dur, A, max_len, dur_prior, lam_dur, switch_logp, reprise_logp):
    import itertools

    T = len(scene_dur)
    cum = np.concatenate([[0.0], np.cumsum(scene_dur)])
    best = -np.inf
    for cuts in itertools.product([0, 1], repeat=T - 1):
        bounds, s = [], 0
        for i, c in enumerate(cuts):
            if c:
                bounds.append((s, i))
                s = i + 1
        bounds.append((s, T - 1))
        if any(e - s + 1 > max_len for s, e in bounds):
            continue
        for tracks in itertools.product(range(A), repeat=len(bounds)):
            sc = 0.0
            for i, ((s, e), a) in enumerate(zip(bounds, tracks)):
                sc += emission(s, e)[a] + lam_dur * dur_prior(cum[e + 1] - cum[s])
                if i > 0:
                    sc += reprise_logp if tracks[i - 1] == a else switch_logp
            best = max(best, sc)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1) exactness vs brute force, several random cases
    for trial in range(20):
        T, A, D = rng.integers(3, 7), rng.integers(2, 4), 8
        scene_emb = rng.normal(size=(T, D))
        tracks = [rng.normal(size=(rng.integers(2, 6), D)) for _ in range(A)]
        dur = rng.uniform(2, 15, size=T)
        em = (SumMaxEmission if trial % 2 else PooledEmission)(scene_emb, tracks, scale=5.0)
        kw = dict(max_len=int(rng.integers(2, T + 1)), dur_prior=LogNormalDuration(),
                  lam_dur=float(rng.uniform(0, 2)), switch_logp=float(rng.uniform(-3, 0)),
                  reprise_logp=float(rng.choice([-np.inf, -4.0])))
        segs, sc = semi_markov_decode(em, dur, A, **kw)
        bf = _brute_force(em, dur, A, **kw)
        assert abs(sc - bf) < 1e-6, (trial, sc, bf)
        assert segs[0][0] == 0 and segs[-1][1] == T - 1
        assert all(segs[i][1] + 1 == segs[i + 1][0] for i in range(len(segs) - 1))
    print("exactness vs brute force: 20/20 passed")

    # 2) recovers planted structure: 3 cues, each matched by one track
    D, T = 32, 15
    # windows of one track share a musical identity (centre) plus local variation
    centres = [rng.normal(size=D) for _ in range(6)]
    tracks = [c + 0.7 * rng.normal(size=(20, D)) for c in centres]
    truth = [0] * 5 + [3] * 6 + [5] * 4
    scene_emb = np.stack([tracks[a][rng.integers(20)] + 1.0 * rng.normal(size=D) for a in truth])
    dur = np.full(T, 2.5)
    for name, E in [("sum-max", SumMaxEmission), ("pooled", PooledEmission)]:
        segs, _ = semi_markov_decode(E(scene_emb, tracks, 10.0), dur, 6, max_len=8,
                                     lam_dur=1.0, switch_logp=-2.0)
        pred = [a for s, e, a in segs for _ in range(s, e + 1)]
        acc = np.mean(np.array(pred) == np.array(truth))
        print(f"planted test ({name}): segments={[(s, e, a) for s, e, a in segs]}  track acc={acc:.2f}")
    print("layout example:", layout_segment(scene_emb[:5], dur[:5], tracks[0], hop_s=2.5))
