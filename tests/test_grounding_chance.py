import numpy as np
from bgm.train.shifts import sample_negative_shifts
from bgm.eval.metrics import lines_up, grounding_rules

fps = 25
def song(bpm, n, rng, noise=0.05):
    t = np.arange(n) / fps
    beats = np.arange(0.2, n / fps, 60 / bpm)
    on = sum(np.exp(-0.5 * ((t - b) / 0.04) ** 2) for b in beats) + noise * rng.standard_normal(n)
    return (on - on.mean()) / on.std(), beats

class Forced:
    """Stub generator that makes sample_negative_shifts try exactly one shift d."""
    def __init__(self, d): self.d = d
    def uniform(self, a, b): return abs(self.d) / fps
    def choice(self, xs): return 1 if self.d > 0 else -1

def test_lines_up_matches_sampler():
    rng, T, checked = np.random.default_rng(0), 125, 0
    for bpm in (70, 100, 120, 150, 180):
        for beats_kind in ("true", "none"):
            on, beats = song(bpm, 1000, rng)
            if beats_kind == "none":
                beats = np.array([])
            for _ in range(60):
                start = int(rng.integers(0, 1000 - T))
                d = int(rng.integers(13, 101)) * int(rng.choice([-1, 1]))       # 0.5-4 s, the sampler's range
                lo, hi = max(0, start - 25), min(1000, start + T + 25)
                if start + d < 0 or start + d + T > 1000 or lo + d < 0 or hi + d > 1000:
                    continue                                                 # the sampler skips these
                accepted = sample_negative_shifts(on, beats, start, T, Forced(d), k=1, max_tries=1) == [start + d]
                assert accepted == (not lines_up(on, beats, start, T, d)), (bpm, beats_kind, start, d)
                checked += 1
    assert checked > 300

def test_rule_sizes_and_truth():
    rng = np.random.default_rng(1)
    on, beats = song(120, 500, rng)
    r = grounding_rules(on, beats, 200, 125)
    assert r["exact"].sum() == 5 and r["strict"].sum() == 25                 # |d| <= 2 and <= 12 frames
    assert all(r[k][200] for k in r) and np.all(r["sync"] >= r["exact"])

def test_random_guess_equals_chance():
    rng = np.random.default_rng(2)
    on, beats = song(120, 500, rng)
    r = grounding_rules(on, beats, 180, 125)
    guesses = rng.integers(0, len(r["sync"]), 20000)
    assert abs(r["sync"][guesses].mean() - r["sync"].mean()) < 0.02
