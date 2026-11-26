"""M5: metrics (exact chance, ranks, cluster bootstrap) and the shifted-negative sampler (SPEC §5.6, §5.12)."""
from __future__ import annotations

import numpy as np
import pytest

from bgm.eval.metrics import chance_mrr, chance_recall, chance_summary, cluster_bootstrap, rank_of_first_hit, recall_summary
from bgm.train.shifts import sample_negative_shifts, shift_in_sync


def test_rank_of_first_hit_hand_computed():
    s = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
    assert rank_of_first_hit(s, np.array([0, 0, 1, 0, 0], bool)) == 3
    assert rank_of_first_hit(s, np.array([0, 1, 1, 0, 0], bool)) == 3       # best-scored hit counts
    assert rank_of_first_hit(s, np.array([1, 0, 0, 0, 0], bool)) == 1
    assert recall_summary([1, 3, 12])["R@10"] == pytest.approx(2 / 3) and recall_summary([1, 3, 12])["MRR"] == pytest.approx((1 + 1 / 3 + 1 / 12) / 3)


@pytest.mark.parametrize("N,h,k", [(250, 1, 10), (250, 3, 10), (50, 2, 5), (1248, 4, 1), (10, 1, 10)])
def test_exact_chance_matches_simulation(N, h, k):
    rng = np.random.default_rng(0)
    trials = 20000
    hits = np.zeros(N, bool)
    hits[:h] = True
    sim = np.array([rank_of_first_hit(rng.random(N), hits) <= k for _ in range(trials)], float)
    exact = chance_recall(N, h, k)
    assert abs(sim.mean() - exact) <= 2 * max(sim.std(), 1e-3) / np.sqrt(trials) + 1e-9
    ranks = np.array([rank_of_first_hit(rng.random(N), hits) for _ in range(5000)])
    assert abs((1 / ranks).mean() - chance_mrr(N, h)) < 4 * (1 / ranks).std() / np.sqrt(5000)


def test_chance_is_not_k_over_n_with_several_hits():
    assert chance_recall(100, 1, 10) == pytest.approx(0.10)
    assert chance_recall(100, 3, 10) > 0.27
    assert chance_summary([100, 100], [1, 3])["R@10"] == pytest.approx((chance_recall(100, 1, 10) + chance_recall(100, 3, 10)) / 2)


def test_cluster_bootstrap_widens_for_correlated_queries():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(40), 10)
    vals = np.repeat(rng.normal(0, 1, 40), 10) + 0.1 * rng.normal(size=400)    # strongly clustered
    b_cl = cluster_bootstrap(vals, cl, 4000)
    b_iid = cluster_bootstrap(vals, np.arange(400), 4000)
    assert b_cl["lo"] < b_cl["mean"] < b_cl["hi"]
    assert (b_cl["hi"] - b_cl["lo"]) > 2 * (b_iid["hi"] - b_iid["lo"])
    assert b_cl["n_clusters"] == 40


def beat_train(bpm, dur=40.0, fps=25, jitter=0.0, rng=None):
    period = 60 / bpm
    beats = np.arange(0.3, dur, period)
    if jitter:
        beats = beats + rng.normal(0, jitter, len(beats))
    t = np.arange(int(dur * fps)) / fps
    onset = sum(np.exp(-0.5 * ((t - b) / 0.03) ** 2) for b in beats)
    onset = (onset - onset.mean()) / onset.std()
    return onset, beats


@pytest.mark.parametrize("bpm", [70, 90, 120, 150, 180])
def test_accepted_shifts_are_off_the_beat_grid(bpm):
    onset, beats = beat_train(bpm)
    period = 60 / bpm
    for item in range(20):
        rng = np.random.default_rng([0, 0, item])
        start = int(rng.integers(150, 500))
        for s in sample_negative_shifts(onset, beats, start, 200, rng):
            d = (s - start) / 25
            off = abs(d / period - round(d / period)) * period
            assert off >= 0.10 - 1e-9, (bpm, d, off)
            assert 0.5 - 1e-9 <= abs(d) <= 4.0 + 0.02
            assert not shift_in_sync(onset, beats, start, 200, s - start)


@pytest.mark.parametrize("bpm", [80, 128, 170])
def test_whole_beat_shifts_count_as_in_sync(bpm):
    onset, beats = beat_train(bpm)
    period_frames = 60 / bpm * 25
    for m in (1, 2, 3):
        d = int(round(m * period_frames))
        assert shift_in_sync(onset, beats, 300, 200, d) or abs(d - m * period_frames) > 2.5   # rounding can move it off-grid


def test_sampler_is_seeded_per_item():
    onset, beats = beat_train(100)
    a = sample_negative_shifts(onset, beats, 300, 200, np.random.default_rng([0, 1, 7]))
    b = sample_negative_shifts(onset, beats, 300, 200, np.random.default_rng([0, 1, 7]))
    c = sample_negative_shifts(onset, beats, 300, 200, np.random.default_rng([0, 1, 8]))
    assert a == b and a != c and len(set(a)) == len(a)
