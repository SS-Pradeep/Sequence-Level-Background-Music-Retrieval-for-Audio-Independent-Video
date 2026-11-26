"""Join probes (src/bgm/cue/probes.py, PROGRESS.md D85): same-music sequences, the stratified AUROC and its cluster
bootstrap. Synthetic data only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from bgm.cue.probes import (auc_draws, build_samemusic_sets, cluster_weights, duration_feature, order_grouped,
                            pair_counts)
from bgm.cue.sequences import build_sets, check_sets, clip_table, by_film, is_timeline
from tests.test_cue import synth_manifest


def test_order_grouped_puts_same_track_clips_together():
    pairs, folds = synth_manifest(seed=5)
    fg = folds["film_group"]
    for k in range(5):
        clips = clip_table(pairs, fg, folds["folds"][str(k)]["test"])
        for f, cs in by_film(clips).items():
            o = order_grouped(cs, np.random.default_rng(k))
            assert sorted(c.pair_id for c in o) == sorted(c.pair_id for c in cs)
            runs = [o[0].track] + [b.track for a, b in zip(o, o[1:]) if a.track != b.track]
            assert len(runs) == len({c.track for c in cs})          # each track group is one contiguous run
            if is_timeline(cs):
                firsts = [min(c.t_film for c in cs if c.track == t) for t in runs]
                assert firsts == sorted(firsts)


def test_samemusic_sets_partition_test_clips_and_keep_main_sets():
    pairs, folds = synth_manifest(seed=5)
    sm = build_samemusic_sets(pairs, folds, 5, 60.0, 180.0)
    check_sets({k: {"test": {"samemusic": v}} for k, v in sm.items()}, folds)
    n_same = 0
    for k, v in sm.items():
        assert sorted(p for q in v for p in q) == sorted(folds["folds"][k]["test"])
        n_same += sum(pairs.loc[a, "track_id"] == pairs.loc[b, "track_id"] for q in v for a, b in zip(q, q[1:]))
    assert n_same > 0
    a = build_sets(pairs, folds, 5, 60.0, 180.0, 40.0, 3)          # the main sets do not depend on the new code
    b = build_sets(pairs, folds, 5, 60.0, 180.0, 40.0, 3)
    assert a == b


def test_pair_counts_equal_sklearn_auc_and_respect_strata():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    z = np.round(rng.normal(y * 0.7, 1.0), 1)                       # rounding makes ties
    cl = rng.integers(0, 12, n)
    C, D = pair_counts(z, y == 1, y == 0, np.zeros(n, int), cl, 12)
    assert np.isclose(C.sum() / D.sum(), roc_auc_score(y, z))
    # a score that only encodes the stratum: plain AUROC high, stratified AUROC exactly 0.5
    s = rng.integers(0, 4, n)
    y2 = (rng.random(n) < 0.15 + 0.2 * s).astype(int)
    z2 = s.astype(float)
    C, D = pair_counts(z2, y2 == 1, y2 == 0, np.zeros(n, int), cl, 12)
    assert C.sum() / D.sum() > 0.6
    C, D = pair_counts(z2, y2 == 1, y2 == 0, s, cl, 12)
    assert np.isclose(C.sum() / D.sum(), 0.5)


def test_cluster_bootstrap_matches_brute_force_resample():
    rng = np.random.default_rng(1)
    n, K = 300, 9
    y = rng.integers(0, 2, n)
    z = rng.normal(y * 0.5, 1.0)
    cl = rng.integers(0, K, n)
    st = rng.integers(0, 3, n)
    C, D = pair_counts(z, y == 1, y == 0, st, cl, K)
    W = cluster_weights(K, nb=5, seed=3)
    _, v = auc_draws([(C, D)], W)
    for b in range(5):
        copies = [c for c in range(K) for _ in range(int(W[b, c]))]          # resampled clusters, with repeats
        idx = np.concatenate([np.where(cl == c)[0] for c in copies])
        new_cl = np.concatenate([np.full(int((cl == c).sum()), j) for j, c in enumerate(copies)])
        Cb, Db = pair_counts(z[idx], y[idx] == 1, y[idx] == 0, st[idx], new_cl, len(copies))
        assert np.isclose(v[b], Cb.sum() / Db.sum())


def test_duration_feature_is_left_right_pairs():
    x = np.log(np.array([2.0, 5.0, 11.0]))
    assert np.allclose(duration_feature(x), [[x[0], x[1]], [x[1], x[2]]])
