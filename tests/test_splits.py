"""M1: fold invariants (SPEC §3.3) and track grouping (SPEC §3.2), on synthetic manifests only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bgm.data.folds import check_folds, make_folds
from bgm.data.tracks import Fingerprint, ber_at_offset, bit_error_rate, group_tracks, union_find


def synthetic_pairs(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 2, 3, 12], size=n)        # mostly singletons, some reused tracks, one big group
    track = np.repeat(np.arange(n), sizes)[:n]
    return pd.DataFrame({"pair_id": [f"p{i:04d}" for i in range(n)], "track_id": [f"trk_{t:04d}" for t in track],
                         "video_duration_s": rng.uniform(10, 170, n)})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_folds_satisfy_invariants(seed):
    pairs = synthetic_pairs(seed=seed)
    folds = make_folds(pairs)
    assert len(folds) == 5
    check_folds(folds, pairs)
    track = dict(zip(pairs.pair_id, pairs.track_id))
    for f in folds.values():                                # val is ~10% of the non-test track groups
        n_val = len({track[p] for p in f["val"]})
        n_rest = len({track[p] for p in f["train"] + f["val"]})
        assert abs(n_val / n_rest - 0.1) < 0.01


def test_folds_are_deterministic():
    pairs = synthetic_pairs()
    assert make_folds(pairs) == make_folds(pairs.sample(frac=1, random_state=3))


def test_check_folds_catches_leakage():
    pairs = synthetic_pairs()
    folds = make_folds(pairs)
    tracks = dict(zip(pairs.pair_id, pairs.track_id))
    shared = next(p for p in folds["0"]["test"] if list(tracks.values()).count(tracks[p]) > 1)
    sibling = next(q for q in folds["0"]["test"] if q != shared and tracks[q] == tracks[shared])
    folds["0"]["test"].remove(sibling)
    folds["0"]["train"].append(sibling)                     # same track now in train and test
    folds["1"]["test"].append(sibling)
    with pytest.raises(AssertionError):
        check_folds(folds, pairs)


def test_bit_error_rate():
    a = np.array([0, 0xFFFFFFFF], np.uint32)
    assert bit_error_rate(a, a) == 0.0
    assert bit_error_rate(a, ~a) == 1.0
    assert bit_error_rate(np.array([0], np.uint32), np.array([0xFFFF], np.uint32)) == 0.5
    x = np.arange(100, dtype=np.uint32)
    assert ber_at_offset(x, x[5:], -5) == (0.0, 95)


def test_union_find():
    lab = union_find(6, [(0, 3), (3, 5), (1, 2)])
    assert lab.tolist() == [0, 1, 1, 0, 4, 0]


def test_group_tracks_links_shifted_noisy_copy():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**32, 400, dtype=np.uint64).astype(np.uint32)
    copy = base[37:337].copy()
    flip = rng.random((len(copy), 32)) < 0.1                # 10% bit errors
    copy ^= (flip * (1 << np.arange(32, dtype=np.uint64))).sum(1).astype(np.uint32)
    other = rng.integers(0, 2**32, 300, dtype=np.uint64).astype(np.uint32)
    fps = [Fingerprint(base, 8.0), Fingerprint(copy, 8.0), Fingerprint(other, 8.0)]
    labels, checks, links = group_tracks(fps, ber_max=0.35, min_overlap_s=10.0, min_votes=3)
    assert labels[0] == labels[1] != labels[2]
    assert len(links) == 1 and links[0].offset == -37 and links[0].ber < 0.15


def test_spectral_corr_matches_shifted_copy_not_noise():
    from bgm.data.tracks import SPEC_HOP, SPEC_SR, spectral_corr
    rng = np.random.default_rng(0)
    fps = SPEC_SR / SPEC_HOP
    a = rng.standard_normal((48, int(30 * fps)))
    b = a[:, int(4 * fps):] + 0.3 * rng.standard_normal((48, int(26 * fps)))   # b[t] = a[t + 4 s] plus noise
    assert spectral_corr(a, b, offset_s=-4.1) > 0.9                          # found within the +/-0.3 s search
    assert abs(spectral_corr(a, rng.standard_normal((48, int(30 * fps))), offset_s=0.0)) < 0.1


def test_fingerprint_offset_is_in_real_seconds(tmp_path):
    """A clip cut 7 s into a longer one must link at an offset of ~7 s (the item rate is constant)."""
    import soundfile as sf
    from bgm.data.tracks import FP_ITEM_RATE, fingerprint, group_tracks
    rng = np.random.default_rng(1)
    sr = 22050
    t = np.arange(40 * sr) / sr
    notes = rng.integers(48, 84, 80)
    y = sum(np.sin(2 * np.pi * 440 * 2 ** ((n - 69) / 12) * t) * (np.abs(t - k * 0.5 - 0.25) < 0.25) for k, n in enumerate(notes))
    y = (0.3 * y / np.abs(y).max()).astype(np.float32)
    sf.write(tmp_path / "long.wav", y, sr)
    sf.write(tmp_path / "cut.wav", y[7 * sr: 22 * sr], sr)
    paths = [str(tmp_path / "long.wav"), str(tmp_path / "cut.wav")]
    fps = [fingerprint(p, highpass_hz=200.0) for p in paths]
    _, _, links = group_tracks(fps, 0.35, 10.0, 3, paths, 0.6)
    assert len(links) == 1
    assert abs(links[0].offset / FP_ITEM_RATE - (-7.0)) < 0.3 and links[0].spec_corr > 0.9
