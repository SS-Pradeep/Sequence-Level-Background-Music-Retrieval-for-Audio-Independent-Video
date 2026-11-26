"""bgm.data.fp_index: the vectorised candidate search agrees with data/tracks.candidates and groups shared audio."""
import numpy as np

from bgm.data.fp_index import candidates_np, group_fingerprints
from bgm.data.tracks import Fingerprint, candidates


def _corpus(seed=0):
    """Random 'songs'; song 1 re-uses 400 items of song 0 (offset 150 -> 30) with 2% of items bit-flipped,
    song 3 is a full copy of song 2 delayed by 57 items."""
    rng = np.random.default_rng(seed)
    fps = [rng.integers(0, 2**32, size=rng.integers(600, 1200), dtype=np.uint64).astype(np.uint32) for _ in range(8)]
    seg = fps[0][150:550].copy()
    flip = rng.random(len(seg)) < 0.02
    seg[flip] ^= np.uint32(1 << 7)
    fps[1][30:430] = seg
    fps[3] = np.concatenate([rng.integers(0, 2**32, 57, dtype=np.uint64).astype(np.uint32), fps[2]])
    return fps


def test_matches_reference_candidates():
    fps = _corpus()
    ref = candidates([Fingerprint(f, 8.07) for f in fps], min_votes=3)
    got = candidates_np(fps, min_votes=3, max_occ=10**6, chunk_pairs=7)     # tiny chunks exercise the merge
    assert got == ref
    assert got[(0, 1)] == 30 - 150 and got[(2, 3)] == 57


def test_frequent_values_are_ignored():
    fps = _corpus()
    common = np.full(50, 12345, np.uint32)
    fps = [np.concatenate([f, common]) for f in fps]          # a 'silence' value in every file
    got = candidates_np(fps, min_votes=3, max_occ=32)
    assert set(got) == {(0, 1), (2, 3)}


def test_group_fingerprints():
    labels, links = group_fingerprints(_corpus(), ber_max=0.35, min_overlap_s=5.0, min_votes=3)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert len(set(labels)) == 6
    assert {(i, j) for i, j, *_ in links} == {(0, 1), (2, 3)}
