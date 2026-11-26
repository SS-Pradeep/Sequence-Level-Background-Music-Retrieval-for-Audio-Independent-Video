import numpy as np

from bgm.segmentation.retrieval_eval import (boundary_labels, from_rel, overlap_accuracy, overlap_pieces, segments_from_cuts,
                                             sequence_motion, to_rel, true_changes)

SPANS = [(0.0, 10.0), (10.0, 8.0), (18.0, 12.0)]       # (t0, d) per clip
TRACKS = ["a", "b", "b"]


def test_rel_round_trip_and_joins():
    starts = [s[0] for s in SPANS]
    rel = to_rel([4.0, 10.0, 15.5, 18.0], starts)
    assert rel == [[0, 4.0], [1, 0.0], [1, 5.5], [2, 0.0]]
    # another timeline of the same clips (slightly different durations): joins stay on joins
    assert from_rel(rel, [0.0, 10.04, 18.08], [10.04, 8.04, 12.0]) == [4.0, 10.04, 15.54, 18.08]
    # an offset past a shorter clip is clipped to that clip's end
    assert from_rel([[1, 9.0]], [0.0, 10.0], [10.0, 8.0]) == [18.0]


def test_segments_from_cuts_drops_duplicates_and_edges():
    assert segments_from_cuts([0.0, 5.0, 5.0, 30.0], 30.0) == [(0.0, 5.0), (5.0, 30.0)]


def test_overlap_accuracy_equals_duration_weighted_without_crossing():
    segs = [(0.0, 6.0), (6.0, 10.0), (10.0, 18.0), (18.0, 30.0)]
    pieces = overlap_pieces(segs, SPANS, TRACKS)
    assert len(pieces[0]) == len(segs)
    chosen = np.array(["a", "b", "b", "a"], object)
    seg_track = np.array(["a", "a", "b", "b"], object)
    w = np.array([b - a for a, b in segs])
    assert overlap_accuracy(chosen, pieces) == float(np.average(chosen == seg_track, weights=w))


def test_overlap_accuracy_scores_a_crossing_segment_by_pieces():
    segs = [(0.0, 14.0), (14.0, 30.0)]                  # first segment: 10 s of a, 4 s of b
    pieces = overlap_pieces(segs, SPANS, TRACKS)
    assert overlap_accuracy(np.array(["a", "b"], object), pieces) == 26.0 / 30.0
    assert overlap_accuracy(np.array(["b", "b"], object), pieces) == 20.0 / 30.0


def test_truth_and_labels():
    assert true_changes(SPANS, TRACKS) == [10.0]
    segs = [(0.0, 9.0), (9.0, 16.0), (16.0, 30.0)]      # the dominant tracks: a, b, b
    assert boundary_labels(segs, overlap_pieces(segs, SPANS, TRACKS)).tolist() == [1, 0]


def test_sequence_motion_hides_clip_starts():
    m = sequence_motion([np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0, 4.0])])
    assert m.tolist() == [0.0, 1.0, 2.0, 3.0, 3.0, 4.0]
