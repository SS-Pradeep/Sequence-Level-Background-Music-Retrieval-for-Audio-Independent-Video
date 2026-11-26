"""tools/build_cleaned.py: folds and track groups carried from the source pairs to the cleaned pairs."""
from __future__ import annotations

import pandas as pd

from tools.build_cleaned import carry_folds, carry_tracks


def test_carry_folds_keeps_each_source_role():
    old = {"folds": {"0": {"train": ["a", "b"], "val": ["c"], "test": ["d"]},
                     "1": {"train": ["c", "d"], "val": ["a"], "test": ["b"]}},
           "film_group": {"a": "F1", "b": "F2", "c": "F3", "d": "F4"}}
    # b excluded; d split into two parts; a, c kept whole
    sources = {"a": "a", "c": "c", "d_p1": "d", "d_p2": "d"}
    folds, film = carry_folds(old, list(sources), sources)
    assert folds["0"] == {"train": ["a"], "val": ["c"], "test": ["d_p1", "d_p2"]}
    assert folds["1"] == {"train": ["c", "d_p1", "d_p2"], "val": ["a"], "test": []}
    assert film == {"a": "F1", "c": "F3", "d_p1": "F4", "d_p2": "F4"}


def test_carry_tracks_rewrites_music_ids_and_drops_empty_groups():
    old = pd.DataFrame({"track_id": ["t1", "t2", "t3"], "music_ids": ["m1;m2", "m3", "m4"],
                        "evidence": ["fingerprint", "none", "none"]})
    pairs = pd.DataFrame({"track_id": ["t1", "t3", "t3"], "music_id": ["m1", "m4_p1", "m4_p2"]})
    out = carry_tracks(old, pairs).set_index("track_id")
    assert list(out.index) == ["t1", "t3"]
    assert out.loc["t1", "music_ids"] == "m1" and out.loc["t1", "evidence"] == "fingerprint"
    assert out.loc["t3", "music_ids"] == "m4_p1;m4_p2"


def test_trimmed_pairs_treats_full_length_keeps_as_whole():
    from bgm.analysis.cleaned_compare import trimmed_pairs
    orig = pd.DataFrame({"pair_id": ["a", "b", "c", "d"], "video_duration_s": ["10.0", "10.0", "10.0", "10.0"]})
    clean = pd.DataFrame({"pair_id": ["a", "b", "c_p1", "c_p2", "d"],
                          "meta_trim_source_id": ["a", "b", "c", "c", ""],
                          "meta_trim_start_s": ["0.0", "2.0", "0.0", "5.0", ""],
                          "meta_trim_end_s": ["9.96", "10.0", "4.0", "10.0", ""]})
    # a: one full-length part (last frame boundary) = kept whole; b: start cut; c: split; d: never edited
    assert trimmed_pairs(clean, orig) == {"b", "c_p1", "c_p2"}
