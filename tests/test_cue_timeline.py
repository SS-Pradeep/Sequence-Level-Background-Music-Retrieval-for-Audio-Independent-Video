"""D93: film times for clips the manifest lacks (cue.timeline_file); default off."""
import pandas as pd
from omegaconf import OmegaConf

from bgm.cue.sequences import _num, with_timeline


def _pairs():
    # read_pairs keeps missing meta values as "" (keep_default_na=False)
    return pd.DataFrame({"pair_id": ["a", "b", "c"], "meta_movie_start_s": ["12.5", "", ""]}).set_index("pair_id", drop=False)


def test_default_returns_pairs_unchanged():
    p = _pairs()
    assert with_timeline(p, OmegaConf.create({"cue": {"timeline_file": None}})) is p


def test_fills_only_missing_times(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame({"clip_id": ["a", "b"], "start_sec": [99.0, 42.25], "film": ["x", "y"]}).to_csv(csv, index=False)
    out = with_timeline(_pairs(), OmegaConf.create({"cue": {"timeline_file": str(csv)}}))
    t = out["meta_movie_start_s"].map(_num)
    assert t["a"] == 12.5                     # an existing manifest time wins
    assert t["b"] == 42.25                    # filled from the file
    assert pd.isna(t["c"])                    # not in the file: still unknown


def test_splice_parts_tokenised_like_clips():
    """D94: two whole-second ends of one clip, each re-segmented and tokenised; the gap is removed."""
    import numpy as np
    from bgm.cue.train import splice_clip
    rng = np.random.default_rng(0)
    T1 = 40
    sem = rng.normal(size=(T1, 16)).astype(np.float32)
    dyn = np.abs(rng.normal(size=(T1 * 25, 5))).astype(np.float32)
    cuts = [3.2, 11.0, 19.5, 27.1, 33.3]
    seg = OmegaConf.create({"merge_cosine": 0.85, "max_len_s": 20.0, "min_shot_s": 2.0})
    for _ in range(50):
        t1, t2, a, b = splice_clip(sem, dyn, cuts, rng, seg, 8.0, (2.0, 30.0))
        assert 8 <= t1 and t2 - t1 >= 2 and T1 - t2 >= 8 and isinstance(t1, int) and isinstance(t2, int)
        assert a.shape[1] == b.shape[1] == 16 + 3
        # the segments tile each part: their durations (log_dur, column -3) add up to the part length
        assert abs(np.exp(a[:, -3]).sum() - t1) < 1e-3 and abs(np.exp(b[:, -3]).sum() - (T1 - t2)) < 1e-3
    assert splice_clip(sem[:15], dyn[:15 * 25], cuts, rng, seg, 8.0, (2.0, 30.0)) is None    # too short for two 8 s ends


def test_splice_join_is_labelled_no_change():
    import numpy as np
    from bgm.cue.sequences import boundary_labels
    parts = ["x", "p§10-14A", "p§10-14B", "y"]
    base = {"x": "x", "p§10-14A": "p", "p§10-14B": "p", "y": "y"}
    track = {"x": "t1", "p": "t2", "y": "t3"}
    film = {"x": "f", "p": "f", "y": "f"}
    y, typ = boundary_labels([0, 1, 2, 3], parts, {q: track[b] for q, b in base.items()}, {q: film[b] for q, b in base.items()})
    assert y.tolist() == [1, 0, 1] and typ == ["film", "film", "film"]
