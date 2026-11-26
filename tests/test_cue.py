"""Cue-switch predictor (src/bgm/cue/): labels, fold leakage, the Viterbi hook, the scene-cut replay, tokens, model.
Synthetic data only (tests/fixtures/synth.py for video)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bgm.cue.evaluate import match_counts
from bgm.cue.model import CueModel, predict
from bgm.cue.sequences import (Clip, boundary_labels, build_sets, by_film, check_sets, clip_table, partition_sequences,
                               sample_sequence, single_film_sequences)
from bgm.cue.stitch import constant_equivalent, decode_lattice_cue
from bgm.cue.tokens import clip_segments, segment_features
from bgm.retrieval.stitch_eval import Layer, decode_lattice, switch_f1


# ---------------------------------------------------------------- synthetic manifest
def synth_manifest(n_films=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, fg = [], {}
    for f in range(n_films):
        n = int(rng.integers(1, 7))
        timed = f % 3 == 0
        t = 0.0
        for j in range(n):
            pid = f"film{f:02d}_clip{j}"
            d = float(rng.uniform(10, 30))
            track = f"trk_{f}_{j}" if not (j and rng.random() < 0.1) else f"trk_{f}_{j - 1}"   # some same-track neighbours
            rows.append({"pair_id": pid, "track_id": track, "video_duration_s": d, "meta_source": "src", "meta_film": str(f),
                         "meta_movie_start_s": t if timed else np.nan})
            t += d + float(rng.uniform(2, 60))
            fg[pid] = f"g{f}"
    big = "gBIG"                                  # a train-only film with many clips (the Konjum case)
    for j in range(30):
        pid = f"big_clip{j}"
        rows.append({"pair_id": pid, "track_id": f"trk_big_{j}", "video_duration_s": 15.0, "meta_source": "src", "meta_film": "big",
                     "meta_movie_start_s": np.nan})
        fg[pid] = big
    pairs = pd.DataFrame(rows).set_index("pair_id", drop=False)
    films = sorted({v for v in fg.values() if v != big})
    rng.shuffle(films)
    folds = {"folds": {}, "film_group": fg, "train_only": [big], "split": "film"}
    for k in range(5):
        test = set(films[k::5])
        val = set(films[(k + 1) % 5::5][:3])
        ids = {s: [p for p, g in fg.items() if (g in test if s == "test" else g in val if s == "val" else
                                                   (g not in test and g not in val))] for s in ("train", "val", "test")}
        folds["folds"][str(k)] = ids
    return pairs, folds


def test_within_clip_boundaries_are_zero():
    parts = ["a", "b", "c", "d"]
    track = {"a": "t1", "b": "t2", "c": "t2", "d": "t3"}
    film = {"a": "f1", "b": "f1", "c": "f1", "d": "f2"}
    seg_part = [0, 0, 0, 1, 1, 2, 3, 3]
    y, typ = boundary_labels(seg_part, parts, track, film)
    assert y.tolist() == [0, 0, 1, 0, 0, 1, 0]
    assert typ == ["clip", "clip", "film", "clip", "film", "cross", "clip"]
    # every within-clip boundary is 0, whatever the tracks
    rng = np.random.default_rng(1)
    for _ in range(50):
        sp = np.sort(rng.integers(0, 4, 20)).tolist()
        y, typ = boundary_labels(sp, parts, track, film)
        assert all(v == 0 for v, t in zip(y, typ) if t == "clip")
        assert all(v == 1 for v, t in zip(y, typ) if t == "cross")


def test_no_fold_leakage_in_sets_and_training_sequences():
    pairs, folds = synth_manifest()
    sets = build_sets(pairs, folds, 5, 60.0, 180.0, 40.0, 3)
    check_sets(sets, folds)
    fg = folds["film_group"]
    for k in range(5):
        f = folds["folds"][str(k)]
        test_films = {fg[p] for p in f["test"]}
        flat = [p for q in sets[str(k)]["test"]["blocked"] for p in q]
        assert sorted(flat) == sorted(f["test"])                      # the blocked test set uses every test clip once
        train = by_film(clip_table(pairs, fg, f["train"]))
        for i in range(200):
            for mode in ("blocked", "cross_film"):
                q = sample_sequence(train, np.random.default_rng([k, i]), 60.0, 180.0, mode, 8, frozenset(folds["train_only"]))
                assert q and set(q) <= set(f["train"])
                assert not {fg[p] for p in q} & test_films
                assert len(set(q)) == len(q)
                assert sum(fg[p] == "gBIG" for p in q) <= 8          # the train-only film: one block at most
                films = [fg[p] for p in q]
                if mode == "cross_film":
                    assert len(set(films)) == len(films)
    bad = {k: {s: {m: [list(x) for x in q] for m, q in ms.items()} for s, ms in v.items()} for k, v in sets.items()}
    bad["0"]["test"]["blocked"][0].append(folds["folds"]["0"]["train"][0])
    with pytest.raises(AssertionError):
        check_sets(bad, folds)


def test_partition_and_single_film_sets():
    pairs, folds = synth_manifest(seed=3)
    clips = clip_table(pairs, folds["film_group"], folds["folds"]["0"]["test"])
    seqs = partition_sequences(clips, np.random.default_rng(0), 60.0, 180.0)
    dur = {p: c.dur for p, c in clips.items()}
    assert sorted(p for q in seqs for p in q) == sorted(clips)
    lens = [sum(dur[p] for p in q) for q in seqs]
    assert all(x >= 60.0 for x in lens[:-1])
    tl = single_film_sequences(clips, np.random.default_rng(0), 40.0, 180.0, timeline=True)
    for q in tl:
        ts = [clips[p].t_film for p in q]
        assert ts == sorted(ts) and len({clips[p].film for p in q}) == 1


# ---------------------------------------------------------------- Viterbi hook
def random_lattice(rng, n_layers=12, n_top=6, n_files=5):
    layers, prev = [], None
    for i in range(n_layers):
        d = float(rng.uniform(2, 6))
        mid = list(rng.integers(0, n_files, n_top))
        off = list(rng.uniform(0, 30, n_top))
        cont_of = None
        if prev is not None:
            cont_of = np.full(len(prev.mid), -1)
            for j, (m, o, pd_) in enumerate(zip(prev.mid, prev.off, prev.dur)):
                if rng.random() < 0.8:
                    cont_of[j] = len(mid)
                    mid.append(int(m)), off.append(float(o + pd_))
        n = len(mid)
        L = Layer(np.array(mid), np.array(off), np.full(n, d), rng.integers(-1, 12, n_files)[np.array(mid)],
                  rng.uniform(60, 160, n_files)[np.array(mid)], rng.normal(0, 1.5, n), n_top, cont_of)
        layers.append(L)
        prev = L
    return layers


@pytest.mark.parametrize("pen", [{"lam_switch": 1.0, "lam_jump": 0.25, "mu": 0.0}, {"lam_switch": 1.0, "lam_jump": 0.5, "mu": 0.0},
                                 {"lam_switch": 2.0, "lam_jump": 2.0, "mu": 1.0}, {"lam_switch": 1.0, "lam_jump": 1.0, "mu": 1.0},
                                 {"lam_switch": 1.0, "lam_jump": 1.0, "mu": 0.5}, {"lam_switch": 8.0, "lam_jump": 0.25, "mu": 0.5}])
def test_constant_p_reproduces_fixed_penalties(pen):
    rng = np.random.default_rng(int(pen["lam_switch"] * 10 + pen["lam_jump"] * 100 + pen["mu"] * 1000))
    for t in range(30):
        layers = random_lattice(rng)
        for beam in (None, 8):
            ref = decode_lattice(layers, pen, beam)
            z, a, lj, mu = constant_equivalent(pen, len(layers) - 1)
            assert decode_lattice_cue(layers, z, a, lj, mu, beam) == ref
            z, a, lj, mu = constant_equivalent(pen, len(layers) - 1, alpha=0.37)    # any alpha, same paths
            assert decode_lattice_cue(layers, z, a, lj, mu, beam) == ref


def test_learned_p_moves_switches():
    rng = np.random.default_rng(5)
    layers = random_lattice(rng, n_layers=10)
    def n_switch(z):
        picks = decode_lattice_cue(layers, z, 1.0, 0.0, 0.0, None)
        mids = [layers[i].mid[j] for i, j in enumerate(picks)]
        return sum(a != b for a, b in zip(mids, mids[1:]))
    assert n_switch(np.full(9, -20.0)) < n_switch(np.full(9, 20.0))         # p ~ 0: avoid switches; p ~ 1: take them


def test_match_counts_agrees_with_switch_f1():
    rng = np.random.default_rng(2)
    for _ in range(200):
        pred = sorted(rng.uniform(0, 60, rng.integers(0, 6)).tolist())
        true = sorted(rng.uniform(0, 60, rng.integers(0, 6)).tolist())
        tp, fp, fn = match_counts(pred, true, 1.0)
        f1 = switch_f1(pred, true, 1.0)
        if pred and true:
            assert np.isclose(f1, 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))


# ---------------------------------------------------------------- tokens and model
def test_tokens_skip_frame0_and_match_concat_filter():
    sem = np.random.default_rng(0).normal(size=(12, 8)).astype(np.float32)
    lm = np.ones(300)
    lm[0] = 0.0                                   # no flow at frame 0
    segs = clip_segments([(0.0, 4.0), (4.0, 12.5), (12.5, 20.0)], 300)
    assert segs == [(0.0, 4.0), (4.0, 12.0)]
    x = segment_features(sem, lm, segs)
    assert np.allclose(x[0, -2:], [1.0, 0.0])     # mean 1, std 0: frame 0 left out
    assert np.allclose(np.linalg.norm(x[:, :8], axis=1), 1.0)
    assert np.isclose(x[1, -3], np.log(8.0))


@pytest.mark.parametrize("context", [True, False])
def test_model_shapes(context):
    torch.manual_seed(0)
    m = CueModel(10, d=16, heads=2, layers=1, ff=32, dropout=0.0, context=context)
    xs = [np.random.default_rng(i).normal(size=(n, 10)).astype(np.float32) for i, n in enumerate((5, 2, 9))]
    z = predict(m, xs)
    assert [len(v) for v in z] == [4, 1, 8]
    # padding does not change a sequence's logits
    alone = predict(m, xs[:1])[0]
    assert np.allclose(alone, z[0], atol=1e-5)


# ---------------------------------------------------------------- R1 scene replay
def test_scene_replay_matches_scenedetect(synth_pairs):
    from scenedetect import ContentDetector
    from scenedetect.common import FrameTimecode

    from bgm.cue.scenes import _downscale, content_scores, detect_cut_frames, pair_score, replay_cuts
    for p in synth_pairs[:2]:
        r = content_scores(p.video_path)
        for thr, ms in ((30.0, 8.0), (30.0, 1.0), (10.0, 2.0)):
            assert replay_cuts(r["scores"], thr, ms) == detect_cut_frames(p.video_path, thr, ms)
    # a joined sequence: replay == the real detector fed the two clips' frames as one stream
    import cv2
    from scenedetect.scene_manager import compute_downscale_factor
    frames = []
    for p in synth_pairs[:2]:
        cap = cv2.VideoCapture(p.video_path)
        f = compute_downscale_factor(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(_downscale(im, f))
        frames.append(None)
    a = content_scores(synth_pairs[0].video_path)
    b = content_scores(synth_pairs[1].video_path)
    joined = np.r_[a["scores"], [pair_score(a["last"], b["first"])], b["scores"][1:]]
    stream = [x for x in frames if x is not None]
    for thr, ms in ((30.0, 1.0), (10.0, 2.0)):
        det = ContentDetector(threshold=thr, min_scene_len=ms)
        cuts = []
        for i, im in enumerate(stream):
            cuts += [int(t.frame_num) for t in det.process_frame(FrameTimecode(i, fps=25.0), im)]
        assert replay_cuts(joined, thr, ms) == sorted(set(cuts))
