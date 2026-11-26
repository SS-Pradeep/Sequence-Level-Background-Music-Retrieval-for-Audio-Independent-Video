"""D140 segmentation study: chained shot sequences, labels, BaSSL windows, TransNetV2 post-processing (synthetic)."""
import numpy as np

from bgm.cue.sequences import Clip, by_film, sample_sequence
from bgm.segmentation.bassl_features import keyframe_index, window_index
from bgm.segmentation.sequences import ClipShots, chain, edge_region
from bgm.segmentation.shots_transnet import predictions_to_scenes, scenes_to_shots, shots_from_cuts


def _clips():
    clips, track, film = {}, {}, {}
    for f in range(6):
        for j in range(6):
            p = f"f{f}_c{j}"
            n = 2 + (j % 3)
            b = np.linspace(0, 12.0 + j, n + 1)
            clips[p] = ClipShots(p, np.stack([b[:-1], b[1:]], 1), np.zeros((n, 4), np.float32), np.zeros(n, np.float32))
            track[p] = f"t{f}_{j // 2}"            # pairs of clips of one film share a track
            film[p] = f"F{f}"
    return clips, track, film


def test_labels_only_at_song_changing_joins():
    clips, track, film = _clips()
    parts = ["f0_c0", "f0_c1", "f0_c2", "f1_c0", "f1_c3"]
    ch = chain(parts, clips, track, film)
    assert len(ch.btimes) == len(ch.shots) - 1
    for s in range(len(ch.btimes)):
        a, b = parts[ch.shot_part[s]], parts[ch.shot_part[s + 1]]
        if a == b:
            assert ch.labels[s] == 0 and ch.types[s] == "clip"
        else:
            assert ch.labels[s] == int(track[a] != track[b])
            assert ch.types[s] == ("film" if film[a] == film[b] else "cross")
    # f0_c0 -> f0_c1 share track t0_0: a same-film join with label 0
    j = np.flatnonzero(ch.is_join)
    assert ch.labels[j[0]] == 0 and ch.types[j[0]] == "film"
    assert ch.labels[j].tolist() == [0, 1, 1, 1]
    assert np.allclose(ch.joins, ch.btimes[j])


def test_training_sequences_have_same_film_blocks():
    clips, track, film = _clips()
    table = {p: Clip(p, film[p], track[p], float(c.shots[-1, 1]), float("nan"), "") for p, c in clips.items()}
    films = by_film(table)
    n_same, n_cross = 0, 0
    for i in range(40):
        parts = sample_sequence(films, np.random.default_rng([0, 0, 0, i]), 60, 180, "blocked", 8)
        ch = chain(parts, clips, track, film)
        t = np.array(ch.types)[ch.is_join]
        n_same += int((t == "film").sum())
        n_cross += int((t == "cross").sum())
        assert 60 <= ch.duration <= 180 + 20
    assert n_same > 0 and n_cross > 0


def test_window_index_matches_bassl_sampler():
    idx, edge = window_index(20)
    assert idx.shape == (19, 17)
    assert idx[10].tolist() == list(range(2, 19))
    assert idx[0, :8].tolist() == [0] * 8 and idx[0, 8] == 0
    assert edge.tolist() == [s < 8 or s > 11 for s in range(19)]
    clips, track, film = _clips()
    ch = chain(["f0_c0", "f1_c1", "f2_c2", "f3_c0", "f4_c1", "f5_c2", "f0_c3", "f1_c4", "f2_c5"], clips, track, film)
    lo, hi = edge_region(ch)
    e = window_index(len(ch.shots))[1]
    assert np.all((ch.btimes <= lo) | (ch.btimes >= hi) == e)


def test_keyframes_inside_shot():
    assert keyframe_index([10, 39]) == [15, 24, 34]
    assert keyframe_index([5, 5]) == [5, 5, 5]


def test_transnet_postprocessing():
    pred = np.zeros(100)
    pred[30:32] = 0.9
    pred[70] = 0.8
    sc = predictions_to_scenes(pred)
    assert sc.tolist() == [[0, 30], [32, 70], [71, 99]]
    shots = scenes_to_shots(sc, 100)
    assert [s[2:] for s in shots] == [[0.0, 1.28], [1.28, 2.84], [2.84, 4.0]]
    assert shots_from_cuts([1.0, 2.5], 4.0) == [[0, 24, 0.0, 1.0], [25, 61, 1.0, 2.5], [62, 99, 2.5, 4.0]]


def test_contrast_features():
    from bgm.segmentation.sequences import shot_contrast, time_contrast
    f = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
    c = shot_contrast(f, ks=(1, 2))
    assert c.shape == (3, 2)
    assert np.allclose(c[:, 0], [1, 0, 1]) and np.allclose(c[1, 1], 0)
    emb = np.array([[1, 0]] * 5 + [[0, 1]] * 5, float)
    tc = time_contrast(emb, np.arange(10) + 0.5, np.array([5.0, 2.0, 0.2, 9.9]))
    assert tc.shape == (4, 2)
    assert np.allclose(tc[0], 0) and np.allclose(tc[1, 0], 1) and np.allclose(tc[1, 1], 3 / np.sqrt(10))
    assert np.all(np.isfinite(tc))


def test_sequence_head_matches_per_sequence():
    import torch
    from bgm.segmentation.head import SequenceHead
    torch.manual_seed(0)
    h = SequenceHead(6).eval()
    xs = [torch.randn(n, 6) for n in (5, 0, 3, 7)]
    z = h(xs)
    assert z.shape == (15,)
    alone = torch.cat([h([x]) for x in xs if len(x)])
    assert torch.allclose(z, alone, atol=1e-5)


def test_losses():
    import torch
    from bgm.segmentation.train_head import loss_fn
    z = torch.tensor([2.0, -1.0, 0.5, -3.0])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0])
    pw = torch.tensor(2.0)
    b, f, r = (loss_fn(k, z, y, pw) for k in ("bce", "focal", "bce_rank"))
    assert f < b < r
    # a perfect ranking gives a smaller ranking term than a reversed one
    good = loss_fn("bce_rank", torch.tensor([3.0, -3.0]), torch.tensor([1.0, 0.0]), pw)
    bad = loss_fn("bce_rank", torch.tensor([-3.0, 3.0]), torch.tensor([1.0, 0.0]), pw)
    assert good < bad
