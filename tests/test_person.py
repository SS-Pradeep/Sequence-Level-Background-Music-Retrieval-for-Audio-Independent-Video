"""Person branch (docs/person_branch_plan.md): pure parts, on synthetic arrays only."""
import numpy as np

from bgm.person.features import face_crop_box, square_box
from bgm.person.subset import subsets


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_square_box_stays_inside_and_is_square():
    rng = np.random.default_rng(0)
    for _ in range(200):
        W, H = 640, 360
        x0, y0 = rng.uniform(-50, W), rng.uniform(-50, H)
        x1, y1 = x0 + rng.uniform(5, 400), y0 + rng.uniform(5, 400)
        a, b, c, d = square_box(x0, y0, x1, y1, W, H)
        assert 0 <= a < c <= W and 0 <= b < d <= H
        assert abs((c - a) - (d - b)) <= 1


def test_face_crop_widens_box():
    a, b, c, d = face_crop_box(np.array([100, 100, 40, 50]), 640, 360, margin=0.3)
    assert (c - a) == (d - b) == 80                     # 50 px tall + 2 x 15 px
    assert a <= 100 and b <= 100 and c >= 140 and d >= 150


def test_subsets_find_same_setting_different_music():
    rng = np.random.default_rng(1)
    n, d = 40, 16
    S, M = unit(rng.normal(size=(n, d))), unit(rng.normal(size=(n, d)))
    video = np.arange(n)
    # segments 0 and 1: near-identical setting, opposite music -> both in 'spec' and 'nn'
    S[1] = unit(S[0] + 0.01 * rng.normal(size=d))
    M[1] = -M[0]
    # segments 2 and 3: near-identical setting, same music -> in neither through each other
    S[3] = unit(S[2] + 0.01 * rng.normal(size=d))
    M[3] = M[2]
    out = subsets(S, M, video)
    assert out["spec"][0] and out["spec"][1] and out["nn"][0] and out["nn"][1]
    assert not out["nn"][2] and not out["nn"][3]


def test_subsets_ignore_same_video_pairs():
    rng = np.random.default_rng(2)
    n, d = 30, 8
    S, M = unit(rng.normal(size=(n, d))), unit(rng.normal(size=(n, d)))
    S[1], M[1] = S[0], -M[0]
    video = np.arange(n)
    video[1] = video[0]                                 # same video: the pair must not count
    out = subsets(S, M, video)
    nn0 = np.where(video != video[0], S @ S[0], -np.inf).argmax()
    assert nn0 != 1


def test_span_index_matches_series_span_mean():
    from bgm.eval.store import Series1Hz
    from bgm.person.train import span_index
    x = np.random.default_rng(3).normal(size=(17, 4)).astype(np.float32)
    s = Series1Hz(x)
    for a, b in [(0, 5), (2.3, 7.9), (16.2, 16.4), (0, 17), (5.5, 5.6), (3.0, 3.0)]:
        assert np.allclose(x[span_index(len(x), a, b)].mean(0), s.span_mean(a, b), atol=1e-6)


def test_person_vector_mask_and_null_vector():
    import torch
    from bgm.person.train import PersonVideoHead, person_vector
    emb = np.random.default_rng(4).normal(size=(6, 1152)).astype(np.float32)
    has = np.array([0, 0, 1, 1, 0, 0], bool)
    v = person_vector(emb, has, np.arange(0, 2))
    assert v.shape == (1153,) and not v.any()
    v = person_vector(emb, has, np.arange(1, 4))
    assert v[-1] == 1 and abs(np.linalg.norm(v[:-1]) - 1) < 1e-5
    torch.manual_seed(0)
    head = PersonVideoHead(1153, 64, 16, 0.0).eval()
    x1 = torch.zeros(2, 1153)
    x2 = torch.randn(2, 1153)
    x2[:, -1] = 0                                    # mask 0: the person part must be ignored (replaced by the null vector)
    assert torch.allclose(head(x1), head(x2), atol=1e-6)
    x3 = x2.clone()
    x3[:, -1] = 1
    assert not torch.allclose(head(x2), head(x3))


def test_tagged_frame_embedding_is_unchanged():
    from bgm.person.pipeline import tag
    g = np.arange(4, dtype=np.float32)
    t = tag(g, np.ones(3))
    M = np.random.default_rng(5).normal(size=(5, 4)).astype(np.float32)
    assert np.array_equal(M @ t, M @ g) and np.array_equal(t.person, np.ones(3))
