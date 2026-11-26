"""D67: temporal video features (features/video_temporal.py)."""
import numpy as np

from bgm.features.video_temporal import FakeTemporalEncoder, clip_indices, encode_video


def test_clip_indices_are_centred_and_clamped():
    idx = clip_indices(100, 4, n=4, stride=2)
    assert idx.shape == (4, 4)
    assert idx[1].tolist() == [34, 36, 38, 40]            # centre 37.5: 37.5 + (-3, -1, 1, 3), rounded half to even
    assert idx[0].tolist() == [10, 12, 14, 16] and idx.max() <= 99
    short = clip_indices(20, 1, n=16, stride=3)           # clip longer than the video: clamped to its ends
    assert short.min() == 0 and short.max() == 19 and (np.diff(short[0]) >= 0).all()


def test_encode_video_gives_one_unit_vector_per_second():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (80, 8, 8, 3), dtype=np.uint8)
    v = encode_video(FakeTemporalEncoder(dim=12), frames, t1=3, batch=2)
    assert v.shape == (3, 12) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    still = np.repeat(frames[:1], 80, 0)
    assert not np.allclose(encode_video(FakeTemporalEncoder(dim=12), still, 3), v)   # motion changes the embedding


def test_to_25hz_repeats_fits_and_removes_the_file_mean():
    from bgm.features.video_temporal import to_25hz
    x12 = np.arange(12, dtype=np.float32).reshape(6, 2) + 5.0
    y = to_25hz(x12, 11)
    assert y.shape == (11, 2) and np.allclose(y[0], y[1]) and not np.allclose(y[1], y[2])   # one sample per 2 frames
    assert np.allclose(y.mean(0), 0.0, atol=1e-5) and np.allclose(y.std(0), 1.0, atol=1e-3)
    assert to_25hz(x12, 15).shape == (15, 2)                                              # padded with the last sample
