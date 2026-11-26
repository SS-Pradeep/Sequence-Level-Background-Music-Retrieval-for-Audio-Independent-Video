import numpy as np

from bgm.render.pace import Reference, pct, video_pace


def ref() -> Reference:
    return Reference(np.arange(0.0, 60.0), np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), np.linspace(-40, 0, 100))


def test_pct_bounds():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    assert pct(r, 0.0) == 0.0 and pct(r, 10.0) == 1.0 and pct(r, 2.0) == 0.5


def test_fast_video_has_higher_pace():
    T = 30 * 25
    slow = np.zeros((T, 5), np.float32)
    fast = slow.copy()
    fast[:, 0] = 0.9
    p_slow = video_pace(slow, np.array([10.0]), 0.0, 30.0, ref())
    p_fast = video_pace(fast, np.arange(0.0, 30.0, 1.0), 0.0, 30.0, ref())
    assert 0.0 <= p_slow < p_fast <= 1.0


def test_short_segment_widened_to_context():
    dyn = np.zeros((30 * 25, 5), np.float32)
    cuts = np.array([14.0, 16.0])                  # both outside a 1 s segment at 10 s, inside its 10 s context
    assert video_pace(dyn, cuts, 9.5, 10.5, ref(), min_ctx_s=10.0) == video_pace(dyn, cuts, 5.0, 15.0, ref())
