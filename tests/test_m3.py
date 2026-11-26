"""M3: features. Event timing on synthetic fixtures, tempo, NaN-freedom, schema, and the no-audio rule."""
from __future__ import annotations

import numpy as np
import pytest

from bgm.features.check import check_file, schema
from bgm.features.dynamics import M_CHANNELS, V_CHANNELS, foote_novelty, music_curves, normalize_dyn, video_curves
from bgm.features.extract import feature_path, frames_1hz, load_features
from bgm.features.keys import estimate_key, fifths_distance, stitch_key
from bgm.features.timing import upsample_1hz_to_25hz
from tests.fixtures.pipeline import run_through_extract

FRAME = 1 / 25


@pytest.fixture(scope="module")
def extracted(synth_pairs, tmp_path_factory):
    cfg, df = run_through_extract(tmp_path_factory.mktemp("m3"), synth_pairs)
    return cfg, df


def test_every_file_matches_the_schema(extracted):
    cfg, df = extracted
    spec = schema(cfg)
    for p, m in zip(df.pair_id, df.music_id):
        assert check_file(load_features(feature_path(cfg, "video", p)), spec["video"]) == []
        assert check_file(load_features(feature_path(cfg, "music", m)), spec["music"]) == []


def test_cut_impulse_peaks_on_the_cuts(extracted, synth_pairs):
    cfg, _ = extracted
    for p in synth_pairs:
        cut = load_features(feature_path(cfg, "video", p.pair_id))["dyn_25hz"][:, V_CHANNELS.index("cut_impulse")]
        for c in p.cut_times_s:
            t0 = int(round(c * 25))
            win = cut[max(0, t0 - 3):t0 + 4]
            assert abs(max(0, t0 - 3) + int(np.argmax(win)) - t0) <= 1 and win.max() > 0.9


def test_beat_impulses_and_tempo(extracted, synth_pairs):
    cfg, _ = extracted
    octave_errors = 0
    for p in synth_pairs:
        d = load_features(feature_path(cfg, "music", p.pair_id))
        beats, true = d["beat_times_s"], np.array(p.beat_times_s)
        assert len(beats) >= 4
        near = np.abs(beats[:, None] - true[None, :]).min(1)
        assert np.mean(near <= 0.040) >= 0.9, (p.pair_id, np.round(near, 3))
        ratio = float(d["tempo_bpm"]) / p.tempo_bpm
        if abs(ratio - 1) > 0.02:
            assert min(abs(ratio - 2), abs(ratio - 0.5)) <= 0.04, f"{p.pair_id}: tempo {d['tempo_bpm']} vs {p.tempo_bpm}"
            octave_errors += 1                                      # flagged, not failed (SPEC M3)
        beat = d["dyn_25hz"][:, M_CHANNELS.index("beat_impulse")]
        for b in beats[1:-1]:
            t0 = int(round(b * 25))
            assert beat[t0] > 0.4
    print(f"tempo octave errors: {octave_errors}/{len(synth_pairs)}")


def test_visual_onset_at_speed_ups(extracted, synth_pairs):
    cfg, _ = extracted
    checked = 0
    for p in synth_pairs:
        dyn = load_features(feature_path(cfg, "video", p.pair_id))["dyn_25hz"]
        vo = dyn[:, V_CHANNELS.index("visual_onset")]
        for k, c in enumerate(p.cut_times_s):
            if p.shot_speeds[k + 1] >= p.shot_speeds[k] + 3:
                t0 = int(round(c * 25))
                assert vo[max(0, t0 - 3):t0 + 4].max() >= np.quantile(vo, 0.9), (p.pair_id, c)
                checked += 1
    assert checked >= 1


def test_upsample_1hz_timing():
    x = np.zeros(10)
    x[4] = 1.0
    y = upsample_1hz_to_25hz(x, 250)
    assert int(np.argmax(y)) in (112, 113)                          # 4.5 s = sample 112.5
    assert np.allclose(upsample_1hz_to_25hz(np.arange(3.0)[:, None].repeat(2, 1), 75)[:, 0],
                       np.interp(np.arange(75) / 25, np.arange(3) + 0.5, np.arange(3.0)))


def test_novelty_marks_a_section_change():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(64), rng.standard_normal(64)
    E = np.vstack([np.tile(a, (20, 1)), np.tile(b, (20, 1))]) + 0.05 * rng.standard_normal((40, 64))
    assert abs(int(np.argmax(foote_novelty(E, 8))) - 20) <= 1


def test_music_curves_are_finite_on_digital_silence():
    sr = 22050
    t = np.arange(8 * sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y[: 2 * sr] = 0.0                                               # 2 s of digital silence
    dyn, beats, tempo = music_curves(y, sr)
    assert np.isfinite(dyn).all() and dyn.shape == (200, 5)
    assert np.isfinite(normalize_dyn(dyn, M_CHANNELS)).all()
    silent, _, tempo0 = music_curves(np.zeros(4 * sr, np.float32), sr)
    assert np.isfinite(silent).all() and tempo0 <= 0
    assert estimate_key(np.zeros(sr, np.float32), sr) == (-1, -1)


def test_normalize_leaves_impulses_alone():
    rng = np.random.default_rng(0)
    d = rng.random((100, 5)).astype(np.float32)
    n = normalize_dyn(d, V_CHANNELS)
    assert np.allclose(n[:, 2], d[:, 2]) and abs(n[:, 0].mean()) < 1e-5 and abs(n[:, 0].std() - 1) < 1e-3


def test_video_extractors_refuse_audio(synth_pairs):
    p = next(p for p in synth_pairs if p.embedded)                  # the original, with an audio stream
    with pytest.raises(AssertionError, match="audio stream"):
        video_curves(p.video_path, [])
    with pytest.raises(AssertionError, match="audio stream"):
        frames_1hz(p.video_path, 5)


def test_key_helpers_from_spec():
    assert stitch_key(9, 1, -1, False) == 0                         # A minor -> relative major C
    assert stitch_key(9, 0, -1, False) == 9 and stitch_key(-1, 0, 4, True) == 4 and stitch_key(-1, 1, -1, False) == -1
    assert fifths_distance(0, 7) == pytest.approx(1 / 6) and fifths_distance(0, 5) == pytest.approx(1 / 6)
    assert fifths_distance(0, 6) == pytest.approx(1.0) and fifths_distance(3, 3) == 0.0 and fifths_distance(-1, 2) == 0.5


def test_key_estimate_on_a_c_major_chord():
    sr = 22050
    t = np.arange(4 * sr) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0)).astype(np.float32) * 0.2
    pc, mode = estimate_key(y, sr)
    assert (pc, mode) in [(0, 0), (4, 1), (9, 1)]                   # C major, or a relative / mediant minor reading


def test_extract_is_resumable_and_hash_gated(extracted):
    from bgm.features.extract import Extractor
    cfg, df = extracted
    ex = Extractor(cfg, fake=True, workers=2)
    assert not any(ex.stale("video", p, "sem_video") for p in df.pair_id)
    from bgm.config import load_config
    cfg2 = load_config("configs/default.yaml", [f"paths.{k}={cfg.paths[k]}" for k in ("manifests", "interim", "features", "reports")]
                       + ["features.dynamics.impulse_sigma_s=0.05"])
    ex2 = Extractor(cfg2, fake=True, workers=2)
    assert all(ex2.stale("video", p, "dyn_video") for p in df.pair_id)       # config change -> recompute
    assert not any(ex2.stale("video", p, "sem_video") for p in df.pair_id)   # other groups untouched


def test_duplicate_frame_repair():
    from bgm.features.dynamics import duplicate_frames, repair_duplicate_frames
    rng = np.random.default_rng(0)
    motion = 2.0 + 0.2 * rng.random(250)
    motion[12::25] = 0.01                                           # one duplicated frame per second
    cam = motion.copy()
    bad = duplicate_frames(motion)
    assert set(np.where(bad)[0]) == set(range(12, 250, 25))
    m2, c2 = repair_duplicate_frames(motion, cam)
    assert m2.min() > 1.9 and np.allclose(m2[~bad], motion[~bad])
    still = np.full(100, 0.01)
    still[50] = 0.0                                                  # a genuinely static shot is left alone
    assert not duplicate_frames(still).any()
    burst = np.r_[np.full(50, 3.0), np.full(50, 0.02)]               # motion that really stops is not "repaired"
    assert not duplicate_frames(burst).any()
