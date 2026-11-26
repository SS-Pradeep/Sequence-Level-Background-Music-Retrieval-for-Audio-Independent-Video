"""M0: CLI surface, config loading, synthetic fixtures, fake encoders."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from bgm.cli import app
from bgm.config import load_config
from bgm.data.media import ffprobe
from bgm.features.encoders import FakeAudioEncoder, FakeFrameEncoder, FakeTextEncoder, fake_encoders

COMMANDS = ["doctor", "smoke", "audit", "folds", "prep", "shots", "peek", "extract", "check-features", "segment",
            "train", "fuse", "tune-stitch", "eval", "render", "check", "report"]


def test_help_lists_every_command():
    out = CliRunner().invoke(app, ["--help"], terminal_width=200)
    assert out.exit_code == 0
    for c in COMMANDS:
        assert c in out.output, f"`{c}` missing from bgm --help"


def test_unbuilt_command_exits_2():
    assert CliRunner().invoke(app, ["report"]).exit_code == 2


def test_config_overrides():
    cfg = load_config("configs/default.yaml", ["train.semantic.lr_finetune=3e-4", "fold=2"])
    assert cfg.train.semantic.lr_finetune == pytest.approx(3e-4) and cfg.fold == 2
    assert cfg.features.sem_music.clap_model == "laion/larger_clap_music_and_speech"
    with pytest.raises(Exception):
        load_config("configs/default.yaml", ["train.semantic.lr_finetuen=3e-4"])   # typo'd key


def test_fixtures_probe_and_durations(synth_pairs):
    assert any(p.embedded for p in synth_pairs) and any(not p.embedded for p in synth_pairs)
    for p in synth_pairs:
        info = ffprobe(p.video_path)
        v = info["video"]
        assert v["fps"] == 25 and (v["width"], v["height"]) == (320, 240)
        assert abs(v["duration_s"] - p.duration_s) < 0.041, (v["duration_s"], p.duration_s)
        if p.embedded:
            assert info["audio"] is not None
            assert abs((info["audio"]["start_s"] - v["start_s"]) - 0.12) < 0.025, info["audio"]
        else:
            assert info["audio"] is None
            y, sr = sf.read(p.music_path)
            assert sr == 44100 and abs(len(y) / sr - p.duration_s) < 0.01


def test_fixture_ground_truth(synth_pairs):
    for p in synth_pairs:
        beats, cuts = np.array(p.beat_times_s), np.array(p.cut_times_s)
        assert 70 <= p.tempo_bpm <= 180 and len(cuts) >= 1
        assert np.all(np.abs(cuts[:, None] - np.array(p.cut_beat_times_s)[None]).min(1) < 1 / 25 + 1e-9)
        assert np.isin(np.round(p.cut_beat_times_s, 6), np.round(beats, 6)).all()
        gaps = np.diff(np.r_[0.0, p.cut_beat_times_s])
        assert np.all((gaps >= 2.0 - 1e-9) & (gaps <= 5.0 + 1e-9))
        assert len(p.shot_palettes) == len(cuts) + 1 and all(a != b for a, b in zip(p.shot_palettes, p.shot_palettes[1:]))


def test_fake_encoders_deterministic_and_unit_norm():
    frames = np.random.default_rng(0).integers(0, 255, (3, 240, 320, 3), dtype=np.uint8)
    a, b = FakeFrameEncoder(64, seed=1).encode_frames(frames), FakeFrameEncoder(64, seed=1).encode_frames(frames)
    assert np.array_equal(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    t = FakeTextEncoder(32).encode_text(["calm music", "calm music", "tense music"])
    assert np.allclose(t[0], t[1]) and not np.allclose(t[0], t[2])
    enc = fake_encoders()
    assert enc["siglip_image"].dim == 1152 and enc["clap_audio"].dim == 512 and enc["mert_audio"].sr == 24000


def test_fake_audio_encoder_asserts_sample_rate():
    enc = FakeAudioEncoder(16, sr=48000)
    with pytest.raises(AssertionError):
        enc.encode_audio(np.zeros((1, 22050), np.float32), 22050)
    sine = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)[None]
    noise = np.random.default_rng(0).standard_normal((1, 48000))
    e = enc.encode_audio(np.vstack([sine, noise]), 48000)
    assert e.shape == (2, 16) and float(e[0] @ e[1]) < 0.99
