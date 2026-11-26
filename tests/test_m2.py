"""M2: media prep (A/V offset correction, silent interim video) and shot detection, on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as ss
import soundfile as sf

from bgm.config import load_config
from bgm.data.audit import probe_row
from bgm.data.media import ffprobe
from bgm.data.prep import interim_music_path, interim_video_path, prep_pair
from bgm.data.shots import detect_cuts

FRAME = 1 / 25


@pytest.fixture(scope="module")
def prepped(synth_pairs, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("interim")
    cfg = load_config("configs/default.yaml", [f"paths.interim={tmp}"])
    rows = [probe_row({"pair_id": p.pair_id, "video_path": p.video_path, "music_path": p.music_path or ""}) for p in synth_pairs]
    for r in rows:
        prep_pair(r, cfg)
    return cfg, rows


def accent_times(y: np.ndarray, sr: int) -> np.ndarray:
    """Times of the accented cut clicks: 2.6-4.5 kHz energy peaks (beat clicks are at 1.5 kHz)."""
    env = np.abs(ss.sosfiltfilt(ss.butter(4, [2600, 4500], "bp", fs=sr, output="sos"), y))
    env = np.convolve(env, np.ones(int(0.005 * sr)) / int(0.005 * sr), "same")
    peaks, _ = ss.find_peaks(env, height=0.4 * env.max(), distance=int(0.5 * sr))
    return peaks / sr


def test_interim_video_is_silent_constant_fps_and_same_length(prepped, synth_pairs):
    cfg, rows = prepped
    for p, r in zip(synth_pairs, rows):
        info = ffprobe(interim_video_path(cfg, p.pair_id))
        assert info["audio"] is None
        assert info["video"]["fps"] == 25
        assert abs(info["video"]["duration_s"] - r["video_duration_s"]) <= 0.1
        assert (info["video"]["width"], info["video"]["height"]) == (320, 240)   # never upscaled (D12)


def test_music_is_mono_48k_and_embedded_offset_is_corrected(prepped, synth_pairs):
    cfg, rows = prepped
    checked = 0
    for p, r in zip(synth_pairs, rows):
        y, sr = sf.read(interim_music_path(cfg, r["music_id"]), dtype="float32")
        assert sr == 48000 and y.ndim == 1
        assert r["music_source"] == ("embedded" if p.embedded else "file")
        acc = accent_times(y, sr)
        cuts = np.array(p.cut_times_s)
        err = np.abs(acc[None, :] - cuts[:, None]).min(1)
        assert np.all(err <= FRAME + 1e-3), (p.pair_id, err)       # every accent within +/-1 frame of its cut
        if p.embedded:
            checked += 1
            assert abs(r["audio_start_s"] - r["video_start_s"] - 0.12) < 0.025
            # without the correction the accents would sit 0.12 s early: the test can see the offset
            assert np.all(np.abs((acc - 0.12)[None, :] - cuts[:, None]).min(1) > FRAME)
    assert checked >= 1


def test_shot_detection_precision_recall(prepped, synth_pairs):
    cfg, _ = prepped
    tp = n_true = n_pred = 0
    for p in synth_pairs:
        pred = np.array(detect_cuts(interim_video_path(cfg, p.pair_id)))
        true = np.array(p.cut_times_s)
        n_true, n_pred = n_true + len(true), n_pred + len(pred)
        tp += sum(np.any(np.abs(pred - t) <= FRAME + 1e-6) for t in true)
    precision, recall = tp / max(n_pred, 1), tp / n_true
    assert precision >= 0.95 and recall >= 0.95, (precision, recall)


def test_prep_video_hwaccel_decode_only_adds_the_decoder_flag(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from bgm.data import prep
    calls = []
    monkeypatch.setattr(prep.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(prep.os, "replace", lambda a, b: None)
    cfg = OmegaConf.load("configs/default.yaml")
    row = {"video_path": "in.mp4", "width": 3840, "height": 2160}
    prep.prep_video(row, tmp_path / "out.mp4", cfg)
    cfg.media.hwaccel_decode = "cuda"
    prep.prep_video(row, tmp_path / "out.mp4", cfg)
    plain, hw = calls
    assert "-hwaccel" not in plain and hw[hw.index("-hwaccel"):hw.index("-hwaccel") + 2] == ["-hwaccel", "cuda"]
    assert [x for x in hw if x not in ("-hwaccel", "cuda")] == plain          # nothing else changes
