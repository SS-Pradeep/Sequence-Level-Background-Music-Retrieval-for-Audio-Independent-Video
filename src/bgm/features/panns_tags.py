"""PANNs CNN14 AudioSet tag distributions per second of music (D111; the KL metric of the evaluation suite).

For second k of a music file: the frozen CNN14 clipwise sigmoid output (527 AudioSet classes) on the 4 s of audio
centred on the second's centre k + 0.5 (zero-padded at the file edges), at 32 kHz. Cached once per music file in
data/features/music_panns/<music_id>.npz with the config hash (CLAUDE.md rule 9).

Run: python -m bgm.features.panns_tags [--highpass] [key=value ...] [music_id ...]   (default: every music file of the
dataset pairs; overrides as for bgm commands)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

log = logging.getLogger(__name__)
SR = 32000
CONFIG = {"model": "Cnn14_mAP=0.431", "sr": SR, "context_s": 4.0, "hop_s": 1.0, "output": "clipwise sigmoid, 527"}
CONFIG_HASH = hashlib.sha1(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def panns_path(cfg, music_id: str) -> Path:
    return Path(cfg.paths.features) / "music_panns" / f"{music_id}.npz"


def load_audio_32k(path: Path) -> np.ndarray:
    """Interim 48 kHz mono FLAC -> float32 mono at 32 kHz (soxr)."""
    import soxr
    y, sr = sf.read(path, dtype="float32")
    assert sr == 48000, f"interim music must be 48 kHz, got {sr}"
    y = y.mean(1) if y.ndim == 2 else y
    out = soxr.resample(y, sr, SR).astype(np.float32)
    return out


def frames_1hz(y32: np.ndarray, n_seconds: int, context_s: float = CONFIG["context_s"]) -> np.ndarray:
    """[T1, context_s * 32000]: the audio centred on each second's centre k + 0.5, zero-padded at the edges."""
    L = int(round(context_s * SR))
    pad = L
    yp = np.concatenate([np.zeros(pad, np.float32), y32, np.zeros(pad, np.float32)])
    starts = [int(round((k + 0.5) * SR)) - L // 2 + pad for k in range(n_seconds)]
    return np.stack([yp[s:s + L] for s in starts]) if n_seconds else np.zeros((0, L), np.float32)


@torch.no_grad()
def tag_file(model, y32: np.ndarray, n_seconds: int, batch: int = 64) -> np.ndarray:
    """[T1, 527] float32 tag probabilities, one row per second."""
    assert model is not None
    fr = frames_1hz(y32, n_seconds)
    out = [model.inference(fr[i:i + batch])[0] for i in range(0, len(fr), batch)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 527), np.float32)


def load_tags(cfg, music_id: str) -> np.ndarray:
    with np.load(panns_path(cfg, music_id), allow_pickle=True) as z:
        assert str(z["config_hash"]) == CONFIG_HASH, f"{music_id}: stale PANNs cache"
        return z["panns_probs_1hz"].astype(np.float32)


def extract(cfg, music_ids: list[str]) -> None:
    from panns_inference import AudioTagging
    from bgm.data.prep import interim_music_path
    from bgm.eval.store import FeatureStore
    store = FeatureStore(cfg)
    todo = [m for m in music_ids if not panns_path(cfg, m).exists()
            or str(np.load(panns_path(cfg, m), allow_pickle=True)["config_hash"]) != CONFIG_HASH]
    log.info("PANNs tags: %d of %d music files to extract (config %s)", len(todo), len(music_ids), CONFIG_HASH)
    if not todo:
        return
    model = AudioTagging(checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()        # fp32: under fp16 autocast CNN14's log-mel front end overflows to NaN (checked, D111)
    for n, m in enumerate(todo):
        T1 = len(store.music(m)["sem_clap_1hz"])                 # one row per 1 Hz sample, as every other 1 Hz key
        probs = tag_file(model, load_audio_32k(interim_music_path(cfg, m)), T1)
        p = panns_path(cfg, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, panns_probs_1hz=probs.astype(np.float16), config_hash=CONFIG_HASH, config=json.dumps(CONFIG))
        if (n + 1) % 200 == 0:
            log.info("PANNs tags %d/%d", n + 1, len(todo))


# ---------------------------------------------------------------- hum-robust variant (D114)
# Same tagger on the music with everything below 150 Hz removed (4th-order Butterworth high-pass, zero phase) and
# the file's RMS set to -20 dBFS, so mains hum (Konjum salangai: 100 Hz, D-hum) does not mask the music.
# Separate cache: data/features/music_panns_hp150/. Analysis only (results/bgm_presence.md).
HP_CONFIG = {**CONFIG, "highpass_hz": 150.0, "highpass_order": 4, "zero_phase": True, "file_rms_dbfs": -20.0}
HP_CONFIG_HASH = hashlib.sha1(json.dumps(HP_CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def panns_hp_path(cfg, music_id: str) -> Path:
    return Path(cfg.paths.features) / "music_panns_hp150" / f"{music_id}.npz"


def highpass_normalize(y32: np.ndarray, hz: float = HP_CONFIG["highpass_hz"],
                       rms_dbfs: float = HP_CONFIG["file_rms_dbfs"]) -> np.ndarray:
    """32 kHz mono -> high-passed (zero-phase Butterworth) and scaled to the target file RMS (dBFS), clipped to ±1."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(HP_CONFIG["highpass_order"], hz, "highpass", fs=SR, output="sos")
    yh = sosfiltfilt(sos, y32.astype(np.float64))
    rms = np.sqrt(np.mean(yh ** 2)) if len(yh) else 0.0
    if rms > 1e-9:
        yh = yh * (10 ** (rms_dbfs / 20) / rms)
    return np.clip(yh, -1, 1).astype(np.float32)


def load_tags_hp(cfg, music_id: str) -> np.ndarray:
    with np.load(panns_hp_path(cfg, music_id), allow_pickle=True) as z:
        assert str(z["config_hash"]) == HP_CONFIG_HASH, f"{music_id}: stale high-pass PANNs cache"
        return z["panns_probs_1hz"].astype(np.float32)


def extract_hp(cfg, music_ids: list[str]) -> None:
    from panns_inference import AudioTagging
    from bgm.data.prep import interim_music_path
    todo = [m for m in music_ids if not panns_hp_path(cfg, m).exists()
            or str(np.load(panns_hp_path(cfg, m), allow_pickle=True)["config_hash"]) != HP_CONFIG_HASH]
    log.info("PANNs tags, 150 Hz high-pass: %d of %d music files to extract (config %s)", len(todo), len(music_ids),
             HP_CONFIG_HASH)
    if not todo:
        return
    model = AudioTagging(checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu")
    model.model.eval()        # fp32, as extract()
    for n, m in enumerate(todo):
        T1 = len(load_tags(cfg, m))                                # same rows as the plain cache
        probs = tag_file(model, highpass_normalize(load_audio_32k(interim_music_path(cfg, m))), T1)
        p = panns_hp_path(cfg, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, panns_probs_1hz=probs.astype(np.float16), config_hash=HP_CONFIG_HASH, config=json.dumps(HP_CONFIG))
        if (n + 1) % 200 == 0:
            log.info("PANNs high-pass tags %d/%d", n + 1, len(todo))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from bgm.config import load_config
    from bgm.data import manifest
    args = sys.argv[1:]
    hp = bool(args) and args[0] == "--highpass"
    args = args[1:] if hp else args
    cfg = load_config("configs/default.yaml", [a for a in args if "=" in a])     # e.g. paths.features=cleaned_dataset/features
    args = [a for a in args if "=" not in a]
    ids = args or sorted(set(manifest.read_pairs(cfg).music_id))
    (extract_hp if hp else extract)(cfg, ids)
