"""`bgm prep` (SPEC §3.4): silent 25 fps interim videos and A/V-aligned 48 kHz mono FLAC music.

* Video: `interim/video/<pair_id>.mp4`, constant 25 fps, no audio, H.264 CRF 18, short side scaled down to
  `media.video_short_side` when larger (never up, PROGRESS.md D12). Frame 0 is the first video frame.
* Music: `interim/music/<music_id>.flac`, mono, `media.music_sr`. Sample 0 is the instant of the first video
  frame: for embedded audio, offset = audio_start_s - video_start_s; a positive offset prepends silence, a
  negative one trims the start. Separate music files are assumed aligned (offset 0; M4 tests this).
Writes are atomic (temp file + rename), so an interrupted run resumes cleanly.
"""
from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.data.media import ffprobe

log = logging.getLogger(__name__)


def interim_video_path(cfg: DictConfig, pair_id: str) -> Path:
    return Path(cfg.paths.interim) / "video" / f"{pair_id}.mp4"


def interim_music_path(cfg: DictConfig, music_id: str) -> Path:
    return Path(cfg.paths.interim) / "music" / f"{music_id}.flac"


def scale_filter(width: int, height: int, short_side: int) -> str | None:
    """ffmpeg scale filter bringing the short side down to `short_side` (even dims), or None if already <=."""
    if min(width, height) <= short_side:
        return None
    return f"scale=-2:{short_side}" if height <= width else f"scale={short_side}:-2"


def av_offset_s(row: dict) -> float:
    """Where music sample 0 sits relative to the first video frame (s). Positive: the audio starts later."""
    if row["music_source"] != "embedded":
        return 0.0
    return float(row["audio_start_s"]) - float(row["video_start_s"])


def prep_video(row: dict, dst: Path, cfg: DictConfig) -> None:
    vf = [f"fps={cfg.media.video_fps}"]
    sc = scale_filter(int(row["width"]), int(row["height"]), cfg.media.video_short_side)
    if sc:
        vf.append(sc)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    hw = cfg.media.get("hwaccel_decode")              # GPU decoding only: the encoded output is byte-identical (D49)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + (["-hwaccel", hw] if hw else []) + ["-i", row["video_path"], "-map", "0:v:0", "-an",
                    "-vf", ",".join(vf), "-c:v", "libx264", "-crf", str(cfg.media.video_crf), "-pix_fmt", "yuv420p",
                    "-threads", "2", str(tmp)], check=True, capture_output=True)
    os.replace(tmp, dst)


def decode_mono(path: str, sr: int, audio_stream: bool = False) -> np.ndarray:
    """Decode to mono float32 at `sr` Hz with the soxr resampler."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path] + (["-map", "0:a:0"] if audio_stream else []) + \
          ["-vn", "-ac", "1", "-af", "aresample=resampler=soxr", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, np.float32)


def prep_music(row: dict, dst: Path, cfg: DictConfig) -> float:
    """Returns the applied offset (s)."""
    sr = cfg.media.music_sr
    embedded = row["music_source"] == "embedded"
    y = decode_mono(row["video_path"] if embedded else row["music_path"], sr, audio_stream=embedded)
    off = av_offset_s(row)
    n = int(round(abs(off) * sr))
    y = np.concatenate([np.zeros(n, np.float32), y]) if off > 0 else y[n:]
    tmp = dst.with_name(dst.stem + ".tmp.flac")
    sf.write(tmp, y, sr, subtype="PCM_24", format="FLAC")
    os.replace(tmp, dst)
    return off


def prep_pair(row: dict, cfg: DictConfig, force: bool = False) -> dict:
    v_dst, m_dst = interim_video_path(cfg, row["pair_id"]), interim_music_path(cfg, row["music_id"])
    v_dst.parent.mkdir(parents=True, exist_ok=True)
    m_dst.parent.mkdir(parents=True, exist_ok=True)
    if force or not v_dst.exists():
        prep_video(row, v_dst, cfg)
    off = av_offset_s(row)
    if force or not m_dst.exists():
        off = prep_music(row, m_dst, cfg)
    info = ffprobe(v_dst)
    assert info["audio"] is None, f"interim video {v_dst} has an audio stream"
    dv = info["video"]["duration_s"] if np.isfinite(info["video"]["duration_s"]) else info["duration_s"]
    assert abs(dv - float(row["video_duration_s"])) <= 0.1, f"interim duration {dv:.3f} vs original {row['video_duration_s']:.3f}"
    return {"pair_id": row["pair_id"], "interim_duration_s": dv, "offset_s": off}


def run_prep(cfg: DictConfig, force: bool = False) -> pd.DataFrame:
    pairs = manifest.read_pairs(cfg)
    rows = pairs.to_dict("records")
    done, failures = [], []
    with ThreadPoolExecutor(cfg.media.prep_workers) as ex:
        futs = [(r, ex.submit(prep_pair, r, cfg, force)) for r in rows]
        for k, (r, f) in enumerate(futs):
            try:
                done.append(f.result())
            except Exception as e:  # noqa: BLE001 -- every failure is reported, never silently skipped
                msg = e.stderr.decode(errors="replace").strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
                failures.append((r["pair_id"], f"{type(e).__name__}: {msg}"))
            if (k + 1) % 200 == 0:
                log.info("prep %d/%d", k + 1, len(rows))
    rep = Path(cfg.paths.reports)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "prep_failures.md").write_text("# Prep failures\n\n" + ("\n".join(f"- `{p}`: {e}" for p, e in failures) or "none") + "\n")
    log.info("prep: %d ok, %d failed (reports/prep_failures.md)", len(done), len(failures))
    return pd.DataFrame(done)
