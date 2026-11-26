"""M8 data (SPEC §3.6): the unlabeled pretraining set from Data/video_bgm_pairs (user-approved source, D28).

Per source video (the video's own embedded audio: its .mp3 copy runs 40 ms late):
  1. speech filter: PANNs CNN14 on 1 s frames at 32 kHz; a frame is kept if Music >= music_min and Speech < speech_max;
  2. dedupe: Chromaprint (hum-robust, verified by log-mel correlation) against every music file of the 1,248 pairs; a
     video with any link is dropped;
  3. clips: up to `clips_per_video` non-overlapping clips of 10-`clip_s` s inside runs of >= min_run_s kept frames;
  4. cut each clip once, by presentation time, into a silent 25 fps video and 48 kHz mono FLAC (so sample 0 is the
     clip's first frame), and write data/unlabeled/manifests/pairs.csv in the labeled-pair format
     (track_id = the source video, so clips of one song are masked as negatives of each other).
After this: `shots`, `extract --what sem_video,sem_music,dyn_video,dyn_music` and the CLAP dedupe on the clip features.
"""
from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from bgm.data import manifest, tracks
from bgm.data.media import ffprobe
from bgm.data.prep import scale_filter

log = logging.getLogger(__name__)
SOURCE = Path("Data/video_bgm_pairs/video")
ROOT = Path("data/unlabeled")


def unlabeled_config(cfg: DictConfig) -> DictConfig:
    """The main config with paths pointing at data/unlabeled/ (same extractors and settings)."""
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"paths.{k}={ROOT / k}" for k in ("manifests", "interim", "features", "reports")]))


def decode(path: Path, sr: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1",
                          "-af", "aresample=resampler=soxr", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


@torch.no_grad()
def tag_frames(model, y32: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(music prob, speech prob) per 1 s frame, PANNs CNN14 at 32 kHz."""
    from panns_inference import labels
    sr = 32000
    n = len(y32) // sr
    frames = y32[: n * sr].reshape(n, sr)
    im, isp = labels.index("Music"), labels.index("Speech")
    out = []
    for i in range(0, n, batch):
        clip, _ = model.inference(frames[i:i + batch])
        out.append(clip[:, [im, isp]])
    p = np.concatenate(out) if out else np.zeros((0, 2))
    return p[:, 0], p[:, 1]


def pick_clips(keep: np.ndarray, rng: np.random.Generator, clip_s: int, min_s: int, min_run_s: int, per_video: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) second spans (min_s..clip_s long) inside runs of >= min_run_s kept seconds."""
    runs, t = [], 0
    while t < len(keep):
        if keep[t]:
            u = t
            while u < len(keep) and keep[u]:
                u += 1
            if u - t >= max(min_run_s, min_s):
                runs.append((t, u))
            t = u
        else:
            t += 1
    spans = []
    for a, b in runs:
        L = b - a
        k = L // clip_s if L >= clip_s else 1
        for j in range(k):
            s0 = a + j * clip_s
            spans.append((s0, min(b, s0 + clip_s)))
    if len(spans) > per_video:
        spans = [spans[i] for i in sorted(rng.choice(len(spans), per_video, replace=False))]
    return spans


def cut_clip(src: Path, start: float, dur: float, v_out: Path, m_out: Path, cfg: DictConfig) -> None:
    v = ffprobe(src)["video"]
    vf = [f"fps={cfg.media.video_fps}"] + ([sc] if (sc := scale_filter(v["width"], v["height"], cfg.media.video_short_side)) else [])
    v_out.parent.mkdir(parents=True, exist_ok=True)
    m_out.parent.mkdir(parents=True, exist_ok=True)
    # input seeking is frame-accurate when transcoding (accurate_seek) and cuts both streams at the same presentation
    # time, so music sample 0 = the clip's first frame
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start}", "-i", str(src), "-t", f"{dur}", "-map", "0:v:0", "-an",
                    "-vf", ",".join(vf), "-c:v", "libx264", "-crf", str(cfg.media.video_crf), "-pix_fmt", "yuv420p", "-threads", "2", str(v_out)],
                   check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start}", "-i", str(src), "-t", f"{dur}", "-map", "0:a:0", "-vn",
                    "-ac", "1", "-af", "aresample=resampler=soxr", "-ar", str(cfg.media.music_sr), "-c:a", "flac", str(m_out)], check=True)


def build_unlabeled(cfg: DictConfig, clip_s: int = 20, min_s: int = 10, per_video: int = 6, workers: int = 8) -> pd.DataFrame:
    from panns_inference import AudioTagging
    u = cfg.unlabeled
    ucfg = unlabeled_config(cfg)
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    report = {"source": str(SOURCE)}
    vids = sorted(p for p in SOURCE.glob("*.mp4"))
    ok = [p for p in vids if ffprobe(p)["audio"] is not None]
    report["videos"], report["without_audio"] = len(vids), [p.name for p in vids if p not in ok]
    # 1. speech filter
    tagger = AudioTagging(checkpoint_path=None, device="cuda")
    tags = {}
    cache = ROOT / "speech_tags.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tags = z["tags"].item()
    for n, p in enumerate(ok):
        if p.stem not in tags:
            tags[p.stem] = np.stack(tag_frames(tagger, decode(p, u.panns_sr)), 1)
        if (n + 1) % 100 == 0:
            log.info("speech filter %d/%d", n + 1, len(ok))
            np.savez(cache, tags=np.array(tags, dtype=object))
    np.savez(cache, tags=np.array(tags, dtype=object))
    del tagger
    torch.cuda.empty_cache()
    # 2. dedupe against the labeled music files (Chromaprint, hum-robust + log-mel verified)
    pairs = manifest.read_pairs(cfg)
    lib_paths = [str(Path(cfg.paths.interim) / "music" / f"{m}.flac") for m in pairs.music_id]
    lib = tracks.fingerprint_all(lib_paths, 16, cfg.tracks.highpass_hz)
    wav_dir = ROOT / "tmp_audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    def to_wav(p):
        w = wav_dir / f"{p.stem}.flac"
        if not w.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(p), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "22050", str(w)], check=True)
        return str(w)
    with ThreadPoolExecutor(workers) as ex:
        wavs = list(ex.map(to_wav, ok))
    ufp = tracks.fingerprint_all(wavs, 16, cfg.tracks.highpass_hz)
    fps = ufp + lib
    cands = {k: v for k, v in tracks.candidates(fps, cfg.tracks.min_votes).items() if k[0] < len(ok) <= k[1]}
    checks = tracks.verify_spectral(tracks.check_candidates(fps, cands), fps, wavs + lib_paths)
    links = [c for c in checks if c.ber <= cfg.tracks.ber_max and c.overlap_s >= cfg.tracks.min_overlap_s and c.spec_corr >= cfg.tracks.spec_min]
    dup = {ok[c.i].stem: pairs.pair_id.iloc[c.j - len(ok)] for c in links}
    report["chromaprint_duplicates"] = dup
    # 3-4. clips
    rows, rng = [], np.random.default_rng(0)
    jobs = []
    kept_seconds = 0
    for p in ok:
        if p.stem in dup:
            continue
        pr = tags[p.stem]
        keep = (pr[:, 0] >= u.music_min) & (pr[:, 1] < u.speech_max)
        kept_seconds += int(keep.sum())
        for k, (a, b) in enumerate(pick_clips(keep, rng, clip_s, min_s, int(u.min_run_s), per_video)):
            cid = f"u{p.stem.split('.')[0]}_{k:02d}"
            jobs.append((p, float(a), float(b - a), cid))
    report["kept_seconds_total"] = kept_seconds

    def do(job):
        p, a, d, cid = job
        v_out, m_out = ROOT / "interim" / "video" / f"{cid}.mp4", ROOT / "interim" / "music" / f"{cid}.flac"
        if not (v_out.exists() and m_out.exists()):
            cut_clip(p, a, d, v_out, m_out, ucfg)
        return cid, p, a, d
    with ThreadPoolExecutor(workers) as ex:
        for cid, p, a, d in ex.map(do, jobs):
            vi = ffprobe(ROOT / "interim" / "video" / f"{cid}.mp4")
            mi = ffprobe(ROOT / "interim" / "music" / f"{cid}.flac")
            rows.append({"pair_id": cid, "video_path": str(ROOT / "interim" / "video" / f"{cid}.mp4"),
                         "music_path": str(ROOT / "interim" / "music" / f"{cid}.flac"), "music_source": "file", "music_id": cid,
                         "video_duration_s": vi["video"]["duration_s"], "music_duration_s": mi["duration_s"], "video_start_s": 0.0,
                         "audio_start_s": np.nan, "fps": 25.0, "width": vi["video"]["width"], "height": vi["video"]["height"],
                         "video_has_audio": False, "music_sr": mi["audio"]["sr"], "music_channels": 1, "track_id": f"src_{p.stem}",
                         "meta_source": "video_bgm_pairs", "meta_start_in_source_s": a})
    df = pd.DataFrame(rows)
    manifest.write_pairs(df, ucfg)
    manifest.write_tracks(df.groupby("track_id").agg(music_ids=("music_id", ";".join)).reset_index().assign(evidence="source-video"), ucfg)
    report.update({"clips": len(df), "source_videos_used": int(df.track_id.nunique()), "hours": float(df.video_duration_s.sum() / 3600)})
    (ROOT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("unlabeled: %d clips from %d videos (%.1f h); %d videos dropped as duplicates of the labeled set",
             len(df), report["source_videos_used"], report["hours"], len(dup))
    return df


def clap_dedupe(cfg: DictConfig, threshold: float = 0.90) -> list[str]:
    """Drop unlabeled clips whose CLAP file-mean cosine to any labeled music file is >= threshold (SPEC §3.6)."""
    from bgm.features.extract import feature_path, load_features
    ucfg = unlabeled_config(cfg)
    lab = manifest.read_pairs(cfg)
    L = np.stack([load_features(feature_path(cfg, "music", m))["sem_clap_1hz"].astype(np.float64).mean(0) for m in lab.music_id])
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    un = manifest.read_pairs(ucfg)
    drop = []
    for m in un.music_id:
        e = load_features(feature_path(ucfg, "music", m))["sem_clap_1hz"].astype(np.float64).mean(0)
        if float((L @ (e / np.linalg.norm(e))).max()) >= threshold:
            drop.append(m)
    keep = un[~un.music_id.isin(drop)]
    manifest.write_pairs(keep, ucfg)
    manifest.write_tracks(keep.groupby("track_id").agg(music_ids=("music_id", ";".join)).reset_index().assign(evidence="source-video"), ucfg)
    rep = json.loads((ROOT / "report.json").read_text())
    rep.update({"clap_duplicates_dropped": drop, "clips_after_dedupe": int(len(keep)), "hours_after_dedupe": float(keep.video_duration_s.sum() / 3600)})
    (ROOT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    log.info("CLAP dedupe: %d of %d clips dropped (cosine >= %.2f to a labeled file)", len(drop), len(un), threshold)
    return drop
