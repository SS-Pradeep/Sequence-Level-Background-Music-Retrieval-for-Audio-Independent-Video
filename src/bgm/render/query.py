"""Featurize an external (off-catalog) video exactly like the dataset: silent 25 fps copy first (the query never
exposes its soundtrack), then shots, SigLIP 1 Hz, raw mood logits, ImageBind 1 Hz, 25 Hz dynamics curves, segments.
Cached at renders/<name>/query.npz."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.data.media import ffprobe
from bgm.data.prep import scale_filter
from bgm.data.segments import make_segments
from bgm.data.shots import detect_cuts
from bgm.features.dynamics import V_CHANNELS, normalize_dyn, video_curves
from bgm.features.extract import EncoderBank, frames_1hz, word_embeddings
from bgm.features.timing import n_1hz


def silent_copy(src: Path, dst: Path, cfg: DictConfig) -> None:
    if dst.exists():
        return
    v = ffprobe(src)["video"]
    vf = [f"fps={cfg.media.video_fps}"] + ([sc] if (sc := scale_filter(v["width"], v["height"], cfg.media.video_short_side)) else [])
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-vf", ",".join(vf),
                    "-c:v", "libx264", "-crf", str(cfg.media.video_crf), "-pix_fmt", "yuv420p", str(dst)], check=True)


def featurize(src: str | Path, workdir: Path, cfg: DictConfig) -> dict:
    src, workdir = Path(src), Path(workdir)
    cache = workdir / "query.npz"
    if cache.exists():
        with np.load(cache) as z:
            d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d["meta"]))
        return d
    silent = workdir / "silent25.mp4"
    silent_copy(src, silent, cfg)
    info = ffprobe(silent)
    dur = info["video"]["duration_s"] if np.isfinite(info["video"]["duration_s"]) else info["duration_s"]
    cuts = np.array(detect_cuts(silent, cfg.shots.adaptive_threshold), float)
    frames = frames_1hz(silent, n_1hz(dur))
    bank = EncoderBank(cfg, fake=False)
    sig = bank.frames("siglip", frames)
    a = cfg.features.attributes
    W = word_embeddings(bank, "siglip", list(a.moods), list(a.video_templates))
    scale, bias = bank.logit_params("siglip")
    attr = (scale * (sig @ W.T) + bias).astype(np.float32)
    ib = bank.frames("imagebind", frames)
    del bank
    d = cfg.features.dynamics
    dyn = video_curves(silent, cuts, d.impulse_sigma_s, d.flow_short_side, d.affine_grid)
    segs = make_segments(cuts, dur, sig, cfg.segments.merge_cosine, cfg.segments.max_len_s, cfg.segments.min_shot_s)
    out = {"sem_siglip_1hz": sig.astype(np.float16), "attr_logits_1hz": attr, "imagebind_1hz": ib.astype(np.float16), "dyn_25hz": dyn,
           "cut_times_s": cuts.astype(np.float32), "segments": np.array([(s, e) for s, e, _ in segs], np.float32),
           "meta": {"source": str(src), "duration_s": dur, "video_start_s": ffprobe(src)["video"]["start_s"]}}
    np.savez(cache, **{k: v for k, v in out.items() if k != "meta"}, meta=np.array(json.dumps(out["meta"])))
    return out


def as_video_query(feat: dict):
    from bgm.retrieval.index import VideoQuery
    return VideoQuery(feat["sem_siglip_1hz"].astype(np.float32), feat["attr_logits_1hz"], normalize_dyn(feat["dyn_25hz"], V_CHANNELS),
                      np.asarray(feat["cut_times_s"], float), [tuple(map(float, s)) for s in feat["segments"]],
                      float(feat["meta"]["duration_s"]), {"parts": []})
