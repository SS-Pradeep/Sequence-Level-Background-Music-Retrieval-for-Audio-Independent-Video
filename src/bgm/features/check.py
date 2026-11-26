"""`bgm check-features` (SPEC M3): validate cached features against SPEC §4.2."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.features.extract import SR, feature_path, load_features

log = logging.getLogger(__name__)

# key -> (dtype, shape with "T1" / "T25" / int / None for free, L2-normalized rows?)
def schema(cfg: DictConfig) -> dict[str, dict]:
    M, O = len(cfg.features.attributes.moods), len(cfg.features.bridge.occasions)
    common = {"attr_logits_1hz": (np.float32, ("T1", M), False), "bridge_logits_1hz": (np.float32, ("T1", O), False),
              "imagebind_1hz": (np.float16, ("T1", 1024), True), "dyn_25hz": (np.float32, ("T25", 5), False)}
    video = {"sem_siglip_1hz": (np.float16, ("T1", 1152), True), "cut_times_s": (np.float32, (None,), False), **common}
    if cfg.features.sem_video.store_clip:
        video["sem_clip_1hz"] = (np.float16, ("T1", 768), True)
    music = {"sem_clap_1hz": (np.float16, ("T1", 512), True), "sem_mert_1hz": (np.float16, ("T1", 1024), True),
             "beat_times_s": (np.float32, (None,), False), "tempo_bpm": (np.float32, (), False),
             "key_pc": (np.int8, (), False), "key_mode": (np.int8, (), False), "tonic_pc": (np.int8, (), False), **common}
    if not cfg.features.imagebind.enabled:
        video.pop("imagebind_1hz"), music.pop("imagebind_1hz")
    return {"video": video, "music": music}


def check_file(d: dict, spec: dict) -> list[str]:
    problems = []
    meta = d.get("meta", {})
    if not meta:
        return ["missing file or meta"]
    if meta.get("sr") != SR:
        problems.append(f"sample rates in meta {meta.get('sr')} != {SR}")
    dims = {"T1": meta.get("T1"), "T25": meta.get("T25")}
    for key, (dtype, shape, unit) in spec.items():
        if key not in d:
            problems.append(f"missing {key}")
            continue
        a = d[key]
        if a.dtype != dtype:
            problems.append(f"{key} dtype {a.dtype} != {np.dtype(dtype)}")
        want = tuple(dims.get(s, s) if isinstance(s, str) else s for s in shape)
        if len(a.shape) != len(want) or any(w is not None and w != g for w, g in zip(want, a.shape)):
            problems.append(f"{key} shape {a.shape} != {want}")
        if a.dtype.kind == "f" and not np.isfinite(a.astype(np.float32)).all():
            problems.append(f"{key} has NaN or inf")
        if unit and a.size:
            n = np.linalg.norm(a.astype(np.float32), axis=-1)
            if np.abs(n - 1).max() > 1e-2:
                problems.append(f"{key} not unit norm (max |norm - 1| = {np.abs(n - 1).max():.3g})")
    return problems


def run_check(cfg: DictConfig) -> dict[str, list[str]]:
    pairs = manifest.read_pairs(cfg)
    spec = schema(cfg)
    items = [("video", p) for p in pairs["pair_id"]] + [("music", m) for m in dict.fromkeys(pairs["music_id"])]
    bad = {}
    for side, i in items:
        probs = check_file(load_features(feature_path(cfg, side, i)), spec[side])
        if probs:
            bad[f"{side}/{i}"] = probs
    rep = Path(cfg.paths.reports)
    rep.mkdir(parents=True, exist_ok=True)
    lines = ["# check-features", "", f"- {len(items)} files checked, **{len(bad)} with problems**", ""]
    lines += [f"- `{k}`: {'; '.join(v)}" for k, v in list(bad.items())[:200]]
    (rep / "check_features.md").write_text("\n".join(lines) + "\n")
    log.info("check-features: %d files, %d with problems (reports/check_features.md)", len(items), len(bad))
    return bad
