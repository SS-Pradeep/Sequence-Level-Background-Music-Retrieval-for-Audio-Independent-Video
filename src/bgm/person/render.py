"""Render with a person-branch system (Step 4): the opt-in hook of `bgm.render.run.plan_for` (`person.enabled=true`).

`bgm render --run runs/filmsplit/person/eval/<system>/fold<k>/seed<s> person.enabled=true ...` takes the system's
spec from that run's metrics.json and its fusion.json / stitch.json as usual. The input's person arrays are computed
from a silent 25 fps copy made by the render's own `silent_copy` (rule 8) and cached in `cache_dir` (person.npz; a
face-branch system, P1, also gets the D77 face features, face.npz). The render's <work>/query.npz is not touched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.features.extract import EncoderBank, frames_1hz
from bgm.person import features
from bgm.person.pipeline import PersonMusicIndex, PersonVideoQuery, load_models
from bgm.render.query import as_video_query

log = logging.getLogger(__name__)


def cache_dir(cfg, source: str) -> Path:
    """<paths.renders>/person_cache/<stem>_<hash of the source path, size and mtime>/: the input's own silent copy and
    person arrays, independent of the output name."""
    import hashlib
    src = Path(source)
    st = src.stat()
    h = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{features.CONFIG_HASH}".encode()).hexdigest()[:10]
    return Path(cfg.paths.renders) / "person_cache" / f"{src.stem}_{h}"


def input_arrays(cfg, feat: dict, need_face: bool) -> dict:
    from bgm.render.query import silent_copy
    T1 = len(feat["sem_siglip_1hz"])
    work = cache_dir(cfg, str(feat["meta"]["source"]))
    silent = work / "silent25.mp4"
    silent_copy(Path(str(feat["meta"]["source"])), silent, cfg)
    p = work / "person.npz"
    if not p.exists():
        import torch
        dev = cfg.features.device if torch.cuda.is_available() else "cpu"
        features.save(p, features.person_arrays(features.PersonCropper(dev), EncoderBank(cfg, fake=False), frames_1hz(silent, T1)))
    out = features.load(p)
    if need_face:
        fp = work / "face.npz"
        if not fp.exists():
            import cv2
            from bgm.features.face import FaceExpression, encode_video
            rgb = frames_1hz(silent, T1)
            np.savez(fp, face_expr_1hz=encode_video(FaceExpression(), [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]))
        with np.load(fp) as z:
            out["face_expr_1hz"] = z["face_expr_1hz"]
    return out


def person_setup(cfg, store, feat: dict, run: Path):
    """-> (models, index, prepared query) for plan_for."""
    spec = json.loads((Path(run) / "metrics.json").read_text())["spec"]
    fold = int(Path(run).parent.name[4:])
    models = load_models(cfg, store, fold, spec)
    index = PersonMusicIndex(store, models, list(dict.fromkeys(store.pairs.music_id)))
    arr = input_arrays(cfg, feat, (spec.get("branch") or {}).get("kind") == "face")
    vq = PersonVideoQuery.wrap(as_video_query(feat), person_emb=arr["person_siglip_1hz"].astype(np.float32),
                               person_has=arr["person_kind_1hz"] > 0, face_1hz=arr.get("face_expr_1hz"))
    vq.prepare(models, store.min_span)
    kinds = arr["person_kind_1hz"]
    log.info("person branch: %s, %d s: face %.0f%%, person box %.0f%%, none %.0f%%", spec, len(kinds), 100 * (kinds == 1).mean(),
             100 * (kinds == 2).mean(), 100 * (kinds == 0).mean())
    return models, index, vq
