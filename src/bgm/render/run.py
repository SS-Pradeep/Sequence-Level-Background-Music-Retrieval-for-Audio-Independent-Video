"""`bgm render --video in.mp4 --out out.mp4 [--run runs/pipeline/fold<k>/seed<s>]` (SPEC M10, §5.9-5.11, §5.13).

Uses one trained run (semantic heads, dynamics encoders, the fold's attribute calibration, its fusion.json and its
tuned stitch.json) against the whole library. Deployment models trained on all pairs (`bgm train --all`) are not
built yet, so a run directory is used (default fold 0 / seed 0; an off-catalog video is leak-free for every fold).
Long videos (>= `long_s`): the coarse shortlist also takes the top-K files of the chunked 15-model ensemble
(bgm.analysis.long_video). Writes the rendered video and renders/<name>.plan.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig

from bgm.data.media import ffprobe
from bgm.data.prep import interim_music_path
from bgm.eval.store import FeatureStore, Series1Hz
from bgm.render.query import as_video_query, featurize
from bgm.render.render import SR, mux, render_soundtrack, runs_from_path, seam_loudness_jumps
from bgm.retrieval.index import FoldModels, MusicIndex
from bgm.retrieval.pipeline import coarse_shortlist, fine
from bgm.retrieval.stitch_eval import build_lattice, chosen_candidates, decode_lattice

log = logging.getLogger(__name__)


def render_library(cfg: DictConfig, store: FeatureStore) -> list[str]:
    """Library music ids for a render, minus any matching `render.exclude_music` (regexes; [] = whole library).
    Opt-in: e.g. the Konjum salangai files, whose audio is mostly DC offset + 100 Hz hum (PROGRESS.md D9, D114)."""
    import re
    lib = list(dict.fromkeys(store.pairs.music_id))
    pats = [re.compile(p) for p in (cfg.render.get("exclude_music") or [])]
    if pats:
        n = len(lib)
        lib = [m for m in lib if not any(p.search(m) for p in pats)]
        log.info("render library: excluded %d of %d music files by render.exclude_music", n - len(lib), n)
    return lib


def long_video_fold(source: str, pair_ids, fold: int) -> int | None:
    """Folds whose heads may score this video in the long-video shortlist: a dataset clip only its render run's fold
    (other folds' heads trained on its film: test-fold leak, 2026-11-17); an outside video every fold (None)."""
    return fold if Path(source).stem in set(pair_ids) else None


def chunked_ensemble_top(cfg, store, feat, K: int, fold: int | None = None) -> list[str]:
    """Top-K library files by the chunked ensemble (SigLIP + CLAP+MERT heads) of every fold, or of `fold` only."""
    from bgm.analysis.long_video import Pooler, load_models, scores_for
    from bgm.retrieval.chunked import ensemble
    lib = render_library(cfg, store)
    m_series = {t: (store.series("music", t, "sem_clap_1hz"), store.series("music", t, "sem_mert_1hz")) for t in lib}
    dur = {t: store.duration("music", t) for t in lib}
    q_dur = float(feat["meta"]["duration_s"])
    pool = Pooler({"q": Series1Hz(feat["sem_siglip_1hz"].astype(np.float32))}, m_series, dur)
    wcache, sc = {}, []
    for _, model in load_models(cfg, fold):
        sc.append(scores_for(model, pool, "q", q_dur, lib, wcache)[1])
    return [lib[i] for i in np.argsort(-ensemble(sc))[:K]]


def plan_for(cfg: DictConfig, store: FeatureStore, feat: dict, run: Path, long_s: float = 60.0,
             brief: list[dict] | None = None) -> dict:
    fold, seed = int(run.parent.name[4:]), int(run.name[4:])
    fusion = json.loads((run / "fusion.json").read_text())
    stitch = json.loads((run / "stitch.json").read_text())
    over = cfg.render.get("penalty_overrides") or {}
    pen = {**stitch["penalties"], **{k: float(v) for k, v in over.items() if v is not None}}   # user-facing render choice (e.g. fewer switches)
    if cfg.get("person", {}).get("enabled", False):          # opt-in (D87): a person-branch system from its eval run
        from bgm.person.render import person_setup
        models, index, vq = person_setup(cfg, store, feat, run)
    else:
        models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant)   # honours pipeline.sem_seeds (D75)
        index = MusicIndex(store, models, render_library(cfg, store))
        vq = as_video_query(feat)
        vq.prepare(models, store.min_span)
    K, J, hop = fusion["K"], fusion["J"], fusion["window_hop_s"]
    w_coarse, w_fine = dict(fusion["w_coarse"]), dict(fusion["w_fine"])
    pace_w = float((cfg.render.get("pace_match") or {}).get("weight", 0.0))
    brief_w = float((cfg.render.get("scene_brief") or {}).get("weight", 0.0))
    if pace_w > 0:                                            # opt-in (D100): fast video -> energetic music
        from bgm.render import pace
        index = pace.setup(store, index, feat)
        w_coarse["pace"] = w_fine["pace"] = pace_w
    if brief_w > 0:                                           # opt-in (D101): the scene brief's music text, via CLAP
        from bgm.render import scene
        index = scene.setup(cfg, index, brief)
        w_coarse["brief"] = w_fine["brief"] = brief_w
    len_w = float((cfg.render.get("length_fit") or {}).get("weight", 0.0))
    if len_w > 0:                                             # opt-in (D105): music long enough for the segment, no loops
        from bgm.render.branches import LengthIndex
        index = LengthIndex(index)
        w_coarse["length"] = w_fine["length"] = len_w
    if pace_w > 0 or brief_w > 0 or len_w > 0:
        from bgm.render.branches import branch_shortlist
        fusion = {**fusion, "w_coarse": w_coarse, "w_fine": w_fine}
        short = branch_shortlist(index, vq, w_coarse, K)
    else:
        short = coarse_shortlist(index, vq, w_coarse, K)
    extra = []
    if vq.duration >= long_s:
        extra = chunked_ensemble_top(cfg, store, feat, K, long_video_fold(feat["meta"]["source"], store.pairs.pair_id, fold))
        short = sorted(set(short) | {index.pos[m] for m in extra})
    results = fine(index, vq, short, hop)
    cap = cfg.render.get("max_windows_per_file")              # opt-in (D101): keep other songs among each segment's candidates
    lat = build_lattice(index, vq, results, w_fine, J, int(cap) if cap else None)
    cue, replay = None, None
    if cfg.cue.get("cuts_render", {}).get("enabled", False):  # opt-in (D104): learned per-candidate change costs at shot cuts
        from bgm.cue.cut_hook import cut_decode
        picks, cue = cut_decode(cfg, store, index, lat, feat, results, fold, stitch["beam"])
        from bgm.retrieval.stitch_eval import decode_lattice as _dl
        decode = lambda L: _dl(L, pen, stitch["beam"], stitch["tol_s"])
    elif cfg.get("cue", {}).get("enabled", False):             # opt-in (D78): learned change probabilities replace the fixed penalties
        from bgm.cue.hook import cue_decoder
        decode, cue = cue_decoder(cfg, feat, len(lat), fold, stitch)
        picks = decode(lat)
    else:
        decode = lambda L: decode_lattice(L, pen, stitch["beam"], stitch["tol_s"])
        picks = decode(lat)
    replay_pen = float(cfg.render.get("replay_penalty", 0.0) or 0.0)
    if replay_pen > 0:                                        # opt-in (D101): no passage played again and again
        from bgm.render.replay import decode_without_replays
        picks, lat, replay = decode_without_replays(decode, lat, replay_pen, int(cfg.render.get("max_plays", 1)))
    chosen = chosen_candidates(index, lat, picks)
    segs = []
    for r, c in zip(results, chosen):
        raw = index.window_scores(vq.segment(r.a, r.b), np.array([index.pos[c.music_id]]), np.array([c.offset]), c.dur)
        z = {b: float((raw[b][0] - r.stats[b][0]) / r.stats[b][1]) for b in raw}
        segs.append({"start_s": r.a, "end_s": r.b, "music_id": c.music_id, "track_id": store.music_track[c.music_id],
                     "offset_s": c.offset, "fused": -c.cost, "z": z, "key": c.key, "tempo": c.tempo})
    runs = runs_from_path(chosen, [(r.a, r.b) for r in results])
    return {"video": feat["meta"]["source"], "duration_s": vq.duration, "run": str(run), "fold": fold, "seed": seed,
            "fusion": fusion, "penalties": pen, "tuned_penalties": stitch["penalties"], "shortlist_size": len(short), "long_video_extra": extra,
            **({"scene_brief": brief} if brief else {}), **({"replay": replay} if replay else {}), "segments": segs, "pieces": [r.__dict__ for r in runs], **({"cue": cue} if cue is not None else {})}


def render_video(cfg: DictConfig, video: str, out: str, run: str | None = None) -> Path:
    video, out = Path(video), Path(out)
    run = Path(run or Path(cfg.paths.runs) / "pipeline" / "fold0" / "seed0")
    name = out.stem
    work = Path(cfg.paths.renders) / name
    work.mkdir(parents=True, exist_ok=True)
    feat = featurize(video, work, cfg)
    store = FeatureStore(cfg)
    brief = None
    if float((cfg.render.get("scene_brief") or {}).get("weight", 0.0)) > 0:
        from bgm.render.scene import brief_path, ensure_brief
        brief = ensure_brief(cfg, brief_path(cfg, out), work / "silent25.mp4", feat["segments"], str(video))
    plan = plan_for(cfg, store, feat, run, brief=brief)
    plan["video"] = str(video)                               # the path used now (the feature cache may predate a move)
    from bgm.render.render import Run, beat_sync_runs, fill_short_runs
    runs = [Run(**p) for p in plan["pieces"]]
    if cfg.render.beat_sync:
        runs, sync = beat_sync_runs(runs, feat["cut_times_s"], lambda m: store.music(m)["beat_times_s"],
                                    lambda m: store.duration("music", m), cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
        plan["pieces_retrieved"], plan["pieces"], plan["beat_sync"] = plan["pieces"], [r.__dict__ for r in runs], sync
        log.info("beat sync: %d pieces, median shift %.2f s; inner cuts on a beat %d -> %d of %d", len(runs),
                 float(np.median(np.abs([s["shift_s"] for s in sync]))), sum(s["on_beat_before"] for s in sync),
                 sum(s["on_beat_after"] for s in sync), sum(s["inner_cuts"] for s in sync))
    runs, loops = fill_short_runs(runs, lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m))
    if loops:                                                 # pieces longer than their file: loop at beats instead of silence
        plan["pieces"], plan["loops"] = [r.__dict__ for r in runs], loops
        log.info("filled %d pieces longer than their music file by looping at beats", len(loops))
    orig = ffprobe(video)["video"]
    total = orig["duration_s"] if np.isfinite(orig["duration_s"]) else ffprobe(video)["duration_s"]
    mix, seams = render_soundtrack(runs, total, lambda m: interim_music_path(cfg, m), cfg.render)
    sil = cfg.render.get("silence") or {}
    if brief and sil.get("enabled", False):                   # opt-in (D107): parts the brief leaves without music
        from bgm.render.render import silence_envelope
        from bgm.render.scene import silent_spans
        spans = silent_spans(brief, float(sil.get("min_s", 3.0)))
        if spans:
            mix = mix * silence_envelope(len(mix), spans, float(sil.get("fade_s", 1.5)))
            log.info("silence: %d spans without music %s", len(spans), [(round(a, 1), round(b, 1)) for a, b in spans])
        plan["silence"] = spans
    wav = work / "bgm.wav"
    # the soundtrack is mono; duplicated to two channels BS.1770 sums them (+3 LU), so scale by 1/sqrt(2)
    stereo = np.stack([mix, mix], 1) / np.sqrt(2)
    sf.write(wav, stereo, SR)
    import pyloudnorm as pyln
    out_lufs = float(pyln.Meter(SR).integrated_loudness(stereo.astype(np.float64)))
    out.parent.mkdir(parents=True, exist_ok=True)
    mux(video, wav, out, orig["start_s"] if np.isfinite(orig["start_s"]) else 0.0, bitrate=str(cfg.render.get("audio_bitrate", "192k")))
    plan.update({"output": str(out), "bgm_wav": str(wav), "seams_s": seams, "seam_loudness_jump_LU": seam_loudness_jumps(mix, seams),
                 "output_lufs": out_lufs, "output_peak_dbfs": float(20 * np.log10(np.abs(stereo).max() + 1e-12)),
                 "render_duration_s": len(mix) / SR})
    path = Path(cfg.paths.renders) / f"{name}.plan.json"
    path.write_text(json.dumps(plan, indent=1, default=float))
    for other in sorted(run.parent.glob("seed*")):           # plans of the fold's other seeds, for the agreement check
        if other != run and (other / "stitch.json").exists():
            p2 = plan_for(cfg, store, feat, other, brief=brief)
            (Path(cfg.paths.renders) / f"{name}.{other.name}.plan.json").write_text(json.dumps(p2, indent=1, default=float))
    log.info("render: %d segments -> %d pieces, %d switches; -> %s, %s", len(plan["segments"]), len(runs),
             sum(a["music_id"] != b["music_id"] for a, b in zip(plan["segments"], plan["segments"][1:])), out, path)
    return path
