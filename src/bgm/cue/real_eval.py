"""`bgm cue real-eval` (Step 5): boundary metrics on real annotated multi-cue excerpts.

Annotation CSV (template: data/annotations/cue_changes_template.csv, header only; the user fills it in):
    youtube_id         the local file <media>/<youtube_id>.mp4 (nothing is downloaded)
    start, end         the excerpt, seconds in that file
    cue_change_times   semicolon-separated seconds in the SAME timeline as start / end (not relative to start)
    film_key           optional: the dataset film the excerpt comes from (a film group id such as
                       "sunnxt:93", or a "meta_source:meta_film" key). Given and found: the excerpt is scored with the
                       fold whose TEST set holds that film (leak-free). Empty or not a dataset film: emission and cue
                       scores are averaged over all 5 fold models (user, 2026-10-19). A train-only film (Konjum) has no
                       leak-free fold and is refused.
Per excerpt: cut [start, end] to a silent copy (the query never sees its soundtrack, rule 8), featurize it exactly as
`bgm render` does (SigLIP 1 Hz, shots, flow curves, segments), retrieve with the D75 system against the whole library
(plus render's long-video shortlist, from the chosen fold's heads only when a fold is chosen), build one lattice and
decode it with every system. Averaged models: p = the mean change probability over the models; transition parameters =
the mean of their tuned values. Metrics: boundary F1 at +-0.5 / +-1 s against the annotated changes, switches / min,
and the share of annotated changes that lie within +-1 s of any segment boundary (the most a segment-level switch can
recall; risk 3 of docs/cue_switch_plan.md). There is no track truth (the music is outside the library).
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.cue.baselines import cosine_logits, load_cosine
from bgm.cue.evaluate import match_counts
from bgm.cue.hook import average_params, cue_logits, eval_params, video_tokens
from bgm.cue.scenes import content_scores, replay_cuts, scene_logits
from bgm.cue.stitch import decode_lattice_cue, prob_to_logit
from bgm.retrieval.stitch_eval import decode_lattice, greedy_lattice

log = logging.getLogger(__name__)
COLUMNS = ("youtube_id", "start", "end", "cue_change_times", "film_key")
TEMPLATE = Path("data/annotations/cue_changes_template.csv")


def write_template(path: Path = TEMPLATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(",".join(COLUMNS) + "\n")
    return path


def read_annotations(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            missing = [c for c in COLUMNS[:4] if c not in r]
            assert not missing, f"{path}: missing columns {missing}"
            start, end = float(r["start"]), float(r["end"])
            assert end > start, f"row {i + 2}: end <= start"
            times = sorted(float(t) for t in (r["cue_change_times"] or "").replace(",", ";").split(";") if t.strip())
            outside = [t for t in times if not start < t < end]
            assert not outside, (f"row {i + 2} ({r['youtube_id']}): change times {outside} are outside ({start}, {end}); "
                                 "they must be seconds in the source video's timeline, like start / end")
            rows.append({"youtube_id": r["youtube_id"].strip(), "start": start, "end": end, "changes": [t - start for t in times],
                         "film_key": (r.get("film_key") or "").strip()})
    return rows


def fold_for(film_key: str, cfg) -> int | None:
    """The fold whose test set holds the film, or None (not a dataset film)."""
    if not film_key:
        return None
    from bgm.data import manifest
    folds, pairs = manifest.read_folds(cfg), manifest.read_pairs(cfg)
    fg = folds["film_group"]
    keys = dict(zip(pairs.meta_source + ":" + pairs.meta_film.astype(str), pairs.pair_id))
    group = film_key if film_key in set(fg.values()) else fg.get(keys.get(film_key, ""), None)
    if group is None:
        return None
    assert group not in set(folds.get("train_only", [])), f"{film_key} is train-only in every fold: no leak-free fold"
    hit = [int(k) for k, f in folds["folds"].items() if any(fg[p] == group for p in f["test"])]
    assert len(hit) == 1, (film_key, hit)
    return hit[0]


def cut_excerpt(src: Path, start: float, end: float, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                        "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def load_models(cfg, store, folds: list[int]):
    """The D75 emission models: one fold's 10-seed ensemble, or every fold's heads as one ensemble (mean cosine)."""
    from bgm.eval.store import fold_ids
    from bgm.retrieval.index import EnsembleSemantic, FoldModels
    ms = [FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=fold_ids(cfg, k)["train"])
          for k in folds]
    if len(ms) == 1:
        return ms[0]
    members = [m for fm in ms for m in (fm.sem.models if isinstance(fm.sem, EnsembleSemantic) else [fm.sem])]
    m0 = ms[0]
    return FoldModels(-1, 0, EnsembleSemantic(members), None, m0.cal, m0.device)


def lattice_for(cfg, store, feat, models, fusion, folds: list[int]) -> tuple[list, list]:
    """render.plan_for's retrieval (coarse shortlist + long-video shortlist + fine + lattice)."""
    from bgm.analysis.long_video import Pooler
    from bgm.analysis.long_video import load_models as chunk_models
    from bgm.analysis.long_video import scores_for
    from bgm.eval.store import Series1Hz
    from bgm.render.query import as_video_query
    from bgm.retrieval.chunked import ensemble
    from bgm.retrieval.index import MusicIndex
    from bgm.retrieval.pipeline import coarse_shortlist, fine
    from bgm.retrieval.stitch_eval import build_lattice
    index = MusicIndex(store, models, list(dict.fromkeys(store.pairs.music_id)))
    vq = as_video_query(feat)
    vq.prepare(models, store.min_span)
    K, J, hop = fusion["K"], fusion["J"], fusion["window_hop_s"]
    short = coarse_shortlist(index, vq, fusion["w_coarse"], K)
    if vq.duration >= 60.0:                                     # render's long-video rule, restricted to the folds used
        lib = list(dict.fromkeys(store.pairs.music_id))
        m_series = {t: (store.series("music", t, "sem_clap_1hz"), store.series("music", t, "sem_mert_1hz")) for t in lib}
        pool = Pooler({"q": Series1Hz(feat["sem_siglip_1hz"].astype(np.float32))}, m_series, {t: store.duration("music", t) for t in lib})
        wcache, sc = {}, []
        for k in folds:
            for _, m in chunk_models(cfg, fold=k):
                sc.append(scores_for(m, pool, "q", vq.duration, lib, wcache)[1])
        short = sorted(set(short) | {index.pos[lib[i]] for i in np.argsort(-ensemble(sc))[:K]})
    results = fine(index, vq, short, hop)
    return build_lattice(index, vq, results, fusion["w_fine"], J), results


def run_real_eval(cfg: DictConfig, csv_path: str, media: str, out: str) -> Path:
    from bgm.eval.store import FeatureStore
    from bgm.render.query import featurize
    write_template()
    rows = read_annotations(csv_path)
    assert rows, f"{csv_path} has no rows: fill it in (template {TEMPLATE})"
    missing = [r["youtube_id"] for r in rows if not (Path(media) / f"{r['youtube_id']}.mp4").exists()]
    assert not missing, f"local videos missing under {media}/ (nothing is downloaded): {missing}"
    store = FeatureStore(cfg)
    sc, st = cfg.cue.scene, cfg.stitch
    beam = st.beam_factor * cfg.retrieval.fine_top_j
    results_all = []
    for r in rows:
        fold = fold_for(r["film_key"], cfg)
        folds = [fold] if fold is not None else list(range(cfg.splits.n_folds))
        name = f"{r['youtube_id']}_{r['start']:.0f}_{r['end']:.0f}"
        work = Path(cfg.paths.renders) / "cue_real" / name
        clip = cut_excerpt(Path(media) / f"{r['youtube_id']}.mp4", r["start"], r["end"], work / "excerpt.mp4")
        feat = featurize(clip, work, cfg)
        em = [Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{k}" / "seed0" for k in folds]
        fusion = json.loads((em[0] / "fusion.json").read_text())
        models = load_models(cfg, store, folds)
        layers, res = lattice_for(cfg, store, feat, models, fusion, folds)
        raw = video_tokens(feat)
        assert len(raw) == len(layers)
        btimes = np.array([x.a for x in res[1:]])
        d75 = [json.loads((e / "stitch.json").read_text())["penalties"] for e in em]
        sys_picks = {"greedy": greedy_lattice(layers), "R0_stored": decode_lattice(layers, average_params(d75), beam, st.tol_s),
                     "R0_tuned": decode_lattice(layers, average_params([eval_params(cfg, k, "R0_tuned") for k in folds]), beam, st.tol_s)}
        sc_raw = content_scores(work / "silent25.mp4")["scores"]
        cuts = [f / 25.0 for f in replay_cuts(sc_raw, sc.threshold, sc.min_scene_s)]
        p1 = average_params([eval_params(cfg, k, "R1") for k in folds])
        sys_picks["R1"] = decode_lattice_cue(layers, scene_logits(cuts, btimes, sc.snap_s, sc.eps), p1["alpha"], p1["lam_jump"], p1["mu"], beam, st.tol_s)
        z2 = prob_to_logit(np.mean([1 / (1 + np.exp(-cosine_logits(load_cosine(cfg, k), raw))) for k in folds], 0))
        p2 = average_params([eval_params(cfg, k, "R2") for k in folds])
        sys_picks["R2"] = decode_lattice_cue(layers, z2, p2["alpha"], p2["lam_jump"], p2["mu"], beam, st.tol_s)
        ps = {}
        for v, sname in (("pair", "R3"), ("ctx", "R4")):
            z, p = cue_logits(cfg, raw, v, folds, list(cfg.cue.seeds))
            sys_picks[sname] = decode_lattice_cue(layers, z, p["alpha"], p["lam_jump"], p["mu"], beam, st.tol_s)
            ps[sname] = (1 / (1 + np.exp(-z))).tolist()
        minutes = feat["meta"]["duration_s"] / 60
        reach = float(np.mean([np.min(np.abs(btimes - t)) <= 1.0 for t in r["changes"]])) if (r["changes"] and len(btimes)) else float("nan")
        row = {"excerpt": name, "fold": fold, "folds": folds, "duration_s": feat["meta"]["duration_s"], "n_changes": len(r["changes"]),
               "n_segments": len(layers), "changes_within_1s_of_a_segment_boundary": reach, "systems": {}, "p": ps}
        for sname, picks in sys_picks.items():
            mids = [L.mid[j] for L, j in zip(layers, picks)]
            sw = [float(t) for t, a, b in zip(btimes, mids, mids[1:]) if a != b]
            row["systems"][sname] = {"switches_per_min": len(sw) / minutes, "switch_times": sw,
                                     **{f"{tol:g}": match_counts(sw, r["changes"], tol) for tol in (0.5, 1.0)}}
        results_all.append(row)
        log.info("real-eval %s: %d changes, %d segments, fold %s", name, len(r["changes"]), len(layers), fold)
    d = Path(cfg.paths.runs) / cfg.cue.exp / "real"
    d.mkdir(parents=True, exist_ok=True)
    (d / "real_eval.json").write_text(json.dumps(results_all, indent=1, default=float))
    return write_real_report(results_all, Path(out))


def write_real_report(rows: list[dict], out: Path) -> Path:
    systems = list(rows[0]["systems"])
    L = ["# Cue-switch predictor on real annotated excerpts", "", "Generated by `bgm cue real-eval` (src/bgm/cue/real_eval.py). "
         "Boundary F1 pooled over excerpts; no CIs are computed here (the set is small and hand-made).", "",
         f"Excerpts: {len(rows)}; annotated changes: {sum(r['n_changes'] for r in rows)}; share of changes within ±1 s of a "
         f"segment boundary (max recall at ±1 s): {np.nanmean([r['changes_within_1s_of_a_segment_boundary'] for r in rows]):.3f}", "",
         "| system | boundary F1 ±1 s | ±0.5 s | precision ±1 s | recall ±1 s | switches / min |", "| --- | --- | --- | --- | --- | --- |"]
    minutes = sum(r["duration_s"] for r in rows) / 60
    for s in systems:
        for tol in ("1", "0.5"):
            c = np.array([r["systems"][s][tol] for r in rows], float).sum(0)
            if tol == "1":
                tp1, fp1, fn1 = c
                f1_1 = 2 * tp1 / max(2 * tp1 + fp1 + fn1, 1e-9)
            else:
                f1_05 = 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1e-9)
        spm = sum(r["systems"][s]["switches_per_min"] * r["duration_s"] / 60 for r in rows) / minutes
        L.append(f"| {s} | {f1_1:.3f} | {f1_05:.3f} | {tp1 / max(tp1 + fp1, 1e-9):.3f} | {tp1 / max(tp1 + fn1, 1e-9):.3f} | {spm:.2f} |")
    L.append(f"| annotated changes | — | — | — | — | {sum(r['n_changes'] for r in rows) / minutes:.2f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out
