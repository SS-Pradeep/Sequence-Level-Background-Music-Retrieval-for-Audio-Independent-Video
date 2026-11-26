"""`bgm tune-stitch` (SPEC M10 / H5): stitching penalties on the val concatenations, test evaluation.

Per (fold, seed), with that run's fusion weights (runs/pipeline/.../fusion.json):
  * val: grid lam_switch x lam_jump x mu (SPEC §6), objective = mean over val concatenations of duration-weighted part
    track accuracy + switch-boundary F1; saved to runs/pipeline/fold<k>/seed<s>/stitch.json;
  * test concatenations, Viterbi vs greedy: part track accuracy, switch-boundary F1, switches and jumps per minute,
    seam loudness jumps (rendered soundtrack), within-piece excess cut-on-beat vs the beat-snapped control and the
    original soundtracks;
  * natural test videos: duration-weighted track / window accuracy after Viterbi and after greedy (H3).
"""
from __future__ import annotations

import itertools
import json
import logging
import time

import numpy as np
from omegaconf import DictConfig

from bgm.data.prep import interim_music_path
from bgm.eval.metrics import cut_on_beat_rate
from bgm.eval.store import FeatureStore, fold_ids
from bgm.render.render import render_soundtrack, runs_from_path, seam_loudness_jumps
from bgm.retrieval.fuse import prepared_queries
from bgm.retrieval.index import FoldModels, MusicIndex, concat_query
from bgm.retrieval.pipeline import path_accuracy, run_video
from bgm.retrieval.stitch_eval import (build_lattice, chosen_candidates, decode_lattice, greedy_lattice, part_track_accuracy,
                                       read_concat_sets, soundtrack_beats, switch_f1, switch_times, within_piece_cuts)
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)


def excess_rate(cuts: np.ndarray, beats: np.ndarray, duration: float, rng, tol: float, draws: int) -> float:
    if not len(cuts):
        return np.nan
    r = cut_on_beat_rate(cuts, beats, tol)
    sh = np.mean([cut_on_beat_rate(cuts, (beats + rng.uniform(0, duration)) % duration, tol) for _ in range(draws)])
    return r - sh


def prepare_concats(store, index, models, sets, cfg, fusion):
    r = cfg.retrieval
    out = []
    for parts in sets:
        vq = concat_query(store, parts)
        vq.prepare(models, store.min_span)
        res = run_video(index, vq, fusion["w_coarse"], r.coarse_top_k, r.window_hop_s)
        out.append((vq, res, build_lattice(index, vq, res, fusion["w_fine"], r.fine_top_j)))
    return out


def evaluate_path(store, index, vq, res, chosen, cfg, rng, beats_of, render=True) -> dict:
    sw, jp = switch_times(chosen, res)
    joins = [t0 for _, t0, _ in vq.meta["parts"][1:]]
    minutes = vq.duration / 60
    out = {"part_track_acc": part_track_accuracy(store, vq, res, chosen), "switch_f1": switch_f1(sw, joins),
           "switches_per_min": len(sw) / minutes, "jumps_per_min": len(jp) / minutes}
    runs = runs_from_path(chosen, [(x.a, x.b) for x in res])
    starts = [ru.start for ru in runs[1:]]
    cuts = within_piece_cuts(vq.cut_times, vq.duration, starts)
    e = cfg.eval
    out["cob_excess"] = excess_rate(cuts, soundtrack_beats(chosen, res, beats_of), vq.duration, rng, e.cut_beat_tol_s, e.circular_shift_draws)
    # beat-snapped control: random library windows with the same piece boundaries and beat-snapped starts
    ctrl = []
    for _ in range(e.circular_shift_draws):
        beats = []
        for ru in runs:
            it = index.items[rng.integers(len(index.items))]
            d = ru.end - ru.start
            ok = it.beats[it.beats + d <= it.duration]
            s = float(rng.choice(ok)) if len(ok) else 0.0
            b = it.beats[(it.beats >= s) & (it.beats < s + d)]
            beats.append(ru.start + b - s)
        bt = np.concatenate(beats)
        ctrl.append(cut_on_beat_rate(cuts, bt, e.cut_beat_tol_s) - np.mean(
            [cut_on_beat_rate(cuts, (bt + rng.uniform(0, vq.duration)) % vq.duration, e.cut_beat_tol_s) for _ in range(5)]) if len(cuts) else np.nan)
    out["cob_excess_control"] = float(np.nanmean(ctrl)) if len(cuts) else np.nan
    out["n_within_piece_cuts"] = int(len(cuts))
    if render:
        mix, seams = render_soundtrack(runs, vq.duration, lambda m: interim_music_path(cfg, m), cfg.render)
        j = [x for x in seam_loudness_jumps(mix, seams) if np.isfinite(x)]
        out["seam_jump_LU_median"] = float(np.median(j)) if j else np.nan
        out["seam_jumps"] = j
    return out


def original_excess(store, vq, cfg, rng) -> float:
    """Excess cut-on-beat of the concatenation's own soundtracks (pieces = its parts)."""
    beats = []
    for p, t0, d in vq.meta["parts"]:
        b = np.asarray(store.music(store.music_of[p])["beat_times_s"], float)
        beats.append(t0 + b[b < d])
    joins = [t0 for _, t0, _ in vq.meta["parts"][1:]]
    cuts = within_piece_cuts(vq.cut_times, vq.duration, joins)
    return excess_rate(cuts, np.concatenate(beats), vq.duration, rng, cfg.eval.cut_beat_tol_s, cfg.eval.circular_shift_draws)


def tune_one(cfg: DictConfig, store: FeatureStore, fold: int, seed: int) -> dict:
    t0 = time.time()
    d = run_dir(cfg, cfg.pipeline.exp, fold, seed)
    fusion = json.loads((d / "fusion.json").read_text())
    f = fold_ids(cfg, fold)
    sets = read_concat_sets(cfg)[str(fold)]
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    assert models.dyn is not None or fusion["w_fine"].get("dyn", 0.0) == 0, \
        "pipeline.dyn_variant=null, but this run's fusion.json gives the dynamics branch a weight"
    st, beam = cfg.stitch, cfg.stitch.beam_factor * cfg.retrieval.fine_top_j
    # ---- val: penalties
    v_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["val"])))
    val = prepare_concats(store, v_index, models, sets["val"], cfg, fusion)
    best, best_pen = None, None
    for ls, lj, mu in itertools.product(st.grid.lam_switch, st.grid.lam_jump, st.grid.mu):
        pen = {"lam_switch": float(ls), "lam_jump": float(lj), "mu": float(mu)}
        scores = []
        for vq, res, lat in val:
            ch = chosen_candidates(v_index, lat, decode_lattice(lat, pen, beam, st.tol_s))
            sw, _ = switch_times(ch, res)
            scores.append(part_track_accuracy(store, vq, res, ch) + switch_f1(sw, [t for _, t, _ in vq.meta["parts"][1:]]))
        if best is None or np.mean(scores) > best:
            best, best_pen = float(np.mean(scores)), pen
    (d / "stitch.json").write_text(json.dumps({"penalties": best_pen, "val_objective": best, "tol_s": st.tol_s, "beam": beam}, indent=1))
    # ---- test concatenations
    t_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    beats_of = lambda m: t_index.items[t_index.pos[m]].beats  # noqa: E731
    test = prepare_concats(store, t_index, models, sets["test"], cfg, fusion)
    rng = np.random.default_rng([seed, fold, 77])
    rows = {"viterbi": [], "greedy": []}
    orig = []
    for vq, res, lat in test:
        for name, picks in (("viterbi", decode_lattice(lat, best_pen, beam, st.tol_s)), ("greedy", greedy_lattice(lat))):
            rows[name].append(evaluate_path(store, t_index, vq, res, chosen_candidates(t_index, lat, picks), cfg, rng, beats_of))
        orig.append(original_excess(store, vq, cfg, rng))
    # ---- natural test videos: duration-weighted accuracy after decoding (H3)
    nat = {"viterbi": [], "greedy": []}
    for p, vq in prepared_queries(store, f["test"], models).items():
        res = run_video(t_index, vq, fusion["w_coarse"], cfg.retrieval.coarse_top_k, cfg.retrieval.window_hop_s)
        lat = build_lattice(t_index, vq, res, fusion["w_fine"], cfg.retrieval.fine_top_j)
        for name, picks in (("viterbi", decode_lattice(lat, best_pen, beam, st.tol_s)), ("greedy", greedy_lattice(lat))):
            nat[name].append(path_accuracy(store, vq, res, chosen_candidates(t_index, lat, picks)))
    per = {}
    for name in rows:
        for k in ("part_track_acc", "switch_f1", "switches_per_min", "jumps_per_min", "cob_excess", "cob_excess_control", "seam_jump_LU_median"):
            per[f"concat/{name}/{k}"] = np.array([x[k] for x in rows[name]], float)
        per[f"concat/{name}/seam_jumps_all"] = np.array([j for x in rows[name] for j in x["seam_jumps"]], float)
        per[f"natural/{name}/track"] = np.array([x["track"] for x in nat[name]])
        per[f"natural/{name}/window"] = np.array([x["window"] for x in nat[name]])
    per["concat/original/cob_excess"] = np.array(orig, float)
    per["natural/clusters"] = np.array([store.track[p] for p in f["test"]])
    per["natural/pair"] = np.array(list(f["test"]))
    per["concat/clusters"] = np.array([f"concat{fold}_{i}" for i in range(len(test))])
    np.savez(d / "stitch_perquery.npz", **per)
    summary = {k: float(np.nanmean(v)) for k, v in per.items() if v.dtype.kind == "f" and not k.endswith("seam_jumps_all")}
    for name in rows:
        sj = per[f"concat/{name}/seam_jumps_all"]
        summary[f"concat/{name}/seam_jump_LU_median_all"] = float(np.median(sj)) if len(sj) else float("nan")
    (d / "stitch_metrics.json").write_text(json.dumps({"penalties": best_pen, "val_objective": best, "test": summary}, indent=1))
    write_run(d, cfg, json.loads((d / "metrics.json").read_text()) | {"stitch": {"penalties": best_pen, "test": summary}}, seed)
    log.info("tune-stitch fold %d seed %d: %s val %.3f | test part acc viterbi %.3f greedy %.3f | F1 %.3f vs %.3f | "
             "switch/min %.2f vs %.2f  [%.0f s]", fold, seed, best_pen, best, summary["concat/viterbi/part_track_acc"],
             summary["concat/greedy/part_track_acc"], summary["concat/viterbi/switch_f1"], summary["concat/greedy/switch_f1"],
             summary["concat/viterbi/switches_per_min"], summary["concat/greedy/switches_per_min"], time.time() - t0)
    return summary


def run_tune_stitch(cfg: DictConfig, folds=None, seeds=None) -> list[dict]:
    from bgm.retrieval.stitch_eval import concat_sets_path, make_concat_sets
    store = FeatureStore(cfg)
    path = concat_sets_path(cfg)
    if not path.exists():
        make_concat_sets(cfg, store, cfg.eval.concat_videos_per_fold)
    return [tune_one(cfg, store, k, s) for k in (range(cfg.splits.n_folds) if folds is None else folds)
            for s in (list(cfg.train.seeds) if seeds is None else seeds)]
