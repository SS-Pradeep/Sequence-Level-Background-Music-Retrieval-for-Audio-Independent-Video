"""Stitching for any person-branch system: `bgm.retrieval.tune_stitch.tune_one` with person-aware queries and index.

Same steps and settings: Viterbi penalties tuned on the fold's val concatenations (objective part track accuracy +
switch F1), then the test concatenations (Viterbi and greedy) and the natural test videos. Only the query / index
construction differs, so a system without a branch reproduces the saved D75 / D60 stitching exactly (checked by
bgm.person.run).
"""
from __future__ import annotations

import itertools

import numpy as np

from bgm.person.pipeline import PersonMusicIndex, person_query
from bgm.retrieval.pipeline import path_accuracy, run_video
from bgm.retrieval.stitch_eval import (build_lattice, chosen_candidates, decode_lattice, greedy_lattice, part_track_accuracy,
                                       read_concat_sets, switch_f1, switch_times)
from bgm.retrieval.tune_stitch import evaluate_path, original_excess


def _prepare(cfg, store, pstore, index, models, sets, fusion):
    r, out = cfg.retrieval, []
    for parts in sets:
        vq = person_query(store, pstore, list(parts))
        vq.prepare(models, store.min_span)
        res = run_video(index, vq, fusion["w_coarse"], fusion["K"], fusion["window_hop_s"])
        out.append((vq, res, build_lattice(index, vq, res, fusion["w_fine"], fusion["J"])))
    return out


def stitch_fold(cfg, store, pstore, models, fold: int, seed: int, fusion: dict, f: dict) -> tuple[dict, dict]:
    """-> (stitch.json content, per-query arrays) for one system on one fold."""
    sets = read_concat_sets(cfg)[str(fold)]
    st, beam = cfg.stitch, cfg.stitch.beam_factor * fusion["J"]
    v_index = PersonMusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["val"])))
    val = _prepare(cfg, store, pstore, v_index, models, sets["val"], fusion)
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
    t_index = PersonMusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    beats_of = lambda m: t_index.items[t_index.pos[m]].beats  # noqa: E731
    test = _prepare(cfg, store, pstore, t_index, models, sets["test"], fusion)
    rng = np.random.default_rng([seed, fold, 77])
    rows, orig = {"viterbi": [], "greedy": []}, []
    for vq, res, lat in test:
        for name, picks in (("viterbi", decode_lattice(lat, best_pen, beam, st.tol_s)), ("greedy", greedy_lattice(lat))):
            rows[name].append(evaluate_path(store, t_index, vq, res, chosen_candidates(t_index, lat, picks), cfg, rng, beats_of))
        orig.append(original_excess(store, vq, cfg, rng))
    nat = {"viterbi": [], "greedy": []}
    for p in f["test"]:
        vq = person_query(store, pstore, [p])
        vq.prepare(models, store.min_span)
        res = run_video(t_index, vq, fusion["w_coarse"], fusion["K"], fusion["window_hop_s"])
        lat = build_lattice(t_index, vq, res, fusion["w_fine"], fusion["J"])
        for name, picks in (("viterbi", decode_lattice(lat, best_pen, beam, st.tol_s)), ("greedy", greedy_lattice(lat))):
            nat[name].append(path_accuracy(store, vq, res, chosen_candidates(t_index, lat, picks)))
    per = {}
    for name in rows:
        for k in ("part_track_acc", "switch_f1", "switches_per_min", "jumps_per_min", "cob_excess", "cob_excess_control", "seam_jump_LU_median"):
            per[f"concat/{name}/{k}"] = np.array([x[k] for x in rows[name]], float)
        per[f"natural/{name}/track"] = np.array([x["track"] for x in nat[name]])
        per[f"natural/{name}/window"] = np.array([x["window"] for x in nat[name]])
    per["concat/original/cob_excess"] = np.array(orig, float)
    per["natural/pair"] = np.array(list(f["test"]))
    per["concat/clusters"] = np.array([f"concat{fold}_{i}" for i in range(len(test))])
    return {"penalties": best_pen, "val_objective": best, "tol_s": st.tol_s, "beam": beam}, per
