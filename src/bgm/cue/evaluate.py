"""`bgm cue eval` (Step 4): R0-R6 on the frozen val / test sequences, per fold (PROGRESS.md D78+).

Emission: the D75 system, unchanged: the 10-seed ensemble heads, semantic-only fusion (runs/<paths.runs>/
<cue.emission_exp>/fold<k>/seed0/fusion.json), coarse-to-fine retrieval and `build_lattice`, run through the existing
code. Each sequence's lattice is built ONCE and decoded by every system R0-R5, so their emission scores (and track R@k)
are identical by construction; each system records a hash of the lattices it decoded, and the report asserts the
R0-R5 hashes are equal. R6 alone decodes lattices built with `retrieval.context_weight = 1` (D72 context score: the
one intended emission change, user 2026-10-19). Before anything else, the fold's 50 D75 test concatenations
(concat_sets_film.json) are decoded with the stored D75 penalties and must reproduce that run's saved
stitch_perquery.npz (part track accuracy and switch F1) exactly.

Systems (transitions only):
  R0_stored  fixed penalties from the D75 stitch.json (lam_switch, lam_jump, mu)
  R0_tuned   the same grid (stitch.grid lam_switch x lam_jump) re-tuned on the new val sequences; mu = D75's mu
  greedy     each segment's best window (reference)
  R1         scene-cut rule (scenes.py)        | the cue hook (cue/stitch.py) with alpha x lam_jump tuned on val;
  R2         cosine rule (baselines.py)        | mu = D75's mu in every run ("applied identically", user Q3)
  R3_s<k>    learned, pair-only (seed k)       |
  R4_s<k>    learned + context (seed k)        |
  R5_s<k>    R4 trained on cross_film          |
  R6_s<k>    R4's p, emission with context     |
  R4nodur_s<k>  diagnostic: R4 without the log-duration feature (D79)
  R3s / R4s_s<k>  R3 / R4 trained with same-music splices (D94; opt-in, `--variants`)
  R3a / R3sa_s<k>  R3 / R3s trained on audio-teacher targets (D98; opt-in, `--variants`)
  oracle     p = 1 - eps at the true changes (upper reference, not a system)
  Rhaz_L<L>  opt-in (`--hazard`): constant duration prior p_i = 1 - exp(-d_i / L), d_i = length (s) of the segment
             ending at boundary i; one system per L, alpha x lam_jump tuned on val like R1-R5; L is picked on val
             (highest val objective) at report time
Tuning objective: the existing one (mean over val sequences of duration-weighted part track accuracy + switch F1
at +-1 s), with truth = the boundaries where the track group changes (identical to the old "every join" whenever two
adjacent clips never share a track group, which holds for every old concatenation).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig

from bgm.cue.baselines import cosine_logits, load_cosine
from bgm.cue.model import predict
from bgm.cue.scenes import SceneCache, replay_cuts, scene_logits, sequence_scores
from bgm.cue.sequences import IN_FILM, boundary_labels, read_sets
from bgm.cue.stitch import decode_lattice_cue, prob_to_logit
from bgm.cue.tokens import build_tokens, sequence_tokens
from bgm.cue.train import load_cue
from bgm.eval.store import FeatureStore, fold_ids
from bgm.retrieval.index import FoldModels, MusicIndex, concat_query
from bgm.retrieval.pipeline import fuse, run_video, truth_for
from bgm.retrieval.stitch_eval import (build_lattice, chosen_candidates, decode_lattice, greedy_lattice, part_track_accuracy,
                                       read_concat_sets, switch_f1, switch_times)
from bgm.retrieval.tune_stitch import evaluate_path
from bgm.runs import git_commit, run_dir

log = logging.getLogger(__name__)
SETS = (("val", "blocked"), ("test", "blocked"), ("test", "same_film"), ("test", "timeline"))
TOLS = (0.5, 1.0)
GROUPS = {"in_film": IN_FILM, "cross": ("cross",), "all": ("clip", "film", "cross")}
RATE = 25


@dataclass
class Lattice:
    layers: list
    cand_track: list[np.ndarray]     # per layer: track group of each candidate
    ranks: np.ndarray                # per segment: rank of the true track group among files by best-window score
    digest: str


@dataclass
class Case:
    split: str
    mode: str
    idx: int
    parts: list[str]
    vq: SimpleNamespace              # cut_times, duration, meta (what evaluate_path needs)
    res: list                        # SimpleNamespace(a, b) per segment
    raw_tokens: np.ndarray
    labels: np.ndarray
    types: list[str]
    btimes: np.ndarray
    seg_track: np.ndarray
    w: np.ndarray
    n25: list[int]
    films: list[str]
    lat: dict = field(default_factory=dict)      # 'base' | 'ctx' -> Lattice


def lattice_digest(layers) -> str:
    h = hashlib.sha256()
    for L in layers:
        for a in (L.mid, L.off, L.dur, L.cost, L.key, L.tempo, np.array([L.n_top]), L.cont_of if L.cont_of is not None else np.zeros(0)):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def emission_rank(index: MusicIndex, store, vq, r, w: dict) -> float:
    """Rank (1-based) of the true track group among the shortlisted files ordered by their best window's fused
    score; inf if the true track is not shortlisted."""
    S = fuse(r.raw, r.stats, w)
    best = np.full(len(index.ids), -np.inf)
    np.maximum.at(best, r.fi, S)
    files = np.where(np.isfinite(best))[0]
    order = files[np.argsort(-best[files], kind="stable")]
    true = store.music_track[truth_for(vq, store, r.a, r.b)[0]]
    hit = [k for k, f in enumerate(order) if store.music_track[index.ids[f]] == true]
    return float(hit[0] + 1) if hit else float("inf")


def make_lattice(store, index, models, vq, fusion, cfg) -> tuple[Lattice, list]:
    r = cfg.retrieval
    res = run_video(index, vq, fusion["w_coarse"], r.coarse_top_k, r.window_hop_s)
    layers = build_lattice(index, vq, res, fusion["w_fine"], r.fine_top_j)
    ct = [np.array([store.music_track[index.ids[m]] for m in L.mid]) for L in layers]
    ranks = np.array([emission_rank(index, store, vq, x, fusion["w_fine"]) for x in res])
    return Lattice(layers, ct, ranks, lattice_digest(layers)), res


def build_case(store, tokens, fg, index, models, fusion, cfg, split, mode, idx, parts, contexts) -> Case:
    vq = concat_query(store, parts)
    raw, seg_part = sequence_tokens(tokens, parts)
    assert len(raw) == len(vq.segments), (parts, len(raw), len(vq.segments))
    y, typ = boundary_labels(seg_part, parts, store.track, fg)
    lat, res = {}, None
    for name, wctx in contexts:
        cfg.retrieval.context_weight = float(wctx)
        try:
            q = concat_query(store, parts)
            q.prepare(models, store.min_span)
            lat[name], res = make_lattice(store, index, models, q, fusion, cfg)
        finally:
            cfg.retrieval.context_weight = 0.0
    light = SimpleNamespace(cut_times=vq.cut_times, duration=vq.duration, meta=vq.meta)
    rs = [SimpleNamespace(a=float(a), b=float(b)) for a, b in vq.segments]
    seg_track = np.array([store.music_track[truth_for(vq, store, a, b)[0]] for a, b in vq.segments])
    n25 = [int(round(d * RATE)) for _, _, d in vq.meta["parts"]]
    return Case(split, mode, idx, list(parts), light, rs, raw, y, typ, np.array([x.a for x in rs[1:]]), seg_track,
                np.array([x.b - x.a for x in rs]), n25, [fg[p] for p in parts], lat)


# ---------------------------------------------------------------- objective and metrics
def fast_objective(case: Case, lat: Lattice, picks: list[int]) -> float:
    """tune_stitch's objective (part track accuracy + switch F1 +-1 s) with track-group-change truth."""
    tr = np.array([ct[j] for ct, j in zip(lat.cand_track, picks)])
    acc = float(np.average(tr == case.seg_track, weights=case.w))
    mids = [L.mid[j] for L, j in zip(lat.layers, picks)]
    sw = [float(t) for t, m0, m1 in zip(case.btimes, mids, mids[1:]) if m0 != m1]
    return acc + switch_f1(sw, [float(t) for t in case.btimes[case.labels == 1]])


def match_counts(pred: list[float], true: list[float], tol: float) -> tuple[int, int, int]:
    """switch_f1's greedy one-to-one matching, as (tp, fp, fn)."""
    used, tp = set(), 0
    for t in true:
        c = [(abs(p - t), k) for k, p in enumerate(pred) if k not in used and abs(p - t) <= tol]
        if c:
            used.add(min(c)[1])
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def boundary_counts(case: Case, lat: Lattice, picks: list[int]) -> dict:
    mids = np.array([L.mid[j] for L, j in zip(lat.layers, picks)])
    sw = mids[1:] != mids[:-1]
    typ = np.array(case.types)
    out = {}
    for g, types in GROUPS.items():
        sel = np.isin(typ, types)
        pred, true = case.btimes[sw & sel].tolist(), case.btimes[(case.labels == 1) & sel].tolist()
        for tol in TOLS:
            tp, fp, fn = match_counts(pred, true, tol)
            out[f"{g}/{tol:g}"] = [tp, fp, fn]
        out[f"{g}/n_true"] = len(true)
        out[f"{g}/n_pred"] = len(pred)
    return out


# ---------------------------------------------------------------- systems
@dataclass
class System:
    name: str
    kind: str                     # fixed | grid | cue | greedy
    ctx: bool = False
    seed: int | None = None
    logits: dict = field(default_factory=dict)   # (split, mode, idx) -> logits
    params: dict = field(default_factory=dict)
    val_objective: float = float("nan")


def decode(sys_: System, case: Case, params: dict, beam: int, tol_s: float) -> tuple[Lattice, list[int]]:
    lat = case.lat["ctx" if sys_.ctx else "base"]
    if sys_.kind == "greedy":
        return lat, greedy_lattice(lat.layers)
    if sys_.kind in ("fixed", "grid"):
        return lat, decode_lattice(lat.layers, params, beam, tol_s)
    z = sys_.logits[(case.split, case.mode, case.idx)]
    return lat, decode_lattice_cue(lat.layers, z, params["alpha"], params["lam_jump"], params["mu"], beam, tol_s)


def tune(sys_: System, val: list[Case], grid: list[dict], beam: int, tol_s: float) -> None:
    best, best_p = None, None
    for p in grid:
        s = float(np.mean([fast_objective(c, *decode(sys_, c, p, beam, tol_s)) for c in val]))
        if best is None or s > best:
            best, best_p = s, p
    sys_.params, sys_.val_objective = best_p, best


def clusters_of(cases: list[Case]) -> list[str]:
    """Cluster id per case: blocked test sequences joined by shared film groups (union-find; a film is in one test
    sequence unless it was split), single-film sequences by their film group."""
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, c in enumerate(cases):
        for f in set(c.films):
            if f in seen:
                parent[find(i)] = find(seen[f])
            else:
                seen[f] = i
    return [f"{cases[0].mode}:{find(i)}" for i in range(len(cases))]


def check_emission(cfg, store, index, models, fusion, fold: int, beam: int) -> dict:
    """Decode the D75 test concatenations with the D75 penalties; must equal the saved stitch_perquery.npz."""
    d = Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{fold}" / "seed0"
    pen = json.loads((d / "stitch.json").read_text())["penalties"]
    saved = np.load(d / "stitch_perquery.npz")
    sets = read_concat_sets(cfg)[str(fold)]["test"]
    acc, f1 = [], []
    for parts in sets:
        vq = concat_query(store, parts)
        vq.prepare(models, store.min_span)
        lat, res = make_lattice(store, index, models, vq, fusion, cfg)
        ch = chosen_candidates(index, lat.layers, decode_lattice(lat.layers, pen, beam, cfg.stitch.tol_s))
        sw, _ = switch_times(ch, res)
        acc.append(part_track_accuracy(store, vq, res, ch))
        f1.append(switch_f1(sw, [t0 for _, t0, _ in vq.meta["parts"][1:]]))
    ok_a = np.array_equal(np.array(acc), saved["concat/viterbi/part_track_acc"])
    ok_f = np.array_equal(np.array(f1), saved["concat/viterbi/switch_f1"])
    assert ok_a and ok_f, f"fold {fold}: emission / Viterbi does not reproduce the D75 run (acc {ok_a}, f1 {ok_f})"
    return {"n": len(sets), "part_track_acc_equal": ok_a, "switch_f1_equal": ok_f}


def hazard_logits(w: np.ndarray, L: float) -> np.ndarray:
    """Logits of p_i = 1 - exp(-d_i / L) at each boundary; w: [n_segments] segment lengths (s), d_i = w[i]."""
    return prob_to_logit(-np.expm1(-np.asarray(w[:-1], float) / float(L)))


def systems_for(cfg, fold: int, cases: list[Case], seeds, variants, hazard=()) -> list[System]:
    sc = cfg.cue.scene
    out = [System("R0_stored", "fixed"), System("R0_tuned", "grid"), System("greedy", "greedy"),
           System("R1", "cue"), System("R2", "cue"), System("oracle", "cue")]
    cache = SceneCache(cfg)
    cos = load_cosine(cfg, fold)
    hi = float(np.log1p(-sc.eps) - np.log(sc.eps))
    for c in cases:
        key = (c.split, c.mode, c.idx)
        cuts = [f / RATE for f in replay_cuts(sequence_scores(cache, c.parts, c.n25), sc.threshold, sc.min_scene_s)]
        out[3].logits[key] = scene_logits(cuts, c.btimes, sc.snap_s, sc.eps)
        out[4].logits[key] = cosine_logits(cos, c.raw_tokens)
        out[5].logits[key] = np.where(c.labels == 1, hi, -hi)
    for L in hazard:
        out.append(System(f"Rhaz_L{L:g}", "cue", logits={(c.split, c.mode, c.idx): hazard_logits(c.w, L) for c in cases}))
    names = {"pair": "R3", "ctx": "R4", "ctx_cross": "R5", "ctx_nodur": "R4nodur", "pair_splice": "R3s", "ctx_splice": "R4s",
             "pair_audio": "R3a", "pair_splice_audio": "R3sa"}
    for v in [v for v in variants if v]:
        for s in seeds:
            model, scaler = load_cue(cfg, v, fold, s)
            z = predict(model, [scaler(c.raw_tokens) for c in cases])
            sy = System(f"{names[v]}_s{s}", "cue", seed=s, logits={(c.split, c.mode, c.idx): zi for c, zi in zip(cases, z)})
            out.append(sy)
            if v == "ctx":
                out.append(System(f"R6_s{s}", "cue", ctx=True, seed=s, logits=dict(sy.logits)))
    return out


def eval_fold(cfg: DictConfig, fold: int, seeds=(0, 1, 2), variants=("pair", "ctx", "ctx_cross", "ctx_nodur"),
              render: bool = True, hazard=()) -> Path:
    t0 = time.time()
    assert cfg.pipeline.dyn_variant is None and cfg.pipeline.sem_seeds, "run with the D75 emission overrides (see reports/cue_chain.sh)"
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    from bgm.data import manifest
    folds = manifest.read_folds(cfg)
    fg = folds["film_group"]
    f = fold_ids(cfg, fold)
    em = Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{fold}" / "seed0"
    fusion = json.loads((em / "fusion.json").read_text())
    d75 = json.loads((em / "stitch.json").read_text())["penalties"]
    models = FoldModels.load(cfg, store, fold, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    st = cfg.stitch
    beam = st.beam_factor * cfg.retrieval.fine_top_j
    v_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["val"])))
    t_index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    check = check_emission(cfg, store, t_index, models, fusion, fold, beam)
    log.info("cue eval fold %d: D75 test concatenations reproduced exactly (%d)", fold, check["n"])
    sets = read_sets(cfg)["sets"][str(fold)]
    cases: list[Case] = []
    for split, mode in SETS:
        index = v_index if split == "val" else t_index
        for i, parts in enumerate(sets[split][mode]):
            cases.append(build_case(store, tokens, fg, index, models, fusion, cfg, split, mode, i, parts, (("base", 0.0), ("ctx", 1.0))))
    log.info("cue eval fold %d: %d lattices built [%.0f s]", fold, 2 * len(cases), time.time() - t0)
    val = [c for c in cases if c.split == "val"]
    test = [c for c in cases if c.split == "test"]
    systems = systems_for(cfg, fold, cases, seeds, variants, hazard)
    mu = float(d75["mu"])
    grid_fixed = [{"lam_switch": float(a), "lam_jump": float(b), "mu": mu} for a, b in itertools.product(st.grid.lam_switch, st.grid.lam_jump)]
    grid_cue = [{"alpha": float(a), "lam_jump": float(b), "mu": mu} for a, b in itertools.product(cfg.cue.tune.alpha, cfg.cue.tune.lam_jump)]
    clusters = {}
    for mode in ("blocked", "same_film", "timeline"):
        cs = [c for c in test if c.mode == mode]
        clusters.update({(c.mode, c.idx): k for c, k in zip(cs, clusters_of(cs))})
    out = {"fold": fold, "git_commit": git_commit(), "check_emission": check, "d75_penalties": d75, "mu": mu,
           "cases": [{"split": c.split, "mode": c.mode, "idx": c.idx, "parts": c.parts, "films": c.films,
                      "cluster": clusters.get((c.mode, c.idx)) if c.split == "test" else None, "duration": c.vq.duration,
                      "labels": c.labels.tolist(), "types": c.types, "btimes": c.btimes.tolist(),
                      "w": c.w.tolist(), "ranks": {k: v.ranks.tolist() for k, v in c.lat.items()},
                      "digest": {k: v.digest for k, v in c.lat.items()}} for c in cases],
           "systems": {}}
    for sy in systems:
        t1 = time.time()
        if sy.kind == "fixed":
            sy.params = dict(d75)
            sy.val_objective = float(np.mean([fast_objective(c, *decode(sy, c, sy.params, beam, st.tol_s)) for c in val]))
        elif sy.kind == "grid":
            tune(sy, val, grid_fixed, beam, st.tol_s)
        elif sy.kind == "cue":
            tune(sy, val, grid_cue, beam, st.tol_s)
        rows = []
        digests = []
        for c in test:
            rng = np.random.default_rng([fold, 77, c.idx, {"blocked": 0, "same_film": 1, "timeline": 2}[c.mode],
                                         int(hashlib.sha256(sy.name.encode()).hexdigest()[:6], 16)])
            lat, picks = decode(sy, c, sy.params, beam, st.tol_s)
            digests.append(lat.digest)
            ch = chosen_candidates(t_index, lat.layers, picks)
            m = evaluate_path(store, t_index, c.vq, c.res, ch, cfg, rng, lambda mid: t_index.items[t_index.pos[mid]].beats, render=render)
            m["switch_f1_joins"] = m.pop("switch_f1")             # the old definition: every join is a change
            m["switch_f1"] = switch_f1(switch_times(ch, c.res)[0], c.btimes[c.labels == 1].tolist())
            m["objective"] = fast_objective(c, lat, picks)
            m.update(boundary_counts(c, lat, picks))
            m["mode"], m["idx"] = c.mode, c.idx
            rows.append(m)
        out["systems"][sy.name] = {"kind": sy.kind, "ctx": sy.ctx, "seed": sy.seed, "params": sy.params,
                                   "val_objective": sy.val_objective, "rows": rows,
                                   "emission_digest": hashlib.sha256("".join(digests).encode()).hexdigest(),
                                   "logits": {f"{k[1]}/{k[2]}": v.tolist() for k, v in sy.logits.items() if k[0] == "test"}}
        log.info("cue eval fold %d %-10s %s val %.3f | test blocked switch F1 %.3f part acc %.3f [%.0f s]", fold, sy.name, sy.params,
                 sy.val_objective, np.mean([r["switch_f1"] for r in rows if r["mode"] == "blocked"]),
                 np.mean([r["part_track_acc"] for r in rows if r["mode"] == "blocked"]), time.time() - t1)
    d = run_dir(cfg, f"{cfg.cue.exp}/eval", fold)
    (d / "eval.json").write_text(json.dumps(out, default=float))
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, d / "config.yaml")
    (d / "git_commit.txt").write_text(git_commit() + "\n")
    (d / "seed.txt").write_text(f"{list(seeds)}\n")
    log.info("cue eval fold %d done -> %s [%.0f s]", fold, d, time.time() - t0)
    return d
