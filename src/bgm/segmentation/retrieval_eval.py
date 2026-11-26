"""D143: the step-1 segmenter inside the retrieval pipeline (C = contrast + BiGRU vs the D140 head vs the current
PySceneDetect + SigLIP merge).

`python -m bgm.segmentation.retrieval_eval segments [--device cpu]`
    per fold, theta on val (as D140 / D142) and the cuts of every segmenter on the frozen title-split val / test
    blocked sequences, with clip joins forced and without (own cuts) -> data/interim/segmentation/retrieval_segments.json
`python -m bgm.segmentation.retrieval_eval eval --fold k [--device cpu]`
    the title-split retrieval system of D121 (10-head ensemble, runs/titlesplit/pipeline_ens10 fusion, cue models
    runs/titlesplit/cue/ctx) on each segmenter's segments: lattice per sequence, R0_tuned (fixed penalties re-tuned
    on val), R4 seeds 0-2 (cue predictor, alpha / lam_jump re-tuned on val), oracle (p at the true changes); every
    arm tuned on its own val segments -> runs/titlesplit/seg_retrieval/fold<k>/eval.json
`python -m bgm.segmentation.retrieval_eval report`
    -> results/seg_retrieval.md, .json (mean over test sequences, paired differences, cluster bootstrap over film
    groups, 10,000 resamples)

Arms: <segmenter>/forced (clip joins are always cuts: the old protocol, where every true change is a candidate) and
<segmenter>/own (each segmenter's own cuts, as on a real silent video). psd_siglip/forced uses concat_query's segments
unchanged (the current pipeline exactly) and must reproduce the D121 eval.json (lattice digests, tuned parameters,
per-sequence objective); the report prints that check. oracle/joins: one segment per clip (cuts = the joins).

Metrics (truth = the joins where the track group changes; times in sequence seconds):
  track accuracy  share of the sequence's time where the chosen file's track group is the true one, by overlap with
                  each clip (a segment that crosses a join is scored piece by piece; equals the old duration-weighted
                  part track accuracy when no segment crosses a join)
  switch F1       predicted switches (chosen file changes) vs true changes, greedy one-to-one within +-1 s
The tuning objective is their sum (the D121 objective with overlap accuracy).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

log = logging.getLogger(__name__)

SEGMENTERS = {"psd_siglip": "S0 PySceneDetect + SigLIP merge (current)", "emb_p": "D140 BaSSL + head", "combo": "C contrast + BiGRU"}
HEADS = {"emb_p": "emb_p", "combo": "combo"}
ARMS = ("psd_siglip/forced", "psd_siglip/own", "emb_p/forced", "emb_p/own", "combo/forced", "combo/own", "oracle/joins")
SEG_FILE = "retrieval_segments.json"
EXP = "seg_retrieval"
RATE = 25
CUE_OVERRIDES = ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "paths.results=results/titlesplit",
                 "pipeline.dyn_variant=null", "eval.concat_sets_file=concat_sets_title.json",
                 "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "cue.sequences_file=cue_sequences_title.json"]
REF_EVAL = Path("runs/titlesplit/cue/eval")


# ---------------------------------------------------------------- pure helpers
def to_rel(cuts: list[float], starts: list[float], tol: float = 1e-6) -> list[list[float]]:
    """Cut times on a sequence timeline -> [[part index, seconds into that part]]; a cut at a join belongs to the
    later part (offset 0)."""
    s = np.asarray(starts, float)
    out = []
    for t in cuts:
        j = int(np.searchsorted(s, t + tol, side="right") - 1)
        out.append([j, max(0.0, float(t) - float(s[j]))])
    return out


def from_rel(rel: list[list[float]], starts: list[float], durs: list[float]) -> list[float]:
    """Inverse of to_rel on another timeline of the same parts (offsets clipped to the part)."""
    return sorted(float(starts[int(j)]) + min(float(r), float(durs[int(j)])) for j, r in rel)


def segments_from_cuts(cuts: list[float], T: float, min_len: float = 1e-3) -> list[tuple[float, float]]:
    b = [0.0] + [c for c in sorted(cuts) if 0.0 < c < T] + [float(T)]
    return [(a, e) for a, e in zip(b[:-1], b[1:]) if e - a > min_len]


def overlap_pieces(segs: list[tuple[float, float]], spans: list[tuple[float, float]], tracks: list[str]
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(segment index, true track, overlap s) for every segment x clip overlap. spans: [(t0, d)] per clip."""
    si, tr, w = [], [], []
    for i, (a, b) in enumerate(segs):
        for (t0, d), t in zip(spans, tracks):
            ov = min(b, t0 + d) - max(a, t0)
            if ov > 1e-9:
                si.append(i)
                tr.append(t)
                w.append(ov)
    return np.array(si, int), np.array(tr, object), np.array(w, float)


def true_changes(spans: list[tuple[float, float]], tracks: list[str]) -> list[float]:
    return [float(t0) for (t0, _), a, b in zip(spans[1:], tracks, tracks[1:]) if a != b]


def boundary_labels(segs: list[tuple[float, float]], pieces) -> np.ndarray:
    """[N - 1]: 1 where the dominant true track of two neighbouring segments differs (the oracle's p)."""
    si, tr, w = pieces
    dom = []
    for i in range(len(segs)):
        sel = si == i
        tracks, ws = tr[sel], w[sel]
        dom.append(tracks[int(np.argmax(ws))])
    return np.array([int(a != b) for a, b in zip(dom, dom[1:])], np.int8)


def overlap_accuracy(chosen_track: np.ndarray, pieces) -> float:
    si, tr, w = pieces
    return float(np.average(chosen_track[si] == tr, weights=w))


def sequence_motion(raw_lm: list[np.ndarray]) -> np.ndarray:
    """Raw log motion of the clips end to end; each later clip's frame 0 (no flow, always 0) takes its frame 1's
    value, so clip starts are not marked (tokens.py: frame 0 is a join tell)."""
    out = []
    for k, m in enumerate(raw_lm):
        m = np.asarray(m, np.float64).copy()
        if k > 0 and len(m) > 1:
            m[0] = m[1]
        out.append(m)
    return np.concatenate(out)


# ---------------------------------------------------------------- stage 1: segments
def stage_segments(device: str) -> Path:
    from bgm.config import load_config
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold, tune_theta
    from bgm.segmentation.run_all import out_dir
    from bgm.segmentation.train_head import SegData
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    data = SegData(cfg, need_siglip=True)
    bassl = BaSSL(device)
    res = {"thetas": {}, "folds": {}}
    for k in range(5):
        F = {s: Fold(data, bassl, k, s, HEADS) for s in ("val", "test")}
        th = {v: tune_theta(F["val"], v) for v in HEADS}
        res["thetas"][str(k)] = th
        res["folds"][str(k)] = {}
        for split, fo in F.items():
            recs = []
            for i, parts in enumerate(fo.seqs):
                r = {"parts": list(parts), "cuts": {}}
                for v in SEGMENTERS:
                    c = fo.chains(v)[i]
                    starts = [0.0] + [float(x) for x in c.joins]
                    for forced in (True, False):
                        r["cuts"][f"{v}/{'forced' if forced else 'own'}"] = to_rel(fo.cuts(v, i, th.get(v), forced), starts)
                recs.append(r)
            res["folds"][str(k)][split] = recs
        log.info("segments fold %d: thetas %s, %d val / %d test sequences", k, th, len(F["val"].seqs), len(F["test"].seqs))
    path = out_dir(cfg) / SEG_FILE
    path.write_text(json.dumps(res))
    log.info("-> %s", path)
    return path


# ---------------------------------------------------------------- stage 2: retrieval
@dataclass
class SegCase:
    split: str
    idx: int
    parts: list[str]
    films: list[str]
    segs: list[tuple[float, float]]
    raw_tokens: np.ndarray
    btimes: np.ndarray
    labels: np.ndarray
    pieces: tuple
    true_sw: list[float]
    duration: float
    lat: object = None
    mode: str = "blocked"
    dev: dict = field(default_factory=dict)


def objective(case: SegCase, lat, picks: list[int]) -> dict:
    from bgm.retrieval.stitch_eval import switch_f1
    tr = np.array([ct[j] for ct, j in zip(lat.cand_track, picks)], object)
    mids = [L.mid[j] for L, j in zip(lat.layers, picks)]
    sw = [float(t) for t, m0, m1 in zip(case.btimes, mids, mids[1:]) if m0 != m1]
    acc = overlap_accuracy(tr, case.pieces)
    f1 = switch_f1(sw, case.true_sw)
    return {"track_acc": acc, "switch_f1": f1, "switch_f1_05": switch_f1(sw, case.true_sw, 0.5), "objective": acc + f1,
            "n_switch": len(sw), "n_true": len(case.true_sw), "n_seg": len(case.segs),
            "switches_per_min": 60 * len(sw) / case.duration, "segments_per_min": 60 * len(case.segs) / case.duration}


def build_case(store, tokens, fg, index, models, fusion, cfg, split, idx, parts, arm, rel) -> SegCase:
    from bgm.cue.evaluate import make_lattice
    from bgm.cue.tokens import segment_features, sequence_tokens
    from bgm.retrieval.index import concat_query
    vq = concat_query(store, parts)
    spans = [(t0, d) for _, t0, d in vq.meta["parts"]]
    starts, durs = [s[0] for s in spans], [s[1] for s in spans]
    tracks = [store.track[p] for p in parts]
    dev = {}
    if arm == "psd_siglip/forced":                  # the current pipeline, unchanged
        segs = [(float(a), float(b)) for a, b in vq.segments]
        raw, _ = sequence_tokens(tokens, parts)
        mine = segments_from_cuts(from_rel(rel, starts, durs), vq.duration)
        dev = {"n_dump": len(mine), "n_pipeline": len(segs),
               "max_dev_s": float(np.max(np.abs(np.array(mine) - np.array(segs)))) if len(mine) == len(segs) else None}
    else:
        cuts = starts[1:] if arm == "oracle/joins" else from_rel(rel, starts, durs)
        segs = segments_from_cuts(cuts, vq.duration)
        vq.segments = segs
        sem = vq.sem_1hz
        lm = sequence_motion([np.asarray(store.video(p)["dyn_25hz"])[:, 0] for p in parts])
        raw = segment_features(sem, lm, segs)
    vq.prepare(models, store.min_span)
    lat, _ = make_lattice(store, index, models, vq, fusion, cfg)
    pieces = overlap_pieces(segs, spans, tracks)
    return SegCase(split, idx, list(parts), [fg[p] for p in parts], segs, raw, np.array([a for a, _ in segs[1:]]),
                   boundary_labels(segs, pieces), pieces, true_changes(spans, tracks), float(vq.duration), lat, dev=dev)


def decode(kind: str, case: SegCase, params: dict, logits, beam: int, tol_s: float) -> list[int]:
    from bgm.cue.stitch import decode_lattice_cue
    from bgm.retrieval.stitch_eval import decode_lattice
    if kind == "grid":
        return decode_lattice(case.lat.layers, params, beam, tol_s)
    return decode_lattice_cue(case.lat.layers, logits, params["alpha"], params["lam_jump"], params["mu"], beam, tol_s)


def tune(kind: str, val: list[SegCase], logits: dict, grid: list[dict], beam: int, tol_s: float) -> tuple[dict, float]:
    best, best_p = None, None
    for p in grid:
        s = float(np.mean([objective(c, c.lat, decode(kind, c, p, logits.get((c.split, c.idx)), beam, tol_s))["objective"] for c in val]))
        if best is None or s > best:
            best, best_p = s, p
    return best_p, best


def reference_check(fold: int, arm_out: dict, cases: list[SegCase]) -> dict:
    """psd_siglip/forced vs the D121 eval.json of this fold (blocked test sequences)."""
    ref = json.loads((REF_EVAL / f"fold{fold}" / "eval.json").read_text())
    rc = [c for c in ref["cases"] if c["split"] == "test" and c["mode"] == "blocked"]
    test = [c for c in cases if c.split == "test"]
    out = {"digest_equal": len(rc) == len(test) and all(c.lat.digest == r["digest"]["base"] for c, r in zip(test, rc))}
    for name, sy in arm_out["systems"].items():
        if name not in ref["systems"]:
            continue
        rs = [r for r in ref["systems"][name]["rows"] if r["mode"] == "blocked"]
        out[name] = {"params_equal": sy["params"] == ref["systems"][name]["params"],
                     "max_obj_diff": float(np.max(np.abs(np.array([r["objective"] for r in sy["rows"]]) - np.array([r["objective"] for r in rs])))),
                     "max_f1_diff": float(np.max(np.abs(np.array([r["switch_f1"] for r in sy["rows"]]) - np.array([r["switch_f1"] for r in rs]))))}
    return out


def stage_eval(fold: int, device: str, arms: tuple[str, ...] = ARMS, seeds=(0, 1, 2)) -> Path:
    import torch
    from omegaconf import OmegaConf
    from bgm.config import load_config
    from bgm.cue.model import predict
    from bgm.cue.tokens import build_tokens
    from bgm.cue.train import load_cue
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.runs import git_commit, run_dir
    from bgm.segmentation.run_all import out_dir
    t0 = time.time()
    cfg = load_config("configs/default.yaml", CUE_OVERRIDES)
    ref_cfg = OmegaConf.load(REF_EVAL / f"fold{fold}" / "config.yaml")
    for sec in ("retrieval", "pipeline", "segments"):
        assert OmegaConf.to_container(cfg[sec]) == OmegaConf.to_container(ref_cfg[sec]), f"cfg.{sec} differs from the D121 run"
    for key in ("emission_exp", "exp", "tune", "variant"):
        assert OmegaConf.to_container(cfg.cue)[key] == OmegaConf.to_container(ref_cfg.cue)[key], f"cfg.cue.{key} differs"
    segfile = json.loads((out_dir(load_config("configs/default.yaml", [])) / SEG_FILE).read_text())["folds"][str(fold)]
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    fg = manifest.read_folds(cfg)["film_group"]
    f = fold_ids(cfg, fold)
    em = Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{fold}" / "seed0"
    fusion = json.loads((em / "fusion.json").read_text())
    d75 = json.loads((em / "stitch.json").read_text())["penalties"]
    models = FoldModels.load(cfg, store, fold, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device=device, train_ids=f["train"])
    st = cfg.stitch
    beam = st.beam_factor * cfg.retrieval.fine_top_j
    idx = {"val": MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["val"]))),
           "test": MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))}
    mu = float(d75["mu"])
    grid_fixed = [{"lam_switch": float(a), "lam_jump": float(b), "mu": mu} for a, b in itertools.product(st.grid.lam_switch, st.grid.lam_jump)]
    grid_cue = [{"alpha": float(a), "lam_jump": float(b), "mu": mu} for a, b in itertools.product(cfg.cue.tune.alpha, cfg.cue.tune.lam_jump)]
    cue_models = [load_cue(cfg, "ctx", fold, s) for s in seeds]
    hi = float(np.log1p(-cfg.cue.scene.eps) - np.log(cfg.cue.scene.eps))
    d = run_dir(cfg, EXP, fold)
    prev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else None
    out = {"fold": fold, "git_commit": git_commit(), "device": device, "mu": mu, "arms": dict(prev["arms"]) if prev else {}}
    for arm in arms:
        if arm in out["arms"]:
            log.info("fold %d %s: already in %s, skipped", fold, arm, d / "eval.json")
            continue
        t1 = time.time()
        cases = [build_case(store, tokens, fg, idx[split], models, fusion, cfg, split, i, r["parts"], arm, r["cuts"].get(arm, []))
                 for split in ("val", "test") for i, r in enumerate(segfile[split])]
        val = [c for c in cases if c.split == "val"]
        test = [c for c in cases if c.split == "test"]
        systems = [("R0_tuned", "grid", {})]
        for s, (model, scaler) in zip(seeds, cue_models):
            z = predict(model, [scaler(c.raw_tokens) for c in cases])
            systems.append((f"R4_s{s}", "cue", {(c.split, c.idx): zi for c, zi in zip(cases, z)}))
        systems.append(("oracle", "cue", {(c.split, c.idx): np.where(c.labels == 1, hi, -hi) for c in cases}))
        a_out = {"systems": {}, "cases": [{"parts": c.parts, "films": c.films, "duration": c.duration, "n_seg": len(c.segs),
                                           "segs": [[round(a, 4), round(b, 4)] for a, b in c.segs], "dev": c.dev,
                                           "ranks": c.lat.ranks.tolist()} for c in test]}
        for name, kind, logits in systems:
            params, vobj = tune(kind, val, logits, grid_fixed if kind == "grid" else grid_cue, beam, st.tol_s)
            rows = []
            for c in test:
                picks = decode(kind, c, params, logits.get((c.split, c.idx)), beam, st.tol_s)
                rows.append(objective(c, c.lat, picks))
            a_out["systems"][name] = {"params": params, "val_objective": vobj, "rows": rows}
            log.info("fold %d %-18s %-9s %s val %.3f | test switch F1 %.3f track acc %.3f", fold, arm, name, params, vobj,
                     np.mean([r["switch_f1"] for r in rows]), np.mean([r["track_acc"] for r in rows]))
        if arm == "psd_siglip/forced":
            a_out["reference_check"] = reference_check(fold, a_out, cases)
            log.info("fold %d reference check (D121 eval.json): %s", fold, a_out["reference_check"])
        out["arms"][arm] = a_out
        (d / "eval.json").write_text(json.dumps(out, default=float))      # after every arm, so a stopped run resumes
        log.info("fold %d %s: %d val + %d test sequences [%.0f s]", fold, arm, len(val), len(test), time.time() - t1)
        del cases
        if device == "cuda":
            torch.cuda.empty_cache()
    (d / "eval.json").write_text(json.dumps(out, default=float))
    OmegaConf.save(cfg, d / "config.yaml")
    (d / "git_commit.txt").write_text(git_commit() + "\n")
    (d / "seed.txt").write_text(f"{list(seeds)}\n")
    log.info("fold %d done -> %s [%.0f s]", fold, d, time.time() - t0)
    return d


# ---------------------------------------------------------------- stage 3: report
METRICS = (("track_acc", "track accuracy"), ("switch_f1", "switch F1 ±1 s"), ("switch_f1_05", "switch F1 ±0.5 s"))


def per_sequence(arm: dict, system: str) -> dict[str, np.ndarray]:
    """{metric: [n_test]}; R4 = mean over its seeds."""
    names = [n for n in arm["systems"] if n.startswith("R4_s")] if system == "R4" else [system]
    rows = [arm["systems"][n]["rows"] for n in names]
    keys = rows[0][0].keys()
    return {k: np.mean([[r[k] for r in rs] for rs in rows], 0) for k in keys}


def stage_report(n_boot: int = 10000) -> Path:
    from bgm.segmentation import evaluate as ev
    folds = [json.loads((Path("runs/titlesplit") / EXP / f"fold{k}" / "eval.json").read_text()) for k in range(5)]
    arms = [a for a in ARMS if all(a in f["arms"] for f in folds)]
    clusters = []
    for k, f in enumerate(folds):
        clusters += ev.seq_clusters([c["films"] for c in f["arms"][arms[0]]["cases"]], f"f{k}")
    bs = ev.Bootstrap(clusters, n_boot, 0)
    C = len(bs.uniq)

    def boot_mean(x):
        per = np.zeros(C)
        n = np.zeros(C)
        np.add.at(per, bs.inv, x)
        np.add.at(n, bs.inv, 1.0)
        return (bs.M @ per) / np.maximum(bs.M @ n, 1e-12)

    res = {"n_seq": len(clusters), "n_clusters": C, "arms": {}, "reference_check": [f["arms"].get("psd_siglip/forced", {}).get("reference_check") for f in folds]}
    vals = {}
    for a in arms:
        for sy in ("R4", "R0_tuned", "oracle"):
            v = {m: np.concatenate([per_sequence(f["arms"][a], sy)[m] for f in folds]) for m in ("track_acc", "switch_f1", "switch_f1_05", "switches_per_min", "segments_per_min")}
            vals[(a, sy)] = v
            res["arms"].setdefault(a, {})[sy] = {m: {"mean": float(x.mean()), "ci": ev.ci(boot_mean(x))} for m, x in v.items()}
    for base in [b for b in ("psd_siglip/own", "psd_siglip/forced") if b in arms]:
        for a in arms:
            if a == base:
                continue
            for sy in ("R4", "R0_tuned"):
                d = {m: vals[(a, sy)][m] - vals[(base, sy)][m] for m, _ in METRICS}
                res["arms"][a][sy][f"delta vs {base}"] = {m: {"mean": float(x.mean()), "ci": ev.ci(boot_mean(x))} for m, x in d.items()}
    Path("results/seg_retrieval.json").write_text(json.dumps(res, indent=1, default=float))
    out = Path("results/seg_retrieval.md")
    out.write_text(render_md(res, arms))
    log.info("\n%s", out.read_text())
    return out


def _fmt(m: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    s = f"{f.format(m['mean'])} [{f.format(m['ci'][0])}, {f.format(m['ci'][1])}]"
    return f"**{s}**" if signed and (m["ci"][0] > 0 or m["ci"][1] < 0) else s


def arm_name(a: str) -> str:
    seg, mode = a.split("/")
    return "oracle: one segment per clip" if seg == "oracle" else f"{SEGMENTERS[seg]}, {'joins forced' if mode == 'forced' else 'own cuts'}"


def render_md(res: dict, arms: list[str]) -> str:
    L = ["# Step-1 segmenter in the retrieval pipeline (D143)", "",
         f"Title split, original dataset, frozen blocked test sequences, 5 folds pooled ({res['n_seq']} sequences, {res['n_clusters']} "
         "film clusters). Emission = the D121 title-split system (10-head ensemble). Every arm's Viterbi penalties (R0) and "
         "cue alpha / lam_jump (R4, mean of seeds 0-2) are tuned on that arm's own val segments. Truth: joins where the track "
         "group changes. Track accuracy by time overlap with each clip. [95% CI]: cluster bootstrap over film groups, 10,000 "
         "resamples. Bold = CI excludes 0.", ""]
    for sy, title in (("R4", "Default system (R4 cue predictor)"), ("R0_tuned", "Fixed penalties (R0, re-tuned)")):
        L += [f"## {title}", "", "| segmenter | track accuracy | switch F1 ±1 s | switch F1 ±0.5 s | segments / min | switches / min |",
              "| --- | --- | --- | --- | --- | --- |"]
        for a in arms:
            m = res["arms"][a][sy]
            L.append(f"| {arm_name(a)} | {_fmt(m['track_acc'])} | {_fmt(m['switch_f1'])} | {_fmt(m['switch_f1_05'])} | "
                     f"{m['segments_per_min']['mean']:.1f} | {m['switches_per_min']['mean']:.1f} |")
        for base in [b for b in ("psd_siglip/own", "psd_siglip/forced") if b in arms]:
            L += ["", f"Paired differences vs {arm_name(base)}:", "", "| segmenter | Δ track accuracy | Δ switch F1 ±1 s | Δ switch F1 ±0.5 s |",
                  "| --- | --- | --- | --- |"]
            for a in arms:
                d = res["arms"][a][sy].get(f"delta vs {base}")
                if d:
                    L.append(f"| {arm_name(a)} | {_fmt(d['track_acc'], True)} | {_fmt(d['switch_f1'], True)} | {_fmt(d['switch_f1_05'], True)} |")
        L.append("")
    L += ["## Oracle change probabilities (upper reference: p = 1 at the true changes)", "", "| segmenter | track accuracy | switch F1 ±1 s |",
          "| --- | --- | --- |"]
    for a in arms:
        m = res["arms"][a]["oracle"]
        L.append(f"| {arm_name(a)} | {_fmt(m['track_acc'])} | {_fmt(m['switch_f1'])} |")
    if "psd_siglip/forced" in arms:
        L += ["", "Reference check (S0 joins forced vs the D121 eval.json, per fold):", ""]
        for k, rc in enumerate(res["reference_check"]):
            L.append(f"- fold {k}: {json.dumps(rc)}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("segments", "eval", "report"))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    if a.stage == "segments":
        stage_segments(a.device)
    elif a.stage == "eval":
        stage_eval(a.fold, a.device, tuple(a.arms.split(",")))
    else:
        stage_report()


if __name__ == "__main__":
    main()
