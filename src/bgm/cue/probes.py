"""`bgm cue probe`: does the cue predictor detect music changes, or clip joins? (PROGRESS.md D85)

In the joined test sequences nearly every music change is also a clip join (558 of 569 same-film joins change the
music, no within-clip boundary does), so boundary F1 and AP cannot tell a model that learned music transitions from
one that finds where clips were joined. Two checks on the predictors' logits (no Viterbi):

Test 2, joins without a music change. Frozen sequences (data/manifests/<cue.sequences_file stem>_samemusic.json)
partition each fold's test clips like the blocked test set, except that clips sharing a track group sit side by side
in their film's block (`order_grouped`). On the in-film boundaries of these sequences:
  * change join vs same-music join: a transition model ranks change joins higher; a join detector does not (~0.5);
  * same-music join vs within-clip boundary: a transition model ~0.5; a join detector >> 0.5;
  * change join vs within-clip boundary: the usual task, for reference.
Test 3, duration-matched AUROC on the frozen blocked test sequences (eval.json): change vs no change over the in-film
boundaries, counting only pairs whose neighbouring segments fall in the same duration bins (8 x 8 quantile bins of
log left / right segment length; 4 x 4 as a sensitivity check). Bin edges come from the fold's val sequences.

Reference predictor Rdur: logistic regression on (log left, log right segment length), fit like R2 (baselines.py) on
2,048 training sequences per fold, class-balanced. It sees nothing but segment length.

AUROC: pairs only inside one fold (and one stratum); folds pooled per seed, then the mean over seeds. CIs: 95%
cluster bootstrap (10,000 resamples) over the report's fold-prefixed clusters (test sequences joined by shared film
groups); deltas are paired (same resamples).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

from bgm.cue.sequences import (Clip, boundary_labels, check_sets, clip_table, is_timeline, partition_sequences,
                               sequences_path, with_timeline)
from bgm.cue.tokens import N_SCALAR, SCALARS, sequence_tokens
from bgm.runs import git_commit, run_dir, write_run

log = logging.getLogger(__name__)
RATE = 25
NB = 10000
LOG_DUR = SCALARS.index("log_dur") - N_SCALAR          # column of the raw log segment length in a token
NAMES = {"pair": "R3", "ctx": "R4", "ctx_cross": "R5", "ctx_nodur": "R4nodur", "pair_splice": "R3s", "ctx_splice": "R4s",
         "pair_audio": "R3a", "pair_splice_audio": "R3sa"}
EXTRA = ("R4s", "R3s", "R3a", "R3sa")                    # opt-in systems, shown when present (D94, D98)
SYSTEMS_BASE = ("R4", "R3", "R5", "R4nodur", "R2", "R1", "Rdur")
LABEL = {"R4": "R4 learned + context", "R3": "R3 learned, pair-only", "R5": "R5 = R4 trained cross_film",
         "R4nodur": "R4 without log duration", "R2": "R2 cosine rule (SigLIP)", "R1": "R1 scene cuts",
         "Rdur": "Rdur: segment length only (reference)", "R4s": "R4s = R4 + same-music splices (D94)",
         "R3s": "R3s = R3 + same-music splices (D94)", "R3a": "R3a = R3 + audio-teacher targets (D98)",
         "R3sa": "R3sa = R3s + audio-teacher targets (D98)"}


# ---------------------------------------------------------------- test 2 sequences
def order_grouped(clips: list[Clip], rng: np.random.Generator) -> list[Clip]:
    """A film's clips with each track group's clips side by side. Timeline films: groups in film-time order of their
    first clip, film time inside a group; other films: groups and clips in random order."""
    groups: dict[str, list[Clip]] = {}
    for c in clips:
        groups.setdefault(c.track, []).append(c)
    names = sorted(groups)
    if is_timeline(clips):
        names.sort(key=lambda t: min((c.t_film, c.pair_id) for c in groups[t]))
        return [c for t in names for c in sorted(groups[t], key=lambda c: (c.t_film, c.pair_id))]
    out = []
    for i in rng.permutation(len(names)):
        g = groups[names[i]]
        out += [g[j] for j in rng.permutation(len(g))]
    return out


def build_samemusic_sets(pairs, folds: dict, n_folds: int, lo: float, hi: float, seed: int = 0) -> dict:
    """{fold: [sequence of pair_ids]}: each fold's test clips once, same-track clips adjacent. Seeds [seed, fold, 1, 3, 0]
    (split test, mode 3; modes 0-2 are the main sets)."""
    fg = folds["film_group"]
    return {str(k): partition_sequences(clip_table(pairs, fg, folds["folds"][str(k)]["test"]),
                                        np.random.default_rng([seed, k, 1, 3, 0]), lo, hi, order_grouped)
            for k in range(n_folds)}


def samemusic_path(cfg) -> Path:
    p = sequences_path(cfg)
    return p.with_name(p.stem + "_samemusic.json")


def samemusic_sets(cfg) -> dict:
    """The frozen test-2 sequences (built and written on first use)."""
    path = samemusic_path(cfg)
    if path.exists():
        return json.loads(path.read_text())["sets"]
    from bgm.data import manifest
    pairs = with_timeline(manifest.read_pairs(cfg).set_index("pair_id", drop=False), cfg)
    folds = manifest.read_folds(cfg)
    assert folds.get("split") == "film", "cue sequences are built on the film split (folds_film.json)"
    c = cfg.cue
    sets = build_samemusic_sets(pairs, folds, cfg.splits.n_folds, c.seq_min_s, c.seq_max_s)
    check_sets({k: {"test": {"samemusic": v}} for k, v in sets.items()}, folds)
    for k, v in sets.items():
        assert sorted(p for q in v for p in q) == sorted(folds["folds"][k]["test"]), (k, "not a partition of the test clips")
    path.write_text(json.dumps({"folds_file": cfg.splits.folds_file, "manifest_hash": folds["manifest_hash"],
                                "settings": {k: c[k] for k in ("seq_min_s", "seq_max_s")}, "sets": sets}, indent=1))
    log.info("cue probe: same-music sequences -> %s", path)
    return sets


# ---------------------------------------------------------------- Rdur, the segment-length rule
def duration_feature(log_dur: np.ndarray) -> np.ndarray:
    """[N] log segment lengths -> [N - 1, 2] (log left, log right) per boundary."""
    x = np.asarray(log_dur, np.float64)
    return np.stack([x[:-1], x[1:]], 1)


def log_lengths(w) -> np.ndarray:
    """Segment lengths [N] s -> log lengths, clipped as in tokens.segment_features."""
    return np.log(np.maximum(np.asarray(w, np.float64), 1e-3))


def fit_duration(data, n_seqs: int) -> dict:
    """data: train.FoldData. The R2 recipe (the same training sequences, class-balanced) on segment length only."""
    xs, ys = [], []
    for i in range(n_seqs):
        parts = data.draw(np.random.default_rng([data.fold, 2, i]), "blocked")
        raw, seg_part = sequence_tokens(data.tokens, parts)
        y, _ = boundary_labels(seg_part, parts, data.store.track, data.fg)
        xs.append(duration_feature(raw[:, LOG_DUR])), ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    lr = LogisticRegression(C=1e4, class_weight="balanced").fit(x, y)
    return {"coef": lr.coef_[0].tolist(), "b": float(lr.intercept_[0]), "n": int(len(y)), "pos": int(y.sum())}


def duration_logits(params: dict, log_dur: np.ndarray) -> np.ndarray:
    return duration_feature(log_dur) @ np.asarray(params["coef"]) + params["b"]


def load_duration(cfg: DictConfig, fold: int) -> dict:
    return json.loads((Path(cfg.paths.runs) / cfg.cue.exp / "duration" / f"fold{fold}" / "params.json").read_text())


# ---------------------------------------------------------------- probe logits on the test-2 sequences
def probe_fold(cfg: DictConfig, fold: int, store, tokens: dict, seeds=(0, 1, 2),
               variants=("pair", "ctx", "ctx_cross", "ctx_nodur")) -> Path:
    from bgm.cue.baselines import cosine_logits, load_cosine
    from bgm.cue.evaluate import clusters_of
    from bgm.cue.model import predict
    from bgm.cue.scenes import SceneCache, replay_cuts, scene_logits, sequence_scores
    from bgm.cue.train import load_cue
    from bgm.data import manifest
    from bgm.retrieval.index import concat_query
    fg = manifest.read_folds(cfg)["film_group"]
    cases = []
    for i, parts in enumerate(samemusic_sets(cfg)[str(fold)]):
        vq = concat_query(store, parts)
        raw, seg_part = sequence_tokens(tokens, parts)
        assert len(raw) == len(vq.segments), (parts, len(raw), len(vq.segments))
        y, typ = boundary_labels(seg_part, parts, store.track, fg)
        cases.append(SimpleNamespace(idx=i, mode="samemusic", parts=list(parts), films=[fg[p] for p in parts], raw=raw,
                                     labels=y, types=typ, w=np.array([b - a for a, b in vq.segments]),
                                     btimes=np.array([a for a, _ in vq.segments[1:]]), duration=vq.duration,
                                     n25=[int(round(d * RATE)) for _, _, d in vq.meta["parts"]]))
    for c, k in zip(cases, clusters_of(cases)):
        c.cluster = k
    sc = cfg.cue.scene
    cache, cos, dur = SceneCache(cfg), load_cosine(cfg, fold), load_duration(cfg, fold)
    systems = {"R1": {"seed": None, "logits": {}}, "R2": {"seed": None, "logits": {}}, "Rdur": {"seed": None, "logits": {}}}
    for c in cases:
        cuts = [f / RATE for f in replay_cuts(sequence_scores(cache, c.parts, c.n25), sc.threshold, sc.min_scene_s)]
        systems["R1"]["logits"][str(c.idx)] = scene_logits(cuts, c.btimes, sc.snap_s, sc.eps).tolist()
        systems["R2"]["logits"][str(c.idx)] = cosine_logits(cos, c.raw).tolist()
        systems["Rdur"]["logits"][str(c.idx)] = duration_logits(dur, c.raw[:, LOG_DUR]).tolist()
    for v in variants:
        for s in seeds:
            model, scaler = load_cue(cfg, v, fold, s)
            z = predict(model, [scaler(c.raw) for c in cases])
            systems[f"{NAMES[v]}_s{s}"] = {"seed": s, "logits": {str(c.idx): zi.tolist() for c, zi in zip(cases, z)}}
    out = {"fold": fold, "git_commit": git_commit(), "sequences_file": samemusic_path(cfg).name,
           "cases": [{"idx": c.idx, "parts": c.parts, "films": c.films, "cluster": c.cluster, "duration": c.duration,
                      "labels": c.labels.tolist(), "types": c.types, "w": c.w.tolist()} for c in cases],
           "systems": systems}
    d = run_dir(cfg, f"{cfg.cue.exp}/probes", fold)
    (d / "probes.json").write_text(json.dumps(out, default=float))
    OmegaConf.save(cfg, d / "config.yaml")
    (d / "git_commit.txt").write_text(git_commit() + "\n")
    (d / "seed.txt").write_text(f"{list(seeds)}\n")
    t = np.concatenate([np.array(c.types) for c in cases])
    y = np.concatenate([c.labels for c in cases])
    log.info("cue probe fold %d: %d sequences, same-music joins %d, change joins %d -> %s", fold, len(cases),
             int(((t == "film") & (y == 0)).sum()), int(((t == "film") & (y == 1)).sum()), d)
    return d


def run_probe(cfg: DictConfig, folds, seeds=(0, 1, 2), variants=("pair", "ctx", "ctx_cross", "ctx_nodur")) -> list[Path]:
    """Test-2 sequences (frozen on first use), Rdur fits, and the probe logits of every fold."""
    from bgm.cue.tokens import build_tokens
    from bgm.cue.train import FoldData
    from bgm.eval.store import FeatureStore
    samemusic_sets(cfg)
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    out = []
    for k in folds:
        p = fit_duration(FoldData(cfg, k, store, tokens), cfg.cue.train.pos_weight_seqs)
        d = run_dir(cfg, f"{cfg.cue.exp}/duration", k)
        (d / "params.json").write_text(json.dumps(p, indent=1))
        write_run(d, cfg, p, None)
        log.info("cue Rdur fold %d: coef %s b %.2f (%d boundaries, %d changes)", k, np.round(p["coef"], 2).tolist(), p["b"],
                 p["n"], p["pos"])
        out.append(probe_fold(cfg, k, store, tokens, seeds, tuple(variants)))
    return out


# ---------------------------------------------------------------- stratified AUROC with a cluster bootstrap
def pair_counts(z: np.ndarray, pos: np.ndarray, neg: np.ndarray, stratum: np.ndarray, cl: np.ndarray,
                K: int) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney counts split by cluster: C[i, j] = sum over (positive in cluster i, negative in cluster j, same
    stratum) of 1[z+ > z-] + 0.5 * 1[z+ == z-]; D[i, j] = the number of such pairs. AUROC = C.sum() / D.sum()."""
    C, D = np.zeros(K * K), np.zeros(K * K)
    for s in np.unique(stratum[pos | neg]):
        p, n = np.where(pos & (stratum == s))[0], np.where(neg & (stratum == s))[0]
        if not len(p) or not len(n):
            continue
        d = z[p][:, None] - z[n][None, :]
        idx = (cl[p][:, None] * K + cl[n][None, :]).ravel()
        C += np.bincount(idx, weights=((d > 0) + 0.5 * (d == 0)).ravel(), minlength=K * K)
        D += np.bincount(idx, minlength=K * K)
    return C.reshape(K, K), D.reshape(K, K)


def cluster_weights(K: int, nb: int = NB, seed: int = 0) -> np.ndarray:
    """[nb, K] multiplicity of each cluster in each resample."""
    draws = np.random.default_rng(seed).integers(0, K, (nb, K))
    W = np.zeros((nb, K))
    np.add.at(W, (np.repeat(np.arange(nb), K), draws.ravel()), 1.0)
    return W


def auc_draws(runs: list[tuple[np.ndarray, np.ndarray]], W: np.ndarray) -> tuple[float, np.ndarray]:
    """runs: [(C, D)] per seed. Point AUROC (mean over seeds) and its value in each resample."""
    point = float(np.mean([C.sum() / D.sum() for C, D in runs]))
    v = np.mean([((W @ C) * W).sum(1) / np.maximum(((W @ D) * W).sum(1), 1e-12) for C, D in runs], 0)
    return point, v


def ci(point: float, v: np.ndarray) -> dict:
    return {"mean": point, "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}


def fmt(d: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


class Boundaries:
    """In-film boundaries of a set of test sequences, pooled over folds: per boundary its fold, cluster, type, label,
    left / right segment length, and each run's logit."""

    def __init__(self, cases_by_fold: dict[int, list[dict]], logits_by_fold: dict[int, dict[str, dict]],
                 rdur: dict[int, dict]):
        cols = {k: [] for k in ("fold", "cluster", "type", "label", "lw_l", "lw_r")}
        self.z: dict[str, dict[int, list]] = {}        # run name -> fold -> [logit arrays]
        for fold, cases in sorted(cases_by_fold.items()):
            for c in cases:
                key = str(c["key"])
                sel = np.isin(c["types"], ("clip", "film"))
                lw = log_lengths(c["w"])
                cols["fold"] += [fold] * int(sel.sum())
                cols["cluster"] += [f"{fold}:{c['cluster']}"] * int(sel.sum())
                cols["type"] += list(np.array(c["types"])[sel])
                cols["label"] += list(np.array(c["labels"])[sel])
                cols["lw_l"] += list(lw[:-1][sel])
                cols["lw_r"] += list(lw[1:][sel])
                runs = dict(logits_by_fold[fold])
                runs["Rdur"] = {"seed": None, "logits": {key: duration_logits(rdur[fold], lw)}}
                for name, s in runs.items():
                    if key in s["logits"]:
                        self.z.setdefault(name, {}).setdefault(fold, []).append(np.asarray(s["logits"][key], float)[sel])
        self.fold = np.array(cols["fold"])
        uniq, self.cl = np.unique(np.array(cols["cluster"]), return_inverse=True)
        self.K = len(uniq)
        self.type, self.label = np.array(cols["type"]), np.array(cols["label"], int)
        self.lw_l, self.lw_r = np.array(cols["lw_l"]), np.array(cols["lw_r"])

    def runs_of(self, system: str) -> list[np.ndarray]:
        """Full-length logit vectors, one per seed (seedless systems: one)."""
        names = sorted(n for n in self.z if n == system or (n.rsplit("_s", 1)[0] == system and n.rsplit("_s", 1)[-1].isdigit()))
        out = []
        for n in names:
            z = np.concatenate([np.concatenate(self.z[n][f]) for f in sorted(self.z[n])])
            assert len(z) == len(self.label), (n, len(z), len(self.label))
            out.append(z)
        return out

    def counts(self, system: str, pos: np.ndarray, neg: np.ndarray, stratum: np.ndarray) -> list:
        return [pair_counts(z, pos, neg, stratum, self.cl, self.K) for z in self.runs_of(system)]


def strata(B: Boundaries, edges: dict[int, tuple[np.ndarray, np.ndarray]] | None) -> np.ndarray:
    """Stratum id per boundary: the fold, or the fold x left bin x right bin."""
    if edges is None:
        return B.fold.copy()
    out = np.zeros(len(B.fold), int)
    for f, (el, er) in edges.items():
        m = B.fold == f
        out[m] = f * 10000 + np.digitize(B.lw_l[m], el) * 100 + np.digitize(B.lw_r[m], er)
    return out


def val_edges(evals: list[dict], nb: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per fold, quantile edges (nb bins) of log left / right segment length over the val in-film boundaries."""
    out = {}
    q = np.linspace(0, 1, nb + 1)[1:-1]
    for e in evals:
        ll, lr = [], []
        for c in e["cases"]:
            if c["split"] == "val":
                sel = np.isin(c["types"], ("clip", "film"))
                lw = log_lengths(c["w"])
                ll.append(lw[:-1][sel]), lr.append(lw[1:][sel])
        out[e["fold"]] = (np.quantile(np.concatenate(ll), q), np.quantile(np.concatenate(lr), q))
    return out


# ---------------------------------------------------------------- report section (called by report.write_report)
def load_probes(cfg: DictConfig, n_folds: int) -> list[dict] | None:
    root = Path(cfg.paths.runs) / cfg.cue.exp / "probes"
    paths = [root / f"fold{k}" / "probes.json" for k in range(n_folds)]
    return [json.loads(p.read_text()) for p in paths] if all(p.exists() for p in paths) else None


def blocked_boundaries(cfg, evals: list[dict]) -> Boundaries:
    cases, logits = {}, {}
    for e in evals:
        f = e["fold"]
        cases[f] = [dict(c, key=f"blocked/{c['idx']}") for c in e["cases"] if c["split"] == "test" and c["mode"] == "blocked"]
        logits[f] = {n: s for n, s in e["systems"].items() if n.rsplit("_s", 1)[0] in SYSTEMS_BASE + EXTRA}
    return Boundaries(cases, logits, {e["fold"]: load_duration(cfg, e["fold"]) for e in evals})


def samemusic_boundaries(cfg, probes: list[dict]) -> Boundaries:
    cases = {p["fold"]: [dict(c, key=str(c["idx"])) for c in p["cases"]] for p in probes}
    logits = {p["fold"]: {n: s for n, s in p["systems"].items() if n != "Rdur"} for p in probes}
    return Boundaries(cases, logits, {p["fold"]: load_duration(cfg, p["fold"]) for p in probes})


def report_lines(cfg: DictConfig, evals: list[dict]) -> tuple[list[str], list[str]]:
    """(summary bullets, section lines); empty if `bgm cue probe` has not run for every fold."""
    probes = load_probes(cfg, len(evals))
    if probes is None:
        return [], []
    extra = tuple(x for x in EXTRA if any(k.startswith(x + "_s") for k in probes[0]["systems"]))   # D94, D98
    SYSTEMS = SYSTEMS_BASE[:2] + extra + SYSTEMS_BASE[2:]
    res = {}
    # ---- test 3: duration-matched AUROC on the blocked test sequences
    B = blocked_boundaries(cfg, evals)
    W = cluster_weights(B.K)
    pos, neg = B.label == 1, B.label == 0
    st = {"plain": strata(B, None), "4x4": strata(B, val_edges(evals, 4)), "8x8": strata(B, val_edges(evals, 8))}
    draws3 = {}
    for s in SYSTEMS:
        for k, S in st.items():
            draws3[s, k] = auc_draws(B.counts(s, pos, neg, S), W)
    L = ["", "## Clip joins or music changes? (D85, `bgm cue probe`)", "",
         "Every music change in these test sequences is also a clip join (558 of 569 same-film joins change the music; no "
         "within-clip boundary does), so the F1 and AP above cannot separate a model that learned music transitions from one "
         "that finds clip joins. Two checks on the predictors' logits (no Viterbi), src/bgm/cue/probes.py. Rdur = logistic "
         "regression on log left / right segment length only, fit like R2 on training sequences. AUROC: pairs inside one "
         "fold, folds pooled per seed, mean over seeds; 95% cluster-bootstrap CIs (10,000 resamples, the clusters above); "
         "deltas paired.", "",
         "### Test 3: AUROC with segment length held fixed (blocked test sequences, in-film boundaries)", "",
         "Change vs no change, counting only pairs whose neighbouring segments fall in the same bins of log left and log "
         "right segment length (quantile edges from the fold's val sequences). A model that only reads length drops to "
         "0.5; Rdur shows how much length the bins leave.", "",
         f"Boundaries: {int(pos.sum())} changes, {int(neg.sum())} without a change; {B.K} clusters.", "",
         "| predictor | AUROC | AUROC within 4 × 4 length bins | within 8 × 8 length bins | Δ vs Rdur, 8 × 8 | Δ vs R2, 8 × 8 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for s in SYSTEMS:
        p, v = draws3[s, "8x8"]
        dd = fmt(ci(p - draws3["Rdur", "8x8"][0], v - draws3["Rdur", "8x8"][1]), True) if s != "Rdur" else "—"
        d2 = fmt(ci(p - draws3["R2", "8x8"][0], v - draws3["R2", "8x8"][1]), True) if s != "R2" else "—"
        L.append(f"| {LABEL[s]} | {fmt(ci(*draws3[s, 'plain']))} | {fmt(ci(*draws3[s, '4x4']))} | {fmt(ci(p, v))} | {dd} | {d2} |")
    res["t3"] = {s: ci(*draws3[s, "8x8"]) for s in SYSTEMS}
    res["t3_plain"] = {s: ci(*draws3[s, "plain"]) for s in SYSTEMS}
    # ---- test 2: joins without a music change
    S2 = samemusic_boundaries(cfg, probes)
    W2 = cluster_weights(S2.K)
    st2 = strata(S2, None)
    within, same, change = S2.type == "clip", (S2.type == "film") & (S2.label == 0), (S2.type == "film") & (S2.label == 1)
    comps = (("change join vs same-music join", change, same, "want > 0.5"),
             ("same-music join vs within-clip", same, within, "want ≈ 0.5"),
             ("change join vs within-clip", change, within, "the usual task"))
    mins = sum(c["duration"] for p in probes for c in p["cases"]) / 60
    L += ["", "### Test 2: joins where the music does not change", "",
          "Frozen sequences (data/manifests/" + samemusic_path(cfg).name + ") that partition each fold's test clips like "
          "the blocked test set, but with clips that share a track group side by side. A predictor that learned music "
          "changes ranks change joins above same-music joins and puts same-music joins with the within-clip boundaries; "
          "a join detector does neither.", "",
          f"Sequences: {sum(len(p['cases']) for p in probes)}, {mins:.0f} min, {S2.K} clusters. In-film boundaries: "
          f"{int(within.sum())} within a clip, {int(same.sum())} same-music joins, {int(change.sum())} change joins.", "",
          "| predictor | " + " | ".join(f"{t} ({w})" for t, _, _, w in comps) + " | Δ vs Rdur, change vs same-music |",
          "| --- |" + " --- |" * (len(comps) + 1)]
    draws2 = {}
    for s in SYSTEMS:
        for t, a, b, _ in comps:
            draws2[s, t] = auc_draws(S2.counts(s, a, b, st2), W2)
    t0 = comps[0][0]
    for s in SYSTEMS:
        p, v = draws2[s, t0]
        dd = fmt(ci(p - draws2["Rdur", t0][0], v - draws2["Rdur", t0][1]), True) if s != "Rdur" else "—"
        L.append(f"| {LABEL[s]} | " + " | ".join(fmt(ci(*draws2[s, t])) for t, _, _, _ in comps) + f" | {dd} |")
    res["t2"] = {s: {t: ci(*draws2[s, t]) for t, _, _, _ in comps} for s in SYSTEMS}
    if "R4s" in SYSTEMS:                                    # D94: paired, the splice model vs R4
        L += ["", "### Same-music splices (D94) and audio-teacher targets (D98): paired AUROC differences", "",
              "| A − B | test 3, within 8 × 8 length bins | test 2, change vs same-music join | test 2, same-music join vs within-clip |",
              "| --- | --- | --- | --- |"]
        for a, b in (("R4s", "R4"), ("R3s", "R3"), ("R3a", "R3"), ("R3sa", "R3s"), ("R3sa", "R3")):
            if a in SYSTEMS:
                L.append(f"| {a} − {b} | {fmt(ci(draws3[a, '8x8'][0] - draws3[b, '8x8'][0], draws3[a, '8x8'][1] - draws3[b, '8x8'][1]), True)} | "
                         + " | ".join(fmt(ci(draws2[a, t][0] - draws2[b, t][0], draws2[a, t][1] - draws2[b, t][1]), True) for t, _, _, _ in comps[:2]) + " |")
        res["d94"] = {"t3": ci(draws3["R4s", "8x8"][0] - draws3["R4", "8x8"][0], draws3["R4s", "8x8"][1] - draws3["R4", "8x8"][1]),
                      "t2": ci(draws2["R4s", t0][0] - draws2["R4", t0][0], draws2["R4s", t0][1] - draws2["R4", t0][1])}
    summary = [f"* Clip joins or music changes (D85): with segment length held fixed (8 × 8 bins) R4's AUROC is "
               f"{fmt(res['t3']['R4'])} (plain {fmt(res['t3_plain']['R4'])}; Rdur, length only, plain "
               f"{fmt(res['t3_plain']['Rdur'])}). On joins that keep the music, R4 ranks change joins above same-music joins "
               f"with AUROC {fmt(res['t2']['R4'][t0])}, and same-music joins above within-clip boundaries with "
               f"{fmt(res['t2']['R4'][comps[1][0]])}."]
    if "d94" in res:
        summary.append(f"* Splices, probes (D94): R4s − R4 with length held fixed {fmt(res['d94']['t3'], True)}; change vs same-music "
                       f"join {fmt(res['d94']['t2'], True)} (R4s {fmt(res['t2']['R4s'][t0])}, R4 {fmt(res['t2']['R4'][t0])}).")
    return summary, L
