"""`bgm report --final` on the 1,248-pair film split (PROGRESS.md D99) -> results/final.md.

The paper tables, regenerated from runs/filmsplit/ with nothing re-trained by the report itself:

1. Track and segment retrieval (H1): exact chance, zero-shot, single heads (seeds 0-2) and the final 10-head ensemble
   (runs/filmsplit/semantic_ens/siglip-both-ens10, run seed e = head seeds 10e .. 10e + 9; `bgm.eval.ensemble`).
2. Paper baselines reimplemented on our features, single and as size-matched 3-seed ensembles.
3. Temporal grounding (H2) with exact chance levels.
4. Pipeline accuracy (H3): semantic only against every fusion variant run on the film split, and the ensemble.
5. Offset precision and 6. stitching (H5) of the final system (runs/filmsplit/pipeline_ens10).
7. Unlabeled pretraining (H4). 8. The film-split rows of the architecture ledger (results/arch_ablation.md).

Every table: mean ± std over runs (5 folds × 3 seeds unless a row says otherwise). Comparisons: per-query values
averaged over a fold's seeds, pooled over folds, paired 95% cluster bootstrap over FILM groups (D39) with
cfg.eval.bootstrap_resamples resamples; concatenations are resampled as units (they join clips of different films).
"""
from __future__ import annotations

from dataclasses import replace

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.eval.final import _ci, _ms
from bgm.eval.metrics import GROUNDING_RULES, cluster_bootstrap
from bgm.eval.report_filmsplit import METRICS, Split, _cell, _chance, _hits, _row
from bgm.eval.report_pipeline_filmsplit import SPECS as PIPE_SPECS, load_runs, per_run, section_offset, section_stitch, seed_mean
from bgm.eval.rescore_grounding import _load, paired, summarize

log = logging.getLogger(__name__)
SINGLE_SEEDS = (0, 1, 2)
ENS = "siglip-both-ens10"
VARIANTS = (("siglip-both", "SigLIP → CLAP+MERT"), ("siglip-clap", "SigLIP → CLAP"), ("siglip-mert", "SigLIP → MERT"),
            ("clip-clap", "CLIP → CLAP"))
BASELINES = (("cfd", "CFD-style"), ("utcmvmr", "UT-CMVMR-style"), ("mvpt", "MVPt-style"))
ZEROSHOT = (("imagebind", "ImageBind zero-shot"), ("text_bridge", "text bridge, zero-shot"), ("attribute", "mood attributes, zero-shot"))
# pipeline experiment -> label; weights are read from each run's fusion.json
PIPELINES = (("pipeline", "semantic only, single heads"), ("pipeline_dyn20", "+ learned dynamics (fine 0.2)"),
             ("pipeline_vjdyn20", "+ V-JEPA dynamics (fine 0.2)"), ("pipeline_attr20", "+ mood (0.2)"),
             ("pipeline_both20", "+ dynamics and mood (0.2 each)"), ("pipeline_w532", "0.5 / 0.3 / 0.2 (sem / dyn / mood)"),
             ("pipeline_w532_vj", "0.5 / 0.3 / 0.2 with V-JEPA dynamics"), ("pipeline_tuned", "weights tuned on val per run"),
             ("pipeline_ens10", "**final: semantic only, 10-head ensemble**"))
SEC_TITLE = {"track_test_fold": "track, test-fold gallery", "track_all": "track, all 1,248 files", "segment_all": "segment, all windows",
             "segment_other": "segment, other-video windows"}


# ---------------------------------------------------------------- loading
def rank_runs(root: Path, kind: str, variant: str, seeds=None) -> dict[int, list[dict]]:
    """{fold: [{section: ranks}]} for runs/<kind>/<variant>/fold*/seed*/perquery.npz, optionally only `seeds`."""
    out: dict[int, list[dict]] = {}
    for p in sorted(root.glob(f"{kind}/{variant}/fold*/seed*/perquery.npz")):
        if seeds is not None and int(p.parent.name[4:]) not in seeds:
            continue
        with np.load(p) as z:
            out.setdefault(int(p.parent.parent.name[4:]), []).append({s: z[s] for s in SEC_TITLE})
    return out


def zeroshot_runs(root: Path, system: str) -> dict[int, list[dict]]:
    out = {}
    for p in sorted(root.glob("zeroshot/fold*/perquery.npz")):
        with np.load(p) as z:
            out[int(p.parent.name[4:])] = [{s: z[f"{s}/{system}"] for s in SEC_TITLE}]
    return out


def n_runs(runs: dict) -> int:
    return sum(len(v) for v in runs.values())


# ---------------------------------------------------------------- 1. retrieval
def _retrieval_row(split: Split, name: str, runs: dict) -> str:
    t = _row(split, runs, "track_test_fold", False)
    a, s, o = (_row(split, runs, sec, False) for sec in ("track_all", "segment_all", "segment_other"))
    return (f"| {name} | {n_runs(runs)} | " + " | ".join(_cell(t[m]) for m in METRICS) +
            f" | {_cell(a['R@10'])} | {_cell(s['R@1'])} | {_cell(s['R@10'])} | {_cell(o['R@10'])} |")


def _delta(split: Split, a: dict, b: dict, sec: str, k: int, B: int) -> dict:
    """Paired Δ hit@k over the folds both systems have (all 5 once every run exists)."""
    common = sorted(set(a) & set(b))
    a, b = {f: a[f] for f in common}, {f: b[f] for f in common}
    ha, cl = _hits(split, a, sec, k, False)
    hb, clb = _hits(split, b, sec, k, False)
    assert np.array_equal(cl, clb)
    return cluster_bootstrap(ha - hb, cl, B)


def section_retrieval(split: Split, root: Path, B: int) -> tuple[list[str], dict]:
    single = {v: rank_runs(root, "semantic", v, SINGLE_SEEDS) for v, _ in VARIANTS}
    ens = rank_runs(root, "semantic_ens", ENS)
    zs = {s: zeroshot_runs(root, s) for s, _ in ZEROSHOT}
    gal = np.concatenate([split.N[(k, "track_test_fold")] for k in split.folds])
    L = ["## 1. Track and segment retrieval (H1)", "",
         f"Queries: the {sum(len(split.q[(k, 'track_test_fold')]) for k in split.folds):,} test clips (track) and "
         f"{sum(len(split.q[(k, 'segment_all')]) for k in split.folds):,} test segments (segment). Track gallery: the test fold's "
         f"music files, {gal.mean():.0f} ({gal.min()}–{gal.max()}), or all 1,248 files; a hit is any file of the query's track "
         "group. Segment gallery: every test segment's aligned music window (other same-track videos left out); "
         "'other-video' also drops the query video's own other windows. Single heads: seeds 0–2. The final ensemble: run seed e "
         "averages the cosines of head seeds 10e … 10e + 9, so its 3 seeds are disjoint 10-head ensembles.", "",
         "| system | runs | track R@1 | track R@5 | track R@10 | track MedR | track MRR | track R@10, all files | segment R@1 | "
         "segment R@10 | segment R@10, other-video |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ch = {sec: _chance(split, sec, False) for sec in SEC_TITLE}
    L.append(f"| random (exact chance) | — | {_cell(ch['track_test_fold']['R@1'])} | {_cell(ch['track_test_fold']['R@5'])} | "
             f"{_cell(ch['track_test_fold']['R@10'])} | — | — | {_cell(ch['track_all']['R@10'])} | {_cell(ch['segment_all']['R@1'])} | "
             f"{_cell(ch['segment_all']['R@10'])} | {_cell(ch['segment_other']['R@10'])} |")
    for s, name in ZEROSHOT:
        L.append(_retrieval_row(split, name, zs[s]))
    for v, name in VARIANTS:
        if single[v]:
            L.append(_retrieval_row(split, f"heads {name}", single[v]))
    if ens:
        L.append(_retrieval_row(split, "**final: 10-head ensemble, SigLIP → CLAP+MERT**", ens))
    base, ib = single["siglip-both"], zs["imagebind"]
    d = {"h1": _delta(split, base, ib, "track_test_fold", 10, B), "h1_r1": _delta(split, base, ib, "track_test_fold", 1, B),
         "both_clap": _delta(split, base, single["siglip-clap"], "track_test_fold", 10, B)}
    L += ["", "Paired differences (hit@k per query averaged over seeds; film-group CIs):", "",
          f"- **H1** single heads − ImageBind zero-shot, track R@10: {_ci(d['h1'])}; R@1: {_ci(d['h1_r1'])}.",
          f"- CLAP+MERT − CLAP-only music input (single heads), track R@10: {_ci(d['both_clap'])}."]
    if ens:
        d["ens_t10"] = _delta(split, ens, base, "track_test_fold", 10, B)
        d["ens_t1"] = _delta(split, ens, base, "track_test_fold", 1, B)
        d["ens_s10"] = _delta(split, ens, base, "segment_all", 10, B)
        d["ens_ib"] = _delta(split, ens, ib, "track_test_fold", 10, B)
        L += [f"- Final ensemble − single heads: track R@10 {_ci(d['ens_t10'])}; track R@1 {_ci(d['ens_t1'])}; segment R@10 {_ci(d['ens_s10'])}.",
              f"- Final ensemble − ImageBind zero-shot, track R@10: {_ci(d['ens_ib'])}."]
    else:
        L.append("- The ensemble runs are not there yet (`python -m bgm.eval.ensemble`).")
    return L + [""], d


# ---------------------------------------------------------------- 2. baselines
def section_baselines(split: Split, root: Path, B: int) -> list[str]:
    heads = rank_runs(root, "semantic", "siglip-both", SINGLE_SEEDS)
    heads3 = rank_runs(root, "semantic_ens", "siglip-both-ens3")
    L = ["## 2. Paper baselines on our features", "",
         "CFD-, UT-CMVMR- and MVPt-style models reimplemented on the same frozen features and trained on the same folds "
         "(D43). Ensembles of 3 average the cosines of seeds 0–2: one ensemble per fold (5 runs), compared size for size.", "",
         "| system | runs | track R@1 | track R@10 | track MedR | segment R@10 | Δ track R@10, heads − this [95% CI] | Δ segment R@10 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]

    def row(name, runs, ref):
        t, s = _row(split, runs, "track_test_fold", False), _row(split, runs, "segment_all", False)
        dt = _ci(_delta(split, ref, runs, "track_test_fold", 10, B)) if ref is not None else "—"
        ds = _ci(_delta(split, ref, runs, "segment_all", 10, B)) if ref is not None else "—"
        return f"| {name} | {n_runs(runs)} | {_cell(t['R@1'])} | {_cell(t['R@10'])} | {_cell(t['MedR'])} | {_cell(s['R@10'])} | {dt} | {ds} |"

    L.append(row("heads SigLIP → CLAP+MERT, single", heads, None))
    for b, name in BASELINES:
        r = rank_runs(root, "baselines", b, SINGLE_SEEDS)
        if r:
            L.append(row(f"{name}, single", r, heads))
    if heads3:
        L.append(row("heads, ensemble of 3", heads3, None))
        for b, name in BASELINES:
            r = rank_runs(root, "semantic_ens", f"{b}-ens3")
            if r:
                L.append(row(f"{name}, ensemble of 3", r, heads3))
    return L + ["", "Single models: the heads lead every baseline only if the CI clears zero; the ensembled rows test whether the "
                "lead survives when every system is ensembled the same way.", ""]


# ---------------------------------------------------------------- 3. grounding
def _hand_grounding(path: Path) -> SimpleNamespace:
    z = dict(np.load(path))
    g = "grounding/hand_dynamics/"
    return SimpleNamespace(hits={r: z[g + r] for r in GROUNDING_RULES}, chance={r: z[f"{g}{r}_chance"] for r in GROUNDING_RULES},
                           true=z[g + "true"], clusters=z["grounding/clusters"], err_s=z[g + "err_s"], n_tied=z[g + "n_tied"])


def section_grounding(cfg: DictConfig, root: Path, B: int) -> tuple[list[str], dict]:
    pairs = manifest.read_pairs(cfg)
    film = manifest.read_folds(cfg)["film_group"]
    film_of_track = {}
    for p, t in zip(pairs.pair_id, pairs.track_id):
        film_of_track.setdefault(t, film[p])
    by = {}
    for v in ("shiftneg", "vjepa-shiftneg"):
        runs = {}
        for p in sorted(root.glob(f"dynamics/{v}/fold*/seed*/perquery.npz")):
            if int(p.parent.name[4:]) in SINGLE_SEEDS:
                runs.setdefault(int(p.parent.parent.name[4:]), []).append(_load(p))
        if runs:
            by[v] = runs
    by["hand"] = {int(p.parent.name[4:]): [_hand_grounding(p)] for p in sorted(root.glob("zeroshot/fold*/perquery.npz"))}
    ref = by["hand"]
    for name, runs in by.items():
        for k, gs in runs.items():
            for g in gs:
                assert np.array_equal(g.true, ref[k][0].true) and np.array_equal(g.clusters, ref[k][0].clusters), (name, k)
    cl = {k: np.array([film_of_track[t] for t in ref[k][0].clusters]) for k in ref}
    res = {name: summarize(runs, cl, B) for name, runs in by.items()}
    labels = {"shiftneg": "learned dynamics, shifted negatives", "vjepa-shiftneg": "learned dynamics on V-JEPA 2 video tokens",
              "hand": "hand-crafted (cut-on-beat + motion/energy)"}
    rules = {"exact": "exact (±0.1 s)", "strict": "within 0.5 s", "sync": "sync (exact, or the shift still lines up)"}
    L = ["## 3. Temporal grounding (H2)", "",
         f"Each query: a test segment at its exact duration against every 25 Hz offset of its own music file "
         f"({res['hand']['n_queries']:,} queries, {res['hand']['n_clusters']} film groups). Hit = the rule at the top offset, "
         "ties sharing it; chance = the share of offsets the rule accepts (exact per query). Learned rows: 5 folds × 3 seeds; "
         "hand-crafted: 5 folds, deterministic. No 'without shifted negatives' encoder was trained on the film split.", "",
         "| system | rule | accuracy [95% CI] | chance | accuracy − chance [95% CI] | median \\|error\\| (s) |",
         "| --- | --- | --- | --- | --- | --- |"]
    for name, s in res.items():
        med = float(np.median(np.abs(np.concatenate([g.err_s for gs in by[name].values() for g in gs]))))
        for r in GROUNDING_RULES:
            L.append(f"| {labels[name]} | {rules[r]} | {_ci_plain(s[r]['accuracy'])} | {s[r]['chance']['mean']:.3f} | "
                     f"{_ci(s[r]['excess'])} | {med:.2f} |")
    L += ["", "Paired, on (hit − chance) per query:", ""]
    for a in [x for x in ("shiftneg", "vjepa-shiftneg") if x in res]:
        p = paired(res[a], res["hand"], B)
        L.append(f"- **H2** {labels[a]} − hand-crafted: " + "; ".join(f"{rules[r]} {_ci(p[r])}" for r in GROUNDING_RULES) + ".")
    if "shiftneg" in res and "vjepa-shiftneg" in res:
        p = paired(res["vjepa-shiftneg"], res["shiftneg"], B)
        L.append("- V-JEPA 2 − hand-made video curves (both learned): " + "; ".join(f"{rules[r]} {_ci(p[r])}" for r in GROUNDING_RULES) + ".")
    return L + [""], res


def _ci_plain(b: dict) -> str:
    return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


# ---------------------------------------------------------------- 4. pipeline
def section_pipeline_variants(root: Path, film: dict, B: int) -> tuple[list[str], dict]:
    runs = {e: load_runs(root / e, "perquery.npz") for e, _ in PIPELINES}
    runs = {e: r for e, r in runs.items() if r}
    base = runs["pipeline"]
    bt, bpairs = seed_mean(base, "fused/track@1", "pair")
    bw, _ = seed_mean(base, "fused/window@1")
    cl = np.array([film[p] for p in bpairs])
    L =["## 4. Pipeline accuracy (H3)", "",
         f"Per test segment ({len(bpairs):,}) after the coarse and fine stages, before Viterbi. Track@1: the top window's file is in "
         "the right track group. Window@1 / @J: the query's own file and an offset within 0.5 s of the aligned one (J = 20). "
         "Fusion: per-query z-scored branches, weighted sum; weights are the runs' fusion.json (fixed by design, or tuned on val). "
         "Δ: paired against semantic only with the same single heads.", "",
         "| system | runs | coarse weights | fine weights | track@1 | window@1 | window@J | Δ track@1 [95% CI] | Δ window@1 [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out = {}
    for e, name in PIPELINES:
        if e not in runs:
            continue
        P = runs[e]
        meta = [json.loads(p.read_text()) for p in sorted((root / e).glob("fold*/seed*/metrics.json"))]
        wc = _weights([m["fusion"]["w_coarse"] for m in meta], ("sem", "attr"))
        wf = _weights([m["fusion"]["w_fine"] for m in meta], ("sem", "dyn", "attr"))
        t, pairs = seed_mean(P, "fused/track@1", "pair")
        w, _ = seed_mean(P, "fused/window@1")
        assert np.array_equal(pairs, bpairs), e
        dt = cluster_bootstrap(t - bt, cl, B) if e != "pipeline" else None
        dw = cluster_bootstrap(w - bw, cl, B) if e != "pipeline" else None
        out[e] = {"track@1": dt, "window@1": dw}
        L.append(f"| {name} | {n_runs(P)} | {wc} | {wf} | " + " | ".join(_ms(per_run(P, f"fused/{m}")) for m in ("track@1", "window@1", "window@J")) +
                 f" | {_ci(dt) if dt else '—'} | {_ci(dw) if dw else '—'} |")
    if "pipeline_ens10" in runs:
        ch_t, _ = seed_mean(runs["pipeline_ens10"], "offset/chance/track")
        ch_w, _ = seed_mean(runs["pipeline_ens10"], "offset/chance/window")
        L += ["", f"Chance (a random window of the gallery): track@1 {ch_t.mean():.3f}, window@1 {ch_w.mean():.4f}."]
    if "pipeline_tuned" in out:
        t = out["pipeline_tuned"]
        L += ["", f"- **H3** (fusion beats the best single branch, semantic): the pre-specified test is fusion with weights tuned on val "
              f"per run (SPEC §5.8): Δ window@1 {_ci(t['window@1'])}, Δ track@1 {_ci(t['track@1'])}."]
    fixed = [e for e in out if e not in ("pipeline", "pipeline_ens10", "pipeline_tuned")]
    if fixed:
        clear = [e for e in fixed if out[e]["window@1"]["lo"] > 0 or out[e]["track@1"]["lo"] > 0]
        L.append(f"- The {len(fixed)} fixed-weight variants were each chosen before seeing test data but are not corrected for multiple "
                 "comparisons. " + ("CIs clear of zero: " + "; ".join(
                     f"{dict(PIPELINES)[e]}: window@1 {_ci(out[e]['window@1'])}, track@1 {_ci(out[e]['track@1'])}" for e in clear) + "."
                     if clear else "None has a CI clear of zero.") +
                 " The final system scores the semantic branch alone (D55, D56); the ensemble is the only change that moves both measures.")
    return L + [""], out


def _weights(ws: list[dict], keys: tuple[str, ...]) -> str:
    """Fixed weights as 'sem 1.0 / attr 0.0'; weights that vary between runs as their mean."""
    def one(k):
        v = np.array([w.get(k, 0.0) for w in ws], float)
        return f"{k} {v[0]:g}" if np.allclose(v, v[0]) else f"{k} {v.mean():.2f} (tuned)"
    return " / ".join(one(k) for k in keys)


# ---------------------------------------------------------------- 6. stitching extras
def stitch_extras(root: Path, B: int) -> list[str]:
    S = load_runs(root / "pipeline_ens10", "stitch_perquery.npz")
    if not S:
        return []
    v, ccl = seed_mean(S, "concat/viterbi/cob_excess", "concat/clusters")
    c, _ = seed_mean(S, "concat/viterbi/cob_excess_control")
    o, _ = seed_mean(S, "concat/original/cob_excess")
    return ["Within-piece excess cut-on-beat rate (share of cuts inside a piece with a soundtrack beat within ±70 ms, minus the "
            "rate under circular shifts), on the test concatenations:", "",
            "| soundtrack | excess cut-on-beat |", "| --- | --- |",
            f"| Viterbi (final) | {_ms(per_run(S, 'concat/viterbi/cob_excess'))} |",
            f"| beat-snapped random control | {_ms(per_run(S, 'concat/viterbi/cob_excess_control'))} |",
            f"| the clips' original soundtracks | {_ms(per_run(S, 'concat/original/cob_excess'))} |", "",
            f"- Viterbi − beat-snapped control: {_ci(cluster_bootstrap(v - c, ccl, B))}; original − control: "
            f"{_ci(cluster_bootstrap(o - c, ccl, B))} (concatenation CIs).", ""]


# ---------------------------------------------------------------- 7. pretraining
def section_pretraining(split: Split, root: Path, B: int) -> tuple[list[str], dict]:
    a, b = rank_runs(root, "semantic", "siglip-both-pt", SINGLE_SEEDS), rank_runs(root, "semantic", "siglip-both", SINGLE_SEEDS)
    if not a:
        return ["## 7. Unlabeled pretraining (H4)", "", "No pretrained runs.", ""], {}
    L = ["## 7. Unlabeled pretraining (H4)", "",
         "Heads pretrained on the M8 pool of unlabeled song-video clips, then fine-tuned on each fold, against the same heads "
         "trained from scratch (single heads, 5 folds × 3 seeds each).", "",
         "| measure | from scratch | pretrained + fine-tuned | Δ [95% CI] |", "| --- | --- | --- | --- |"]
    out = {}
    for sec, k in (("track_test_fold", 10), ("track_test_fold", 1), ("segment_all", 10)):
        ha, _ = _hits(split, a, sec, k, False)
        hb, _ = _hits(split, b, sec, k, False)
        out[(sec, k)] = d = _delta(split, a, b, sec, k, B)
        L.append(f"| {SEC_TITLE[sec]} R@{k} | {hb.mean():.3f} | {ha.mean():.3f} | {_ci(d)} |")
    return L + [""], out


# ---------------------------------------------------------------- 8. ledger
def section_ledger(path: Path = Path("results/arch_ablation.md")) -> list[str]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text().splitlines():
        c = [x.strip() for x in ln.split("|")[1:-1]]
        if len(c) == 9 and c[2] == "film" and c[0][:4].isdigit():
            rows.append(f"| {c[0]} | {c[1]} | {c[5]} | {c[7]} | {c[8]} |")
    return ["## 8. Architecture ablations (ledger)", "",
            f"Every film-split row of {path} as recorded ({len(rows)} rows): each change against the single heads; paired Δ R@10, "
            "95% cluster bootstrap over film groups (5,000 resamples). KEEP needs the CI to clear zero.", "",
            "| date | change | R@10 (or the row's measure) | Δ [95% CI] | verdict |", "| --- | --- | --- | --- | --- |"] + rows + [""]


# ---------------------------------------------------------------- assembly
def _verdict(b: dict | None) -> str:
    if b is None:
        return "not run"
    return "holds" if b["lo"] > 0 else ("reversed" if b["hi"] < 0 else "not shown")


def write_final_film(cfg: DictConfig) -> Path:
    root, B = Path(cfg.paths.runs), int(cfg.eval.bootstrap_resamples)
    folds = cfg.splits.folds_file
    assert folds in ("folds_film.json", "folds_title.json"), "the final report is on a film split (folds_film.json or folds_title.json, D102)"
    title_split = folds == "folds_title.json"
    split = Split(cfg, folds, "film")
    film = manifest.read_folds(cfg)["film_group"]
    s1, d1 = section_retrieval(split, root, B)
    s2 = section_baselines(split, root, B)
    s3, g = section_grounding(cfg, root, B)
    s4, p4 = section_pipeline_variants(root, film, B)
    spec = replace(PIPE_SPECS["ens"], runs=root / "pipeline_ens10", concat_file=f"data/manifests/{cfg.eval.concat_sets_file}")
    P = load_runs(spec.runs, "perquery.npz")
    s5, _ = section_offset(P, film, B)
    s6, st = section_stitch(P, film, B, spec)
    s5[0], s6[0] = "## 5. Offset precision of the final system's top-1 window", "## 6. Stitching (H5): Viterbi vs greedy, final system"
    s6 = s6 + stitch_extras(root, B)
    s7, p7 = section_pretraining(split, root, B)
    h3 = p4.get("pipeline_tuned", {}).get("window@1")
    h2 = paired(g["shiftneg"], g["hand"], B)["sync"] if "shiftneg" in g else None
    hyp = ["## Summary of the hypotheses", "", "| # | hypothesis | measure | Δ [95% CI] | verdict |", "| --- | --- | --- | --- | --- |",
           f"| H1 | trained heads beat ImageBind zero-shot | track R@10, single heads | {_ci(d1['h1'])} | {_verdict(d1['h1'])} |",
           f"| H2 | learned dynamics beat hand-crafted on sync | grounding sync excess, shifted negatives − hand | "
           f"{_ci(h2) if h2 else '—'} | {_verdict(h2)} |",
           f"| H3 | fusion beats the best single branch | window@1, val-tuned fusion − semantic only | "
           f"{_ci(h3) if h3 else '—'} | {_verdict(h3)} |",
           f"| H4 | unlabeled pretraining helps | track R@10, pretrained − scratch | "
           f"{_ci(p7[('track_test_fold', 10)]) if p7 else '—'} | {_verdict(p7.get(('track_test_fold', 10)))} |",
           f"| H5 | Viterbi beats greedy | switch-boundary F1 on concatenations, final system | "
           f"{_ci(st['switch_f1']['ci']) if st else '—'} | {_verdict(st['switch_f1']['ci']) if st else 'not run'} |", "",
           "The listening study (M12, H5's listener preference) has not been run.", ""]
    ens_runs = n_runs(rank_runs(root, "semantic_ens", ENS))
    pipe_runs = n_runs(P)
    grouping = ("film groups merged by **movie title** (`data/manifests/folds_title.json`, D102: SunNXT / YouTube uploads of one "
                "movie are one group; 219 groups, 228 movies)" if title_split else "**film groups** (`data/manifests/folds_film.json`, D36)")
    L = [f"# Final results ({'title-grouped film split' if title_split else 'film split'})", "",
         f"Generated by `bgm report --final` from {root}/ (src/bgm/eval/final_film.py). 1,248 pairs; folds over {grouping}: "
         "no film is in both training and test; Konjum salangai (266 clips) is train-only, so "
         "the test folds hold the other 982 clips. Test data never touches training, early stopping, calibration, fusion weights or "
         "penalties. Encoders frozen; SigLIP so400m video, CLAP (`larger_clap_music_and_speech`, D1) + MERT-v1-330M music.", "",
         "**Final system (D75):** a 10-head ensemble of SigLIP → CLAP+MERT semantic heads (mean cosine), semantic-only coarse-to-fine "
         "retrieval, Viterbi stitching with penalties tuned on val concatenations. Dynamics, mood, whole-video context, person branch "
         f"and the cue-switch predictor are off: none gave a significant gain. Ensemble runs: {ens_runs} retrieval, {pipe_runs} pipeline "
         f"({'5 folds × 3 disjoint 10-head ensembles' if ens_runs == pipe_runs == 15 else 'INCOMPLETE: 5 folds × 3 disjoint ensembles expected'}).", "",
         f"Tables: mean ± std over runs. Comparisons: paired, per-query values averaged over seeds, 95% cluster bootstrap over film groups, "
         f"{B:,} resamples (concatenations resampled as units). The earlier track_id-split report, whose test folds share films "
         "with training, is results/final_trackid.md.", ""]
    L += hyp + s1 + s2 + s3 + s4 + s5 + s6 + s7 + ([] if title_split else section_ledger())
    out = Path(cfg.paths.results) / "final.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out
