"""`bgm cue report` -> results/cue_switch.md (Step 4; PROGRESS.md D78+). Reads runs/<paths.runs>/<cue.exp>/eval/fold*/eval.json.

Aggregation (CLAUDE.md rule 11, D39, D45):
  * per test sequence, a system's values are averaged over its seeds first (R0-R2 have none);
  * boundary F1 is pooled over sequences (micro: 2TP / (2TP + FP + FN)); its CI resamples clusters, and paired
    differences resample the same clusters for both systems;
  * per-sequence metrics (part track accuracy, switch F1, switches / min, ...) use `cluster_bootstrap` of the
    per-sequence values (differences: per-sequence differences);
  * clusters: blocked test sequences joined by shared film groups (union-find, so resampling sequences resamples
    disjoint film groups); single-film sequences by film group; fold-prefixed;
  * "mean ± std" columns are over the 15 fold x seed runs (5 for seedless systems).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, roc_auc_score

from bgm.eval.metrics import cluster_bootstrap

log = logging.getLogger(__name__)
SEEDLESS = ("R0_stored", "R0_tuned", "R1", "R2", "greedy", "oracle")
SEEDED = ("R3", "R4", "R5", "R6", "R4nodur", "R3s", "R4s", "R3a", "R3sa")
LABEL = {"R0_stored": "R0 fixed penalties, stored D75", "R0_tuned": "R0 fixed penalties, re-tuned", "R1": "R1 scene cuts",
         "R2": "R2 cosine rule", "R3": "R3 learned, pair-only", "R4": "R4 learned + context", "R5": "R5 = R4 trained cross_film",
         "R6": "R6 = R4 + D72 context score (**emission changed**)", "R4nodur": "R4 without log duration (diagnostic)",
         "greedy": "greedy (no transitions, reference)", "oracle": "oracle p (true changes, reference)",
         "R3s": "R3s = R3 + same-music splices (D94)", "R4s": "R4s = R4 + same-music splices (D94)",
         "R3a": "R3a = R3 + audio-teacher targets (D98)", "R3sa": "R3sa = R3s + audio-teacher targets (D98)"}
MODES = ("blocked", "same_film", "timeline")
SEQ_METRICS = (("part_track_acc", "part track accuracy"), ("switch_f1", "switch F1 ±1 s (per sequence)"),
               ("switches_per_min", "switches / min"), ("jumps_per_min", "jumps / min"),
               ("seam_jump_LU_median", "seam loudness jump, median LU"), ("cob_excess", "within-piece excess cut-on-beat"))
NB = 10000


def load_evals(cfg: DictConfig) -> list[dict]:
    root = Path(cfg.paths.runs) / cfg.cue.exp / "eval"
    out = [json.loads((root / f"fold{k}" / "eval.json").read_text()) for k in range(cfg.splits.n_folds)
           if (root / f"fold{k}" / "eval.json").exists()]
    assert out, f"no eval.json under {root}"
    return out


def base(name: str) -> str:
    return name.rsplit("_s", 1)[0] if name.rsplit("_s", 1)[-1].isdigit() and name.split("_")[0] in SEEDED else name


class Table:
    """Per (system, mode): per-case arrays (seed-averaged) and per-run values."""

    def __init__(self, evals: list[dict]):
        self.evals = evals
        self.systems = [s for s in SEEDLESS + SEEDED if any(base(n) == s for n in evals[0]["systems"])]
        self.cases = {m: [] for m in MODES}             # (fold, idx, cluster, case dict)
        for e in evals:
            for c in e["cases"]:
                if c["split"] == "test":
                    self.cases[c["mode"]].append((e["fold"], c["idx"], f"{e['fold']}:{c['cluster']}", c))

    def runs(self, sysname: str):
        """[(fold, seed, system dict)] of one base system."""
        out = []
        for e in self.evals:
            for n, s in e["systems"].items():
                if base(n) == sysname:
                    out.append((e["fold"], s["seed"], s))
        return out

    def rows(self, sysname: str, mode: str) -> list[list[dict]]:
        """Per test case of `mode` (in self.cases order): the list of that case's rows over seeds."""
        by = {}
        for fold, _, s in self.runs(sysname):
            for r in s["rows"]:
                if r["mode"] == mode:
                    by.setdefault((fold, r["idx"]), []).append(r)
        return [by[(f, i)] for f, i, _, _ in self.cases[mode]]

    def per_case(self, sysname: str, mode: str, key: str) -> np.ndarray:
        return np.array([np.nanmean([r[key] for r in rs]) if any(np.isfinite(r[key]) for r in rs) else np.nan
                         for rs in self.rows(sysname, mode)], float)

    def counts(self, sysname: str, mode: str, key: str) -> np.ndarray:
        """[n_cases, 3] tp, fp, fn averaged over seeds."""
        return np.array([np.mean([r[key] for r in rs], 0) for rs in self.rows(sysname, mode)], float)

    def clusters(self, mode: str) -> np.ndarray:
        return np.array([c for _, _, c, _ in self.cases[mode]])


def f1_of(c: np.ndarray) -> float:
    tp, fp, fn = c.sum(0)
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")


def boot_f1(c: np.ndarray, clusters: np.ndarray, c2: np.ndarray | None = None, seed: int = 0) -> dict:
    """Pooled F1 (or F1(c) - F1(c2)) with a cluster-bootstrap CI; NaN for an empty set (D123: no timeline sequence
    reaches 300 s)."""
    if len(c) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0}
    uniq, inv = np.unique(clusters, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=c[:, j], minlength=len(uniq)) for j in range(3)], 1)
    S2 = None if c2 is None else np.stack([np.bincount(inv, weights=c2[:, j], minlength=len(uniq)) for j in range(3)], 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (NB, len(uniq)))

    def f1s(Sx):
        t = Sx[draws].sum(1)
        den = 2 * t[:, 0] + t[:, 1] + t[:, 2]
        return np.where(den > 0, 2 * t[:, 0] / np.maximum(den, 1e-12), np.nan)
    v = f1s(S) if S2 is None else f1s(S) - f1s(S2)
    point = f1_of(c) if c2 is None else f1_of(c) - f1_of(c2)
    return {"mean": point, "lo": float(np.nanquantile(v, 0.025)), "hi": float(np.nanquantile(v, 0.975)), "n_clusters": len(uniq)}


def fmt_ci(d: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def run_values(T: Table, sysname: str, mode: str, fn) -> np.ndarray:
    """fn(rows of one run, cases of that fold) per (fold, seed) run."""
    out = []
    for fold, seed, s in T.runs(sysname):
        rows = [r for r in s["rows"] if r["mode"] == mode]
        out.append(fn(rows))
    return np.array(out, float)


def mean_std(x: np.ndarray, k: int = 3) -> str:
    x = x[np.isfinite(x)]
    return f"{x.mean():.{k}f} ± {x.std():.{k}f}" if len(x) else "—"


def predictor_scores(T: Table, sysname: str, mode: str, group: tuple[str, ...]) -> np.ndarray:
    """AP per run over the boundaries of `group` in `mode` test cases."""
    out = []
    cases = {(f, c["idx"]): c for f, _, _, c in T.cases[mode]}
    for fold, seed, s in T.runs(sysname):
        ys, zs = [], []
        for key, z in s["logits"].items():
            m, i = key.split("/")
            if m != mode:
                continue
            c = cases[(fold, int(i))]
            sel = np.isin(c["types"], group)
            ys.append(np.array(c["labels"])[sel]), zs.append(np.array(z)[sel])
        y, z = np.concatenate(ys), np.concatenate(zs)
        out.append((average_precision_score(y, z), roc_auc_score(y, z)) if len(np.unique(y)) == 2 else (np.nan, np.nan))
    return np.array(out, float)


# ---------------------------------------------------------------- the report
def write_report(cfg: DictConfig) -> Path:
    evals = load_evals(cfg)
    T = Table(evals)
    full = Path(str(cfg.paths.manifests)).name == "manifests_full"           # D93: the full dataset
    title = str(cfg.splits.get("folds_file", "")) == "folds_title.json"
    split = ("Extended film split of the full dataset (4,246 pairs; D49, D93)" if full
             else "Title split (folds_title.json, D102)" if title else "Film split (D36)")
    L = ["# Cue-switch predictor: learned music-change probabilities in Viterbi stitching", "",
         f"Generated by `bgm cue report` (src/bgm/cue/report.py) from {cfg.paths.runs}/{cfg.cue.exp}/eval/fold*/eval.json; plan in "
         "docs/cue_switch_plan.md; decisions D78 onward in docs/PROGRESS.md.", "",
         f"{split}, {len(evals)} folds; learned systems × 3 seeds. Emission = the D75 system (10-seed ensemble, "
         "semantic only), identical for every system except R6. Penalties / α, λ_jump tuned on val sequences only; μ (key / "
         "tempo) = the D75 value of the fold in every run. CIs: 95% cluster bootstrap (10,000 resamples) over film-group "
         "clusters; deltas are paired.", ""]
    # ---- summary (numbers computed here; interpretation in PROGRESS.md)
    cb = T.clusters("blocked")
    from bgm.cue.probes import report_lines
    probe_summary, probe_section = report_lines(cfg, evals)
    from bgm.cue import within_clip                     # D98: join-free test, when the audio-teacher systems ran
    w_summary, w_section = within_clip.report_lines(cfg, evals)
    probe_summary, probe_section = probe_summary + w_summary, probe_section + w_section

    def d(a, b, key="in_film/1", md="blocked"):
        return fmt_ci(boot_f1(T.counts(a, md, key), T.clusters(md), T.counts(b, md, key)), True)
    if all(x in T.systems for x in ("R0_tuned", "R1", "R2", "R3", "R4", "R5", "R6")):
        f4 = boot_f1(T.counts("R4", "blocked", "in_film/1"), cb)
        L += ["## Summary", "",
              f"* Headline (blocked test sequences, music changes inside one film, boundary F1 ±1 s): R4 {fmt_ci(f4)}; "
              f"R4 − R0 re-tuned {d('R4', 'R0_tuned')}, − R0 stored {d('R4', 'R0_stored')}, − R1 {d('R4', 'R1')}, − R2 {d('R4', 'R2')}.",
              f"* Context: R4 − R3 (pair-only) {d('R4', 'R3')}. Shortcut check: R5 − R4 {d('R5', 'R4')} inside films, "
              f"{d('R5', 'R4', 'cross/1')} between films.",
              f"* D72 conflict: R6 − R4 {d('R6', 'R4')} inside films, {d('R6', 'R4', 'cross/1')} between films; part track accuracy "
              f"{fmt_ci(cluster_bootstrap(T.per_case('R6', 'blocked', 'part_track_acc') - T.per_case('R4', 'blocked', 'part_track_acc'), cb, NB), True)}.",
              f"* Duration diagnostic: R4 without log duration − R4 {d('R4nodur', 'R4')}; − R0 re-tuned {d('R4nodur', 'R0_tuned')}.",
              f"* Secondary single-film sets, R4 − R0 re-tuned: same_film {d('R4', 'R0_tuned', md='same_film')}, timeline "
              f"{d('R4', 'R0_tuned', md='timeline')}.",
              f"* Oracle p (true changes) reaches {fmt_ci(boot_f1(T.counts('oracle', 'blocked', 'in_film/1'), cb))}: the room left for a better predictor.",
              "* ±0.5 s and ±1 s give the same numbers on concatenations: switches happen only at segment boundaries, every true "
              "change is a segment boundary and segments are ≥ 2 s long."] + probe_summary
        if "R3a" in T.systems:                               # D98
            L += [f"* Audio-teacher targets (D98): R3a − R0 re-tuned {d('R3a', 'R0_tuned')}, R3a − R3 {d('R3a', 'R3')}; "
                  f"R3sa − R0 re-tuned {d('R3sa', 'R0_tuned')}, R3sa − R3s {d('R3sa', 'R3s')}; between films R3a − R3 "
                  f"{d('R3a', 'R3', 'cross/1')}; timeline R3a − R3 {d('R3a', 'R3', md='timeline')}, R3sa − R3s "
                  f"{d('R3sa', 'R3s', md='timeline')}."] if "R3sa" in T.systems else []
        if "R4s" in T.systems:                               # D94
            L += [f"* Same-music splices (D94): R4s − R4 {d('R4s', 'R4')} inside films, {d('R4s', 'R4', 'cross/1')} between films; "
                  f"R4s − R0 re-tuned {d('R4s', 'R0_tuned')}; timeline R4s − R4 {d('R4s', 'R4', md='timeline')}."
                  + (f" R3s − R3 {d('R3s', 'R3')}." if "R3s" in T.systems else "")]
        L += [""]
    # ---- data
    L += ["## Test sequences", "", "| set | sequences | clusters | minutes | boundaries: inside a clip / same-film join / cross-film join | "
          "music changes inside a film / between films | true changes per min |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in MODES:
        cs = [c for _, _, _, c in T.cases[m]]
        if not cs:                                        # D123: a set can be empty (no timeline film reaches 300 s)
            L.append(f"| {m} | 0 | 0 | 0 | — | — | — |")
            continue
        typ = np.concatenate([np.array(c["types"]) for c in cs])
        lab = np.concatenate([np.array(c["labels"]) for c in cs])
        mins = sum(c["duration"] for c in cs) / 60
        L.append(f"| {m} | {len(cs)} | {len(set(T.clusters(m)))} | {mins:.0f} | {(typ == 'clip').sum()} / {(typ == 'film').sum()} / "
                 f"{(typ == 'cross').sum()} | {lab[np.isin(typ, ['clip', 'film'])].sum()} / {lab[typ == 'cross'].sum()} | "
                 f"{lab.sum() / mins:.2f} |")
    L += ["", "Val (tuning, early stopping): " + ", ".join(f"fold {e['fold']} {sum(c['split'] == 'val' for c in e['cases'])}" for e in evals)
          + " blocked sequences.", ""]
    # ---- emission identity
    L += ["## Emission check", ""]
    ok = all(e["check_emission"]["part_track_acc_equal"] and e["check_emission"]["switch_f1_equal"] for e in evals)
    L.append(f"* The D75 test concatenations, decoded through this code with the stored D75 penalties, reproduce the saved D75 "
             f"stitch_perquery.npz exactly in {sum(1 for _ in evals)} of {len(evals)} folds: **{ok}**.")
    same = True
    for e in evals:
        dg = {n: s["emission_digest"] for n, s in e["systems"].items()}
        ref = dg["R0_stored"]
        same &= all(v == ref for n, v in dg.items() if base(n) != "R6")
        same &= all(v != ref for n, v in dg.items() if base(n) == "R6")
    L.append(f"* Every system except R6 decoded byte-identical lattices (sha256 of every layer's candidates and unary costs), "
             f"and R6 decoded different ones, in every fold: **{same}**.")
    assert ok and same, "emission identity check failed"
    rk = {}
    for kind in ("base", "ctx"):
        r = np.concatenate([np.array(c["ranks"][kind], float) for m in MODES for _, _, _, c in T.cases[m]])
        rk[kind] = {k: float((r <= k).mean()) for k in (1, 5, 10)}
    L += ["", "| per test segment, emission only | track R@1 | track R@5 | track R@10 |", "| --- | --- | --- | --- |",
          f"| R0–R5 (and R4nodur, greedy, oracle): one lattice per sequence | {rk['base'][1]:.3f} | {rk['base'][5]:.3f} | {rk['base'][10]:.3f} |",
          f"| R6 (context score in the emission) | {rk['ctx'][1]:.3f} | {rk['ctx'][5]:.3f} | {rk['ctx'][10]:.3f} |", "",
          "Track R@k = the true track group among the top k shortlisted files ranked by their best window. Identical for "
          "R0–R5 by construction (same lattices, asserted above).", ""]
    # ---- headline
    mode = "blocked"
    cl = T.clusters(mode)
    L += ["## Headline: boundaries inside one film, blocked test sequences", "",
          "Boundary F1 pooled over sequences, at the track-group changes inside a film (same-film joins) against the switches "
          "Viterbi places inside a film (any within-clip switch is a false positive). Predictor AP / AUROC: the change logits on "
          "the same boundaries (mean ± std over runs).", "",
          "| system | boundary F1 ±1 s | boundary F1 ±0.5 s | precision ±1 s | recall ±1 s | predictor AP | AUROC | mean ± std over runs (F1 ±1 s) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in T.systems:
        c1 = T.counts(s, mode, "in_film/1")
        c05 = T.counts(s, mode, "in_film/0.5")
        tp, fp, fn = c1.sum(0)
        ap = predictor_scores(T, s, mode, ("clip", "film")) if s not in ("R0_stored", "R0_tuned", "greedy") else None
        per_run = run_values(T, s, mode, lambda rows: f1_of(np.array([r["in_film/1"] for r in rows], float)))
        L.append(f"| {LABEL[s]} | {fmt_ci(boot_f1(c1, cl))} | {fmt_ci(boot_f1(c05, cl))} | {tp / max(tp + fp, 1e-9):.3f} | "
                 f"{tp / max(tp + fn, 1e-9):.3f} | {mean_std(ap[:, 0]) if ap is not None else '—'} | "
                 f"{mean_std(ap[:, 1]) if ap is not None else '—'} | {mean_std(per_run)} |")
    # ---- paired comparisons
    comps = [("R4", "R0_tuned"), ("R4", "R0_stored"), ("R4", "R1"), ("R4", "R2"), ("R4", "R3"), ("R5", "R4"), ("R6", "R4"),
             ("R4nodur", "R4"), ("R3", "R0_tuned"), ("R0_tuned", "R0_stored"), ("R2", "R0_tuned"), ("R1", "R0_tuned"),
             ("R4s", "R4"), ("R3s", "R3"), ("R4s", "R0_tuned"), ("R3s", "R0_tuned"), ("R4s", "R1"),           # D94 rows when present
             ("R3a", "R0_tuned"), ("R3sa", "R0_tuned"), ("R3a", "R3"), ("R3sa", "R3s"), ("R3sa", "R3")]      # D98 rows when present
    comps = [(a, b) for a, b in comps if a in T.systems and b in T.systems]
    L += ["", "### Paired differences (blocked test sequences)", "",
          "| A − B | in-film boundary F1 ±1 s | ±0.5 s | cross-film boundary F1 ±1 s | part track accuracy | switch F1 ±1 s (per sequence) | switches / min |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for a, b in comps:
        row = [f"{a} − {b}"]
        for key in ("in_film/1", "in_film/0.5", "cross/1"):
            row.append(fmt_ci(boot_f1(T.counts(a, mode, key), cl, T.counts(b, mode, key)), True))
        for key in ("part_track_acc", "switch_f1", "switches_per_min"):
            d = T.per_case(a, mode, key) - T.per_case(b, mode, key)
            row.append(fmt_ci(cluster_bootstrap(d, cl, NB), True))
        L.append("| " + " | ".join(row) + " |")
    # ---- other boundary groups and sets
    for title, md, key, grp in (("Boundaries between films (blocked test sequences)", "blocked", "cross", ("cross",)),
                                ("All boundaries (blocked test sequences)", "blocked", "all", ("clip", "film", "cross")),
                                (f"Secondary: same_film test sequences (one film each, random order, ≥ {cfg.cue.single_film_min_s:.0f} s)", "same_film", "in_film", ("clip", "film")),
                                (f"Secondary: timeline test sequences (one timestamped film each, film order, ≥ {cfg.cue.single_film_min_s:.0f} s)", "timeline", "in_film", ("clip", "film"))):
        if not T.cases[md]:
            L += ["", f"## {title}", "", "No sequences in this set."]
            continue
        clm = T.clusters(md)
        L += ["", f"## {title}", "", "| system | boundary F1 ±1 s | ±0.5 s | predictor AP | Δ F1 ±1 s vs R0 re-tuned | Δ vs R4 |",
              "| --- | --- | --- | --- | --- | --- |"]
        for s in T.systems:
            c1 = T.counts(s, md, f"{key}/1")
            ap = predictor_scores(T, s, md, grp) if s not in ("R0_stored", "R0_tuned", "greedy") else None
            d0 = fmt_ci(boot_f1(c1, clm, T.counts("R0_tuned", md, f"{key}/1")), True) if s != "R0_tuned" else "—"
            d4 = fmt_ci(boot_f1(c1, clm, T.counts("R4", md, f"{key}/1")), True) if s != "R4" and "R4" in T.systems else "—"
            L.append(f"| {LABEL[s]} | {fmt_ci(boot_f1(c1, clm))} | {fmt_ci(boot_f1(T.counts(s, md, f'{key}/0.5'), clm))} | "
                     f"{mean_std(ap[:, 0]) if ap is not None else '—'} | {d0} | {d4} |")
    # ---- existing stitching metrics
    for md in (m for m in MODES if T.cases[m]):
        clm = T.clusters(md)
        L += ["", f"## Existing stitching metrics: {md} test sequences", "",
              "| system | " + " | ".join(t for _, t in SEQ_METRICS) + " |", "| --- |" + " --- |" * len(SEQ_METRICS)]
        for s in T.systems:
            cells = []
            for k, _ in SEQ_METRICS:
                v = T.per_case(s, md, k)
                cells.append(fmt_ci(cluster_bootstrap(v, clm, NB)) if np.isfinite(v).any() else "—")
            L.append(f"| {LABEL[s]} | " + " | ".join(cells) + " |")
    # ---- tuned parameters
    L += ["", "## Tuned transition parameters (val)", "", "| system | per fold (seed 0 for learned systems) | val objective, mean |", "| --- | --- | --- |"]
    for s in T.systems:
        ps, vo = [], []
        for fold, seed, sd in T.runs(s):
            vo.append(sd["val_objective"])
            if seed in (None, 0):
                p = sd["params"]
                ps.append(f"{fold}: " + ", ".join(f"{k} {v:g}" for k, v in p.items()))
        vo = [v for v in vo if np.isfinite(v)]
        L.append(f"| {LABEL[s]} | {'; '.join(ps) if any(p.split(': ', 1)[1] for p in ps) else '—'} | "
                 f"{np.mean(vo):.3f} |" if vo else f"| {LABEL[s]} | — | — |")
    L += probe_section
    out = Path(cfg.cue.get("report_file", None) or "results/cue_switch.md")   # the brief: results/cue_switch.md, whatever paths.results is
    out.write_text("\n".join(L) + "\n")
    log.info("cue report -> %s", out)
    return out
