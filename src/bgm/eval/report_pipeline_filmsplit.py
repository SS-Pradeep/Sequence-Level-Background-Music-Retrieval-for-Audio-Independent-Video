"""M9 (coarse-to-fine pipeline accuracy), the offset precision of the top-1 window and M10 (Viterbi vs greedy) on
the film-level splits: the 1,248-pair film split (PROGRESS.md D60 -> results/pipeline_filmsplit.md) and the full
4,246-pair dataset on the extended film split (D61 -> results/pipeline_full.md).

Run after `bgm fuse --grid` and `bgm tune-stitch --grid` with that split's overrides (reports/pipeline_<split>_chain.sh):
    python -m bgm.eval.report_pipeline_filmsplit [film | full]

* Runs: runs/<filmsplit | full>/pipeline/fold<k>/seed<s>; heads runs/<...>/semantic/siglip-both; semantic-only scoring
  (fusion.dyn_weight 0, fusion.attr_weight 0; D55 / D56) and no dynamics model (pipeline.dyn_variant null).
* Tables: mean ± std over the runs. Comparisons: per-query values averaged over the fold's seeds, pooled over folds,
  95% cluster bootstrap over FILM groups (D39) for everything per segment or per natural video. A concatenation joins
  3-4 clips of different films, so it has no film group: Viterbi vs greedy on concatenations resamples concatenations.
* Offset precision: `pipeline.offset_rules`. Hit = the rule at the top-1 window, ties sharing it (D34). Chance = the
  rule's mean over every window of every test-fold gallery file (the set a random choice draws from), as
  `grounding_rules` defines chance (D33).
"""
from __future__ import annotations

import collections
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from bgm.analysis.arch_phase0 import film_cfg
from bgm.analysis.full_dataset import BASE_FOLDS, full_cfg
from bgm.data import manifest
from bgm.eval.final import _ci, _ms
from bgm.eval.metrics import cluster_bootstrap, cluster_bootstrap_ratio
from bgm.retrieval.pipeline import OFFSET_TOLS_S

log = logging.getLogger(__name__)
LEDGER = Path("results/arch_ablation.md")
TOLS = [f"{t:g}" for t in OFFSET_TOLS_S]


@dataclass(frozen=True)
class Spec:
    """One split's report: where its runs are and how its text differs."""
    cfg: Callable
    runs: Path
    out: Path
    date: str
    decision: str
    title: str
    data: str                  # opening words of the intro: which data and folds
    heads: str
    ref_runs: Path             # runs shown for reference only (never paired)
    ref_row: str               # their label in the M9 table
    ref_name: str              # their name in sentences
    ref_note: str              # closing sentence of the intro
    concat_file: str
    ledger_split: str
    ledger_where: str          # "the film split" / "the full dataset ..."
    origin: bool = False       # add the old-clips vs new-clips section (the full dataset)


SPECS = {
    "film": Spec(film_cfg, Path("runs/filmsplit/pipeline"), Path("results/pipeline_filmsplit.md"), "2026-10-05", "D60",
                 "# Pipeline on the film split: coarse-to-fine retrieval, offset precision and stitching (D60)",
                 "Film split (`data/manifests/folds_film.json`)", "runs/filmsplit/semantic/siglip-both", Path("runs/pipeline"),
                 "_track_id split, semantic only (old runs, gallery ≈ 250, films shared with training)_",
                 "the track_id split (old runs)",
                 "The track_id-split runs in `runs/pipeline` are untouched and shown only for reference; they are not paired with these.",
                 "data/manifests/concat_sets_film.json", "film", "the film split"),
    "ens": Spec(film_cfg, Path("runs/filmsplit/pipeline_ens10"), Path("results/pipeline_filmsplit_ens10.md"), "2026-10-06", "D65",
                "# Pipeline on the film split with the 10-seed ensemble of heads (D65)",
                "Film split (`data/manifests/folds_film.json`), heads = the ensemble of seeds 0–9 (mean cosine; one run per fold)",
                "runs/filmsplit/semantic/siglip-both (seeds 0–9)", Path("runs/filmsplit/pipeline"),
                "_single heads, same split (D60; 15 runs)_", "the single heads on the same split (D60)",
                "The single-head runs in `runs/filmsplit/pipeline` are shown for reference; the paired comparison is in docs/PROGRESS.md D65.",
                "data/manifests/concat_sets_film.json", "film", "the film split with the 10-seed ensemble"),
    "fullens": Spec(full_cfg, Path("runs/full/pipeline_ens10"), Path("results/pipeline_full_ens10.md"), "2026-10-09", "D73",
                    "# Pipeline on the full dataset (4,246 pairs) with the 10-seed ensemble of heads (D73)",
                    "Full dataset, all 4,246 pairs of `dataset_merged`, extended film split, heads = the ensemble of seeds 0–9 (one run per fold)",
                    "runs/full/semantic/siglip-both (seeds 0–9)", Path("runs/full/pipeline"),
                    "_single heads, same data (D61; 15 runs)_", "the single heads on the same data (D61)",
                    "The single-head runs in `runs/full/pipeline` are shown for reference.",
                    "data/manifests_full/concat_sets_film.json", "extended (4,246)", "the full dataset with the 10-seed ensemble", True),
    "full": Spec(full_cfg, Path("runs/full/pipeline"), Path("results/pipeline_full.md"), "2026-10-06", "D61",
                 "# Pipeline on the full dataset (4,246 pairs): coarse-to-fine retrieval, offset precision and stitching (D61)",
                 "Full dataset, all 4,246 pairs of `dataset_merged`, on the extended film split (`data/manifests_full/folds_film.json`)",
                 "runs/full/semantic/siglip-both", Path("runs/filmsplit/pipeline"),
                 "_1,248-pair film split, semantic only (D60; gallery 196, so not comparable row for row)_",
                 "the 1,248-pair film split (D60)",
                 "The 1,248-pair film-split runs in `runs/filmsplit/pipeline` are untouched and shown only for reference: their galleries "
                 "are 4× smaller, so they are not paired with these.",
                 "data/manifests_full/concat_sets_film.json", "extended (4,246)", "the full dataset (4,246 pairs, extended film split)", True),
}


def load_runs(root: Path, name: str) -> dict[int, list[dict]]:
    """{fold: [one dict of arrays per seed]} for a per-query file of every run under root."""
    out = collections.defaultdict(list)
    for p in sorted(root.glob(f"fold*/seed*/{name}")):
        with np.load(p, allow_pickle=True) as z:
            out[int(p.parent.parent.name[4:])].append({k: z[k] for k in z.files})
    return dict(out)


def seed_mean(runs: dict[int, list[dict]], key: str, pair_key: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """(per-query values averaged over each fold's seeds and pooled over folds, the matching pair_ids)."""
    vals, pairs = [], []
    for k in sorted(runs):
        vals.append(np.nanmean([r[key].astype(float) for r in runs[k]], 0))
        if pair_key:
            assert all((r[pair_key] == runs[k][0][pair_key]).all() for r in runs[k]), f"fold {k}: seeds disagree on the queries"
            pairs.append(runs[k][0][pair_key])
    return np.concatenate(vals), (np.concatenate(pairs) if pair_key else None)


def per_run(runs: dict[int, list[dict]], key: str) -> list[float]:
    return [float(np.nanmean(r[key])) for k in sorted(runs) for r in runs[k]]


def _pct(x: float) -> str:
    return f"{x:.3f}" if x >= 0.0005 or x == 0 else f"{x:.4f}"


def _verdict(b: dict, marginal: float = 0.0) -> str:
    """marginal: a CI lower bound in (0, marginal] reads 'marginally above chance' (used for the conditional offset rows)."""
    if b["lo"] > marginal:
        return "above chance"
    return "marginally above chance" if b["lo"] > 0 else ("below chance" if b["hi"] < 0 else "at chance")


def section_pipeline(P: dict, film: dict, B: int, meta: list[dict], spec: Spec) -> tuple[list[str], dict]:
    fu = meta[0]["fusion"]
    cl = lambda pairs: np.array([film[p] for p in pairs])  # noqa: E731
    names = {"fused": "**coarse-to-fine, semantic only**", "fused_no_coarse": "no coarse stage (every window of every file)",
             "attr": "attributes only (reference; weight 0 in the pipeline)"}
    L = ["## 1. Pipeline accuracy (M9)", "",
         f"Per test segment after the coarse and fine stages, before Viterbi. Shortlist: top {fu['K']} files for the whole "
         f"video and for each segment; windows every {fu['window_hop_s']} s; J = {fu['J']}. Track@1: the top window's file is "
         "in the right track group. Window@1 / @J: the query's own file and an offset within 0.5 s of the aligned one.", "",
         "| system | track@1 | window@1 | window@J |", "| --- | --- | --- | --- |"]
    for s, title in names.items():
        L.append(f"| {title} | " + " | ".join(_ms(per_run(P, f"{s}/{m}")) for m in ("track@1", "window@1", "window@J")) + " |")
    old = [json.loads(p.read_text()) for p in sorted(spec.ref_runs.glob("fold*/seed*/metrics.json"))]
    if old:
        L.append(f"| {spec.ref_row} | " +
                 " | ".join(_ms([m["test"]["sem"][x] for m in old]) for x in ("track@1", "window@1", "window@J")) + " |")
    ch_t, pairs = seed_mean(P, "offset/chance/track", "pair")
    ch_w, _ = seed_mean(P, "offset/chance/window")
    d = {}
    for m in ("track@1", "window@1", "window@J"):
        a, _ = seed_mean(P, f"fused/{m}")
        b, _ = seed_mean(P, f"fused_no_coarse/{m}")
        d[m] = cluster_bootstrap(a - b, cl(pairs), B)
    t1, _ = seed_mean(P, "fused/track@1")
    w1, _ = seed_mean(P, "fused/window@1")
    vs_t, vs_w = cluster_bootstrap(t1 - ch_t, cl(pairs), B), cluster_bootstrap(w1 - ch_w, cl(pairs), B)
    ws, _ = seed_mean(P, "offset/fused/window")
    L += ["", f"- Chance (a random window of the gallery): track@1 {_pct(ch_t.mean())}, window@1 {_pct(ch_w.mean())}. "
          f"Coarse-to-fine minus chance: track@1 {_ci(vs_t)}; window@1 {_ci(vs_w)}.",
          f"- Coarse-to-fine minus no coarse stage: track@1 {_ci(d['track@1'])}; window@1 {_ci(d['window@1'])}; "
          f"window@J {_ci(d['window@J'])}.",
          f"- Tie check: neighbouring 0.5 s windows often pool the same seconds and tie exactly. Window@1 with ties sharing "
          f"the hit is {ws.mean():.3f}, against {w1.mean():.3f} for the first-index choice in the table.", ""]
    return L, {"track@1": per_run(P, "fused/track@1"), "window@1": per_run(P, "fused/window@1"),
               "window@J": per_run(P, "fused/window@J"), "coarse_vs_nc": d["track@1"], "n": len(pairs)}


def section_offset(P: dict, film: dict, B: int) -> tuple[list[str], dict]:
    _, pairs = seed_mean(P, "offset/chance/track", "pair")
    cl = np.array([film[p] for p in pairs])
    dur, _ = seed_mean(P, "dur")
    L = ["## 2. Offset precision of the top-1 window", "",
         "For each test segment: the top-1 (file, offset) after the coarse and fine stages. A hit needs the file in the "
         "right track group AND the offset within the tolerance of the true (aligned) offset; windows tied at the top "
         "share the hit. Chance is exact per query: the share of all gallery windows (every test-fold file, every 0.5 s) "
         "that the same rule accepts.", "",
         "| rule | coarse-to-fine | chance | minus chance [95% CI] | verdict | no coarse stage |", "| --- | --- | --- | --- | --- | --- |"]
    out = {}
    for k, title in [("track", "right track group, any offset")] + [(t, f"right track group and within ±{t} s") for t in TOLS]:
        h, _ = seed_mean(P, f"offset/fused/{k}")
        c, _ = seed_mean(P, f"offset/chance/{k}")
        n, _ = seed_mean(P, f"offset/fused_no_coarse/{k}")
        b = cluster_bootstrap(h - c, cl, B)
        out[k] = {"hit": per_run(P, f"offset/fused/{k}"), "chance": float(c.mean()), "ci": b}
        L.append(f"| {title} | {_ms(out[k]['hit'])} | {_pct(c.mean())} | {_ci(b)} | {_verdict(b)} | {_ms(per_run(P, f'offset/fused_no_coarse/{k}'))} |")
    tr, _ = seed_mean(P, "offset/fused/track")
    L += ["", "The joint numbers are mostly track retrieval: the chance levels are tiny because a random window is rarely in "
          "the right track group at all. The next table asks the offset question on its own. Among the segments whose "
          "top-1 file is in the right track group, how often is the offset within the tolerance, against a random window "
          "of that track group's files?", "",
          "| tolerance | within tolerance, given the right track group | chance, given the right track group | minus chance [95% CI] | verdict |",
          "| --- | --- | --- | --- | --- |"]
    for t in TOLS:
        h, _ = seed_mean(P, f"offset/fused/{t}")
        c, _ = seed_mean(P, f"offset/chance/cond/{t}")
        c = np.nan_to_num(c)
        rate = cluster_bootstrap_ratio(h, tr, cl, B)
        exp = cluster_bootstrap_ratio(tr * c, tr, cl, B)
        diff = cluster_bootstrap_ratio(h - tr * c, tr, cl, B)
        out[f"cond/{t}"] = {"rate": rate["mean"], "chance": exp["mean"], "ci": diff}
        L.append(f"| ±{t} s | {rate['mean']:.3f} | {exp['mean']:.3f} | {_ci(diff)} | {_verdict(diff, 0.01)} |")
    c1 = out["cond/1"]
    L += ["", f"Once the right track group is found, the offset is within ±1 s {c1['rate']:.3f} of the time against "
          f"{c1['chance']:.3f} for a random window of it. The three rows are nested and share the same segments, so they are "
          "one finding, not three independent tests: the offset is only slightly better than random.", "",
          f"Context for the chance levels: segments last {np.median(dur):.1f} s (median) and the files are short, so a file "
          f"offers few distinct windows and a wide tolerance covers most of them. {tr.sum():.0f} of {len(tr):,} segments "
          "(seed-averaged) have their top-1 in the right track group.", ""]
    return L, out


def section_stitch(P: dict, film: dict, B: int, spec: Spec) -> tuple[list[str], dict]:
    S = load_runs(spec.runs, "stitch_perquery.npz")
    if not S:
        return ["## 3. Stitching (M10)", "", "No stitching runs yet.", ""], {}
    M = [json.loads(p.read_text()) for p in sorted(spec.runs.glob("fold*/seed*/stitch_metrics.json"))]
    rows = [("part track accuracy (duration-weighted)", "part_track_acc"), ("switch-boundary F1 (±1 s)", "switch_f1"),
            ("switches / min", "switches_per_min"), ("jumps / min", "jumps_per_min"),
            ("seam loudness jump, median LU per video", "seam_jump_LU_median")]
    _, ccl = seed_mean(S, "concat/viterbi/part_track_acc", "concat/clusters")
    L = ["## 3. Stitching (M10): Viterbi vs greedy", "",
         f"Test concatenations: {len(ccl) // len(S)} per fold, each 3–4 test clips with different track groups joined end to "
         f"end ({spec.concat_file}). Penalties are tuned per run on the val concatenations (objective: "
         "part track accuracy + switch-boundary F1). Greedy takes each segment's top window. CIs resample concatenations.", "",
         "| metric | Viterbi | greedy | Viterbi − greedy [95% CI] |", "| --- | --- | --- | --- |"]
    out = {}
    for title, k in rows:
        v, _ = seed_mean(S, f"concat/viterbi/{k}")
        g, _ = seed_mean(S, f"concat/greedy/{k}")
        b = cluster_bootstrap(v - g, ccl, B)
        out[k] = {"viterbi": per_run(S, f"concat/viterbi/{k}"), "greedy": per_run(S, f"concat/greedy/{k}"), "ci": b}
        L.append(f"| {title} | {_ms(out[k]['viterbi'])} | {_ms(out[k]['greedy'])} | {_ci(b)} |")
    old = [json.loads(p.read_text()) for p in sorted(spec.ref_runs.glob("fold*/seed*/stitch_metrics.json"))]
    if old:
        o = lambda k: _ms([s["test"][k] for s in old])  # noqa: E731
        L += ["", f"For reference, {spec.ref_name}: part track accuracy {o('concat/viterbi/part_track_acc')} vs "
              f"{o('concat/greedy/part_track_acc')}; switch F1 {o('concat/viterbi/switch_f1')} vs {o('concat/greedy/switch_f1')}; "
              f"switches / min {o('concat/viterbi/switches_per_min')} vs {o('concat/greedy/switches_per_min')}."]
    _, npairs = seed_mean(S, "natural/viterbi/track", "natural/pair")
    ncl = np.array([film[p] for p in npairs])
    L += ["", "Natural test videos after decoding, duration-weighted per video (film-group CIs):", "",
          "| measure | Viterbi | greedy | Viterbi − greedy [95% CI] |", "| --- | --- | --- | --- |"]
    for title, k in (("track accuracy", "track"), ("window accuracy (own file, within 0.5 s)", "window")):
        v, _ = seed_mean(S, f"natural/viterbi/{k}")
        g, _ = seed_mean(S, f"natural/greedy/{k}")
        b = cluster_bootstrap(v - g, ncl, B)
        out[f"natural/{k}"] = b
        L.append(f"| {title} | {_ms(per_run(S, f'natural/viterbi/{k}'))} | {_ms(per_run(S, f'natural/greedy/{k}'))} | {_ci(b)} |")
    pens = collections.Counter(json.dumps(m["penalties"], sort_keys=True) for m in M)
    L += ["", f"Tuned penalties (val), most common of {len(M)} runs: " + ", ".join(f"{k} ×{n}" for k, n in pens.most_common(3)) + ".", ""]
    return L, out


def section_origin(P: dict, film: dict, B: int) -> list[str]:
    """Full dataset only: the 1,248 earlier pairs ('old') against the clips of the added films ('new'), same runs."""
    old_ids = set(json.loads(BASE_FOLDS.read_text())["film_group"])
    _, pairs = seed_mean(P, "offset/chance/track", "pair")
    is_old = np.array([p in old_ids for p in pairs])
    cl = np.array([film[p] for p in pairs])
    get = lambda k: seed_mean(P, k)[0]  # noqa: E731
    tr, ch, w1, o1, c1 = get("offset/fused/track"), get("offset/chance/track"), get("fused/window@1"), get("offset/fused/1"), get("offset/chance/1")
    L = ["## 4. Old clips and new clips", "",
         "The same runs and galleries, with the test segments split by origin: the 1,248 pairs of the earlier dataset (old) and "
         "the Demucs-separated clips of the added films (new). Per-segment values averaged over seeds; film-group CIs.", "",
         "| segments of | n | films | track@1 | chance | minus chance [95% CI] | right track and within ±1 s | chance | window@1 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in (("all clips", np.ones(len(pairs), bool)), ("old clips", is_old), ("new clips", ~is_old)):
        b = cluster_bootstrap(tr[m] - ch[m], cl[m], B)
        L.append(f"| {name} | {int(m.sum()):,} | {len(set(cl[m]))} | {tr[m].mean():.3f} | {_pct(ch[m].mean())} | {_ci(b)} | "
                 f"{o1[m].mean():.3f} | {_pct(c1[m].mean())} | {w1[m].mean():.3f} |")
    return L + [""]


def ledger_rows(pl: dict, off: dict, st: dict, gallery: str, spec: Spec) -> list[str]:
    c1 = off["1"]
    DATE, where = spec.date, spec.ledger_where
    tag = "" if where == "the film split" else f", {where}"
    rows = [f"| {DATE} | M9 on {where}: semantic-only coarse-to-fine pipeline (dyn 0, attr 0; pipeline accuracy per test "
            f"segment, not R@k) | {spec.ledger_split} | {gallery} | track@1 {_ms(pl['track@1'])} | window@J {_ms(pl['window@J'])} | "
            f"window@1 {_ms(pl['window@1'])} | coarse − no coarse, track@1: **{pl['coarse_vs_nc']['mean']:+.3f}** "
            f"[{pl['coarse_vs_nc']['lo']:+.3f}, {pl['coarse_vs_nc']['hi']:+.3f}] | informational ({spec.decision}) |",
            f"| {DATE} | Offset precision of the top-1 window (right track group and within ±1 / ±3 / ±5 s; ties shared{tag}) | "
            f"{spec.ledger_split} | "
            f"{gallery} | ±1 s {_ms(c1['hit'])} (chance {_pct(c1['chance'])}) | ±3 s {_ms(off['3']['hit'])} (chance "
            f"{_pct(off['3']['chance'])}) | ±5 s {_ms(off['5']['hit'])} (chance {_pct(off['5']['chance'])}) | ±1 s vs chance: "
            f"**{c1['ci']['mean']:+.3f}** [{c1['ci']['lo']:+.3f}, {c1['ci']['hi']:+.3f}]; given the right track group: "
            f"**{off['cond/1']['ci']['mean']:+.3f}** [{off['cond/1']['ci']['lo']:+.3f}, {off['cond/1']['ci']['hi']:+.3f}] | "
            f"joint {_verdict(c1['ci'])} (mostly track retrieval); offset alone {_verdict(off['cond/1']['ci'], 0.01)} |"]
    if st:
        a, f1, sw = st["part_track_acc"], st["switch_f1"], st["switches_per_min"]
        ok = a["ci"]["lo"] > 0 and f1["ci"]["lo"] > 0
        rows.append(f"| {DATE} | M10 on {where}: Viterbi vs greedy on test concatenations (V / G) | {spec.ledger_split} | {gallery} | "
                    f"part acc {_ms(a['viterbi'])} / {_ms(a['greedy'])} | switch F1 {_ms(f1['viterbi'])} / {_ms(f1['greedy'])} | "
                    f"switches/min {_ms(sw['viterbi'])} / {_ms(sw['greedy'])} | Viterbi − greedy: part acc **{a['ci']['mean']:+.3f}** "
                    f"[{a['ci']['lo']:+.3f}, {a['ci']['hi']:+.3f}]; F1 **{f1['ci']['mean']:+.3f}** [{f1['ci']['lo']:+.3f}, "
                    f"{f1['ci']['hi']:+.3f}] | {'H5 holds (automatic measures)' if ok else 'H5 not shown on both measures'} |")
    return rows


def append_ledger(rows: list[str]) -> int:
    """Append the rows whose 'change' cell is not in the ledger yet (re-running the report does not duplicate them)."""
    have = LEDGER.read_text()
    new = [r for r in rows if r.split("|")[2].strip() not in have]
    if new:
        with open(LEDGER, "a") as f:
            f.write("\n".join(new) + "\n")
    return len(new)


def main(which: str = "film") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    spec = SPECS[which]
    cfg = spec.cfg()
    B = int(cfg.eval.bootstrap_resamples)
    folds = manifest.read_folds(cfg)
    film, doc = folds["film_group"], folds["folds"]
    P = load_runs(spec.runs, "perquery.npz")
    n_runs = sum(len(v) for v in P.values())
    assert n_runs, f"no runs under {spec.runs}"
    meta = [json.loads(p.read_text()) for p in sorted(spec.runs.glob("fold*/seed*/metrics.json"))]
    pairs = manifest.read_pairs(cfg)
    music_of = dict(zip(pairs.pair_id, pairs.music_id))
    gal = [len({music_of[p] for p in doc[str(k)]["test"]}) for k in sorted(P)]
    gallery = f"{np.mean(gal):.0f} ({min(gal)}–{max(gal)})"
    s1, pl = section_pipeline(P, film, B, meta, spec)
    s2, off = section_offset(P, film, B)
    s3, st = section_stitch(P, film, B, spec)
    s4 = section_origin(P, film, B) if spec.origin else []
    L = [spec.title, "",
         f"{spec.data}, {len(P)} folds × {n_runs // len(P)} seeds = {n_runs} runs under `{spec.runs}`; heads "
         f"`{spec.heads}`. Scoring is semantic only (`fusion.dyn_weight` 0, `fusion.attr_weight` 0; D55, D56) "
         f"and no dynamics model is loaded. Test-fold gallery {gallery} music files; {pl['n']:,} test segments. Tables: mean ± std over "
         f"the runs. CIs: per-query values averaged over seeds, 95% cluster bootstrap over film groups (D39), {B:,} resamples, "
         f"unless a section says otherwise. {spec.ref_note}", ""] + s1 + s2 + s3 + s4
    spec.out.write_text("\n".join(L) + "\n")
    n = append_ledger(ledger_rows(pl, off, st, gallery, spec))
    log.info("wrote %s; %d ledger rows appended", spec.out, n)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "film")
