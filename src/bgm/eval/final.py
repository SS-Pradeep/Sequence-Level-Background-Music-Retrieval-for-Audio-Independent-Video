"""`bgm report --final --split trackid` (SPEC M11): the track_id-split report, results/final_trackid.md, from runs/.
The paper's report is on the film split: bgm.eval.final_film (results/final.md).

Every table: mean ± std over runs (5 folds x 3 seeds unless noted). Comparisons: per-query values averaged over seeds
first, then a paired cluster bootstrap (10,000 resamples) over track_id groups (natural clips) or concatenations.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.eval.metrics import cluster_bootstrap


def _ms(vals) -> str:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "—"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def _ci(b: dict) -> str:
    return f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"


def _hit_at(k: int):
    """Per-run transform for `_seed_mean`: rank -> hit (rank <= k), so seeds average hits, not ranks."""
    return lambda x: (x <= k).astype(float)


def _seed_mean(root: str, pattern: str, keys: list[str], per_run=None) -> tuple[dict, dict]:
    """Per fold: per-query arrays averaged over seeds; returns ({key: pooled array}, {"clusters": pooled}).
    per_run: optional transform applied to each run's array before the seed average (e.g. `_hit_at(10)` on ranks)."""
    by_fold = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{root}/fold*/seed*/{pattern}")):
        by_fold[int(Path(p).parent.parent.name[4:])].append(dict(np.load(p, allow_pickle=True)))
    f = per_run or (lambda x: x.astype(float))
    out = {k: [] for k in keys}
    extra = collections.defaultdict(list)
    for k in sorted(by_fold):
        runs = by_fold[k]
        for key in keys:
            out[key].append(np.nanmean([f(r[key]) for r in runs], 0))
        for key in runs[0]:
            if key.endswith("clusters"):
                extra[key].append(runs[0][key])
    return {k: np.concatenate(v) for k, v in out.items() if v}, {k: np.concatenate(v) for k, v in extra.items()}


def section_zeroshot_semantic(cfg) -> list[str]:
    runs = Path(cfg.paths.runs)
    zs = [json.loads(p.read_text()) for p in sorted(runs.glob("zeroshot/fold*/metrics.json"))]
    L = ["## 1. Track and segment retrieval (M5, M6)", "",
         "Test-fold gallery ≈ 250 files. Zero-shot rows: mean ± std over 5 folds; trained rows: 5 folds × 3 seeds.", "",
         "| system | track R@1 | track R@10 | track MedR | track R@10, all 1,248 | segment R@10 |", "| --- | --- | --- | --- | --- | --- |"]
    for name in ("random", "imagebind", "text_bridge", "attribute"):
        g = lambda s, m: [z[s][name].get(m) for z in zs]  # noqa: E731
        L.append(f"| {name.replace('_', ' ')} | {_ms(g('track_test_fold', 'R@1'))} | {_ms(g('track_test_fold', 'R@10'))} | "
                 f"{_ms(g('track_test_fold', 'MedR')) if name != 'random' else '—'} | {_ms(g('track_all', 'R@10'))} | {_ms(g('segment_all', 'R@10'))} |")
    for v in ("siglip-clap", "siglip-mert", "siglip-both", "clip-clap"):
        M = [json.loads(p.read_text())["test"] for p in sorted(runs.glob(f"semantic/{v}/fold*/seed*/metrics.json"))]
        if not M:
            continue
        g = lambda s, m: [x[s][m] for x in M]  # noqa: E731
        L.append(f"| heads {v} | {_ms(g('track_test_fold', 'R@1'))} | {_ms(g('track_test_fold', 'R@10'))} | {_ms(g('track_test_fold', 'MedR'))} | "
                 f"{_ms(g('track_all', 'R@10'))} | {_ms(g('segment_all', 'R@10'))} |")
    # H1 and variant comparisons
    L.append("")
    base, cl = None, None
    for v in ("siglip-both", "siglip-clap"):
        a, ex = _seed_mean(str(runs / "semantic" / v), "perquery.npz", ["track_test_fold"], per_run=_hit_at(10))
        if "track_test_fold" not in a:
            continue
        hit = a["track_test_fold"]
        ib = np.concatenate([(np.load(p)["track_test_fold/imagebind"] <= 10).astype(float) for p in sorted(runs.glob("zeroshot/fold*/perquery.npz"))])
        cl = ex["track_test_fold/clusters"]
        L.append(f"- **H1** heads {v} − ImageBind, track R@10: {_ci(cluster_bootstrap(hit - ib, cl, cfg.eval.bootstrap_resamples))}")
        if base is None:
            base = hit
        else:
            L.append(f"- siglip-both − siglip-clap, track R@10: {_ci(cluster_bootstrap(base - hit, cl, cfg.eval.bootstrap_resamples))}")
    return L + [""]


def section_dynamics(cfg) -> list[str]:
    """Grounding with chance levels, from results/grounding_chance.json (`bgm eval --exp grounding`)."""
    from bgm.eval.report_dynamics import h2_lines, load_grounding_chance
    from bgm.eval.rescore_grounding import markdown
    res = load_grounding_chance(cfg)
    L = ["## 2. Temporal grounding (M7, H2)", ""] + markdown(res, "###")
    return L + ["**H2**", ""] + h2_lines(res) + [""]


def section_pipeline(cfg, exp: str = "pipeline", title: str = "## 3. Pipeline accuracy (M9, H3)") -> list[str]:
    runs = Path(cfg.paths.runs) / exp
    M = [json.loads(p.read_text()) for p in sorted(runs.glob("fold*/seed*/metrics.json"))]
    if not M:
        return [title, "", "No runs yet.", ""]
    J = M[0]["fusion"]["J"]
    L = [title, "",
         f"Per test segment after the coarse and fine stages, before Viterbi (test-fold gallery; windows every "
         f"{M[0]['fusion']['window_hop_s']} s; J = {J}). Window hit: right file and offset within 0.5 s of the aligned one.", "",
         "| system | track@1 | window@1 | window@J |", "| --- | --- | --- | --- |"]
    names = {"sem": "semantic only", "dyn": "dynamics only", "attr": "attributes only", "fused": "**fused**", "fused_no_coarse": "fused, no coarse stage"}
    for s, title in names.items():
        L.append(f"| {title} | " + " | ".join(_ms([m["test"][s][x] for m in M]) for x in ("track@1", "window@1", "window@J")) + " |")
    keys = [f"{s}/{x}" for s in names for x in ("track@1", "window@1", "window@J")]
    a, ex = _seed_mean(str(runs), "perquery.npz", keys)
    cl = ex["clusters"]
    best_single = max(("sem", "dyn", "attr"), key=lambda s: a[f"{s}/window@1"].mean())
    L += ["", f"- **H3** fused − best single branch ({names[best_single]}), window@1: "
          f"{_ci(cluster_bootstrap(a['fused/window@1'] - a[f'{best_single}/window@1'], cl, cfg.eval.bootstrap_resamples))}; track@1: "
          f"{_ci(cluster_bootstrap(a['fused/track@1'] - a[f'{best_single}/track@1'], cl, cfg.eval.bootstrap_resamples))}",
          f"- fused − fused without the coarse stage, window@1: "
          f"{_ci(cluster_bootstrap(a['fused/window@1'] - a['fused_no_coarse/window@1'], cl, cfg.eval.bootstrap_resamples))}; track@1: "
          f"{_ci(cluster_bootstrap(a['fused/track@1'] - a['fused_no_coarse/track@1'], cl, cfg.eval.bootstrap_resamples))}"]
    wc = [m["fusion"]["w_coarse"] for m in M]
    wf = [m["fusion"]["w_fine"] for m in M]
    L += ["", f"Chosen weights (val), over {len(M)} runs: coarse " + ", ".join(f"{b} {_ms([w[b] for w in wc])}" for b in ("sem", "attr")) +
          "; fine " + ", ".join(f"{b} {_ms([w[b] for w in wf])}" for b in ("sem", "dyn", "attr")) + ".",
          "Most common fine weights: " + ", ".join(f"{k} ×{n}" for k, n in collections.Counter(
              json.dumps(w, sort_keys=True) for w in wf).most_common(3)) + "."]
    S = [json.loads(p.read_text()) for p in sorted(runs.glob("fold*/seed*/stitch_metrics.json"))]
    if S:
        L += ["", "After decoding, duration-weighted over each natural test video:", "",
              "| decoder | track | window |", "| --- | --- | --- |"]
        for d in ("viterbi", "greedy"):
            L.append(f"| {d} | {_ms([s['test'][f'natural/{d}/track'] for s in S])} | {_ms([s['test'][f'natural/{d}/window'] for s in S])} |")
    return L + [""]


def section_stitch(cfg, exp: str = "pipeline", title: str = "## 4. Stitching (M10, H5)") -> list[str]:
    runs = Path(cfg.paths.runs) / exp
    S = [json.loads(p.read_text()) for p in sorted(runs.glob("fold*/seed*/stitch_metrics.json"))]
    if not S:
        return [title, "", "No runs yet.", ""]
    t = lambda k: [s["test"][k] for s in S]  # noqa: E731
    L = [title, "",
         f"Test concatenations: {cfg.eval.concat_videos_per_fold} per fold, 3–4 clips with different tracks. Penalties tuned per run on "
         "the val concatenations (objective: part track accuracy + switch-boundary F1).", "",
         "| metric | Viterbi | greedy |", "| --- | --- | --- |"]
    rows = [("part track accuracy (duration-weighted)", "part_track_acc"), ("switch-boundary F1 (±1 s)", "switch_f1"),
            ("switches / min", "switches_per_min"), ("jumps / min", "jumps_per_min"),
            ("seam loudness jump, median LU (all seams)", "seam_jump_LU_median_all"),
            ("within-piece excess cut-on-beat", "cob_excess"), ("… beat-snapped control", "cob_excess_control")]
    for title, k in rows:
        L.append(f"| {title} | {_ms(t(f'concat/viterbi/{k}'))} | {_ms(t(f'concat/greedy/{k}'))} |")
    L.append(f"| within-piece excess cut-on-beat, original soundtracks | {_ms(t('concat/original/cob_excess'))} | |")
    keys = [f"concat/{d}/{k}" for d in ("viterbi", "greedy") for k in ("part_track_acc", "switch_f1", "cob_excess", "cob_excess_control")]
    a, ex = _seed_mean(str(runs), "stitch_perquery.npz", keys)
    cl = ex["concat/clusters"]
    L += [""]
    for title, k in (("part track accuracy", "part_track_acc"), ("switch-boundary F1", "switch_f1")):
        L.append(f"- **H5** Viterbi − greedy, {title}: {_ci(cluster_bootstrap(a[f'concat/viterbi/{k}'] - a[f'concat/greedy/{k}'], cl, cfg.eval.bootstrap_resamples))}")
    L.append(f"- Viterbi − beat-snapped control, within-piece excess cut-on-beat: "
             f"{_ci(cluster_bootstrap(a['concat/viterbi/cob_excess'] - a['concat/viterbi/cob_excess_control'], cl, cfg.eval.bootstrap_resamples))}")
    pens = collections.Counter(json.dumps(s["penalties"], sort_keys=True) for s in S)
    L += ["", "Tuned penalties (val), most common: " + ", ".join(f"{k} ×{n}" for k, n in pens.most_common(3)) + "."]
    return L + [""]


def section_pretraining(cfg) -> list[str]:
    runs = Path(cfg.paths.runs)
    L = ["## 6. Unlabeled pretraining (M8, H4)", ""]
    rep = Path("data/unlabeled/report.json")
    if rep.exists():
        r = json.loads(rep.read_text())
        L += [f"Pretraining set: {r.get('clips_after_dedupe', r.get('clips'))} clips from {r.get('source_videos_used')} song videos "
              f"(Data/video_bgm_pairs, {r.get('hours_after_dedupe', r.get('hours', 0)):.1f} h); {len(r.get('chromaprint_duplicates', {}))} videos "
              f"dropped as Chromaprint duplicates of the labeled set, {len(r.get('clap_duplicates_dropped', []))} clips dropped by CLAP "
              f"(cosine ≥ 0.90); {len(r.get('without_audio', []))} files without audio skipped.", ""]
    rows = []
    # grounding: the scratch runs' re-scored arrays (grounding_rules); the fine-tuned runs were scored with them directly
    for kind, scratch, pt, key, metric, scratch_file in (
            ("semantic", "siglip-both", "siglip-both-pt", "track_test_fold", "track R@10", "perquery.npz"),
            ("dynamics", "shiftneg", "shiftneg-pt", "sync", "grounding sync", "grounding_rescored.npz")):
        per_run = _hit_at(10) if kind == "semantic" else None
        a, ex = _seed_mean(str(runs / kind / pt), "perquery.npz", [key], per_run)
        b, _ = _seed_mean(str(runs / kind / scratch), scratch_file, [key], per_run)
        if key not in a or key not in b:
            continue
        va, vb = a[key], b[key]
        cl = ex[f"{key}/clusters"] if kind == "semantic" else ex["clusters"]
        bs = cluster_bootstrap(va - vb, cl, cfg.eval.bootstrap_resamples)
        rows.append(f"| {metric} | {vb.mean():.3f} | {va.mean():.3f} | {_ci(bs)} |")
    if rows:
        L += ["Per-query values averaged over seeds; paired cluster bootstrap over track groups.", "",
              "| metric | from scratch | pretrained + fine-tuned | Δ [95% CI] |", "| --- | --- | --- | --- |"] + rows
    else:
        L.append("No pretrained runs yet.")
    return L + [""]


def section_long_video(cfg) -> list[str]:
    L = ["## 5. Long videos and section-level alignment", ""]
    p = Path("reports/long_video.json")
    if p.exists():
        j = json.loads(p.read_text())
        one, lt = j["1.mp4"], j["long_test"]
        L += ["Chunked querying (≈ 20 s chunks, 20 s library windows every 5 s; track score = mean over chunks of the best window) "
              "and ensembles of per-model standardized scores (reports/long_video.json).", "",
              "| query set | whole, single | chunked, single | whole, ensemble | chunked, ensemble |", "| --- | --- | --- | --- | --- |",
              f"| 1.mp4: rank of its original soundtrack among {one['gallery']} (15 models) | median {one['whole']['single_median']:.0f} | "
              f"median {one['chunked']['single_median']:.0f} | {one['whole']['ensemble_rank']} | {one['chunked']['ensemble_rank']} |",
              f"| {lt['n_queries']} test clips ≥ {lt['min_long_s']:.0f} s: R@10 (fold's own 3 seeds) | {lt['whole_single']['R@10']:.3f} | "
              f"{lt['chunked_single']['R@10']:.3f} | {lt['whole_ens']['R@10']:.3f} | {lt['chunked_ens']['R@10']:.3f} |", "",
              f"- chunked ensemble − whole-clip single model, R@10 on long test clips: {_ci(lt['delta_R@10 chunked_ens - whole_single'])}",
              "- 1.mp4's original soundtrack matches no library file (Chromaprint, 17 chunks), so per-chunk track labels do not exist.", ""]
    q = Path("reports/section_alignment.md")
    if q.exists():
        rows = [ln for ln in q.read_text().splitlines() if ln.startswith("| novelty (4 s) or loudness")]
        if rows:
            L += ["Section-level alignment (scene boundaries within ±1 s of a novelty peak or ≥ 3 dB loudness change, vs circular "
                  "shifts; reports/section_alignment.md): all pairs " + rows[0].split("|")[5].strip() + " excess. Beat-level cut "
                  "alignment (M4) is absent (Wilcoxon p = 0.76). 1,238 of 1,248 music clips carry no vocal stem (Demucs), so a "
                  "song / background split is not possible.", ""]
    return L


def write_final(cfg: DictConfig) -> Path:
    L = ["# Final results", "",
         "Generated by `bgm report --final` from runs/. Semantic heads: SigLIP + CLAP+MERT unless named. "
         "Splits: 5 folds grouped by track_id; test data never touches training, calibration, fusion weights or penalties. "
         "Comparisons: paired, per-query values averaged over seeds, 95% cluster-bootstrap CI over track groups (concatenations "
         "for §4). Deviations from docs/SPEC.md: docs/PROGRESS.md D1–D34.", ""]
    for sec in (section_zeroshot_semantic, section_dynamics, section_pipeline, section_stitch):
        L += sec(cfg)
    L += section_long_video(cfg)
    L += section_pretraining(cfg)
    if any((Path(cfg.paths.runs) / "pipeline_pt").glob("fold*/seed*/metrics.json")):
        L += section_pipeline(cfg, "pipeline_pt", "## 7. Pipeline accuracy with pretrained models (H3 with M8)")
        L += section_stitch(cfg, "pipeline_pt", "## 8. Stitching with pretrained models (H5 with M8)")
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_trackid.md").write_text("\n".join(L) + "\n")
    return out / "final_trackid.md"
