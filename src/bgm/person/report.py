"""Step 3 report: results/person_branch.md from runs/filmsplit/person/eval/<system>/fold<k>/seed<s>/perquery.npz.

Run: python -m bgm.person.report

Per query, values are averaged over the system's seeds first (ensembles have one run per fold). Paired differences
against P0 (D75) and against P0s (single siglip-both heads, same seeds) with a 95% cluster bootstrap, 10,000
resamples: film groups (D39) for videos and segments, concatenations for the stitching rows (D60). Per-segment metrics
are also reported on the Step 1 subsets (data/manifests/person_subsets_film.json).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.person import subset
from bgm.person.diagnose import film_cfg

log = logging.getLogger(__name__)
EVAL = Path("runs/filmsplit/person/eval")
OUT = Path("results/person_branch.md")
B = 10000
NAMES = {"P0": "P0 current system (D75, 10-seed ensemble)", "P0s": "P0s single siglip-both heads (reference)",
         "P1": "P1 + D77 facial branch, w on val", "P2": "P2 + person-crop head (late fusion), w on val",
         "P3": "P3 frame ‖ person in one head (early fusion)", "P4-P1": "P4 = P1 with the 10-seed ensemble",
         "P4-P2": "P4 ensemble: D75 frames + 10 regularised person heads, w on val",
         "P4-P2-kd": "P4 distilled: D75 frames + one distilled person head per seed, w on val",
         "P4-P3": "P4 ensemble: 10 regularised early-fusion heads", "P4-P3-kd": "P4 distilled: one distilled early-fusion head per seed"}
# metric -> (array key, level, transform); level: vid | seg | concat
METRICS = {
    "track R@1": ("ret/track_test_fold", "vid", lambda r: (r <= 1).astype(float)),
    "track R@10": ("ret/track_test_fold", "vid", lambda r: (r <= 10).astype(float)),
    "track R@10, all 1,248": ("ret/track_all", "vid", lambda r: (r <= 10).astype(float)),
    "segment R@10": ("ret/segment_all", "seg", lambda r: (r <= 10).astype(float)),
    "pipeline track@1": ("pipe/track@1", "seg", None),
    "pipeline track R@10": ("pipe/track@10", "seg", None),
    "window@1": ("pipe/window@1", "seg", None),
    "window@J": ("pipe/window@J", "seg", None),
    "right track ±1 s": ("pipe/offset/1", "seg", None),
    "switch F1 (Viterbi)": ("stitch/concat/viterbi/switch_f1", "concat", None),
    "part track accuracy": ("stitch/concat/viterbi/part_track_acc", "concat", None),
    "switches / min": ("stitch/concat/viterbi/switches_per_min", "concat", None),
    "natural videos, track after Viterbi": ("stitch/natural/viterbi/track", "vid", None),
}
SUB_METRICS = ("segment R@10", "pipeline track@1", "pipeline track R@10", "window@1", "right track ±1 s")


RANK_KEYS = ("ret/track_test_fold", "ret/track_all", "ret/segment_all")


def per_seed(key: str, v: np.ndarray) -> np.ndarray:
    """Ranks become [hit@1, hit@10] so that hits, not ranks, are averaged over seeds."""
    if key in RANK_KEYS:
        return np.stack([v <= 1, v <= 10], 1).astype(float)
    return v.astype(float)


def load_system(name: str) -> dict | None:
    """{key: array} pooled over folds, each query's value averaged over the system's seeds; None if not complete."""
    folds = sorted(f for f in (EVAL / name).glob("fold*") if f.is_dir())     # Step 1 caches P0/fold<k>.npz sit beside them
    if len(folds) < 5 or not all(list(f.glob("seed*/metrics.json")) for f in folds):
        return None
    out: dict[str, list] = {}
    for f in folds:
        zs = [dict(np.load(r)) for r in sorted(f.glob("seed*/perquery.npz"))]
        for k, v in zs[0].items():
            numeric = v.dtype.kind in "fiub" and not k.endswith(("seg_idx",))
            out.setdefault(k, []).append(np.mean([per_seed(k, z[k]) for z in zs], 0) if numeric else v)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["_w"] = np.array([json.loads(p.read_text())["w_fine"].get("person", np.nan) for f in folds for p in sorted(f.glob("seed*/fusion.json"))])
    return res


def value(sys: dict, metric: str) -> np.ndarray:
    key, _, tf = METRICS[metric]
    v = sys[key]
    if key.startswith("ret/"):
        return v[:, 0] if metric == "track R@1" else v[:, 1]
    return v.astype(float) if tf is None else tf(v)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg([])
    film = manifest.read_folds(cfg)["film_group"]
    S = {n: s for n in NAMES if (s := load_system(n)) is not None}
    assert "P0" in S, "P0 has not been run"
    p0 = S["P0"]
    vid_cl = np.array([film[p] for p in p0["ret/pair"]])
    seg_cl = np.array([film[p] for p in p0["pipe/pair"]])
    cl = {"vid": vid_cl, "seg": seg_cl, "concat": p0["stitch/concat/clusters"], "nat": np.array([film[p] for p in p0["stitch/natural/pair"]])}
    for n, s in S.items():
        assert (s["pipe/pair"] == p0["pipe/pair"]).all() and (s["pipe/seg_idx"] == p0["pipe/seg_idx"]).all(), n
        assert (s["ret/pair"] == p0["ret/pair"]).all() and (s["stitch/concat/clusters"] == p0["stitch/concat/clusters"]).all(), n
    level_cl = lambda m: cl["nat"] if METRICS[m][0].startswith("stitch/natural") else cl[METRICS[m][1]]  # noqa: E731
    L = ["# Person-focused branch (D87–)", "",
         "1,248-pair film split (D36), 5 folds, test-fold galleries (195–199 files). Konjum salangai is train-only. Every weight and "
         "stopping epoch was chosen on the fold's val films. Per query, values are averaged over seeds (0–2) first; ensemble rows are one "
         f"run per fold. Paired Δ with a 95% cluster bootstrap ({B:,} resamples): film groups for videos and segments, "
         "concatenations for stitching. Plan: docs/person_branch_plan.md.", ""]
    v = ["## Verdict (computed)", "", "Does a person-branch system beat P0? A metric counts as better / worse only when its 95% CI excludes 0. P1–P3 are single "
         "models, so most of their deficit is the 10-seed ensemble that P0 has and they lack (P0s, the single frame heads, is behind P0 by "
         "about as much); their own effect is the Δ vs P0s table below. P4 is the like-for-like test against P0.", ""]
    for n in [x for x in S if x not in ("P0", "P0s")]:
        better, worse = [], []
        for m in METRICS:
            if m == "switches / min":
                continue
            d = cluster_bootstrap(value(S[n], m) - value(p0, m), level_cl(m), B)
            (better if d["lo"] > 0 else worse if d["hi"] < 0 else []).append(f"{m} {fmt(d)}")
        v.append(f"- **{NAMES[n]}**: better on {', '.join(better) or 'nothing'}; worse on {', '.join(worse) or 'nothing'}.")
    L += v + [""]
    step1 = Path("reports/person_step1.md")
    if step1.exists():
        L += ["## Step 1: diagnosis", "", *[ln.replace("# Step 1:", "### Step 1:") for ln in step1.read_text().splitlines()[2:]], ""]
    L += ["## All test segments / videos: value (Δ vs P0 [95% CI])", "", "| system | " + " | ".join(METRICS) + " |", "|" + " --- |" * (len(METRICS) + 1)]
    for n, s in S.items():
        cells = []
        for m in METRICS:
            v = value(s, m)
            cells.append(f"{np.nanmean(v):.3f}" + ("" if n == "P0" else f" ({fmt(cluster_bootstrap(v - value(p0, m), level_cl(m), B))})"))
        L.append(f"| {NAMES[n]} | " + " | ".join(cells) + " |")
    if "P0s" in S:
        L += ["", "## Single-model comparison (choice of the P4 base): Δ vs P0s, same seeds", "",
              "| system | " + " | ".join(METRICS) + " |", "|" + " --- |" * (len(METRICS) + 1)]
        for n in ("P1", "P2", "P3"):
            if n in S:
                L.append(f"| {NAMES[n]} | " + " | ".join(fmt(cluster_bootstrap(value(S[n], m) - value(S["P0s"], m), level_cl(m), B)) for m in METRICS) + " |")
    subs = subset.read(cfg)
    keys = list(zip(p0["pipe/pair"], p0["pipe/seg_idx"].astype(int)))
    for sname, label in (("nn", "nearest-neighbour subset (diagnostic)"), ("spec", "subset as specified")):
        ins = np.array([k in subs[sname] for k in keys])
        L += ["", f"## Step 1 {label}: {ins.sum()} segments; value (Δ vs P0 [95% CI])", "",
              "| system | " + " | ".join(SUB_METRICS) + " |", "|" + " --- |" * (len(SUB_METRICS) + 1)]
        for n, s in S.items():
            cells = []
            for m in SUB_METRICS:
                v, v0 = value(s, m)[ins], value(p0, m)[ins]
                cells.append(f"{np.nanmean(v):.3f}" + ("" if n == "P0" else f" ({fmt(cluster_bootstrap(v - v0, seg_cl[ins], B))})"))
            L.append(f"| {NAMES[n]} | " + " | ".join(cells) + " |")
    L += ["", "## Fusion weights chosen on val (w of the branch; frame weight 1)", "", "| system | w per run | mean |", "| --- | --- | --- |"]
    for n, s in S.items():
        if np.isfinite(s["_w"]).any():
            L.append(f"| {NAMES[n]} | {', '.join(f'{x:g}' for x in s['_w'])} | {np.nanmean(s['_w']):.2f} |")
    cov = Path("runs/filmsplit/person/coverage.json")
    if cov.exists():
        c = json.loads(cov.read_text())
        L += ["", f"Person features (Step 2): {c['seconds']:,} seconds of {c['pairs']:,} clips; faces {c['face']:.1%}, person boxes only "
              f"{c['body']:.1%}, no person {c['none']:.1%}; segments without a person second {c['segments_none']:.1%}.", ""]
    rend = Path("outputs/person_branch/_work/outputs_report.json")
    if rend.exists():
        r = json.loads(rend.read_text())
        L += ["## Step 4: rendered outputs (outputs/person_branch/)", "",
              f"baseline/ = P0 (./render_best.sh); person/ = `{r['person_run']}` with `person.enabled=true`. Both: fold 0, audio 320k; outputs "
              "under 1080 lines re-encoded to 1080 (Lanczos, CRF 16, source timestamps kept). Sync: music sample 0 vs the first frame, by "
              "cross-correlation with the render's soundtrack. Pieces per system: outputs/person_branch/index.csv.", "",
              "| output | picture | segments | switches | music files | LUFS | music vs first frame |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in r["rendered"].items():
            n, f = v["native"], v["final"]
            pic = f"{n['width']}x{n['height']} kept" if (n["width"], n["height"]) == (f["width"], f["height"]) else \
                f"{n['width']}x{n['height']} -> {f['width']}x{f['height']}"
            L.append(f"| {k} | {pic} | {v['segments']} | {v['switches']} | {v['files']} | {v['lufs']:.1f} | {v['sync_error_ms']:+.2f} ms |")
        L += ["", f"Failed: {', '.join(r['failed']) or 'none'}.", ""]
    OUT.write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
