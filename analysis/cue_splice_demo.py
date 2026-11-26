"""Output videos for the best transition model (R3s, D94-D95) on held-out test films, scored against the true changes.

Run: python -m bgm.analysis.cue_splice_demo [--per-fold 6]   -> outputs/cue_splice_demo/, results/cue_splice_demo.md

Sequences: the frozen test sets of data/manifests/cue_sequences_film.json (1,248-pair film split), per fold the first
`per_fold` sequences of 60-150 s with at least one true music change and a film not used yet, film-order (timeline)
sequences first, then same_film. Each is joined into one silent 640x360 video (as D84) and rendered like any input with
its OWN fold's runs (leak-free; render's long-video shortlist restricted to that fold): D75 (fixed penalties) and R3s
(`cue.enabled=true cue.exp=cue_splice cue.variant=pair_splice`, seeds 0-2). The clips' own soundtracks joined are the
truth. Scores per video (cue_demo.score): boundary F1 at +-1 s against the true changes, switches / min, right track.
Final videos are upscaled to 1080 lines with the source timestamps kept and sync-checked (person/outputs.py):
  outputs/cue_splice_demo/{d75,r3s,original}/<name>.mp4,  timelines/<name>.png;  work files in _work/ (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from bgm.analysis import cue_demo
from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
OUT = Path("outputs/cue_splice_demo")
WORK = OUT / "_work"
BASE = ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", "pipeline.dyn_variant=null",
        "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.audio_bitrate=320k", f"paths.renders={WORK}"]
R3S = ["cue.enabled=true", "cue.exp=cue_splice", "cue.variant=pair_splice", "cue.seeds=[0,1,2]"]


def pick(cfg, store, per_fold: int, lo: float = 60.0, hi: float = 150.0, one_per_film: bool = True) -> list[dict]:
    """Defaults = the first batch (D96). `--all` (D97): 40-180 s, a film's timeline and same_film sequences both."""
    sets = json.loads((manifest.manifests_dir(cfg) / cfg.cue.sequences_file).read_text())["sets"]
    out = []
    for k in range(cfg.splits.n_folds):
        used, n = set(), 0
        for mode in ("timeline", "same_film"):
            for i, parts in enumerate(sets[str(k)]["test"][mode]):
                d = sum(store.duration("video", p) for p in parts)
                tracks = [store.track[p] for p in parts]
                film = store.pairs.loc[parts[0], "meta_film"]
                if n >= per_fold or not lo <= d <= hi or all(a == b for a, b in zip(tracks, tracks[1:])):
                    continue
                if one_per_film and film in used:
                    continue
                out.append({"fold": k, "mode": mode, "idx": i, "parts": list(parts), "film": str(film)})
                used.add(film)
                n += 1
    return out


def render(video: Path, name: str, fold: int, r3s: bool) -> dict:
    import bgm.analysis.long_video as lv
    from bgm.render.run import render_video
    cfg = load_config("configs/default.yaml", BASE + (R3S if r3s else []))
    tag = "r3s" if r3s else "d75"
    if (WORK / f"{name}_{tag}.plan.json").exists() and (WORK / f"{name}_{tag}.mp4").exists():
        return json.loads((WORK / f"{name}_{tag}.plan.json").read_text())
    orig = lv.load_models
    lv.load_models = lambda c, fold_=None, _k=fold: orig(c, fold=_k)          # leak-free long-video shortlist
    try:
        if r3s:                                                                 # reuse the D75 render's features
            src, dst = WORK / f"{name}_d75", WORK / f"{name}_r3s"
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("query.npz", "silent25.mp4"):
                if (src / f).exists() and not (dst / f).exists():
                    (dst / f).write_bytes((src / f).read_bytes())
        plan = render_video(cfg, str(video), str(WORK / f"{name}_{tag}.mp4"), f"runs/filmsplit/pipeline_ens10/fold{fold}/seed0")
    finally:
        lv.load_models = orig
    return json.loads(Path(plan).read_text())


def finish(native: Path, final: Path, ref_wav: Path | None) -> dict:
    """Upscale to 1080 lines and check sync (the original-soundtrack version has no bgm.wav: start times and frames only)."""
    from bgm.person.outputs import check_sync, probe, upscale
    how = upscale(native, final)
    if ref_wav is not None:
        return {"how": how, **check_sync(native, final, ref_wav)}
    a, b = probe(native), probe(final)
    assert abs(a["v_start"] - b["v_start"]) < 1e-3 and abs(a["a_start"] - b["a_start"]) < 1e-3 and a["frames"] == b["frames"], (a, b)
    return {"how": how}


def main() -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", type=int, default=6)
    ap.add_argument("--all", action="store_true", help="every test sequence of 40-180 s with a music change (D97)")
    a = ap.parse_args()
    cfg = load_config("configs/default.yaml", BASE)
    store = FeatureStore(cfg)
    WORK.mkdir(parents=True, exist_ok=True)
    rows, failed = [], {}
    picks = pick(cfg, store, 10**6, 40.0, 180.0, False) if a.all else pick(cfg, store, a.per_fold)
    for c in picks:
        name = f"f{c['fold']}_{c['mode']}{c['idx']}_{c['film'].replace(' ', '_')[:24]}"
        try:
            video = WORK / f"{name}_input.mp4"
            counts = cue_demo.build_input(cfg, c["parts"], video)
            starts, ends = np.r_[0, np.cumsum(counts)[:-1]] / 25.0, np.cumsum(counts) / 25.0
            tracks = [store.track[p] for p in c["parts"]]
            joins = [float(t) for t, x, y in zip(starts[1:], tracks, tracks[1:]) if x != y]
            truth = lambda t, s=starts, tr=tracks: tr[int(np.searchsorted(s, t, side="right") - 1)]  # noqa: E731
            orig = WORK / f"{name}_original.mp4"
            if not orig.exists():
                cue_demo.original_soundtrack(cfg, store, c["parts"], counts, video, orig)
            plans = {"d75": render(video, name, c["fold"], False), "cue": render(video, name, c["fold"], True)}
            scores = {k: cue_demo.score(v, joins, truth) for k, v in plans.items()}
            (OUT / "timelines").mkdir(parents=True, exist_ok=True)
            title = (f"{c['film']}  ·  fold {c['fold']} test, {c['mode']}  ·  {ends[-1]:.0f} s, {len(c['parts'])} clips, "
                     f"{len(joins)} true music changes")
            cue_demo.plot(name, title, float(ends[-1]), list(zip(starts, ends, tracks)), joins, plans, scores, truth,
                          OUT / "timelines" / f"{name}.png", cue_label="R3s (new)")
            sync = {"d75": finish(WORK / f"{name}_d75.mp4", OUT / "d75" / f"{name}.mp4", Path(plans["d75"]["bgm_wav"])),
                    "r3s": finish(WORK / f"{name}_r3s.mp4", OUT / "r3s" / f"{name}.mp4", Path(plans["cue"]["bgm_wav"])),
                    "original": finish(orig, OUT / "original" / f"{name}.mp4", None)}
            rows.append({**c, "name": name, "duration_s": float(ends[-1]), "true_changes": joins,
                         "segments": len(plans["d75"]["segments"]), "scores": scores, "sync": sync})
            log.info("%s: D75 F1 %.2f (%d sw) | R3s F1 %.2f (%d sw) | %d true changes", name, scores["d75"]["f1_1"],
                     len(scores["d75"]["switches"]), scores["cue"]["f1_1"], len(scores["cue"]["switches"]), len(joins))
        except Exception as e:                                    # noqa: BLE001  skip and list
            failed[name] = repr(e)
            log.error("%s failed: %r", name, e)
    (WORK / "demo.json").write_text(json.dumps({"rows": rows, "failed": failed}, indent=1, default=float))
    return write_md(rows, failed)


def write_md(rows: list[dict], failed: dict) -> Path:
    from bgm.eval.metrics import cluster_bootstrap
    L = ["# Transition model R3s on held-out test films: output videos and scores", "",
         "Generated by `python -m bgm.analysis.cue_splice_demo` (PROGRESS.md D96). Each row is a frozen test sequence of its fold "
         "(films never trained on), joined into one silent video and rendered with that fold's runs: **D75** (fixed transition "
         "penalties) and **R3s** (the learned transition model trained with same-music splices, D94). The clips' own soundtracks "
         "joined are the truth. Videos (1080 lines): outputs/cue_splice_demo/d75/, r3s/, original/; timelines: "
         "outputs/cue_splice_demo/timelines/. F1 = music switches within ±1 s of a true music change (greedy one-to-one).", "",
         "| video | length | true changes | D75 F1 | R3s F1 | Δ | D75 switches / min | R3s switches / min | true changes / min |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    tot = {k: np.zeros(3) for k in ("d75", "cue")}
    for r in rows:
        s, m = r["scores"], r["duration_s"] / 60
        for k in tot:
            tot[k] += s[k]["tp_fp_fn_1"]
        L.append(f"| {r['name']} | {r['duration_s']:.0f} s | {len(r['true_changes'])} | {s['d75']['f1_1']:.2f} | {s['cue']['f1_1']:.2f} | "
                 f"{s['cue']['f1_1'] - s['d75']['f1_1']:+.2f} | {s['d75']['switches_per_min']:.1f} | {s['cue']['switches_per_min']:.1f} | "
                 f"{len(r['true_changes']) / m:.1f} |")
    f = {k: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1e-9) for k, v in tot.items()}
    d = np.array([r["scores"]["cue"]["f1_1"] - r["scores"]["d75"]["f1_1"] for r in rows])
    ci = cluster_bootstrap(d, np.array([r["film"] for r in rows]), 10000) if len(rows) > 1 else None
    L += ["", f"**Pooled over {len(rows)} videos (±1 s):** D75 F1 {f['d75']:.3f} (TP {tot['d75'][0]:.0f}, FP {tot['d75'][1]:.0f}, "
          f"FN {tot['d75'][2]:.0f}); R3s F1 {f['cue']:.3f} (TP {tot['cue'][0]:.0f}, FP {tot['cue'][1]:.0f}, FN {tot['cue'][2]:.0f}). "
          f"R3s better on {int((d > 0).sum())}, equal on {int((d == 0).sum())}, worse on {int((d < 0).sum())} videos"
          + (f"; mean per-video Δ F1 {ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] (bootstrap over films)." if ci else "."),
          "", "These videos are a sample of the test sets for listening; the test-set evaluation (results/cue_switch_splice.md, "
          "results/cue_switch_full_splice.md) is the evidence.", "", f"Failed: {', '.join(failed) or 'none'}.", ""]
    out = Path("results/cue_splice_demo.md")
    out.write_text("\n".join(L) + "\n")
    return out


def render_inputs() -> Path:
    """media/inputs/*.mp4 with R3s (fold 0, like ./render_best.sh) -> outputs/cue_splice/<input>.mp4 (1080 lines,
    sync-checked) and outputs/cue_splice/index.csv. The D75 versions of the same inputs are outputs/person_branch/baseline/."""
    import csv
    from bgm.render.run import render_video
    out, work = Path("outputs/cue_splice"), Path("outputs/cue_splice/_work")
    cfg = load_config("configs/default.yaml", BASE[:-1] + [f"paths.renders={work}"] + R3S)
    rows, report = [], {}
    for video in sorted(Path("media/inputs").glob("*.mp4")):
        try:
            native = work / "native" / video.name
            native.parent.mkdir(parents=True, exist_ok=True)
            plan_path = work / f"{video.stem}.plan.json"
            if not (native.exists() and plan_path.exists()):
                render_video(cfg, str(video), str(native), "runs/filmsplit/pipeline_ens10/fold0/seed0")
            plan = json.loads(plan_path.read_text())
            sync = finish(native, out / video.name, Path(plan["bgm_wav"]))
            segs = plan["segments"]
            report[video.name] = {"switches": sum(x["music_id"] != y["music_id"] for x, y in zip(segs, segs[1:])), "segments": len(segs), **sync}
            rows += [{"input": video.name, "piece": i, "video_start_s": round(q["start"], 3), "video_end_s": round(q["end"], 3),
                      "music_id": q["music_id"], "music_offset_s": round(q["offset"], 3)} for i, q in enumerate(plan["pieces"])]
            log.info("input %s: %d segments, %d switches, music vs first frame %.2f ms", video.name, len(segs), report[video.name]["switches"],
                     sync["sync_error_ms"])
        except Exception as e:                                    # noqa: BLE001  skip and list
            report[video.name] = {"failed": repr(e)}
            log.error("input %s failed: %r", video.name, e)
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["input", "piece", "video_start_s", "video_end_s", "music_id", "music_offset_s"])
        w.writeheader()
        w.writerows(rows)
    (work / "inputs_report.json").write_text(json.dumps(report, indent=1, default=float))
    return out


if __name__ == "__main__":
    import sys
    if "--inputs" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        print(render_inputs())
    else:
        print(main())
