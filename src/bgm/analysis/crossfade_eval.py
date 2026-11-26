"""Seam rendering variants: crossfade shape (equal power / linear) x seam gain ramps (off / on). Render only; the
retrieved songs, offsets and piece boundaries are identical across variants, so every seam is compared paired.

Run: python -m bgm.analysis.crossfade_eval          -> reports/crossfade_eval.json, results/crossfade.md
     python -m bgm.analysis.crossfade_eval report   -> results/crossfade.md only (from the JSON and the 1.mp4 renders)
     python -m bgm.analysis.crossfade_eval listen [n] -> renders/listen_test/: n fold-0 test concatenations as videos

Test bed: the held-out test concatenations of every fold (SPEC M10 / H5; 50 per fold), decoded with Viterbi and the
run's tuned stitch penalties (runs/pipeline/fold<k>/seed<s>, 5 folds x 3 seeds), then put through the same beat sync
and loop fill as `bgm render`. Per seam, on the rendered mix:
  * jump_3s: |loudness of the 3 s after - the 3 s before| (LU; the `bgm check` / stitching metric; gain ramps match
    3 s windows by construction, so this one is partly circular for them);
  * jump_1s: the same with 1 s windows (not what the ramps match);
  * dip: momentary (400 ms) loudness centred on the seam minus the mean momentary loudness just outside the fade on
    both sides (LU; 0 = no hole or bump mid-fade).
CIs: 95 % cluster bootstrap over concatenation videos (a video's seams across seeds form one cluster), 2,000 draws.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.config import load_config
from bgm.data.prep import interim_music_path
from bgm.eval.store import FeatureStore, fold_ids
from bgm.render.render import (SR, _loud, beat_sync_runs, crossfade_lengths, fill_short_runs, render_soundtrack,
                               runs_from_path, seam_loudness_jumps)
from bgm.retrieval.index import FoldModels, MusicIndex
from bgm.retrieval.stitch_eval import chosen_candidates, decode_lattice, read_concat_sets
from bgm.retrieval.tune_stitch import prepare_concats

log = logging.getLogger(__name__)
VARIANTS = {"equal_power": ("equal_power", 0.0), "linear": ("linear", 0.0),
            "equal_power+ramp": ("equal_power", 4.0), "linear+ramp": ("linear", 4.0)}
RAMP_S = 4.0


def variant_cfg(cfg: DictConfig, name: str, variants: dict | None = None) -> DictConfig:
    r = cfg.render.copy()
    r.crossfade_shape, r.gain_ramp_s = (variants or VARIANTS)[name]
    return r


def seam_dips(mix: np.ndarray, seams: list[float], xf: list[float], win_s: float = 0.4) -> list[float]:
    """Momentary loudness centred on each seam minus the mean momentary loudness just outside its fade, LU (NaN when
    a window falls off the mix or is silent)."""
    import pyloudnorm as pyln
    meter = pyln.Meter(SR, block_size=win_s)
    lo = lambda a: _loud(meter, mix[int(a * SR):int(a * SR) + int(win_s * SR)].astype(np.float64))  # noqa: E731
    out = []
    for t, x in zip(seams, xf):
        a, b = t - x / 2 - win_s, t + x / 2
        if a < 0 or int((b + win_s) * SR) > len(mix):
            out.append(float("nan"))
            continue
        mid, before, after = lo(t - win_s / 2), lo(a), lo(b)
        out.append(mid - (before + after) / 2 if np.all(np.isfinite([mid, before, after])) else float("nan"))
    return out


def seam_metrics(mix: np.ndarray, seams: list[float], xf: list[float]) -> dict[str, list[float]]:
    return {"jump_3s": seam_loudness_jumps(mix, seams, 3.0), "jump_1s": seam_loudness_jumps(mix, seams, 1.0),
            "dip": seam_dips(mix, seams, xf)}


def render_variants(cfg: DictConfig, runs, duration_s: float, variants: dict | None = None) -> dict[str, dict]:
    """Per variant: seam metrics and integrated loudness of the rendered mix."""
    import pyloudnorm as pyln
    xf = crossfade_lengths(runs, cfg.render)
    out = {}
    for name in (variants or VARIANTS):
        mix, seams = render_soundtrack(runs, duration_s, lambda m: interim_music_path(cfg, m), variant_cfg(cfg, name, variants))
        out[name] = {**seam_metrics(mix, seams, xf), "lufs": float(pyln.Meter(SR).integrated_loudness(mix.astype(np.float64)))}
    return out


def concat_video(cfg: DictConfig, parts: list[tuple[str, float, float]], out: Path, size=(640, 360)) -> None:
    """Silent video of the concatenation: each part's interim 25 fps clip cut to its 25 Hz frame count (the timeline of
    `concat_query`), letterboxed to `size`, joined end to end."""
    import subprocess

    from bgm.data.prep import interim_video_path
    w, h = size
    ins, flt = [], []
    for i, (p, _, d) in enumerate(parts):
        ins += ["-i", str(interim_video_path(cfg, p))]
        flt.append(f"[{i}:v]trim=end_frame={int(round(d * 25))},setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]")
    flt.append("".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *ins, "-filter_complex", ";".join(flt), "-map", "[v]", "-r", "25",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)


def listen(cfg: DictConfig, n: int = 4, fold: int = 0, seed: int = 0, min_seams: int = 3) -> Path:
    """Listening set on held-out test data: n test concatenations of `fold` (drawn at random, seed 0, among those with
    >= min_seams seams; decoded with the fold's run), each as one video per variant plus the original soundtracks
    (each part's own music from its first frame, joined with the default crossfade). Writes renders/listen_test/."""
    import soundfile as sf

    from bgm.render.render import Run, mux
    store = FeatureStore(cfg)
    d = Path(cfg.paths.runs) / cfg.pipeline.exp / f"fold{fold}" / f"seed{seed}"
    fusion, stitch = json.loads((d / "fusion.json").read_text()), json.loads((d / "stitch.json").read_text())
    f = fold_ids(cfg, fold)
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    beats_of, dur_of = lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m)  # noqa: E731
    sets = read_concat_sets(cfg)[str(fold)]["test"]
    cands = []
    for i, (vq, res, lat) in enumerate(prepare_concats(store, index, models, sets, cfg, fusion)):
        chosen = chosen_candidates(index, lat, decode_lattice(lat, stitch["penalties"], stitch["beam"], stitch["tol_s"]))
        runs, _ = beat_sync_runs(runs_from_path(chosen, [(x.a, x.b) for x in res]), vq.cut_times, beats_of, dur_of,
                                 cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
        runs, _ = fill_short_runs(runs, beats_of, dur_of)
        if len(runs) - 1 >= min_seams:
            cands.append((i, vq, runs))
    pick = sorted(np.random.default_rng(0).choice(len(cands), min(n, len(cands)), replace=False))
    root = Path(cfg.paths.renders) / "listen_test"
    root.mkdir(parents=True, exist_ok=True)
    index_md = ["# Listening set: held-out test data", "",
                f"Fold {fold} test concatenations (test clips joined end to end; the fold-{fold} / seed-{seed} models never saw them), "
                f"drawn at random among those with >= {min_seams} seams. Per video: `original.mp4` (each clip's own music, joined "
                "with the default crossfade), then the system's soundtrack rendered four ways; only the joins differ.", ""]
    for i, vq, runs in (cands[k] for k in pick):
        out = root / f"fold{fold}_concat{i:02d}"
        out.mkdir(exist_ok=True)
        video = out / "silent.mp4"
        concat_video(cfg, vq.meta["parts"], video)
        orig = [Run(store.music_of[p], 0.0, t0, t0 + dd, float(store.music(store.music_of[p])["tempo_bpm"])) for p, t0, dd in vq.meta["parts"]]
        orig, _ = fill_short_runs(orig, beats_of, dur_of)
        tracks = {"original": (orig, cfg.render), **{v: (runs, variant_cfg(cfg, v)) for v in VARIANTS}}
        for name, (rr, rcfg) in tracks.items():
            mix, _ = render_soundtrack(rr, vq.duration, lambda m: interim_music_path(cfg, m), rcfg)
            wav = out / f"{name}.wav"
            sf.write(wav, np.stack([mix, mix], 1) / np.sqrt(2), SR)
            mux(video, wav, out / f"{name.replace('+', '_')}.mp4")
            wav.unlink()
        seams = [r.start for r in runs[1:]]
        index_md += [f"## {out.name} ({vq.duration:.1f} s, {len(vq.meta['parts'])} clips)", "",
                     "Clip joins (s): " + ", ".join(f"{t0:.2f}" for _, t0, _ in vq.meta["parts"][1:]), "",
                     "Seams in the system's soundtrack (s): " + ", ".join(
                         f"{t:.2f}{'' if runs[k].music_id != runs[k + 1].music_id else ' (same song)'}" for k, t in enumerate(seams)), "",
                     "| from (s) | to (s) | music | track matches the clip's own |", "| --- | --- | --- | --- |"]
        for r in runs:
            own = {store.music_track[store.music_of[p]] for p, t0, dd in vq.meta["parts"] if t0 < r.end and t0 + dd > r.start}
            index_md.append(f"| {r.start:.2f} | {r.end:.2f} | {r.music_id} | {'yes' if store.music_track[r.music_id] in own else 'no'} |")
        index_md.append("")
        log.info("listening set: %s (%d seams)", out, len(seams))
    (root / "README.md").write_text("\n".join(index_md) + "\n")
    return root


def fold_seed(cfg: DictConfig, store: FeatureStore, fold: int, seed: int, variants: dict | None = None) -> list[dict]:
    d = Path(cfg.paths.runs) / cfg.pipeline.exp / f"fold{fold}" / f"seed{seed}"
    fusion, stitch = json.loads((d / "fusion.json").read_text()), json.loads((d / "stitch.json").read_text())
    f = fold_ids(cfg, fold)
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    beats_of, dur_of = lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m)  # noqa: E731
    rows = []
    for i, (vq, res, lat) in enumerate(prepare_concats(store, index, models, read_concat_sets(cfg)[str(fold)]["test"], cfg, fusion)):
        chosen = chosen_candidates(index, lat, decode_lattice(lat, stitch["penalties"], stitch["beam"], stitch["tol_s"]))
        runs = runs_from_path(chosen, [(x.a, x.b) for x in res])
        runs, _ = beat_sync_runs(runs, vq.cut_times, beats_of, dur_of, cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
        runs, _ = fill_short_runs(runs, beats_of, dur_of)
        rows.append({"fold": fold, "seed": seed, "video": f"f{fold}c{i}", "duration_s": vq.duration, "n_seams": len(runs) - 1,
                     **render_variants(cfg, runs, vq.duration, variants)})
    log.info("fold %d seed %d: %d concatenations, %d seams", fold, seed, len(rows), sum(r["n_seams"] for r in rows))
    return rows


def paired(rows: list[dict], metric: str, name: str, base: str = "equal_power", absolute: bool = False,
           draws: int = 2000, rng=None) -> dict:
    """Seam-level median of `name`, and the mean paired difference name - base with a cluster-bootstrap 95 % CI."""
    rng = rng or np.random.default_rng(0)
    f = np.abs if absolute else (lambda x: x)
    by_video: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        for a, b in zip(r[name][metric], r[base][metric]):
            if np.isfinite(a) and np.isfinite(b):
                by_video.setdefault(r["video"], []).append((f(a), f(b)))
    vids = list(by_video)
    vals = np.array([a for v in vids for a, _ in by_video[v]])
    sums = np.array([sum(a - b for a, b in by_video[v]) for v in vids])
    counts = np.array([len(by_video[v]) for v in vids])
    boot = []
    for _ in range(draws):
        k = rng.integers(len(vids), size=len(vids))
        boot.append(sums[k].sum() / counts[k].sum())
    return {"median": float(np.median(vals)), "diff": float(sums.sum() / counts.sum()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n_seams": int(counts.sum()), "n_videos": len(vids)}


def summarize(rows: list[dict], variants: dict | None = None, base: str = "equal_power") -> dict:
    out = {}
    for name in (variants or VARIANTS):
        out[name] = {m: paired(rows, m, name, base=base, absolute=(m == "dip")) for m in ("jump_3s", "jump_1s", "dip")}
        dips = np.array([x for r in rows for x in r[name]["dip"] if np.isfinite(x)])
        out[name]["dip_signed_median"] = float(np.median(dips))
        out[name]["dips_below_-3LU"] = float(np.mean(dips < -3))
        j3 = np.array([x for r in rows for x in r[name]["jump_3s"] if np.isfinite(x)])
        out[name]["jumps_3s_over_2LU"] = float(np.mean(j3 > 2))
        lufs = np.array([r[name]["lufs"] for r in rows if r["duration_s"] >= 10])
        out[name]["lufs_median"], out[name]["lufs_within_1LU"] = float(np.median(lufs)), float(np.mean(np.abs(lufs - (-16.0)) <= 1))
    return out


RENDERS_1MP4 = {"equal_power": "1_bgm_eqpow", "linear": "1_bgm_linear", "equal_power+ramp": "1_bgm_eqpow_ramp",
                "linear+ramp": "1_bgm_linear_ramp"}


def render_file_metrics(cfg: DictConfig, plan_path: Path) -> dict:
    """Seam metrics of a `bgm render` output, read back from its bgm.wav (stereo = mono / sqrt 2)."""
    import soundfile as sf

    from bgm.render.render import Run
    plan = json.loads(plan_path.read_text())
    mix = sf.read(plan["bgm_wav"])[0][:, 0] * np.sqrt(2)
    m = seam_metrics(mix, plan["seams_s"], crossfade_lengths([Run(**q) for q in plan["pieces"]], cfg.render))
    ok = {k: np.array([x for x in v if np.isfinite(x)]) for k, v in m.items()}
    return {"n_seams": len(plan["seams_s"]), "jump_3s_median": float(np.median(ok["jump_3s"])), "jump_3s_over_2": int((ok["jump_3s"] > 2).sum()),
            "jump_1s_median": float(np.median(ok["jump_1s"])), "dip_signed_median": float(np.median(ok["dip"])),
            "dips_below_-3": int((ok["dip"] < -3).sum()), "lufs": plan["output_lufs"], "peak_dbfs": plan["output_peak_dbfs"],
            "duration_s": plan["render_duration_s"]}


def report(cfg: DictConfig) -> Path:
    ev = json.loads((Path(cfg.paths.reports) / "crossfade_eval.json").read_text())
    S, rows = ev["summary"], ev["rows"]
    one = {n: render_file_metrics(cfg, Path(cfg.paths.renders) / f"{f}.plan.json") for n, f in RENDERS_1MP4.items()
           if (Path(cfg.paths.renders) / f"{f}.plan.json").exists()}
    ci = lambda x: f"{x['diff']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"  # noqa: E731
    n_seams, n_vid = S["equal_power"]["jump_3s"]["n_seams"], S["equal_power"]["jump_3s"]["n_videos"]
    L = ["# Seam rendering: crossfade shape and gain ramps", "",
         "Render-only comparison (D46, D47). The songs, offsets and piece boundaries are the same in every variant; only how "
         "neighbouring pieces are joined changes:", "",
         "- **equal_power**: cos / sin crossfade gains (the default, SPEC §5.11).",
         "- **linear**: 1 − t / t gains (linear interpolation between the two pieces).",
         f"- **+ramp**: linear gain ramps in dB (up to {ev['ramp_s']:.0f} s on each side) so the two pieces meet in loudness at the seam.", "",
         "Crossfades are 4 beats at the incoming tempo, clamped to 1–2 s. Per seam, measured on the rendered mix:", "",
         "- **jump 3 s**: |loudness of the 3 s after − the 3 s before| (the `bgm check` / stitching metric; under 2 LU is good). "
         "The ramps match 3 s windows by construction, so this metric partly favours them.",
         "- **jump 1 s**: the same with 1 s windows (not what the ramps match).",
         "- **dip**: momentary loudness at the middle of the fade minus the level just outside it (0 = smooth; negative = a hole).", "",
         f"## Held-out test concatenations ({len(ev['folds'])} folds × {len(ev['seeds'])} seeds, {n_vid} videos, {n_seams} seams)", "",
         "Viterbi paths with each run's tuned penalties, then beat sync and loop fill as in `bgm render`. Models and folds are the "
         "track_id-split pipeline runs (runs/pipeline); only the rendering differs. Differences are paired against equal_power, "
         "mean over seams, 95 % cluster-bootstrap CI over videos.", "",
         "| variant | jump 3 s median | Δ jump 3 s | share > 2 LU | jump 1 s median | Δ jump 1 s | dip median (signed) | |dip| Δ | dips < −3 LU | LUFS median | within −16 ± 1 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in S.items():
        base = n == "equal_power"
        L.append(f"| {n} | {v['jump_3s']['median']:.2f} | {'—' if base else ci(v['jump_3s'])} | {v['jumps_3s_over_2LU']:.0%} | "
                 f"{v['jump_1s']['median']:.2f} | {'—' if base else ci(v['jump_1s'])} | {v['dip_signed_median']:+.2f} | "
                 f"{'—' if base else ci(v['dip'])} | {v['dips_below_-3LU']:.0%} | {v['lufs_median']:.2f} | {v['lufs_within_1LU']:.0%} |")
    if one:
        L += ["", "## media/inputs/1.mp4 (347 s, run fold0/seed0)", "",
              "Renders: " + ", ".join(f"`renders/{f}.mp4`" for f in RENDERS_1MP4.values()) + ". `bgm check` reports: `reports/check_<name>.md` "
              "(identical in every row but the seam row, since only the rendering changed). A/B clips of three song switches "
              "(22.28, 207.52, 326.03 s): `renders/ab_crossfade/`.", "",
              "| variant | seams | jump 3 s median | jumps > 2 LU | jump 1 s median | dip median (signed) | dips < −3 LU | LUFS | peak dBFS | duration s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n, v in one.items():
            L.append(f"| {n} | {v['n_seams']} | {v['jump_3s_median']:.2f} | {v['jump_3s_over_2']} | {v['jump_1s_median']:.2f} | "
                     f"{v['dip_signed_median']:+.2f} | {v['dips_below_-3']} | {v['lufs']:.2f} | {v['peak_dbfs']:.2f} | {v['duration_s']:.2f} |")
    L += ["", f"Rows: {len(rows)} concatenation renders per variant. Raw per-seam values: reports/crossfade_eval.json."]
    out = Path("results") / "crossfade.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main(folds=(0, 1, 2, 3, 4), seeds=(0, 1, 2)) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", [])
    store = FeatureStore(cfg)
    rows = [r for fold in folds for seed in seeds for r in fold_seed(cfg, store, fold, seed)]
    summary = summarize(rows)
    rep = Path(cfg.paths.reports) / "crossfade_eval.json"
    rep.write_text(json.dumps({"folds": list(folds), "seeds": list(seeds), "ramp_s": RAMP_S, "summary": summary, "rows": rows},
                              default=float))
    log.info("summary: %s", json.dumps(summary, default=float))
    log.info("wrote %s", report(cfg))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["report"]:
        logging.basicConfig(level=logging.INFO)
        print(report(load_config("configs/default.yaml", [])))
    elif sys.argv[1:2] == ["listen"]:
        logging.basicConfig(level=logging.INFO)
        print(listen(load_config("configs/default.yaml", []), *(int(x) for x in sys.argv[2:3])))
    else:
        main()
