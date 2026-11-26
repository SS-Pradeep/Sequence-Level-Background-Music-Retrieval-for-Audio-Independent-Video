"""All crossfade interpolation shapes (PROGRESS.md D50/D51): every gain family of `crossfade_gains` - including the
Whittaker-Shannon (sinc) band-limited step and the quintic C2 Hermite (smootherstep); the cubic Hermite step with flat
tangents is smoothstep itself, and a natural cubic spline on two knots is exactly linear - each with the default seam
gain ramps (D48), plus the ramp-less equal-power reference. Same test bed and metrics as
results/crossfade.md (bgm.analysis.crossfade_eval); paired against the shipped default, equal power + ramp.

Run: python -m bgm.analysis.crossfade_shapes            -> reports/crossfade_shapes.json, results/crossfade_shapes.md
     python -m bgm.analysis.crossfade_shapes onemp4     -> only the 1.mp4 renders (renders/shapes/) and their table
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from bgm.analysis.crossfade_eval import crossfade_lengths, fold_seed, seam_metrics, summarize
from bgm.config import load_config
from bgm.data.prep import interim_music_path
from bgm.eval.store import FeatureStore
from bgm.render.render import SR, Run, mux, render_soundtrack

log = logging.getLogger(__name__)
RAMP = 4.0
SHAPES = {"equal_power": ("equal_power", RAMP), "linear": ("linear", RAMP), "sqrt": ("sqrt", RAMP),
          "smoothstep": ("smoothstep", RAMP), "smoothstep_power": ("smoothstep_power", RAMP),
          "quadratic": ("quadratic", RAMP), "slow": ("slow", RAMP), "sinc": ("sinc", RAMP),
          "smootherstep": ("smootherstep", RAMP), "smootherstep_power": ("smootherstep_power", RAMP),
          "equal_power_no_ramp": ("equal_power", 0.0)}
THEORY_DB = {"equal_power": 0.0, "linear": -3.0, "sqrt": 0.0, "smoothstep": -3.0, "smoothstep_power": 0.0,
             "quadratic": -9.0, "slow": +0.5, "sinc": -3.0, "smootherstep": -3.0, "smootherstep_power": 0.0,
             "equal_power_no_ramp": 0.0}   # sinc: -3 + Gibbs ripple; smootherstep = quintic C2 Hermite (cubic Hermite = smoothstep)
AB_SEAMS = (22.28, 207.52, 326.03)


def variant_render(cfg, name: str):
    from bgm.analysis.crossfade_eval import variant_cfg
    return variant_cfg(cfg, name, SHAPES)


def onemp4(cfg) -> dict:
    """Render 1.mp4's final plan (same pieces as renders/1_bgm_final) in every shape: renders/shapes/1_bgm_<shape>.mp4,
    A/B clips of three song switches in renders/shapes/ab/, and per-shape seam metrics."""
    import pyloudnorm as pyln
    plan = json.loads(Path("renders/1_bgm_final.plan.json").read_text())
    runs = [Run(**q) for q in plan["pieces"]]
    xf = crossfade_lengths(runs, cfg.render)
    video = plan["video"]
    out_dir = Path(cfg.paths.renders) / "shapes"
    (out_dir / "ab").mkdir(parents=True, exist_ok=True)
    res = {}
    for name in SHAPES:
        mix, seams = render_soundtrack(runs, plan["duration_s"], lambda m: interim_music_path(cfg, m), variant_render(cfg, name))
        m = seam_metrics(mix, seams, xf)
        ok = {k: np.array([x for x in v if np.isfinite(x)]) for k, v in m.items()}
        stereo = np.stack([mix, mix], 1) / np.sqrt(2)
        wav = out_dir / f"{name}.wav"
        sf.write(wav, stereo, SR)
        mp4 = out_dir / f"1_bgm_{name}.mp4"
        mux(video, wav, mp4)
        wav.unlink()
        for t in AB_SEAMS:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t - 4:.2f}", "-i", str(mp4), "-t", "8",
                            "-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-b:a", "192k",
                            str(out_dir / "ab" / f"seam_{t:06.2f}s_{name}.mp4")], check=True)
        res[name] = {"mp4": str(mp4), "n_seams": len(seams), "jump_3s_median": float(np.median(ok["jump_3s"])),
                     "jump_3s_over_2": int((ok["jump_3s"] > 2).sum()), "jump_1s_median": float(np.median(ok["jump_1s"])),
                     "dip_signed_median": float(np.median(ok["dip"])), "dips_below_-3": int((ok["dip"] < -3).sum()),
                     "lufs": float(pyln.Meter(SR).integrated_loudness(stereo.astype(np.float64)))}
        log.info("1.mp4 %s: jump3s %.2f LU, dip %+0.2f LU -> %s", name, res[name]["jump_3s_median"], res[name]["dip_signed_median"], mp4)
    return res


def _ci(x: dict) -> str:
    return f"{x['diff']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def write_report(summary: dict, one: dict, n_rows: int) -> Path:
    base = "equal_power"
    L = ["# Crossfade interpolation shapes", "",
         "Every gain family of `crossfade_gains` (D50), each with the default 4 s seam gain ramps (D48); "
         "`equal_power_no_ramp` is the pre-D48 renderer for reference. Same held-out test bed and metrics as "
         "results/crossfade.md; Δ: paired against **equal_power (+ ramp, the shipped default)**, mean over seams, "
         "95% cluster-bootstrap CI over videos. Theory: mid-fade level on uncorrelated material.", "",
         f"## Held-out test concatenations ({n_rows} renders per shape)", "",
         "| shape | theory mid-fade dB | jump 3 s median | Δ jump 3 s | > 2 LU | jump 1 s median | Δ jump 1 s | dip median (signed) | |dip| Δ | dips < −3 LU | LUFS |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in summary.items():
        b = name == base
        L.append(f"| {name} | {THEORY_DB[name]:+.1f} | {v['jump_3s']['median']:.2f} | {'—' if b else _ci(v['jump_3s'])} | "
                 f"{v['jumps_3s_over_2LU']:.0%} | {v['jump_1s']['median']:.2f} | {'—' if b else _ci(v['jump_1s'])} | "
                 f"{v['dip_signed_median']:+.2f} | {'—' if b else _ci(v['dip'])} | {v['dips_below_-3LU']:.0%} | {v['lufs_median']:.2f} |")
    L += ["", "## 1.mp4 (same pieces as renders/1_bgm_final; videos in renders/shapes/, A/B clips in renders/shapes/ab/)", "",
          "| shape | jump 3 s median | > 2 LU | jump 1 s median | dip median (signed) | dips < −3 LU | LUFS | file |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in one.items():
        L.append(f"| {name} | {r['jump_3s_median']:.2f} | {r['jump_3s_over_2']} | {r['jump_1s_median']:.2f} | "
                 f"{r['dip_signed_median']:+.2f} | {r['dips_below_-3']} of {r['n_seams']} | {r['lufs']:.2f} | {Path(r['mp4']).name} |")
    out = Path("results/crossfade_shapes.md")
    out.write_text("\n".join(L) + "\n")
    return out


def main(folds=(0, 1, 2, 3, 4), seeds=(0, 1, 2)) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", [])
    one = onemp4(cfg)
    store = FeatureStore(cfg)
    rows = [r for fold in folds for seed in seeds for r in fold_seed(cfg, store, fold, seed, SHAPES)]
    summary = summarize(rows, SHAPES, base="equal_power")
    Path(cfg.paths.reports, "crossfade_shapes.json").write_text(json.dumps(
        {"ramp_s": RAMP, "summary": summary, "onemp4": one, "rows": rows}, default=float))
    log.info("wrote %s", write_report(summary, one, len(rows)))


if __name__ == "__main__":
    if sys.argv[1:] == ["onemp4"]:
        logging.basicConfig(level=logging.INFO)
        print(json.dumps(onemp4(load_config("configs/default.yaml", [])), indent=1, default=float))
    else:
        main()
