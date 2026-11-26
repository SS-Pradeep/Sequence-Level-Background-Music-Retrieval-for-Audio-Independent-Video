"""Cue-switch demo renders on held-out test sequences, one per dominant visual mood (PROGRESS.md D84).

Each demo is a frozen single-film test sequence (cue_sequences_film.json, test split of its fold), joined into one silent
640x360 25 fps video. It is rendered like any input video: featurize, segments from its own shots, retrieval, Viterbi,
beat-synced render. Three versions:
  <name>_d75.mp4       the D75 system (fixed penalties)
  <name>_cue.mp4       the same render with cue.enabled=true (R4, the fold's 3 seeds)
  <name>_original.mp4  the clips' own soundtracks joined (the truth, for listening)
Leak-free: the run is the sequence's own fold (its test films were never trained on), and render's long-video
shortlist is restricted to that fold's heads (in a stock render it pools all folds). Mood = the SigLIP zero-shot mood
tags (features.attributes.moods) of the frames, z-scored over the candidate sequences: a visual estimate, not a label.
True music changes = joins between clips of different track groups. Metrics per render: boundary F1 at +-1 / +-0.5 s
(greedy one-to-one matching), switches / min, duration-weighted share of the video on the right track group.

  python -m bgm.analysis.cue_demo <picks.json>     -> renders/cue_demo/, results/cue_demo.md
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from bgm.config import load_config
from bgm.cue.evaluate import match_counts
from bgm.data.prep import interim_music_path, interim_video_path
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
OUT = Path("renders/cue_demo")
BASE = ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", "pipeline.dyn_variant=null",
        "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", f"paths.renders={OUT}"]
SR = 48000


def n_frames(path: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def build_input(cfg, parts: list[str], dst: Path) -> list[int]:
    """Join the clips' silent interim videos (scaled / padded to 640x360, 25 fps). Returns each clip's frame count."""
    srcs = [interim_video_path(cfg, p) for p in parts]
    counts = [n_frames(s) for s in srcs]
    if not dst.exists():
        args = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        for s in srcs:
            args += ["-i", str(s)]
        chains = [f"[{i}:v]fps=25,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
                  for i in range(len(srcs))]
        fc = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(srcs))) + f"concat=n={len(srcs)}:v=1:a=0[out]"
        subprocess.run(args + ["-filter_complex", fc, "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                               str(dst)], check=True)
    assert n_frames(dst) == sum(counts), (dst, n_frames(dst), sum(counts))
    return counts


def original_soundtrack(cfg, store, parts, counts, video: Path, dst: Path) -> None:
    """The clips' own BGM (interim 48 kHz, sample 0 = first frame), each cut / padded to its clip, at -16 LUFS."""
    if dst.exists():
        return
    import pyloudnorm as pyln
    pieces = []
    for p, n in zip(parts, counts):
        x, sr = sf.read(interim_music_path(cfg, store.music_of[p]), dtype="float64")
        assert sr == SR
        x = x if x.ndim == 1 else x.mean(1)
        m = int(round(n / 25 * SR))
        pieces.append(np.r_[x[:m], np.zeros(max(0, m - len(x)))])
    y = np.concatenate(pieces)
    y = y * 10 ** ((-16.0 - pyln.Meter(SR).integrated_loudness(y)) / 20)
    y = np.clip(y, -0.99, 0.99)
    wav = dst.with_suffix(".wav")
    sf.write(wav, np.stack([y, y], 1) / np.sqrt(2), SR)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)], check=True)
    wav.unlink()


def render(video: Path, name: str, fold: int, cue: bool) -> dict:
    import bgm.analysis.long_video as lv
    from bgm.render.run import render_video
    cfg = load_config("configs/default.yaml", BASE + (["cue.enabled=true"] if cue else []))
    orig = lv.load_models
    lv.load_models = lambda c, fold_=None, _k=fold: orig(c, fold=_k)      # leak-free long-video shortlist
    try:
        tag = "cue" if cue else "d75"
        if cue:                                                          # same cached features as the D75 render
            src, dst = OUT / f"{name}_d75", OUT / f"{name}_cue"
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("query.npz", "silent25.mp4"):
                if (src / f).exists() and not (dst / f).exists():
                    (dst / f).write_bytes((src / f).read_bytes())
        plan = render_video(cfg, str(video), str(OUT / f"{name}_{tag}.mp4"), f"runs/filmsplit/pipeline_ens10/fold{fold}/seed0")
    finally:
        lv.load_models = orig
    return json.loads(Path(plan).read_text())


def score(plan: dict, joins: list[float], seg_track_true) -> dict:
    s = plan["segments"]
    sw = [y["start_s"] for x, y in zip(s, s[1:]) if y["music_id"] != x["music_id"]]
    mins = plan["duration_s"] / 60
    out = {"switches": sw, "switches_per_min": len(sw) / mins}
    for tol in (1.0, 0.5):
        tp, fp, fn = match_counts(sw, joins, tol)
        out[f"f1_{tol:g}"] = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        out[f"tp_fp_fn_{tol:g}"] = [tp, fp, fn]
    bounds = np.array([x["start_s"] for x in s[1:]])
    out["reachable"] = int(sum(len(bounds) > 0 and np.abs(bounds - j).min() <= 1.0 for j in joins))   # changes a switch can hit
    w = np.array([x["end_s"] - x["start_s"] for x in s])
    ok = np.array([x["track_id"] == seg_track_true((x["start_s"] + x["end_s"]) / 2) for x in s])
    out["right_track"] = float(np.average(ok, weights=w))
    return out


def plot(name: str, title: str, dur: float, clips: list[tuple[float, float, str]], joins: list[float], plans: dict, scores: dict,
         truth, path: Path, cue_label: str = "cue (R4)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid, surf = "#0b0b0b", "#52514e", "#c3c2b7", "#e8e7e3", "#fcfcfb"
    right, other, true_c = "#2a78d6", "#d6d5cf", "#8a8983"
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(12, 3.6), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True, facecolor=surf)
    lanes = [("True music", None), ("D75", "d75"), (cue_label, "cue")]
    gap = dur * 0.005
    for row, (label, key) in enumerate(lanes):
        y = len(lanes) - 1 - row
        if key is None:
            pieces = [(a, b, True) for a, b, _ in clips]
            color = lambda ok: true_c  # noqa: E731
        else:
            segs = plans[key]["segments"]
            pieces, cur = [], None
            for x in segs:                                   # merge consecutive segments of one music file into a piece
                if cur and cur[2] == x["music_id"]:
                    cur[1] = x["end_s"]
                    cur[3].append(x)
                else:
                    cur = [x["start_s"], x["end_s"], x["music_id"], [x]]
                    pieces.append(cur)
            pieces = [(a, b, np.mean([s["track_id"] == truth((s["start_s"] + s["end_s"]) / 2) for s in ss]) > 0.5) for a, b, _, ss in pieces]
            color = lambda ok: right if ok else other  # noqa: E731
        for a, b, ok in pieces:
            ax.barh(y, max(b - a - gap, gap), left=a + gap / 2, height=0.62, color=color(ok), edgecolor="none")
        ax.text(-dur * 0.01, y, label, ha="right", va="center", fontsize=9, color=ink)
        if key is not None:
            sc = scores[key]
            for t in sc["switches"]:                         # filled = within 1 s of a true change, hollow = not
                hit = any(abs(t - j) <= 1.0 for j in joins)
                ax.scatter([t], [y + 0.47], marker="v", s=34, zorder=4, linewidths=1.2,
                           facecolors=ink if hit else surf, edgecolors=ink)
            ax.text(dur * 1.01, y, f"F1 {sc['f1_1']:.2f} · {sc['switches_per_min']:.1f}/min · right track {sc['right_track']:.0%}",
                    ha="left", va="center", fontsize=8.5, color=ink2)
    for t in joins:
        for a_ in (ax, axp):
            a_.axvline(t, color=ink2, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks([])
    ax.tick_params(axis="x", bottom=False)
    ax.set_ylim(-0.6, len(lanes) - 0.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=ink, pad=8)
    cs = plans["cue"]["segments"]
    bt = np.array([x["start_s"] for x in cs[1:]])
    p = np.array(plans["cue"]["cue"]["p"])
    axp.vlines(bt, 0, p, color=right, lw=2)
    axp.scatter(bt, p, s=16, color=right, zorder=3)
    axp.axhline(0.5, color=muted, lw=1)
    axp.set_ylim(0, 1.05)
    axp.set_yticks([0, 0.5, 1])
    axp.tick_params(colors=ink2, labelsize=8)
    axp.set_ylabel("predicted p", fontsize=8.5, color=ink2)
    axp.set_xlabel("time (s) · dashed = true music change · ▼ switch (filled: within 1 s of a true change) · blue = right track, grey = other",
                   fontsize=8.5, color=ink2)
    axp.grid(axis="y", color=grid, lw=0.8)
    for k in ("top", "right"):
        axp.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        axp.spines[k].set_color(muted)
    axp.set_xlim(0, dur)
    for a_ in (ax, axp):
        a_.set_facecolor(surf)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.84, bottom=0.17, hspace=0.12)
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main(picks_path: str, replot_only: bool = False) -> Path:
    from bgm.log import setup_logging
    setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    picks = json.loads(Path(picks_path).read_text())
    cfg = load_config("configs/default.yaml", BASE)
    store = FeatureStore(cfg)
    rows = []
    for mood, c in picks.items():
        name = f"{mood}_f{c['fold']}_{c['mode']}{c['idx']}"
        video = OUT / f"{name}_input.mp4"
        counts = build_input(cfg, c["parts"], video)
        starts = np.r_[0, np.cumsum(counts)[:-1]] / 25.0
        ends = np.cumsum(counts) / 25.0
        tracks = [store.track[p] for p in c["parts"]]
        joins = [float(t) for t, a, b in zip(starts[1:], tracks, tracks[1:]) if a != b]
        truth = lambda t, s=starts, tr=tracks: tr[int(np.searchsorted(s, t, side="right") - 1)]  # noqa: E731
        original_soundtrack(cfg, store, c["parts"], counts, video, OUT / f"{name}_original.mp4")
        plans = ({k: json.loads((OUT / f"{name}_{k}.plan.json").read_text()) for k in ("d75", "cue")} if replot_only else
                 {"d75": render(video, name, c["fold"], False), "cue": render(video, name, c["fold"], True)})
        scores = {k: score(v, joins, truth) for k, v in plans.items()}
        film = store.pairs.loc[c["parts"][0], "meta_film"]
        title = (f"{mood}  ·  film {film}, fold {c['fold']} test, {c['mode']}  ·  {ends[-1]:.0f} s, {len(c['parts'])} clips, "
                 f"{len(joins)} true changes\nmood tags (SigLIP, zero-shot): {', '.join(c['top_moods'])}")
        plot(name, title, float(ends[-1]), list(zip(starts, ends, tracks)), joins, plans, scores, truth, OUT / f"{name}_timeline.png")
        rows.append({"mood": mood, "name": name, "film": str(film), "film_group": c["film"], "fold": c["fold"], "mode": c["mode"],
                     "duration_s": float(ends[-1]), "clips": len(c["parts"]), "true_changes": joins, "top_moods": c["top_moods"],
                     "segments": len(plans["d75"]["segments"]), "scores": scores})
        log.info("demo %s: D75 F1 %.2f (%d sw) | cue F1 %.2f (%d sw) | right track %.2f vs %.2f", name, scores["d75"]["f1_1"],
                 len(scores["d75"]["switches"]), scores["cue"]["f1_1"], len(scores["cue"]["switches"]),
                 scores["d75"]["right_track"], scores["cue"]["right_track"])
    (OUT / "demo.json").write_text(json.dumps(rows, indent=1, default=float))
    return write_md(rows)


def write_md(rows: list[dict]) -> Path:
    L = ["# Cue-switch demo renders by visual mood", "", "Generated by `python -m bgm.analysis.cue_demo` (PROGRESS.md D84). One held-out "
         "single-film test sequence per mood (SigLIP zero-shot mood tags of the frames, not human labels), joined into one silent "
         "video and rendered like any input: D75 vs the cue hook (R4, the sequence's own fold, leak-free). Files in renders/cue_demo/: "
         "`<name>_d75.mp4`, `<name>_cue.mp4`, `<name>_original.mp4` (the true soundtrack), `<name>_timeline.png`.", "",
         "| mood | film (fold, set) | length | true changes | reachable | D75 F1 ±1 s | cue F1 ±1 s | D75 switches / min | cue switches / min | true changes / min | right track D75 / cue |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    tot = {k: np.zeros(3) for k in ("d75", "cue")}
    for r in rows:
        s, m = r["scores"], r["duration_s"] / 60
        for k in tot:
            tot[k] += s[k]["tp_fp_fn_1"]
        L.append(f"| {r['mood']} | {r['film']} ({r['fold']}, {r['mode']}) | {r['duration_s']:.0f} s | {len(r['true_changes'])} | {s['d75']['reachable']} | "
                 f"{s['d75']['f1_1']:.2f} | {s['cue']['f1_1']:.2f} | {s['d75']['switches_per_min']:.1f} | {s['cue']['switches_per_min']:.1f} | "
                 f"{len(r['true_changes']) / m:.1f} | {s['d75']['right_track']:.0%} / {s['cue']['right_track']:.0%} |")
    f = {k: 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1e-9) for k, v in tot.items()}
    L += ["", f"Pooled over the {len(rows)} demos (±1 s): D75 F1 {f['d75']:.2f} (TP {tot['d75'][0]:.0f}, FP {tot['d75'][1]:.0f}, "
          f"FN {tot['d75'][2]:.0f}); cue F1 {f['cue']:.2f} (TP {tot['cue'][0]:.0f}, FP {tot['cue'][1]:.0f}, FN {tot['cue'][2]:.0f}). "
          "A handful of hand-picked videos: illustrations, not evidence (results/cue_switch.md has the test-set numbers). "
          "Reachable = true changes with a segment boundary within 1 s (the segments come from the joined video's own shots; "
          "a change inside a segment cannot be switched at). Right track is 0% for both because the whole 1,248-file library is "
          "searched and these films are unseen: the true songs rank median 235 of 1,248 by whole-clip score."]
    out = Path("results/cue_demo.md")
    out.write_text("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    print(main(sys.argv[1], replot_only="--replot" in sys.argv))
