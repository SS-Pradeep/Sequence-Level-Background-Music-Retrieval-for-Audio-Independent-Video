"""Worked example of the whole system on one held-out test case, for the paper's methodology section.

python -m bgm.analysis.worked_example   -> outputs/worked_example/

Case: title split, fold 1, frozen blocked test sequence 21 (data/manifests/cue_sequences_title.json): four test clips of
one film (7aum Arivu), four different songs, joined into one 72.3 s silent video. Every model is the fold-1 model, so the
clips are unseen in training, early stopping and tuning. The gallery is the fold-1 test music (the evaluation protocol).

Stages, each dumped to example.json: input -> shots and segments -> features -> coarse shortlist -> fine windows ->
lattice -> change predictor -> Viterbi (greedy / fixed penalties / change predictor) -> render. Also writes the silent
input, the original-music reference, one render per decoder, keyframes, tables.md and timeline.png.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bgm.config import load_config
from bgm.cue.hook import cue_logits, video_tokens
from bgm.cue.model import predict
from bgm.cue.stitch import decode_lattice_cue
from bgm.cue.train import load_cue
from bgm.data.manifest import read_folds
from bgm.data.prep import interim_music_path, interim_video_path
from bgm.eval.store import FeatureStore
from bgm.render.query import as_video_query, featurize
from bgm.render.render import SR, beat_sync_runs, fill_short_runs, mux, render_soundtrack, runs_from_path, seam_loudness_jumps
from bgm.retrieval.index import FoldModels, MusicIndex
from bgm.retrieval.pipeline import coarse_shortlist, fine, fuse, zstats
from bgm.retrieval.stitch_eval import (build_lattice, chosen_candidates, decode_lattice, fifths_matrix, greedy_lattice,
                                       switch_f1, switch_times, tempo_matrix)

log = logging.getLogger(__name__)
OUT = Path("outputs/worked_example")
FOLD, CASE_IDX = 1, 21
RUN = Path("runs/titlesplit/pipeline_ens10/fold1/seed0")
OVERRIDES = ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "pipeline.dyn_variant=null",
             "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "cue.sequences_file=cue_sequences_title.json", "cue.enabled=true"]
FPS = 25


def sh(*cmd: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *cmd], check=True)


def n_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    return int(out.strip())


def load_case() -> dict:
    e = json.loads((Path("runs/titlesplit/cue/eval") / f"fold{FOLD}" / "eval.json").read_text())
    return next(c for c in e["cases"] if c["split"] == "test" and c["mode"] == "blocked" and c["idx"] == CASE_IDX)


def build_input(cfg, store, case: dict) -> list[dict]:
    """Join the clips' silent 25 fps interim videos -> input_silent.mp4; the clips' own music -> original.wav."""
    clips, t = [], 0.0
    for p in case["parts"]:
        v = interim_video_path(cfg, p)
        d = n_frames(v) / FPS
        m = store.music_of[p]
        clips.append({"pair_id": p, "music_id": m, "track_id": store.music_track[m], "start_s": t, "end_s": t + d})
        t += d
    inp = OUT / "input_silent.mp4"
    if not inp.exists():
        ins = sum((["-i", str(interim_video_path(cfg, c["pair_id"]))] for c in clips), [])
        graph = "".join(f"[{i}:v]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[v]"
        sh(*ins, "-filter_complex", graph, "-map", "[v]", "-r", str(FPS), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(inp))
    parts = []
    for c in clips:                                           # music sample 0 = the clip's first frame (SPEC §3.4)
        y, sr = sf.read(interim_music_path(cfg, c["music_id"]), dtype="float32")
        assert sr == 48000, sr
        n = int(round((c["end_s"] - c["start_s"]) * sr))
        parts.append(np.pad(y[:n], (0, max(0, n - len(y)))))
    sf.write(OUT / "original.wav", np.concatenate(parts), 48000)
    return clips


def clip_at(clips: list[dict], a: float, b: float) -> tuple[int, float]:
    """The clip that covers most of [a, b), and the share of [a, b) it covers."""
    ov = [max(0.0, min(b, c["end_s"]) - max(a, c["start_s"])) for c in clips]
    k = int(np.argmax(ov))
    return k, ov[k] / (b - a)


def coarse_stage(index, vq, w, K: int, clips: list[dict], store) -> dict:
    """Per span (whole video, then each segment): top files and the rank of each true song (track rule)."""
    tracks = np.array([store.music_track[m] for m in index.ids])
    spans = [(0.0, vq.duration)] + list(vq.segments)
    keep, rows = set(), []
    for n, (a, b) in enumerate(spans):
        gv, pv = vq.pooled(a, b)
        s = fuse(index.file_scores(gv, pv), zstats(index.file_scores(gv, pv)), w)
        order = np.argsort(-s)
        keep.update(order[:K].tolist())
        rank = {t: int(np.flatnonzero(tracks[order] == t)[0]) + 1 for t in dict.fromkeys(c["track_id"] for c in clips)}
        rows.append({"span": "whole video" if n == 0 else f"seg {n}", "a": a, "b": b, "rank_of_true_songs": rank,
                     "top5": [{"music_id": index.ids[i], "z": float(s[i])} for i in order[:5]]})
    return {"gallery": len(index.ids), "K": K, "shortlist": sorted(keep), "rows": rows}


def fine_stage(index, results, w, clips, store) -> list[dict]:
    """Per segment: windows scored, top 5, the true song's best window rank and the exact true window's rank."""
    out = []
    for n, r in enumerate(results, 1):
        S = fuse(r.raw, r.stats, w)
        order = np.argsort(-S)
        k, share = clip_at(clips, r.a, r.b)
        c = clips[k]
        same_track = np.array([store.music_track[index.ids[f]] == c["track_id"] for f in r.fi])
        true_off = r.a - c["start_s"]
        exact = same_track & (r.fi == index.pos[c["music_id"]]) & (np.abs(r.st - true_off) <= 0.5)
        pos = np.empty(len(S), int)
        pos[order] = np.arange(1, len(S) + 1)
        out.append({"seg": n, "a": r.a, "b": r.b, "dur": r.b - r.a, "true_clip": k + 1, "true_share": share, "true_offset_s": true_off,
                    "n_windows": int(len(S)), "n_files": int(len(np.unique(r.fi))),
                    "top5": [{"music_id": index.ids[r.fi[i]], "start_s": float(r.st[i]), "z": float(S[i])} for i in order[:5]],
                    "true_song_best_rank": int(pos[same_track].min()) if same_track.any() else None,
                    "true_song_best_z": float(S[same_track].max()) if same_track.any() else None,
                    "true_window_rank": int(pos[exact].min()) if exact.any() else None})
    return out


def tokens_stage(feat: dict, raw: np.ndarray) -> list[dict]:
    sem = raw[:, :-3]
    cos = [float(sem[i] @ sem[i + 1] / (np.linalg.norm(sem[i]) * np.linalg.norm(sem[i + 1]))) for i in range(len(raw) - 1)]
    return [{"seg": i + 1, "log_dur": float(raw[i, -3]), "motion_mean": float(raw[i, -2]), "motion_std": float(raw[i, -1]),
             "siglip_cos_to_next": cos[i] if i < len(cos) else None} for i in range(len(raw))]


def per_seed_p(cfg, raw: np.ndarray) -> list[list[float]]:
    out = []
    for s in cfg.cue.seeds:
        model, scaler = load_cue(cfg, cfg.cue.variant, FOLD, s)
        out.append((1 / (1 + np.exp(-predict(model, [scaler(raw)])[0]))).tolist())
    return out


def move_costs(chosen, p: np.ndarray | None, prm: dict, pen: dict, tol: float = 0.04) -> list[dict]:
    """Move and transition cost at each boundary of a path. p None = the fixed-penalty decoder."""
    rows = []
    for i, (c0, c1) in enumerate(zip(chosen, chosen[1:])):
        if c1.music_id == c0.music_id:
            move = "continue" if abs(c1.offset - (c0.offset + c0.dur)) <= tol else "jump"
        else:
            move = "switch"
        music = float(fifths_matrix(np.array([c0.key]), np.array([c1.key]))[0, 0] + tempo_matrix(np.array([c0.tempo]), np.array([c1.tempo]))[0, 0])
        if p is None:
            cost = {"continue": 0.0, "jump": pen["lam_jump"], "switch": pen["lam_switch"] + pen["mu"] * music}[move]
        else:
            a = prm["alpha"]
            cost = {"continue": -a * np.log(1 - p[i]), "jump": -a * np.log(p[i]) + prm["lam_jump"],
                    "switch": -a * np.log(p[i]) + prm["mu"] * music}[move]
        rows.append({"boundary": i + 1, "move": move, "cost": float(cost), "key_tempo_distance": music})
    return rows


def path_report(name, chosen, results, clips, store, p, prm, pen) -> dict:
    truth = [c["start_s"] for c in clips[1:]]                 # every join here is a song change inside one film
    sw, jp = switch_times(chosen, results)
    acc = 0.0
    for c, r in zip(chosen, results):
        for cl in clips:
            ov = max(0.0, min(r.b, cl["end_s"]) - max(r.a, cl["start_s"]))
            acc += ov * (store.music_track[c.music_id] == cl["track_id"])
    moves = move_costs(chosen, p, prm, pen)
    segs = [{"seg": n, "music_id": c.music_id, "track_id": store.music_track[c.music_id], "offset_s": c.offset, "z": -c.cost}
            for n, c in enumerate(chosen, 1)]
    dur = results[-1].b
    return {"name": name, "segments": segs, "moves": moves, "switch_times": sw, "jump_times": jp,
            "switch_f1_1s": switch_f1(sw, truth, 1.0), "part_track_accuracy": acc / dur, "switches_per_min": 60 * len(sw) / dur,
            "emission_cost": float(sum(c.cost for c in chosen)), "transition_cost": float(sum(m["cost"] for m in moves))}


def render(cfg, store, feat, chosen, results, name: str, total: float) -> dict:
    runs = runs_from_path(chosen, [(r.a, r.b) for r in results])
    sync = []
    if cfg.render.beat_sync:
        runs, sync = beat_sync_runs(runs, feat["cut_times_s"], lambda m: store.music(m)["beat_times_s"],
                                    lambda m: store.duration("music", m), cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
    runs, loops = fill_short_runs(runs, lambda m: store.music(m)["beat_times_s"], lambda m: store.duration("music", m))
    mix, seams = render_soundtrack(runs, total, lambda m: interim_music_path(cfg, m), cfg.render)
    stereo = np.stack([mix, mix], 1) / np.sqrt(2)
    wav = OUT / "work" / f"{name}.wav"
    sf.write(wav, stereo, SR)
    mux(OUT / "input_silent.mp4", wav, OUT / f"{name}.mp4", 0.0, bitrate=str(cfg.render.get("audio_bitrate", "192k")))
    import pyloudnorm as pyln
    return {"pieces": [r.__dict__ for r in runs], "beat_sync": sync, "loops": loops, "seams_s": seams,
            "seam_loudness_jump_LU": seam_loudness_jumps(mix, seams),
            "lufs": float(pyln.Meter(SR).integrated_loudness(stereo.astype(np.float64))),
            "peak_dbfs": float(20 * np.log10(np.abs(stereo).max() + 1e-12))}


def keyframes(results) -> list[str]:
    out = []
    for n, r in enumerate(results, 1):
        f = OUT / "keyframes" / f"seg{n:02d}.jpg"
        f.parent.mkdir(parents=True, exist_ok=True)
        sh("-ss", f"{(r.a + r.b) / 2:.2f}", "-i", str(OUT / "input_silent.mp4"), "-frames:v", "1", "-vf", "scale=320:-2", str(f))
        out.append(str(f))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(exist_ok=True)
    cfg = load_config("configs/default.yaml", OVERRIDES)
    store = FeatureStore(cfg)
    case = load_case()
    timing = {}

    t0 = time.time()
    clips = build_input(cfg, store, case)
    feat = featurize(OUT / "input_silent.mp4", OUT / "work", cfg)
    timing["featurize_s"] = time.time() - t0
    segments = [tuple(map(float, s)) for s in feat["segments"]]
    dur = float(feat["meta"]["duration_s"])

    t0 = time.time()
    fusion = json.loads((RUN / "fusion.json").read_text())
    stitch = json.loads((RUN / "stitch.json").read_text())
    models = FoldModels.load(cfg, store, FOLD, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant)
    gallery = list(dict.fromkeys(store.music_of[p] for p in read_folds(cfg)["folds"][str(FOLD)]["test"]))
    assert all(c["music_id"] in gallery for c in clips)
    index = MusicIndex(store, models, gallery)
    vq = as_video_query(feat)
    vq.prepare(models, store.min_span)
    K, J, hop = fusion["K"], fusion["J"], fusion["window_hop_s"]
    coarse = coarse_stage(index, vq, fusion["w_coarse"], K, clips, store)
    assert coarse["shortlist"] == coarse_shortlist(index, vq, fusion["w_coarse"], K)
    results = fine(index, vq, coarse["shortlist"], hop)
    fine_rows = fine_stage(index, results, fusion["w_fine"], clips, store)
    lat = build_lattice(index, vq, results, fusion["w_fine"], J)
    timing["retrieval_s"] = time.time() - t0

    t0 = time.time()
    raw = video_tokens(feat)
    z, prm = cue_logits(cfg, raw, cfg.cue.variant, [FOLD], list(cfg.cue.seeds))
    p = 1 / (1 + np.exp(-z))
    seeds_p = per_seed_p(cfg, raw)
    timing["predictor_s"] = time.time() - t0

    t0 = time.time()
    pen = stitch["penalties"]
    picks = {"greedy": greedy_lattice(lat),
             "viterbi_fixed": decode_lattice(lat, pen, stitch["beam"], stitch["tol_s"]),
             "final": decode_lattice_cue(lat, z, prm["alpha"], prm["lam_jump"], prm["mu"], stitch["beam"], stitch["tol_s"])}
    timing["decode_s"] = time.time() - t0
    paths = {k: path_report(k, chosen_candidates(index, lat, v), results, clips, store, p if k == "final" else None, prm, pen)
             for k, v in picks.items()}
    paths["greedy"]["moves"] = [{**m, "cost": None} for m in paths["greedy"]["moves"]]     # greedy has no transition costs
    paths["greedy"]["transition_cost"] = None

    t0 = time.time()
    renders = {k: render(cfg, store, feat, chosen_candidates(index, lat, v), results, k, dur) for k, v in picks.items()}
    mux(OUT / "input_silent.mp4", OUT / "original.wav", OUT / "original.mp4", 0.0, bitrate="192k")
    timing["render_s"] = time.time() - t0

    ex = {"case": {"fold": FOLD, "set": "test / blocked", "idx": CASE_IDX, "film": "7aum Arivu (title group sunnxt:261 + sunnxt:429)",
                   "duration_s": dur, "clips": clips, "true_change_times_s": [c["start_s"] for c in clips[1:]]},
          "config": {"overrides": OVERRIDES, "run": str(RUN), "fusion": fusion, "stitch": stitch, "cue_params": prm,
                     "cue_variant": cfg.cue.variant, "cue_seeds": list(cfg.cue.seeds)},
          "shots": {"cut_times_s": [float(x) for x in feat["cut_times_s"]], "n_shots": int(len(feat["cut_times_s"]) + 1)},
          "segments": [{"seg": n, "a": a, "b": b, "dur": b - a, "true_clip": clip_at(clips, a, b)[0] + 1,
                        "true_share": clip_at(clips, a, b)[1]} for n, (a, b) in enumerate(segments, 1)],
          "features": {"sem_siglip_1hz": list(feat["sem_siglip_1hz"].shape), "dyn_25hz": list(feat["dyn_25hz"].shape)},
          "coarse": coarse, "fine": fine_rows,
          "lattice": [{"seg": n, "n_top": int(L.n_top), "n_candidates": int(len(L.mid))} for n, L in enumerate(lat, 1)],
          "predictor": {"tokens_shape": list(raw.shape), "tokens": tokens_stage(feat, raw), "p": p.tolist(), "p_per_seed": seeds_p,
                        "boundary_times_s": [b for _, b in segments[:-1]]},
          "paths": paths, "renders": renders, "timing_s": timing, "keyframes": keyframes(results)}
    (OUT / "example.json").write_text(json.dumps(ex, indent=1, default=float))
    figure(ex)
    log.info("done -> %s", OUT)


# ---------------------------------------------------------------- figure
TRUE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (dataviz reference palette): the 4 true songs
OTHER = "#b9b8b2"                                             # every other song: one neutral, told apart by its label
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#dcdbd5"


def tags_for(ex: dict) -> tuple[dict, dict]:
    """Track -> (label, colour): the true songs S1-S4 in clip order, every other song a-z in order of first use."""
    tag = {c["track_id"]: (f"S{k}", TRUE_COLORS[k - 1]) for k, c in enumerate(ex["case"]["clips"], 1)}
    n = 0
    for name in ("greedy", "viterbi_fixed", "final"):
        for s in ex["paths"][name]["segments"]:
            if s["track_id"] not in tag:
                tag[s["track_id"]] = (chr(ord("a") + n), OTHER)
                n += 1
    files = {}
    for name in ("greedy", "viterbi_fixed", "final"):
        for s in ex["paths"][name]["segments"]:
            files.setdefault(tag[s["track_id"]][0], s["music_id"])
    return tag, files


def figure(ex: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    ex = ex or json.loads((OUT / "example.json").read_text())
    tag, _ = tags_for(ex)
    segs, dur = ex["segments"], ex["case"]["duration_s"]
    changes = ex["case"]["true_change_times_s"]
    lanes = [("Original music\n(ground truth)", None), ("Greedy\n(best window per segment)", "greedy"),
             ("Viterbi,\nfixed penalties", "viterbi_fixed"), ("Viterbi + change\npredictor (final)", "final")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(13, 8.2), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 3.2, 1.35], hspace=0.28, left=0.15, right=0.985, top=0.95, bottom=0.07)

    ax0 = fig.add_subplot(gs[0])                              # keyframes, one per segment
    n = len(segs)
    for k, f in enumerate(ex["keyframes"]):
        ax0.imshow(imread(f), extent=(k + 0.04, k + 0.96, 0, 0.56), aspect="auto")
        ax0.text(k + 0.5, 0.62, f"seg {k + 1}\n{segs[k]['dur']:.1f} s", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax0.set_xlim(0, n), ax0.set_ylim(0, 0.95), ax0.axis("off")
    ax0.set_title("Held-out test case: 4 clips of one film (7aum Arivu), 4 different songs, joined into one 72 s silent video",
                  loc="left", fontsize=10.5, color=INK)

    ax = fig.add_subplot(gs[1])
    h = 0.62
    for row, (label, key) in enumerate(lanes):
        y = len(lanes) - 1 - row
        if key is None:
            pieces = [(c["start_s"], c["end_s"], c["track_id"]) for c in ex["case"]["clips"]]
        else:
            ps, moves = ex["paths"][key]["segments"], ex["paths"][key]["moves"]
            pieces, a = [], segs[0]["a"]
            for i, s in enumerate(ps):
                last = i == len(ps) - 1 or moves[i]["move"] != "continue"
                if last:
                    pieces.append((a, segs[i]["b"], s["track_id"]))
                    a = segs[i]["b"]
        for a, b, t in pieces:
            lab, col = tag[t]
            ax.add_patch(plt.Rectangle((a + 0.12, y - h / 2), b - a - 0.24, h, facecolor=col, edgecolor="none"))
            if b - a > 1.6:
                ax.text((a + b) / 2, y, lab, ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if col != OTHER else INK)
        if key is not None:
            r = ex["paths"][key]
            ax.text(dur + 0.8, y, f"F1 {r['switch_f1_1s']:.2f}\n{r['switches_per_min']:.1f} sw/min", va="center", fontsize=8, color=MUTED)
    for s in segs[1:]:
        ax.axvline(s["a"], color=GRID, lw=0.8, zorder=0)
    for t in changes:
        ax.axvline(t, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l for l, _ in lanes][::-1], fontsize=9, color=INK)
    ax.set_xlim(0, dur + 6.5), ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0, len(lanes) - 0.25, "S1–S4 = the clips' own songs (coloured); a, b, … = other library songs (grey). "
            "A gap inside one colour = a jump within the same file.  Dashed lines = true music changes.",
            fontsize=8, color=MUTED, va="bottom")

    ax2 = fig.add_subplot(gs[2], sharex=ax)
    bt, p = ex["predictor"]["boundary_times_s"], ex["predictor"]["p"]
    ax2.vlines(bt, 0, p, color="#2a78d6", lw=2)
    ax2.scatter(bt, p, s=28, color="#2a78d6", zorder=3, edgecolor="white", linewidth=1)
    for t, v in zip(bt, p):
        ax2.text(t, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5, color=INK, zorder=4,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    for t in changes:
        ax2.axvline(t, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax2.set_ylim(0, 1.18), ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylabel("p (music changes)", fontsize=9)
    ax2.set_xlabel("video time (s)", fontsize=9)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_xticks(range(0, int(dur) + 1, 10))
    ax2.set_title("Change predictor output at each segment boundary (mean of 3 seeds)", loc="left", fontsize=9.5, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"timeline.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    figure() if sys.argv[1:] == ["figure"] else main()
