"""Demo outputs for the Experimental Results section (slides): setup and qualitative results.

Nothing is retrained and no pipeline file is changed. Everything is read from the manifests, the saved runs and the
cached features, and the pipeline's own functions are called in the order `bgm render` (bgm.render.run) calls them:
  cleaned_dataset/manifests/pairs.csv, segments.csv, tracks.csv, folds_title.json   cleaned 1,011 pairs, scenes, music library, title split
  runs/cleaned/semantic/siglip-both/fold<k>/seed<s>/                      training config, log and metrics (50 runs)
  runs/cleaned/pipeline_ens10/fold<k>/seed0/fusion.json, stitch.json      retrieval settings and tuned stitch penalties
  (settings: entry_point() below, equal to runs/cleaned/pipeline_ens10/fold0/seed0/config.yaml)
  bgm.retrieval.index.FoldModels.load, MusicIndex, pair_query             10-seed ensemble, music library, video query
  bgm.retrieval.pipeline.coarse_shortlist, fine, fuse, zstats, truth_for  coarse-to-fine retrieval, ground truth
  bgm.retrieval.stitch_eval.build_lattice, decode_lattice                 Viterbi stitching
  bgm.render.render.runs_from_path, beat_sync_runs, fill_short_runs, render_soundtrack, mux

Held-out rule: every example video is run with the models of the fold that holds it in its TEST split, so its film
(and its music) was never in that fold's training or validation set. This is asserted and printed per video.
The music library is the whole library (every music file of the dataset), as in `bgm render` and demos 05-07; a
chosen file's own split in the fold is reported. Video features are the cached ones of the silent 25 fps interim
video (computed once by `bgm extract`), and the models run on CPU.

Usage:
  python demo_outputs/08_experimental_results.py setup                    08_setup.txt, 08_dataset_stats.png
  python demo_outputs/08_experimental_results.py candidates               test clips by zero-shot mood (to choose examples)
  python demo_outputs/08_experimental_results.py videos <pair_id> ...     08_video_<id>/ for each video
  python demo_outputs/08_experimental_results.py cases [--recompute]      08_good_vs_failure.txt / .png, 08_test_scenes/fold<k>.csv
  python demo_outputs/08_experimental_results.py cases --fold k           score one fold only (about 5 min on CPU with OMP_NUM_THREADS=4; the 5 folds can run in parallel)
"""
from __future__ import annotations

import argparse
import functools
import importlib.metadata as md
import inspect
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import OmegaConf

from bgm.config import load_config
from bgm.data import manifest
from bgm.data.media import ffprobe
from bgm.data.prep import interim_music_path, interim_video_path
from bgm.eval.store import FeatureStore
from bgm.render.render import SR, beat_sync_runs, fill_short_runs, mux, render_soundtrack, runs_from_path
from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
from bgm.retrieval.pipeline import coarse_shortlist, fine, fuse, truth_for, zstats
from bgm.retrieval.stitch_eval import build_lattice, chosen_candidates, decode_lattice

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ENTRY = ROOT / "render_best.sh"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE = COLOURS[0], COLOURS[1]
INK, INK2, MUTED, GRID, OTHER = "#0b0b0b", "#52514e", "#898781", "#e4e2dc", "#c9c7c1"
W = 118                                                     # width of the text reports
TOL_S = 0.5                                                 # window hit: right file, start within 0.5 s (segment_hits)
NEAR_TIE_Z = 0.10                                           # top-1 and top-2 closer than this (standardized score) = near-tie

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
log = logging.getLogger("demo08")


# ---------------------------------------------------------------- the entry point's settings
def entry_point() -> tuple[list[str], Path]:
    """(config overrides, run directory) of the FINAL system: cleaned 1,011 pairs, title split, 10-seed ensemble
    (runs/cleaned/pipeline_ens10; these overrides reproduce its saved config.yaml)."""
    return ["paths.manifests=cleaned_dataset/manifests", "paths.features=cleaned_dataset/features",
            "paths.interim=cleaned_dataset/interim", "paths.runs=runs/cleaned", "paths.reports=reports/cleaned",
            "paths.results=results/cleaned", "splits.folds_file=folds_title.json",
            "eval.concat_sets_file=concat_sets_title.json", "cue.enabled=false", "pipeline.exp=pipeline_ens10",
            "pipeline.dyn_variant=null", "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"], \
        Path("runs/cleaned/pipeline_ens10/fold0/seed0")


OVERRIDES, ENTRY_RUN = entry_point()
RUN_ROOT = ENTRY_RUN.parent.parent                          # runs/cleaned/pipeline_ens10


def short_name(music_id: str) -> str:
    """'combined_full_bgm_dataset_youtube__14__14_bgm_0002' -> 'youtube 14 #0002' (source or film, clip number)."""
    m = re.fullmatch(r"sunnxt_(\d+)_\d+_bgm_(\d+)", music_id)
    if m:
        return f"sunnxt {m[1]} #{m[2]}"
    m = re.fullmatch(r"combined_full_bgm_dataset_?(.*?)__(.+?)__.*_bgm_(\d+)", music_id)
    if m:
        return f"{m[1]} {m[2]} #{m[3]}" if m[1] else f"{m[2]} #{m[3]}"
    return music_id


def slug(pair_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", short_name(pair_id)).strip("_")


def table(headers: list[str], rows: list[list], left: tuple[int, ...] = (0,)) -> list[str]:
    """Column-aligned text table; columns in `left` are left-aligned, the rest right-aligned."""
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    fmt = lambda r: "  ".join(c.ljust(w[i]) if i in left else c.rjust(w[i]) for i, c in enumerate(r)).rstrip()
    return [fmt(headers), "  ".join("-" * x for x in w)] + [fmt(r) for r in cells]


def kv(key: str, value, width: int = 30) -> str:
    return f"{key:<{width}}{value}"


def head(title: str) -> list[str]:
    return ["-" * W, title, "-" * W]


def emit(lines: list[str], path: Path) -> None:
    text = "\n".join(lines)
    path.write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- data shared by every part
class Data:
    def __init__(self):
        self.cfg = load_config(ROOT / "configs/default.yaml", OVERRIDES)
        self.store = FeatureStore(self.cfg)
        doc = manifest.read_folds(self.cfg)
        self.folds, self.film = doc["folds"], doc["film_group"]
        self.train_only = list(doc.get("train_only", []))
        self.pair_of_music = dict(zip(self.store.pairs.music_id, self.store.pairs.pair_id))

    def test_fold(self, pair_id: str) -> int | None:
        return next((int(k) for k, f in self.folds.items() if pair_id in f["test"]), None)

    def split_of_music(self, fold: int, music_id: str) -> str:
        p = self.pair_of_music[music_id]
        return next(s for s in ("train", "val", "test") if p in self.folds[str(fold)][s])

    def held_out(self, pair_id: str, fold: int) -> dict:
        """Counts proving the pair's film (and track) is outside the fold's training and validation sets."""
        f, g, t = self.folds[str(fold)], self.film[pair_id], self.store.track[pair_id]
        out = {"film": g, "track": t, "in_test": pair_id in f["test"],
               "film_pairs": sum(self.film[p] == g for p in self.film),
               **{f"{s}_film_pairs": sum(self.film[p] == g for p in f[s]) for s in ("train", "val", "test")},
               **{f"{s}_films": len({self.film[p] for p in f[s]}) for s in ("train", "val", "test")},
               "train_track_pairs": sum(self.store.track[p] == t for p in f["train"] + f["val"])}
        out["ok"] = out["in_test"] and not out["train_film_pairs"] and not out["val_film_pairs"] and not out["train_track_pairs"]
        return out


@functools.lru_cache(maxsize=1)
def fold_ctx(data: Data, fold: int) -> dict:
    """The fold's 10-seed ensemble, the whole-library index and the run's saved retrieval / stitch settings."""
    run = RUN_ROOT / f"fold{fold}" / ENTRY_RUN.name
    cfg, store = data.cfg, data.store
    models = FoldModels.load(cfg, store, fold, int(run.name[4:]), cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device="cpu")
    index = MusicIndex(store, models, list(dict.fromkeys(store.pairs.music_id)))
    return {"fold": fold, "run": run, "models": models, "index": index,
            "gtrack": np.array([store.music_track[m] for m in index.ids]),
            "fusion": json.loads((run / "fusion.json").read_text()), "stitch": json.loads((run / "stitch.json").read_text())}


def analyse(data: Data, ctx: dict, pair_id: str) -> dict:
    """Coarse-to-fine retrieval and Viterbi for one dataset video, with the rank of its own music at each stage.
    Ranks count the candidates that score strictly higher, plus one (tied candidates share the rank)."""
    store, index, fusion, stitch = data.store, ctx["index"], ctx["fusion"], ctx["stitch"]
    vq = pair_query(store, pair_id)
    vq.prepare(ctx["models"], store.min_span)
    short = coarse_shortlist(index, vq, fusion["w_coarse"], fusion["K"])
    results = fine(index, vq, short, fusion["window_hop_s"])
    lat = build_lattice(index, vq, results, fusion["w_fine"], fusion["J"])
    chosen = chosen_candidates(index, lat, decode_lattice(lat, stitch["penalties"], stitch["beam"], stitch["tol_s"]))
    scenes = []
    for k, (r, c) in enumerate(zip(results, chosen)):
        m_true, off = truth_for(vq, store, r.a, r.b)
        group = ctx["gtrack"] == store.music_track[m_true]
        raw = index.file_scores(*vq.pooled(r.a, r.b))
        zc = fuse(raw, zstats(raw), fusion["w_coarse"])                       # what the coarse stage ranks the library by
        order = np.argsort(-zc)
        S = fuse(r.raw, r.stats, fusion["w_fine"])                            # what the fine stage ranks the windows by
        win_ok = (r.fi == index.pos[m_true]) & (np.abs(r.st - off) <= TOL_S + 1e-9)
        best = int(np.argmax(S))
        per_file = pd.Series(S).groupby(r.fi).max().sort_values(ascending=False)
        scenes.append({
            "scene": k, "start_s": r.a, "end_s": r.b, "true_id": m_true, "true_off_s": off,
            "file_rank": int(1 + (zc > zc[group].max()).sum()), "n_files": len(zc), "true_z": float(zc[group].max()),
            "top_files": [(index.ids[i], float(zc[i])) for i in order[:5]], "coarse_margin": float(zc[order[0]] - zc[order[1]]),
            "shortlisted": bool(np.isin(np.flatnonzero(group), short).any()), "n_short": len(short), "n_windows": len(S),
            "window_rank": int(1 + (S > S[win_ok].max()).sum()) if win_ok.any() else None,
            "fine_top_id": index.ids[r.fi[best]], "fine_top_start_s": float(r.st[best]),
            "fine_margin": float(per_file.iloc[0] - per_file.iloc[1]) if len(per_file) > 1 else float("nan"),
            "chosen_id": c.music_id, "chosen_off_s": c.offset, "chosen_split": data.split_of_music(ctx["fold"], c.music_id),
            "chosen_track": bool(store.music_track[c.music_id] == store.music_track[m_true]),
            "chosen_window": bool(c.music_id == m_true and abs(c.offset - off) <= TOL_S + 1e-9)})
    runs = runs_from_path(chosen, [(r.a, r.b) for r in results], stitch["tol_s"])
    joins = ["switch" if a.music_id != b.music_id else "jump" for a, b in zip(runs, runs[1:])]
    return {"pair_id": pair_id, "fold": ctx["fold"], "duration_s": vq.duration, "cuts": vq.cut_times, "scenes": scenes,
            "runs": runs, "switches": joins.count("switch"), "jumps": joins.count("jump")}


def frame_at(cfg, pair_id: str, t: float) -> np.ndarray:
    """RGB frame of the silent interim video at time t (s)."""
    cap = cv2.VideoCapture(str(interim_video_path(cfg, pair_id)))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * t)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"no frame at {t:.2f} s in {pair_id}"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def held_out_lines(pair_id: str, fold: int, h: dict) -> list[str]:
    return [kv("Test fold", f"{fold}  (the models used are fold {fold}'s; they are trained on fold {fold}'s train split only)"),
            kv("Film", f"{h['film']}  ({h['film_pairs']} pair{'s' * (h['film_pairs'] != 1)} in the dataset)"),
            kv("Film in this fold's", f"train: {h['train_film_pairs']} pairs of {h['train_films']} films   "
               f"val: {h['val_film_pairs']} pairs of {h['val_films']} films   test: {h['test_film_pairs']} pairs of {h['test_films']} films"),
            kv("Own music track", f"{h['track']}  (pairs with this track in train + val: {h['train_track_pairs']})"),
            kv("Held-out check", "[OK] film and music are not in this fold's training or validation set" if h["ok"]
               else "[FAIL] this video is NOT held out in this fold")]


# ---------------------------------------------------------------- 1. setup
def hf_snapshot(repo: str) -> str:
    ref = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}" / "refs/main"
    return ref.read_text().strip()[:12] if ref.exists() else "not in the local cache"


def hardware() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), platform.processor())
    ram = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) / 2**20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda = torch.cuda.is_available()
    if cuda:
        gpu = f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB"
    else:
        info = [p.read_text() for p in Path("/proc/driver/nvidia/gpus").glob("*/information")] if Path("/proc/driver/nvidia/gpus").exists() else []
        gpu = ", ".join(re.search(r"Model:\s+(.+)", t)[1].strip() for t in info) or "none found"
    return {"cpu": cpu, "threads": os.cpu_count(), "ram_gib": ram, "gpu": gpu, "cuda": cuda}


def training_runs(cfg) -> dict:
    """Settings, epochs and learned temperature of every semantic run the ensemble uses (5 folds x 10 seeds)."""
    root = ROOT / cfg.paths.runs / "semantic" / cfg.pipeline.sem_variant
    runs = [root / f"fold{k}" / f"seed{s}" for k in range(cfg.splits.n_folds) for s in cfg.pipeline.sem_seeds]
    confs = [OmegaConf.load(r / "config.yaml") for r in runs]
    sem = [OmegaConf.to_container(c.train.semantic) for c in confs]
    common = set.intersection(*(set(s) for s in sem))      # later commits added keys (all switched off) to the config
    assert all(s[k] == sem[0][k] for s in sem for k in common), "the 50 semantic runs do not share one training config"
    assert [int((r / "seed.txt").read_text()) for r in runs] == [int(r.name[4:]) for r in runs]
    met = [json.loads((r / "metrics.json").read_text())["val"] for r in runs]
    logs = [pd.read_csv(r / "train_log.csv") for r in runs]
    return {"runs": runs, "conf": confs[0], "sem": sem[0], "epochs": np.array([m["epochs_run"] for m in met]),
            "best": np.array([m["best_epoch"] for m in met]),
            "tau": np.array([float(l.loc[l.epoch == m["best_epoch"], "tau"].iloc[0]) for l, m in zip(logs, met)]),
            "seconds": np.array([l.seconds.sum() for l in logs]),
            "commits": sorted({(r / "git_commit.txt").read_text().strip()[:7] for r in runs})}


def stats(x) -> str:
    x = np.asarray(x, float)
    return (f"min {x.min():.2f}   median {np.median(x):.2f}   mean {x.mean():.2f}   max {x.max():.2f}   (s)")


def setup(data: Data) -> None:
    cfg, pairs = data.cfg, data.store.pairs
    segs, tracks = manifest.read_segments(cfg), manifest.read_tracks(cfg)
    film = pairs.pair_id.map(data.film)
    per_film = film.value_counts()
    scene_len = (segs.end_s - segs.start_s).to_numpy()
    per_clip = segs.groupby("pair_id").size()
    mdir = manifest.manifests_dir(cfg)
    L = ["=" * W, "EXPERIMENTAL SETUP", "=" * W,
         kv("Read from", f"{mdir}/pairs.csv, segments.csv, tracks.csv, {manifest.folds_file(cfg)}"),
         kv("", f"{RUN_ROOT.parent}/semantic/{cfg.pipeline.sem_variant}/fold*/seed*/ (config, log, metrics), {ENTRY.name}")]
    L += head("DATASET")
    lang = [c for c in pairs.columns if "lang" in c.lower()]
    L += [kv("Video-music pairs", f"{len(pairs)}"),
          kv("Films", f"{film.nunique()} film groups (the unit of the split); {(pairs.meta_source + ':' + pairs.meta_film.astype(str)).nunique()} source:film names"),
          kv("Sources", "   ".join(f"{k} {v}" for k, v in pairs.meta_source.value_counts().items())),
          kv("Languages", ", ".join(lang) if lang else "not recorded in pairs.csv (Indian films; no language column)"),
          kv("Pairs per film", f"min {per_film.min()}   median {per_film.median():g}   mean {per_film.mean():.1f}   max {per_film.max()}  ({per_film.idxmax()})"),
          kv("Clip duration", stats(pairs.video_duration_s)),
          kv("Scenes", f"{len(segs)}  ({per_clip.mean():.2f} per clip, median {per_clip.median():g}, max {per_clip.max()})"),
          kv("Scene duration", stats(scene_len)),
          kv("Total video", f"{pairs.video_duration_s.sum() / 3600:.2f} h"),
          kv("Total music", f"{pairs.music_duration_s.sum() / 3600:.2f} h"),
          kv("Video as used", f"silent, {cfg.media.video_fps} fps, short side {cfg.media.video_short_side} px (data/interim)")]
    L += head(f"FOLDS  ({len(data.folds)} folds split by film; test folds do not overlap)")
    rows = []
    for k, f in data.folds.items():
        films = {s: {data.film[p] for p in f[s]} for s in ("train", "val", "test")}
        shared = len(films["test"] & (films["train"] | films["val"]))
        rows.append([k] + [len(f[s]) for s in ("train", "val", "test")] + [len(films[s]) for s in ("train", "val", "test")]
                    + [shared, f"{pairs.loc[f['test'], 'video_duration_s'].sum() / 60:.1f}"])
    L += table(["fold", "train pairs", "val pairs", "test pairs", "train films", "val films", "test films",
                "films shared test/train+val", "test video (min)"], rows, left=())
    n_test = sum(len(f["test"]) for f in data.folds.values())
    only = {g: int((film == g).sum()) for g in data.train_only}
    L += [f"Every pair is tested once ({n_test} test pairs), except train-only films: "
          + ", ".join(f"{g} ({n} pairs, in every fold's train set)" for g, n in only.items()) + "."]
    L += head("MUSIC LIBRARY")
    mus = pairs.drop_duplicates("music_id")
    L += [kv("Music files", f"{len(mus)}  (one per pair; the retrieval library)"),
          kv("Track groups", f"{len(tracks)}  (files that share audio count as one track)"),
          kv("Total duration", f"{mus.music_duration_s.sum() / 3600:.2f} h"),
          kv("File duration", stats(mus.music_duration_s)),
          kv("Audio as used", f"{cfg.media.music_sr} Hz mono FLAC (data/interim)")]
    t = training_runs(cfg)
    s, c = t["sem"], t["conf"]
    import bgm.train.semantic as sem_mod
    src = inspect.getsource(sem_mod.train_one)
    L += head(f"TRAINING SETTINGS  (semantic heads '{cfg.pipeline.sem_variant}', {len(t['runs'])} runs = {cfg.splits.n_folds} folds x {len(cfg.pipeline.sem_seeds)} seeds)")
    L += [kv("Trained", f"two projection heads (video, music), hidden {s['hidden']}, output {s['out_dim']}; the encoders stay frozen"),
          kv("Inputs", f"video {s['video_feature']}; music {s['music_feature']} (sem_clap_1hz + sem_mert_1hz)"),
          kv("Epochs", f"max {s['max_epochs']}, early stopping patience {s['patience']} (val segment R@10)"),
          kv("Epochs run", f"mean {t['epochs'].mean():.1f}   min {t['epochs'].min()}   max {t['epochs'].max()}"),
          kv("Best epoch", f"mean {t['best'].mean():.1f}   min {t['best'].min()}   max {t['best'].max()}"),
          kv("Batch size", f"{s['batch']}  (one scene per video per batch; <= {s['max_clips_per_film_per_epoch']} clips per film per epoch)"),
          kv("Learning rate", f"{s['lr_scratch']:g}" + (f", cosine schedule with {c.train.warmup_frac:.0%} warm-up" if "cosine_warmup" in src else "")),
          kv("Optimiser", f"{re.search(r'torch\.optim\.(\w+)', src)[1]}, weight decay {s['weight_decay']:g}"),
          kv("Temperature", f"learned; start {s['tau_init']:g}, floor {s['tau_min']:g}; at the best epoch mean {t['tau'].mean():.4f} (min {t['tau'].min():.4f}, max {t['tau'].max():.4f})"),
          kv("Dropout", f"{s['dropout']:g}"),
          kv("Augmentation", f"frame drop {s['frame_drop']:g}, music jitter ±{s['music_jitter_s']:g} s"),
          kv("Seeds", f"{len(cfg.pipeline.sem_seeds)} per fold: {list(cfg.pipeline.sem_seeds)}  (averaged as one ensemble at retrieval)"),
          kv("Training time", f"mean {t['seconds'].mean():.0f} s per run, {t['seconds'].sum() / 60:.0f} min for all {len(t['runs'])} runs"),
          kv("Code commit of the runs", ", ".join(t["commits"]))]
    L += head(f"RETRIEVAL AND STITCHING  ({RUN_ROOT}/fold<k>/{ENTRY_RUN.name}: fusion.json, stitch.json)")
    fu = [json.loads((RUN_ROOT / f"fold{k}" / ENTRY_RUN.name / "fusion.json").read_text()) for k in data.folds]
    st = [json.loads((RUN_ROOT / f"fold{k}" / ENTRY_RUN.name / "stitch.json").read_text()) for k in data.folds]
    L += [kv("Coarse shortlist", f"top K = {fu[0]['K']} files for the whole video and for each scene (union)"),
          kv("Fine windows", f"every {fu[0]['window_hop_s']:g} s over the shortlisted files; top J = {fu[0]['J']} per scene go to Viterbi"),
          kv("Score weights", f"coarse {fu[0]['w_coarse']}   fine {fu[0]['w_fine']}")]
    L += table(["fold", "switch penalty", "jump penalty", "key/tempo weight", "beam"],
               [[k, s_["penalties"]["lam_switch"], s_["penalties"]["lam_jump"], s_["penalties"]["mu"], s_["beam"]] for k, s_ in zip(data.folds, st)], left=())
    L += ["Penalties are tuned per fold on validation concatenations, never on the test fold."]
    hw = hardware()
    meta = data.store.video(pairs.pair_id.iloc[0])["meta"]
    L += head("HARDWARE AND SOFTWARE  (this machine)")
    L += [kv("CPU", f"{hw['cpu']}  ({hw['threads']} threads)"),
          kv("RAM", f"{hw['ram_gib']:.0f} GiB"),
          kv("GPU", hw["gpu"]),
          kv("CUDA in this session", f"available ({torch.version.cuda})" if hw["cuda"] else "NOT available: this script ran on CPU"),
          kv("OS", platform.platform()),
          kv("Python", platform.python_version()),
          kv("PyTorch", torch.__version__),
          kv("transformers", f"{md.version('transformers')}  (features were extracted with {meta['versions']['transformers']})"),
          kv("librosa", f"{md.version('librosa')}  (features were extracted with {meta['versions']['librosa']})"),
          kv("ffmpeg", subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0].split(" Copyright")[0])]
    f = c.features
    L += table(["frozen encoder", "checkpoint (Hugging Face)", "revision", "sample rate"],
               [["SigLIP (video)", f.sem_video.model, hf_snapshot(f.sem_video.model), "1 frame / s"],
                ["CLAP (music)", f.sem_music.clap_model, hf_snapshot(f.sem_music.clap_model), f"{f.sem_music.clap_sr} Hz"],
                ["MERT (music)", f.sem_music.mert_model, hf_snapshot(f.sem_music.mert_model), f"{f.sem_music.mert_sr} Hz"]], left=(0, 1, 2))
    L += ["Revision = the local Hugging Face cache's refs/main.", "=" * W]
    emit(L, OUT / "08_setup.txt")
    plot_dataset_stats(per_film, scene_len, data.train_only, cfg, OUT / "08_dataset_stats.png")


def clean(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=13)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_dataset_stats(per_film: pd.Series, scene_len: np.ndarray, train_only: list[str], cfg, out_png: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6))
    shown = per_film[~per_film.index.isin(train_only)]
    a.hist(shown, bins=np.arange(0.5, shown.max() + 1.5, 1), color=BLUE, edgecolor="white", linewidth=1.5)
    a.set_title(f"Pairs per film  ({len(per_film)} films, {per_film.sum()} pairs)", fontsize=17, fontweight="bold", color=INK, loc="left", pad=14)
    a.set_xlabel("video-music pairs in the film", fontsize=14, color=INK2)
    a.set_ylabel("films", fontsize=14, color=INK2)
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    note = f"median {per_film.median():g} pairs per film"
    if len(shown) < len(per_film):
        note += "\nnot shown: " + ", ".join(f"{g.split(':', 1)[1]} ({per_film[g]} pairs, train-only)" for g in train_only)
    a.text(0.98, 0.95, note, transform=a.transAxes, ha="right", va="top", fontsize=13, color=INK2)
    b.hist(scene_len, bins=np.arange(0, np.ceil(scene_len.max()) + 1, 1), color=BLUE, edgecolor="white", linewidth=1.5)
    med = float(np.median(scene_len))
    b.axvline(med, color=INK, linewidth=1.5, linestyle="--")
    b.text(med + 0.3, b.get_ylim()[1] * 0.97, f"median {med:.1f} s", ha="left", va="top", fontsize=13, color=INK)
    b.set_title(f"Scene duration  ({len(scene_len)} scenes)", fontsize=17, fontweight="bold", color=INK, loc="left", pad=14)
    b.set_xlabel(f"scene duration (s); scenes are capped at {cfg.segments.max_len_s:g} s", fontsize=14, color=INK2)
    b.set_ylabel("scenes", fontsize=14, color=INK2)
    for ax in (a, b):
        clean(ax)
    fig.tight_layout(w_pad=3)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- candidates
def candidates(data: Data, min_scenes: int = 4, lo_s: float = 30.0, hi_s: float = 70.0) -> None:
    """Test clips long enough to stitch, with the mood whose zero-shot SigLIP / CLAP logit is highest relative to the
    other test clips (a hint for choosing examples of different moods; not used by the pipeline here)."""
    st, moods = data.store, list(data.cfg.features.attributes.moods)
    ids = [p for p in st.pairs.pair_id if data.test_fold(p) is not None]
    z = {}
    for side, key in (("video", lambda p: st.video(p)), ("music", lambda p: st.music(st.music_of[p]))):
        x = np.stack([key(p)["attr_logits_1hz"].astype(np.float32).mean(0) for p in ids])
        z[side] = (x - x.mean(0)) / x.std(0)
    rows = [[p, data.test_fold(p), f"{st.duration('video', p):.1f}", len(st.segments[p]), moods[z["video"][i].argmax()],
             f"{z['video'][i].max():.1f}", moods[z["music"][i].argmax()]]
            for i, p in enumerate(ids) if len(st.segments[p]) >= min_scenes and lo_s <= st.duration("video", p) <= hi_s]
    rows.sort(key=lambda r: (r[4], -float(r[5])))
    print("\n".join(table(["pair_id", "test fold", "duration (s)", "scenes", "picture mood", "z", "own music mood"], rows, left=(0, 4, 6))))


# ---------------------------------------------------------------- 3. end-to-end examples
def render(data: Data, res: dict, out_dir: Path) -> dict:
    """Beat snap, soundtrack and mux, as bgm.render.run.render_video does after the plan."""
    cfg, store, pair_id = data.cfg, data.store, res["pair_id"]
    beats_of, dur_of = (lambda m: store.music(m)["beat_times_s"]), (lambda m: store.duration("music", m))
    runs, sync = res["runs"], []
    if cfg.render.beat_sync:
        runs, sync = beat_sync_runs(runs, res["cuts"], beats_of, dur_of, cfg.render.beat_sync_max_shift_s, cfg.render.beat_sync_tol_s)
    runs, loops = fill_short_runs(runs, beats_of, dur_of)
    video = interim_video_path(cfg, pair_id)
    info = ffprobe(video)["video"]
    mix, seams = render_soundtrack(runs, info["duration_s"], lambda m: interim_music_path(cfg, m), cfg.render)
    wav, mp4 = out_dir / "soundtrack.wav", out_dir / "output_with_bgm.mp4"
    sf.write(wav, np.stack([mix, mix], 1) / np.sqrt(2), SR)             # mono soundtrack on two channels (see render_video)
    mux(video, wav, mp4, 0.0)
    probe = ffprobe(mp4)
    return {"pieces": runs, "sync": sync, "loops": loops, "seams": seams, "mp4": mp4, "video_s": info["duration_s"],
            "audio_s": len(mix) / SR, "mp4_s": probe["duration_s"]}


def window(off: float, d: float) -> str:
    return f"{off:.1f}-{off + d:.1f} s"


def video_summary(data: Data, res: dict, h: dict, rd: dict, mood: str, secs: dict) -> list[str]:
    sc, pair_id = res["scenes"], res["pair_id"]
    true_id = sc[0]["true_id"]
    dur = np.array([s["end_s"] - s["start_s"] for s in sc])
    on_track = np.array([s["chosen_track"] for s in sc])
    on_win = np.array([s["chosen_window"] for s in sc])
    L = ["=" * W, f"END-TO-END EXAMPLE  {short_name(pair_id)}", "=" * W,
         kv("Video", f"{pair_id}.mp4  ({res['duration_s']:.2f} s, silent 25 fps interim)"),
         kv("Picked as", mood)] + held_out_lines(pair_id, res["fold"], h)
    L += [kv("Models", f"{RUN_ROOT}/fold{res['fold']}/{ENTRY_RUN.name}: ensemble of seeds {list(data.cfg.pipeline.sem_seeds)}"),
          kv("Music library", f"{sc[0]['n_files']} files (whole library); shortlist {sc[0]['n_short']} files; {sc[0]['n_windows']} windows scored for scene 0"),
          kv("Original music (truth)", f"{true_id}  ({data.store.duration('music', true_id):.1f} s)")]
    L += head("SCENES AND CHOSEN MUSIC  (window = seconds inside the music file, as retrieved)")
    rows = [[s["scene"], f"{s['start_s']:.2f}", f"{s['end_s']:.2f}", f"{s['end_s'] - s['start_s']:.2f}", short_name(s["chosen_id"]),
             window(s["chosen_off_s"], s["end_s"] - s["start_s"]), s["chosen_split"],
             "yes" if s["chosen_track"] else "no", "yes" if s["chosen_window"] else "no",
             f"{s['file_rank']}", s["window_rank"] if s["window_rank"] else "not shortlisted"] for s in sc]
    L += table(["scene", "start (s)", "end (s)", "length (s)", "chosen music file", "window", "file's split", "original track",
                "original segment", "true file rank", "true window rank"], rows, left=(4,))
    L += [f"File's split: where the chosen file's own video sits in fold {res['fold']} (test = unseen music; train = music seen in training).",
          f"Original segment: the original file, starting within {TOL_S:g} s of where it plays in the film.",
          f"True file rank: the original music among the {sc[0]['n_files']} library files (coarse stage). "
          "True window rank: among all fine windows; tied windows share a rank."]
    L += head("STITCHING AND GROUND TRUTH")
    files = list(dict.fromkeys(s["chosen_id"] for s in sc))
    L += [kv("Scenes", f"{len(sc)}"),
          kv("Music files used", f"{len(files)}"),
          kv("Switches (to another file)", f"{res['switches']}  ({60 * res['switches'] / res['duration_s']:.1f} per minute)"),
          kv("Jumps (inside a file)", f"{res['jumps']}"),
          kv("Original track chosen", f"{on_track.sum()} of {len(sc)} scenes  ({dur[on_track].sum():.1f} of {dur.sum():.1f} s)"),
          kv("Original segment chosen", f"{on_win.sum()} of {len(sc)} scenes  ({dur[on_win].sum():.1f} of {dur.sum():.1f} s)"),
          kv("Chosen files by split", "   ".join(f"{k} {v}" for k, v in pd.Series([data.split_of_music(res['fold'], m) for m in files]).value_counts().items()))]
    L += head("RENDERED PIECES  (after beat snap; a piece = one continuous stretch of one file)")
    L += table(["piece", "video start", "video end", "length (s)", "music file", "window"],
               [[i, f"{r.start:.2f}", f"{r.end:.2f}", f"{r.end - r.start:.2f}", short_name(r.music_id), window(r.offset, r.end - r.start)]
                for i, r in enumerate(rd["pieces"])], left=(4,))
    if rd["loops"]:
        L += [f"{len(rd['loops'])} extra piece(s): a stretch longer than what is left of its music file loops back at a beat."]
    L += head("OUTPUT")
    L += [kv("File", f"{rd['mp4'].relative_to(ROOT)}  ({rd['mp4'].stat().st_size / 2**20:.2f} MB)"),
          kv("Duration check", f"soundtrack {rd['audio_s']:.3f} s  vs  video {rd['video_s']:.3f} s  ->  "
             f"difference {1000 * (rd['audio_s'] - rd['video_s']):+.1f} ms  [{'OK' if abs(rd['audio_s'] - rd['video_s']) < 0.04 else 'CHECK'}]"),
          kv("Time", f"plan {secs['plan']:.1f} s (retrieve + Viterbi, CPU), render {secs['render']:.1f} s; "
             f"loading the fold's models and library index {secs['load']:.1f} s"), "=" * W]
    return L


def file_colours(ids: list[str]) -> dict[str, str]:
    """Fixed colour per music file in order of first appearance; files beyond the palette share one grey."""
    order = list(dict.fromkeys(ids))
    return {m: COLOURS[i] if i < len(COLOURS) else OTHER for i, m in enumerate(order)}


def plot_contact_sheet(data: Data, res: dict, out_png: Path, cols: int = 5) -> None:
    sc = res["scenes"]
    cols = min(cols, len(sc))
    rows = int(np.ceil(len(sc) / cols))
    frames = [frame_at(data.cfg, res["pair_id"], (s["start_s"] + s["end_s"]) / 2) for s in sc]
    col = file_colours([s["chosen_id"] for s in sc])
    h, w = frames[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * (3.6 * h / w + 1.45)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, fr, s in zip(axes.flat, frames, sc):
        ax.imshow(fr)
        ax.axis("on")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(col[s["chosen_id"]])
        ax.spines["bottom"].set_linewidth(8)
        ax.set_title(f"Scene {s['scene']}   {s['start_s']:.1f}-{s['end_s']:.1f} s", fontsize=13, fontweight="bold", color=INK, pad=6)
        mark = "  (original track)" if s["chosen_track"] else ""
        ax.set_xlabel(f"{short_name(s['chosen_id'])}{mark}\nwindow {window(s['chosen_off_s'], s['end_s'] - s['start_s'])}",
                      fontsize=12, color=INK, labelpad=9)
    fig.suptitle(f"{short_name(res['pair_id'])}: chosen music per scene  (test fold {res['fold']}; bar colour = music file)",
                 fontsize=16, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(w_pad=1.0, h_pad=1.4, rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_timeline(data: Data, res: dict, out_png: Path) -> None:
    sc, D = res["scenes"], res["duration_s"]
    col = file_colours([s["chosen_id"] for s in sc])
    fig, ax = plt.subplots(figsize=(16, 5.2))
    for s in sc:                                                          # row 2: scenes
        ax.barh(2, s["end_s"] - s["start_s"], left=s["start_s"], height=0.5, color=GRID, edgecolor="white", linewidth=2.5)
        if s["end_s"] - s["start_s"] >= 0.02 * D:
            ax.text((s["start_s"] + s["end_s"]) / 2, 2, f"S{s['scene']}", ha="center", va="center", fontsize=12, color=INK)
    for r in runs_from_path_like(sc):                                     # row 1: chosen music, one bar per continuous stretch
        ax.barh(1, r["end"] - r["start"], left=r["start"], height=0.5, color=col[r["id"]], edgecolor="white", linewidth=2.5)
        if r["end"] - r["start"] >= 0.07 * D:
            ax.text((r["start"] + r["end"]) / 2, 0.68, window(r["off"], r["end"] - r["start"]), ha="center", va="top", fontsize=11, color=INK2)
    true_id = sc[0]["true_id"]
    ax.barh(0, D, left=0, height=0.5, color=col.get(true_id, "white"), edgecolor=INK, linewidth=1.5,
            hatch=None if true_id in col else "///")
    ax.text(D / 2, -0.32, f"{short_name(true_id)}, 0.0-{D:.1f} s" + ("" if true_id in col else "  (never chosen)"),
            ha="center", va="top", fontsize=11, color=INK2)
    seen = {}
    for s in sc:
        seen.setdefault(s["chosen_id"], 0.0)
        seen[s["chosen_id"]] += s["end_s"] - s["start_s"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[m]) for m in seen]
    labels = [f"{short_name(m)}{' (original)' if data.store.music_track[m] == data.store.music_track[true_id] else ''}  {t:.0f} s" for m, t in seen.items()]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, len(seen)), frameon=False, fontsize=12)
    ax.set_yticks([2, 1, 0], ["Scenes", "Chosen music", "Original music"], fontsize=14, color=INK)
    ax.set_ylim(-0.75, 2.5)
    ax.set_xlim(0, D)
    ax.set_xlabel("video time (s)", fontsize=14, color=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2)
    ax.set_title(f"{short_name(res['pair_id'])}: music timeline  ({len(seen)} file{'s' * (len(seen) != 1)}, {res['switches']} switch{'es' * (res['switches'] != 1)}, {res['jumps']} jump{'s' * (res['jumps'] != 1)}; test fold {res['fold']})",
                 fontsize=16, fontweight="bold", color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def runs_from_path_like(sc: list[dict], tol: float = 0.04) -> list[dict]:
    """Continuous stretches of the per-scene table (same rule as render.runs_from_path, on plain dicts)."""
    out = []
    for s in sc:
        if out and out[-1]["id"] == s["chosen_id"] and abs(s["chosen_off_s"] - (out[-1]["off"] + out[-1]["end"] - out[-1]["start"])) <= tol:
            out[-1]["end"] = s["end_s"]
        else:
            out.append({"id": s["chosen_id"], "off": s["chosen_off_s"], "start": s["start_s"], "end": s["end_s"]})
    return out


def videos(data: Data, pair_ids: list[str], moods: list[str]) -> None:
    for pair_id, mood in zip(pair_ids, moods):
        fold = data.test_fold(pair_id)
        assert fold is not None, f"{pair_id} is in no test fold"
        h = data.held_out(pair_id, fold)
        assert h["ok"], f"{pair_id} is not held out in fold {fold}: {h}"
        out_dir = OUT / f"08_video_{slug(pair_id)}"
        out_dir.mkdir(exist_ok=True)
        t0 = time.time()
        ctx = fold_ctx(data, fold)
        t1 = time.time()
        res = analyse(data, ctx, pair_id)
        t2 = time.time()
        rd = render(data, res, out_dir)
        secs = {"load": t1 - t0, "plan": t2 - t1, "render": time.time() - t2}
        emit(video_summary(data, res, h, rd, mood, secs), out_dir / "summary.txt")
        plot_contact_sheet(data, res, out_dir / "contact_sheet.png")
        plot_timeline(data, res, out_dir / "timeline.png")
        (out_dir / "plan.json").write_text(json.dumps({"pair_id": pair_id, "fold": fold, "run": str(ctx["run"]), "held_out": h,
                                                       "scenes": res["scenes"], "pieces": [r.__dict__ for r in rd["pieces"]]}, indent=1, default=float))


# ---------------------------------------------------------------- 4. good vs failure
PARTS = OUT / "08_test_scenes"                               # one csv per fold, so folds can be scored in parallel


def score_fold(data: Data, k: int) -> pd.DataFrame:
    """One row per scene of every test video of fold k, scored by fold k's models."""
    ctx, rows, t0 = fold_ctx(data, k), [], time.time()
    for p in data.folds[str(k)]["test"]:
        assert data.held_out(p, k)["ok"], p
        res = analyse(data, ctx, p)
        for s in res["scenes"]:
            rows.append({"pair_id": p, "fold": k, "n_scenes": len(res["scenes"]), "switches": res["switches"], "jumps": res["jumps"],
                         **{a: b for a, b in s.items() if a != "top_files"}, "top_files": json.dumps(s["top_files"])})
    log.warning("fold %d: %d test videos in %.0f s", k, len(data.folds[str(k)]["test"]), time.time() - t0)
    return pd.DataFrame(rows)


def test_scenes(data: Data, recompute: bool, only: int | None = None) -> pd.DataFrame:
    """Every test video of every fold (or of fold `only`); a fold's csv is reused unless `recompute`."""
    PARTS.mkdir(exist_ok=True)
    out = []
    for k in ([only] if only is not None else sorted(int(k) for k in data.folds)):
        path = PARTS / f"fold{k}.csv"
        if recompute or not path.exists():
            score_fold(data, k).to_csv(path, index=False)
        out.append(pd.read_csv(path))
    return pd.concat(out, ignore_index=True)


def pick_cases(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Good: a multi-scene video stitched as one continuous piece of the original music at the original position
    (every scene on its true window, 0 switches, 0 jumps) with a true window ranked 1st in at least one scene; the one
    with most scenes. The scene shown is the one whose true window ranks 1st.
    Failure: the scene, in a multi-scene video, whose original music ranks lowest in the library."""
    multi = df[df.n_scenes >= 2]
    g = multi.groupby("pair_id").agg(n=("scene", "size"), all_first=("window_rank", lambda x: bool((x == 1).all())),
                                     any_first=("window_rank", lambda x: bool((x == 1).any())),
                                     all_window=("chosen_window", "all"), switches=("switches", "first"), jumps=("jumps", "first"),
                                     lead=("fine_margin", "min"))
    ok = g[g.any_first & g.all_window & (g.switches == 0) & (g.jumps == 0)].sort_values(["n", "lead"], ascending=False)
    assert len(ok), "no multi-scene test video is stitched as one continuous piece of its original music"
    good = multi[(multi.pair_id == ok.index[0]) & (multi.window_rank == 1)].sort_values("fine_margin", ascending=False).iloc[0]
    bad = multi.sort_values(["file_rank", "pair_id"], ascending=[False, True]).iloc[0]
    counts = {"videos": df.pair_id.nunique(), "scenes": len(df), "multi_videos": len(g), "good_videos": len(ok), "all_first_videos": int(g.all_first.sum()),
              "scene_window": int(df.chosen_window.sum()), "scene_track": int(df.chosen_track.sum()),
              "scene_rank1": int((df.window_rank == 1).sum()), "scene_top10_file": int((df.file_rank <= 10).sum()),
              "median_file_rank": float(df.file_rank.median()), "n_files": int(df.n_files.iloc[0])}
    return good, bad, counts


def reason(row: pd.Series, min_span: float, video: pd.DataFrame) -> str:
    """One line read from the scores of the scene."""
    d = row.end_s - row.start_s
    top = json.loads(row.top_files)
    if row.window_rank == 1 and row.chosen_window:
        others = video[video.scene != row.scene]
        return (f"in this scene the true window is ranked 1st, {row.fine_margin:.2f} standardized units ahead of the best other file; "
                f"in the other {len(others)} scenes it ranks {', '.join(str(int(r)) for r in others.window_rank)}, "
                f"and Viterbi keeps the same music playing on, so all {row.n_scenes} scenes land on the original segment")
    parts = [f"the original music scores {row.true_z:+.2f} standardized units for this scene (library mean 0, top file {top[0][1]:+.2f}), "
             f"so it ranks {row.file_rank} of {row.n_files} and never reaches the top-{row.n_short} shortlist" if not row.shortlisted else
             f"the original music ranks {row.file_rank} of {row.n_files} for this scene (score {row.true_z:+.2f} vs top {top[0][1]:+.2f})"]
    if d < min_span:
        parts.append(f"the scene is only {d:.1f} s, under the {min_span:g} s minimum span")
    if row.coarse_margin < NEAR_TIE_Z:
        parts.append(f"near-tie at the top: the two best files differ by {row.coarse_margin:.2f}")
    best_other = int(video.file_rank.min())
    parts.append(f"best rank of the original over the video's {row.n_scenes} scenes: {best_other}")
    return "; ".join(parts)


def case_lines(data: Data, name: str, row: pd.Series, video: pd.DataFrame, why: str) -> list[str]:
    h = data.held_out(row.pair_id, int(row.fold))
    assert h["ok"]
    d = row.end_s - row.start_s
    top = json.loads(row.top_files)
    L = head(f"{name}:  {short_name(row.pair_id)}, scene {row.scene}  ({row.start_s:.2f}-{row.end_s:.2f} s, {d:.2f} s; video of {row.n_scenes} scenes)")
    L += held_out_lines(row.pair_id, int(row.fold), h)
    L += [kv("Ground truth", f"{short_name(row.true_id)}, window {window(row.true_off_s, d)}"),
          kv("Prediction (after Viterbi)", f"{short_name(row.chosen_id)}, window {window(row.chosen_off_s, d)}  (file's split in this fold: {row.chosen_split})"),
          kv("Original track / segment", f"{'yes' if row.chosen_track else 'no'} / {'yes' if row.chosen_window else 'no'}"),
          kv("True file rank (coarse)", f"{row.file_rank} of {row.n_files}  (standardized score {row.true_z:+.2f})"),
          kv("True window rank (fine)", f"{int(row.window_rank)} of {row.n_windows}" if pd.notna(row.window_rank) else f"not scored: the file is outside the {row.n_short}-file shortlist"),
          kv("Stitch of the whole video", f"{row.switches} switches, {row.jumps} jumps over {row.n_scenes} scenes; "
             f"original track on {int(video.chosen_track.sum())} of {row.n_scenes} scenes"),
          *[kv("Why" if i == 0 else "", t) for i, t in enumerate(textwrap.wrap(why, W - 30))], ""]
    L += table(["coarse rank", "music file", "score (standardized)", ""],
               [[i + 1, short_name(m), f"{z:+.2f}", "<- original" if data.store.music_track[m] == data.store.music_track[row.true_id] else ""]
                for i, (m, z) in enumerate(top)], left=(1, 3))
    return L


def plot_cases(data: Data, cases: list[tuple[str, pd.Series, pd.DataFrame, str]], out_png: Path) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 5.6 * len(cases)), gridspec_kw={"width_ratios": [1.7, 1.55, 1.3]}, squeeze=False)
    for (name, row, video, why), (ai, at, ab) in zip(cases, axes):
        d = row.end_s - row.start_s
        ai.imshow(frame_at(data.cfg, row.pair_id, (row.start_s + row.end_s) / 2))
        ai.set_xticks([]), ai.set_yticks([])
        for sp in ai.spines.values():
            sp.set_visible(False)
        ai.set_title(name, fontsize=20, fontweight="bold", color=INK, loc="left", pad=12)
        ai.set_xlabel(f"{short_name(row.pair_id)}, scene {row.scene} of {row.n_scenes}\n{row.start_s:.1f}-{row.end_s:.1f} s   ·   test fold {row.fold}",
                      fontsize=13, color=INK2, labelpad=10)
        at.axis("off")
        wr = f"{int(row.window_rank)} of {row.n_windows}" if pd.notna(row.window_rank) else "not shortlisted"
        items = [("Ground truth", f"{short_name(row.true_id)}\nwindow {window(row.true_off_s, d)}"),
                 ("Prediction", f"{short_name(row.chosen_id)}\nwindow {window(row.chosen_off_s, d)}"),
                 ("Rank of the true music", f"file {row.file_rank} of {row.n_files}   ·   window {wr}"),
                 ("Stitch (whole video)", f"{row.switches} switches, {row.jumps} jumps over {row.n_scenes} scenes"),
                 ("Why", "\n".join(textwrap.wrap(why.split(";")[0], 50)))]
        y = 0.98
        for label, value in items:
            at.text(0.0, y, label.upper(), fontsize=11, color=MUTED, fontweight="bold", va="top", transform=at.transAxes)
            at.text(0.0, y - 0.05, value, fontsize=13.5, color=INK, va="top", transform=at.transAxes, linespacing=1.3)
            y -= 0.09 + 0.058 * (1 + value.count("\n"))
        top = json.loads(row.top_files)
        bars = [(f"{i + 1}. {short_name(m)}", z, data.store.music_track[m] == data.store.music_track[row.true_id]) for i, (m, z) in enumerate(top)]
        if not any(b[2] for b in bars):
            bars.append((f"{row.file_rank}. {short_name(row.true_id)}", row.true_z, True))
        ypos = np.arange(len(bars))[::-1]
        ab.barh(ypos, [b[1] for b in bars], height=0.6, color=[BLUE if b[2] else OTHER for b in bars])
        for yv, b in zip(ypos, bars):
            ab.text(max(b[1], 0) + 0.06, yv, f"{b[1]:+.2f}", va="center", ha="left", fontsize=12, color=INK2)
        ab.set_yticks(ypos, [b[0] for b in bars], fontsize=12, color=INK)
        ab.axvline(0, color=MUTED, linewidth=1)
        ab.set_xlim(min(0, min(b[1] for b in bars)) - 0.3, max(b[1] for b in bars) + 0.9)
        ab.set_xlabel("score, standardized over the library", fontsize=12, color=INK2)
        ab.set_title("Library files ranked for this scene (blue = original music)", fontsize=13, fontweight="bold", color=INK, loc="right", pad=10)
        for sp in ("top", "right", "left"):
            ab.spines[sp].set_visible(False)
        ab.spines["bottom"].set_color(MUTED)
        ab.tick_params(axis="y", length=0)
        ab.tick_params(axis="x", colors=INK2)
    fig.tight_layout(w_pad=2.5, h_pad=4)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def cases(data: Data, recompute: bool, only: int | None = None) -> None:
    df = test_scenes(data, recompute, only)
    if only is not None:                                      # scoring one fold only (run the folds in parallel, then `cases`)
        return
    good, bad, n = pick_cases(df)
    L = ["=" * W, "GOOD CASE VS FAILURE CASE  (test folds, 10-seed ensemble + Viterbi, whole-library retrieval)", "=" * W,
         kv("Test videos / scenes", f"{n['videos']} / {n['scenes']}  (each video scored by the models of the fold that tests it; {PARTS.name}/fold<k>.csv)"),
         kv("Music library", f"{n['n_files']} files"),
         kv("True window ranked 1st", f"{n['scene_rank1']} of {n['scenes']} scenes"),
         kv("True file in the top 10", f"{n['scene_top10_file']} of {n['scenes']} scenes  (median rank {n['median_file_rank']:g})"),
         kv("Chosen after Viterbi", f"original track on {n['scene_track']} scenes, original segment on {n['scene_window']} scenes"),
         kv("Good-case rule", f"video with >= 2 scenes stitched as one continuous piece of the original music at the original position,"),
         kv("", f"with a true window ranked 1st: {n['good_videos']} of {n['multi_videos']} multi-scene videos qualify; most scenes first"),
         kv("", f"(videos with the true window ranked 1st in EVERY scene: {n['all_first_videos']})"),
         kv("Failure-case rule", "the scene, in a multi-scene video, whose original music ranks lowest in the library"),
         "These two cases are chosen as the extremes; they are not typical. The counts above show how common each is."]
    out = []
    for name, row in (("GOOD CASE", good), ("FAILURE CASE", bad)):
        video = df[df.pair_id == row.pair_id]
        why = reason(row, data.store.min_span, video)
        L += case_lines(data, name, row, video, why)
        out.append((name, row, video, why))
    L += ["=" * W]
    emit(L, OUT / "08_good_vs_failure.txt")
    plot_cases(data, out, OUT / "08_good_vs_failure.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["setup", "candidates", "videos", "cases"])
    ap.add_argument("pair_ids", nargs="*")
    ap.add_argument("--moods", nargs="*", default=None, help="one label per video, for the summary")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="cases: only score this fold's test videos (no report)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    data = Data()
    if a.part == "setup":
        setup(data)
    elif a.part == "candidates":
        candidates(data)
    elif a.part == "videos":
        assert a.pair_ids, "give at least one pair_id"
        videos(data, a.pair_ids, a.moods or ["(not labelled)"] * len(a.pair_ids))
    else:
        cases(data, a.recompute, a.fold)


if __name__ == "__main__":
    main()
