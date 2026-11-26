"""MEMA's evaluation protocols applied to the final system on the cleaned dataset (D118).

MEMA (Zhang et al., "MEMA: Multimodal Aesthetic Evaluation of Music in Visual Contexts", Sensors 2026, 26, 1395)
scores soundtracks on three aesthetic dimensions (NEC, TI, TIO) and reports LCC / SRCC / KTAU / MSE / MAE against
LLM-derived and human ratings. Neither its model nor its ratings are public, and we have no aesthetic ratings, so
those numbers cannot be computed here. Its protocols that apply to any video-music scorer are reproduced:

  pairwise matching accuracy (MEMA §5.6)  P(score(video, own music) > score(video, mismatched music)), ties 1/2.
      Track level: own music file vs every test-fold file of another track group. Segment level: own aligned window
      vs every gallery window from another music file (the window protocol's gallery, SPEC §5.12). MEMA draws one
      random mismatch per query; we take the exact expectation over all mismatches (same quantity, no sampling noise).
      Chance 0.5.
  retrieval R@1/5/10 and median rank (MEMA §5.7, Table 4)  the existing track and segment protocols.
  robustness (MEMA §5.4)  the same trained heads on features re-extracted from degraded inputs. MEMA scales its 480p
      video to 360p (mild) and 240p (severe), i.e. x0.75 and x0.5; 98% of our interim videos are 360p, so mild is
      270p and severe 180p (libx264 CRF 23, frame count kept). Audio: MP3 (libmp3lame) at 128 / 64 / 32 kbps,
      decoded back to 48 kHz mono FLAC at the original length (zero lag checked in the tests). Combined levels:
      mild = 270p + 64 kbps, severe = 180p + 32 kbps.
  inference time (MEMA §5.8)  ms per track query for the ensemble on cached features, and encoder time per clip.

Degraded media and features go to cleaned_dataset/robust/<side condition>/{interim,features}; a condition's root
links each side to the clean or degraded folder. Runs: runs/mema/<condition>/<system>/fold<k>/seed<e>/.

Run: python -m bgm.analysis.mema_protocols [build|score|report|all]  -> results/cleaned/mema_protocols.md
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import DictConfig

log = logging.getLogger(__name__)

ROOT = Path("cleaned_dataset/robust")
CLEAN = {"interim": Path("cleaned_dataset/interim"), "features": Path("cleaned_dataset/features")}
RUNS = Path("runs/mema")
OUT = Path("results/cleaned/mema_protocols.md")
HEADS = Path("runs/cleaned/semantic/siglip-both")
SAVED_ENS = Path("runs/cleaned/semantic_ens/siglip-both-ens10")
ENS_SIZE, ENS_RUNS = 10, (0, 1, 2)
VIDEO_LEVELS = {"v270": 270, "v180": 180}
AUDIO_LEVELS = {"a128": 128, "a64": 64, "a32": 32}
CONDITIONS = {"clean": (None, None), "v270": ("v270", None), "v180": ("v180", None), "a128": (None, "a128"),
              "a64": (None, "a64"), "a32": (None, "a32"), "mild": ("v270", "a64"), "severe": ("v180", "a32")}
LABELS = {"clean": "clean (360p, lossless)", "v270": "video 270p", "v180": "video 180p", "a128": "audio MP3 128 kbps",
          "a64": "audio MP3 64 kbps", "a32": "audio MP3 32 kbps", "mild": "mild: 270p + 64 kbps",
          "severe": "severe: 180p + 32 kbps"}


# ---------------------------------------------------------------- metrics (pure)
def pairwise_accuracy(scores: np.ndarray, true_idx: int, mismatch: np.ndarray) -> float:
    """Share of mismatched candidates scored below the true one (ties count 1/2). scores [N], mismatch [N] bool."""
    s = np.asarray(scores, float)
    m = s[np.asarray(mismatch, bool)]
    assert len(m), "no mismatched candidate"
    t = s[true_idx]
    return float(((t > m).sum() + 0.5 * (t == m).sum()) / len(m))


def relative_drop(clean: float, degraded: float) -> float:
    """MEMA's degradation rate: (clean - degraded) / clean."""
    return (clean - degraded) / clean if clean else float("nan")


# ---------------------------------------------------------------- degraded media (I/O)
def _probe(path: Path) -> tuple[int, int, int]:
    """(width, height, frame count) of a video's first stream."""
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True).stdout.strip()
    w, h, n = (int(x) for x in o.split(",")[:3])
    return w, h, n


def degrade_video(src: Path, dst: Path, short_side: int) -> None:
    """Silent 25 fps video with its short side scaled down to short_side (never up); same frame count."""
    w, h, n = _probe(src)
    target = min(short_side, w, h)
    vf = f"scale=-2:{target}" if w >= h else f"scale={target}:-2"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(dst)],
                   check=True)
    assert _probe(dst)[2] == n, f"{dst}: frame count changed"


def degrade_audio(src: Path, dst: Path, kbps: int, tmp: Path) -> None:
    """48 kHz mono FLAC -> MP3 at kbps -> 48 kHz mono FLAC, trimmed / zero-padded to the source length."""
    y, sr = sf.read(src, dtype="float32")
    assert sr == 48000, f"{src}: interim music must be 48 kHz, got {sr}"
    tmp.mkdir(parents=True, exist_ok=True)
    mp3, wav = tmp / f"{dst.stem}.mp3", tmp / f"{dst.stem}.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-c:a", "libmp3lame", "-b:a", f"{kbps}k",
                    str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(mp3), "-ar", "48000", "-ac", "1", str(wav)],
                   check=True)
    z, sr2 = sf.read(wav, dtype="float32")
    assert sr2 == 48000
    z = z[:len(y)] if len(z) >= len(y) else np.concatenate([z, np.zeros(len(y) - len(z), np.float32)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, z, 48000, subtype="PCM_24", format="FLAC")
    mp3.unlink()
    wav.unlink()


def _link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)              # a link to a missing folder breaks the extractor's mkdir
    if not link.is_symlink():
        link.symlink_to(target.resolve(), target_is_directory=True)


def side_dir(level: str | None, side: str, kind: str) -> Path:
    """Folder holding one side (video / music) of the interim media or features at a degradation level."""
    return CLEAN[kind] / side if level is None else ROOT / level / kind / side


def condition_root(cond: str) -> Path:
    """cleaned_dataset/robust/_cond/<cond>/{interim,features}/{video,music}: links to the side folders."""
    v, a = CONDITIONS[cond]
    root = ROOT / "_cond" / cond
    for kind in ("interim", "features"):
        _link(root / kind / "video", side_dir(v, "video", kind))
        _link(root / kind / "music", side_dir(a, "music", kind))
    return root


def make_cfg(cond: str) -> DictConfig:
    from bgm.config import load_config
    over = ["splits.folds_file=folds_title.json", "paths.manifests=cleaned_dataset/manifests", f"paths.runs={RUNS}"]
    if cond == "clean":
        over += [f"paths.interim={CLEAN['interim']}", f"paths.features={CLEAN['features']}"]
    else:
        r = condition_root(cond)
        over += [f"paths.interim={r / 'interim'}", f"paths.features={r / 'features'}"]
    return load_config("configs/default.yaml", over)


def build(workers: int = 8) -> None:
    """Degrade every interim video / music file at each single-side level, then extract its semantic + ImageBind
    features (the other side links to the clean folders, whose features are current, so nothing is rewritten)."""
    from bgm.data import manifest
    from bgm.features.extract import Extractor
    cfg = make_cfg("clean")
    pairs = manifest.read_pairs(cfg)
    vids, mus = list(pairs.pair_id), list(dict.fromkeys(pairs.music_id))
    for lvl, px in VIDEO_LEVELS.items():
        out = side_dir(lvl, "video", "interim")
        todo = [p for p in vids if not (out / f"{p}.mp4").exists()]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda p: degrade_video(CLEAN["interim"] / "video" / f"{p}.mp4", out / f"{p}.mp4", px), todo))
        log.info("%s: %d videos degraded", lvl, len(todo))
    for lvl, kb in AUDIO_LEVELS.items():
        out = side_dir(lvl, "music", "interim")
        todo = [m for m in mus if not (out / f"{m}.flac").exists()]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda m: degrade_audio(CLEAN["interim"] / "music" / f"{m}.flac", out / f"{m}.flac", kb,
                                                ROOT / lvl / "tmp"), todo))
        log.info("%s: %d music files degraded", lvl, len(todo))
    for lvl in (*VIDEO_LEVELS, *AUDIO_LEVELS):
        groups = ["sem_video", "imagebind"] if lvl in VIDEO_LEVELS else ["sem_music", "imagebind"]
        Extractor(make_cfg(lvl)).run(groups)


# ---------------------------------------------------------------- scoring
class ImageBindScorer:
    """ImageBind zero-shot cosine (as ZeroShot.imagebind_*), without the attribute calibration ZeroShot also fits."""

    def __init__(self, store):
        self.store = store

    def track(self, q, gallery):
        from bgm.eval.store import l2
        v = l2(self.store.whole("video", q, "imagebind_1hz"))
        return np.array([l2(self.store.whole("music", m, "imagebind_1hz")) @ v for m in gallery])

    def window(self, q, a, b, windows):
        from bgm.eval.store import l2
        v = l2(self.store.pooled("video", q, "imagebind_1hz", a, b))
        return np.array([l2(self.store.pooled("music", m, "imagebind_1hz", s, s + d)) @ v for m, s, d in windows])


def score_queries(store, test: list[str], scorer) -> dict:
    """Per query: track rank / pairwise accuracy (test-fold gallery) and segment rank / pairwise accuracy (window
    gallery, variant 'all'), with the film-group cluster added by the caller."""
    from bgm.eval.metrics import rank_of_first_hit
    from bgm.eval.protocol import segment_gallery, test_segments
    gal = list(dict.fromkeys(store.music_of[p] for p in test))
    gtrack = np.array([store.music_track[m] for m in gal])
    tr, tp = [], []
    for q in test:
        s = scorer.track(q, gal)
        tr.append(rank_of_first_hit(s, gtrack == store.track[q]))
        tp.append(pairwise_accuracy(s, gal.index(store.music_of[q]), gtrack != store.track[q]))
    segs = test_segments(store, test)
    sr, sp, sq = [], [], []
    for q in segs:
        windows, hit = segment_gallery(store, segs, q, "all")
        s = np.asarray(scorer.window(q[0], q[2], q[3], windows))
        hits = np.zeros(len(windows), bool)
        hits[hit] = True
        sr.append(rank_of_first_hit(s, hits))
        sp.append(pairwise_accuracy(s, hit, np.array([m != store.music_of[q[0]] for m, _, _ in windows])))
        sq.append(q[0])
    return {"track_rank": np.array(tr), "track_pair": np.array(tp), "track_q": np.array(test),
            "seg_rank": np.array(sr), "seg_pair": np.array(sp), "seg_q": np.array(sq)}


def _summary(per: dict) -> dict:
    from bgm.eval.metrics import recall_summary
    return {"track": recall_summary(per["track_rank"]), "track_pairwise": float(per["track_pair"].mean()),
            "segment": recall_summary(per["seg_rank"]), "segment_pairwise": float(per["seg_pair"].mean())}


def score(conds=tuple(CONDITIONS), device: str = "cuda") -> None:
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.person.evaluate import MeanScorer
    from bgm.runs import write_run
    for cond in conds:
        cfg = make_cfg(cond)
        store = FeatureStore(cfg)
        for k in range(cfg.splits.n_folds):
            test = fold_ids(cfg, k)["test"]
            jobs = [("imagebind", 0, lambda: ImageBindScorer(store))]
            jobs += [("ens10", e, lambda e=e: MeanScorer([load_scorer(HEADS / f"fold{k}" / f"seed{m}", store, device)
                                                         for m in range(ENS_SIZE * e, ENS_SIZE * (e + 1))]))
                     for e in ENS_RUNS]
            for system, e, make in jobs:
                d = RUNS / cond / system / f"fold{k}" / f"seed{e}"
                if (d / "perquery.npz").exists():
                    continue
                with torch.no_grad():
                    per = score_queries(store, test, make())
                if cond == "clean" and system == "ens10":       # the scoring path must reproduce the saved run
                    with np.load(SAVED_ENS / f"fold{k}" / f"seed{e}" / "perquery.npz") as z:
                        assert np.array_equal(z["track_test_fold"], per["track_rank"]), (k, e, "track")
                        assert np.array_equal(z["segment_all"], per["seg_rank"]), (k, e, "segment")
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "perquery.npz", **per)
                write_run(d, cfg, _summary(per), seed=e)
                torch.cuda.empty_cache()
            log.info("%s: fold %d scored", cond, k)


def time_queries(device: str = "cuda", fold: int = 0) -> dict:
    """ms per track query for the 10-head ensemble on cached features: cold (embeds the gallery) and warm."""
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.person.evaluate import MeanScorer
    cfg = make_cfg("clean")
    store = FeatureStore(cfg)
    test = fold_ids(cfg, fold)["test"]
    gal = list(dict.fromkeys(store.music_of[p] for p in test))
    for q in test:                                           # load features into memory first
        store.video(q)
    for m in gal:
        store.music(m)
    sc = MeanScorer([load_scorer(HEADS / f"fold{fold}" / f"seed{m}", store, device) for m in range(ENS_SIZE)])
    out = {"queries": len(test), "gallery": len(gal)}
    with torch.no_grad():
        for name in ("cold_ms", "warm_ms"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for q in test:
                sc.track(q, gal)
            torch.cuda.synchronize()
            out[name] = 1000 * (time.perf_counter() - t0) / len(test)
    return out


# ---------------------------------------------------------------- report
def _load(cond: str, system: str) -> list[dict]:
    runs = sorted((RUNS / cond / system).glob("fold*/seed*/perquery.npz"))
    return [dict(np.load(p)) for p in runs]


def _pooled(runs: list[dict], key: str, qkey: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-query values averaged over the runs that share a fold (ensemble seeds), in a fixed query order."""
    by_fold: dict[tuple, list] = {}
    for r in runs:
        by_fold.setdefault(tuple(r[qkey]), []).append(r[key].astype(float))
    vals = np.concatenate([np.mean(v, 0) for v in by_fold.values()])
    qs = np.concatenate([np.array(q) for q in by_fold])
    return vals, qs


def _stat(runs, fn) -> str:
    v = [fn(r) for r in runs]
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def _ci(vals, clusters) -> str:
    from bgm.eval.metrics import cluster_bootstrap
    b = cluster_bootstrap(vals, clusters)
    return f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"


def report(timing: dict | None = None) -> Path:
    from bgm.data import manifest
    cfg = make_cfg("clean")
    film = manifest.read_folds(cfg)["film_group"]
    sysname = {"ens10": "final: 10-head ensemble", "imagebind": "ImageBind zero-shot"}
    R = {(c, s): _load(c, s) for c in CONDITIONS for s in sysname}
    clean = {s: R[("clean", s)] for s in sysname}
    n_t, n_s = len(clean["ens10"][0]["track_rank"]), len(clean["ens10"][0]["seg_rank"])
    nq_t = sum(len(r["track_rank"]) for r in clean["imagebind"])
    nq_s = sum(len(r["seg_rank"]) for r in clean["imagebind"])

    out = ["# MEMA's evaluation protocols on the final system (cleaned dataset, D118)", "",
           "Generated by `python -m bgm.analysis.mema_protocols` (src/bgm/analysis/mema_protocols.py). Protocols from "
           "MEMA (Zhang et al., *Sensors* 2026, 26, 1395): pairwise matching accuracy (§5.6), retrieval R@K and median "
           "rank (§5.7), robustness to degraded inputs (§5.4) and inference time (§5.8). Final system: the 10-head "
           "SigLIP → CLAP+MERT ensemble trained on cleaned_dataset/ (D116); test: the cleaned title-split test folds, "
           f"{nq_t} clips and {nq_s} segments. Ensemble rows: 15 runs (5 folds × 3 disjoint 10-head ensembles); "
           "ImageBind: 5 runs. Mean ± std over runs; Δ: per query averaged over ensembles, 95% cluster bootstrap over "
           "film groups (10,000 resamples). The clean ensemble runs reproduce the saved D116 ranks exactly (asserted).", "",
           "**Not computable here:** MEMA's NEC / TI / TIO aesthetic scores need its trained model (not public), and "
           "its LCC / SRCC / KTAU / MSE / MAE and human-correlation numbers need aesthetic ratings, which this dataset "
           "does not have.", ""]

    out += ["## 1. Pairwise matching accuracy (MEMA §5.6)", "",
            "Does the system score a video's own music above a mismatched one? Track: own music file vs every test-fold "
            "file of another track group. Segment: own aligned window vs every gallery window cut from another music "
            "file. Exact expectation over all mismatches (MEMA samples one); ties count ½. Chance 0.5.", "",
            "| system | runs | track pairwise acc. | segment pairwise acc. | track − chance [95% CI] | segment − chance [95% CI] |",
            "| --- | --- | --- | --- | --- | --- |", "| random | — | 0.500 | 0.500 | — | — |"]
    for s, name in sysname.items():
        runs = clean[s]
        tv, tq = _pooled(runs, "track_pair", "track_q")
        sv, sq = _pooled(runs, "seg_pair", "seg_q")
        out.append(f"| {name} | {len(runs)} | {_stat(runs, lambda r: r['track_pair'].mean())} | "
                   f"{_stat(runs, lambda r: r['seg_pair'].mean())} | {_ci(tv - 0.5, [film[q] for q in tq])} | "
                   f"{_ci(sv - 0.5, [film[q] for q in sq])} |")
    tv_e, tq = _pooled(clean["ens10"], "track_pair", "track_q")
    tv_i, tq_i = _pooled(clean["imagebind"], "track_pair", "track_q")
    sv_e, sq = _pooled(clean["ens10"], "seg_pair", "seg_q")
    sv_i, sq_i = _pooled(clean["imagebind"], "seg_pair", "seg_q")
    assert list(tq) == list(tq_i) and list(sq) == list(sq_i)
    out += ["", f"- Ensemble − ImageBind: track {_ci(tv_e - tv_i, [film[q] for q in tq])}; segment "
            f"{_ci(sv_e - sv_i, [film[q] for q in sq])}.", ""]

    out += ["## 2. Retrieval in MEMA's Table 4 format (§5.7)", "",
            "Track: query = a test video, gallery = its test fold's music files, a hit is any file of the right track "
            "group. Segment: query = a test segment, target = its own aligned music window among every test segment's "
            "window. MR = median rank. Chance is exact per query.", "",
            "| system | track R@1 | track R@5 | track R@10 | track MR | segment R@1 | segment R@5 | segment R@10 | segment MR |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out.append(_chance_row(cfg))
    for s, name in sysname.items():
        runs = clean[s]
        cell = lambda key, f: _stat(runs, lambda r: f(r[key]))   # noqa: E731
        out.append(f"| {name} | " + " | ".join(
            [cell("track_rank", lambda x, k=k: (x <= k).mean()) for k in (1, 5, 10)] + [cell("track_rank", np.median)]
            + [cell("seg_rank", lambda x, k=k: (x <= k).mean()) for k in (1, 5, 10)] + [cell("seg_rank", np.median)]) + " |")
    out += ["", "Not comparable with MEMA's SymMV numbers (R@10 0.536): different data, gallery sizes and hit rules.", ""]

    out += ["## 3. Robustness to degraded inputs (MEMA §5.4)", "",
            "The same trained heads (no retraining) on features re-extracted from degraded test media. MEMA scales its "
            "480p video to 360p (mild) and 240p (severe), ×0.75 and ×0.5; our interim video is 360p (991 of 1,012 "
            "files), so the same factors give 270p and 180p (libx264 CRF 23). Audio: MP3 at 128 / 64 / 32 kbps, decoded "
            "back to 48 kHz. Drop = (clean − degraded) / clean, MEMA's degradation rate (its Figure 4 uses average LCC; "
            "here track R@10). Δ R@10: degraded − clean per query, film-group CI.", ""]
    for s, name in sysname.items():
        out += [f"**{name}**", "",
                "| condition | track R@10 | drop | Δ track R@10 [95% CI] | segment R@10 | drop | track pairwise | segment pairwise |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        c_runs = R[("clean", s)]
        c_t = np.mean([(r["track_rank"] <= 10).mean() for r in c_runs])
        c_s = np.mean([(r["seg_rank"] <= 10).mean() for r in c_runs])
        c_hit, cq = _hit_avg(c_runs, "track_rank")
        for cond in CONDITIONS:
            runs = R[(cond, s)]
            if not runs:
                continue
            t = np.mean([(r["track_rank"] <= 10).mean() for r in runs])
            sg = np.mean([(r["seg_rank"] <= 10).mean() for r in runs])
            d_hit, dq = _hit_avg(runs, "track_rank")
            assert list(dq) == list(cq)
            delta = "—" if cond == "clean" else _ci(d_hit - c_hit, [film[q] for q in cq])
            out.append(f"| {LABELS[cond]} | {_stat(runs, lambda r: (r['track_rank'] <= 10).mean())} | "
                       f"{'—' if cond == 'clean' else f'{100 * relative_drop(c_t, t):+.1f}%'} | {delta} | "
                       f"{_stat(runs, lambda r: (r['seg_rank'] <= 10).mean())} | "
                       f"{'—' if cond == 'clean' else f'{100 * relative_drop(c_s, sg):+.1f}%'} | "
                       f"{_stat(runs, lambda r: r['track_pair'].mean())} | {_stat(runs, lambda r: r['seg_pair'].mean())} |")
        out.append("")

    if timing:
        out += ["## 4. Inference time (MEMA §5.8)", "",
                f"RTX 4080 SUPER. One track query = the 10-head ensemble scoring a test video against its fold's "
                f"{timing['gallery']} music files ({timing['queries']} queries, features in memory). Cold: the gallery's "
                "music embeddings are computed on first use; warm: they are cached, as in a deployed index. Encoder times "
                "per clip are from the cleaned extraction log (reports/cleaned/extract.log); MEMA's 2.49 ms excludes its "
                "frozen encoders too.", "",
                "| step | time |", "| --- | --- |",
                f"| ensemble track query, cold | {timing['cold_ms']:.2f} ms |",
                f"| ensemble track query, warm | {timing['warm_ms']:.2f} ms |",
                "| SigLIP (+ CLIP) video features, per clip | 0.28 s |",
                "| CLAP + MERT music features, per file | 0.22 s |", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    log.info("wrote %s", OUT)
    return OUT


def _hit_avg(runs: list[dict], key: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Per-query hit@k averaged over the runs of a fold, concatenated over folds, and the query order (as _pooled)."""
    return _pooled([{**r, key: (r[key] <= k).astype(float)} for r in runs], key,
                   "track_q" if key.startswith("track") else "seg_q")


def _chance_row(cfg) -> str:
    from bgm.eval.metrics import chance_recall
    from bgm.eval.report_filmsplit import Split
    sp = Split(cfg, "folds_title.json", "film")
    cells = []
    for sec in ("track_test_fold", "segment_all"):
        keys = [key for key in sp.N if key[1] == sec]
        N = np.concatenate([sp.N[key] for key in keys])
        h = np.concatenate([sp.h[key] for key in keys])
        cells += [f"{np.mean([chance_recall(n, hh, k) for n, hh in zip(N, h)]):.3f}" for k in (1, 5, 10)] + ["—"]
    return "| random (exact chance) | " + " | ".join(cells) + " |"


def main(what: str = "all") -> None:
    if what in ("build", "all"):
        build()
    if what in ("score", "all"):
        score()
    if what in ("report", "all"):
        t = Path("reports/cleaned/mema_timing.json")
        if what == "all" or not t.exists():
            t.write_text(json.dumps(time_queries()))
        report(json.loads(t.read_text()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
