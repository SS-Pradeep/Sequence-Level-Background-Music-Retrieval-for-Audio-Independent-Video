#!/usr/bin/env python3
"""
Stream a download_yt_dataset.py folder into cached features and delete the media as it goes (user decision
2026-11-15: the full 200k list does not fit on disk as media; keep only features).

Per batch of finished downloads (<src>/manifest.jsonl status ok, <src>/video/<id>.mp4 + <src>/audio/<id>.*):
  1. Demucs htdemucs two-stem -> BGM stem (tools/separate_bgm.py's call), stem RMS checked;
  2. Chromaprint of the original mix -> <out>/fingerprints/<id>.npy (uint32; later track_id grouping);
  3. the standard pipeline on a per-batch manifest: `bgm prep` (silent 25 fps video, 48 kHz mono FLAC music,
     sample 0 = first video frame), `bgm shots`, `bgm extract` -> <out>/features/{video,music}/<id>.npz.
     MERT layers and CLIP are not stored; ImageBind is kept only as its whole-clip mean (imagebind_mean [1024]),
     which is all the zero-shot baseline uses (disk: ~1.2 MB per video instead of ~2.2 MB);
  4. after the features check out: delete the video, audio, stem and interim files of that batch.
The batch manifests (pairs.csv, shots.csv) stay under <out>/batches/<b>/manifests. One JSON line per video goes to
<out>/stream_manifest.jsonl (ok | failed | silent), so re-running skips finished videos. Failed videos keep their
media and are retried with --retry-failed.

Three stages overlap: Demucs on batch k+1, prep / shots / GPU features on batch k, and the CPU features
(Farneback dynamics, ~110 core-s per 4-min video), check and deletion of batch k-1.

Usage:
    python tools/stream_youtube.py dataset_himv --out dataset_himv_stream
    python tools/stream_youtube.py dataset_himv --out dataset_himv_stream --batch 64 --max-batches 1   # timing test
    python tools/stream_youtube.py dataset_himv --out dataset_himv_stream --follow --pause-match "download_yt_dataset.py.*--out dataset_himv( |$)"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.separate_bgm import AUDIO_EXT, SILENT_RMS, rms, run_demucs  # noqa: E402

from bgm.config import load_config  # noqa: E402
from bgm.data import manifest  # noqa: E402
from bgm.data.media import ffprobe  # noqa: E402
from bgm.data.prep import run_prep  # noqa: E402
from bgm.data.shots import run_shots  # noqa: E402
from bgm.data.tracks import fingerprint  # noqa: E402
from bgm.features.extract import EncoderBank, Extractor, feature_path, load_features, save_features  # noqa: E402
from bgm.features.timing import n_1hz  # noqa: E402

log = logging.getLogger("stream_youtube")
GPU_GROUPS = ["sem_video", "sem_music", "attr", "bridge", "imagebind"]
CPU_GROUPS = ["dyn_video", "dyn_music", "key"]
NEED = {"video": ["sem_siglip_1hz", "attr_logits_1hz", "bridge_logits_1hz", "imagebind_mean", "dyn_25hz", "cut_times_s"],
        "music": ["sem_clap_1hz", "sem_mert_1hz", "attr_logits_1hz", "bridge_logits_1hz", "imagebind_mean", "dyn_25hz",
                  "beat_times_s", "tempo_bpm", "key_pc"]}
OVERRIDES = ["features.sem_video.store_clip=false", "features.sem_music.store_all_mert_layers=false"]


# ---------------------------------------------------------------- bookkeeping
def last_status(path: Path) -> dict[str, dict]:
    """Last JSON line per id of a jsonl manifest ({} if missing); unparsable lines are skipped."""
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["id"]] = r
    return out


def audio_file(src: Path, vid: str) -> Path | None:
    return next((p for p in sorted((src / "audio").glob(f"{vid}.*")) if p.suffix.lower() in AUDIO_EXT and ".part" not in p.name), None)


def pending_ids(src: Path, out: Path, retry_failed: bool) -> list[str]:
    """Downloaded ok, both files on disk, not yet streamed (failed ones only with retry_failed). Download order."""
    dl, done = last_status(src / "manifest.jsonl"), last_status(out / "stream_manifest.jsonl")
    skip = {i for i, r in done.items() if r["status"] != "failed" or not retry_failed}
    videos = {p.stem for p in (src / "video").glob("*.mp4")}
    audios = {p.stem for p in (src / "audio").iterdir() if p.suffix.lower() in AUDIO_EXT and ".part" not in p.name}
    return [i for i, r in dl.items() if r.get("status") == "ok" and i not in skip and i in videos and i in audios]


def pair_row(vid: str, video: Path, bgm: Path) -> dict:
    """pairs.csv row: the BGM stem is a separate music file aligned with the video (both DASH streams of one upload)."""
    v, m = ffprobe(video), ffprobe(bgm)
    vd = v["video"]["duration_s"] if np.isfinite(v["video"]["duration_s"]) else v["duration_s"]
    return {"pair_id": vid, "video_path": str(video), "music_path": str(bgm), "music_source": "file", "music_id": vid,
            "video_duration_s": vd, "music_duration_s": m["duration_s"], "video_start_s": v["video"]["start_s"],
            "audio_start_s": m["audio"]["start_s"], "fps": v["video"]["fps"], "width": v["video"]["width"],
            "height": v["video"]["height"], "video_has_audio": v["audio"] is not None, "music_sr": m["audio"]["sr"],
            "music_channels": m["audio"]["channels"], "track_id": vid}


def compact_imagebind(d: dict) -> dict:
    """Replace imagebind_1hz [T1, 1024] by its l2-normalised mean imagebind_mean [1024] f16 (the whole-clip baseline)."""
    if "imagebind_1hz" in d:
        x = d.pop("imagebind_1hz").astype(np.float32).mean(0)
        d["imagebind_mean"] = (x / max(np.linalg.norm(x), 1e-8)).astype(np.float16)
    return d


def check_item(cfg, vid: str) -> str | None:
    """None if both feature files hold every key with consistent 1 Hz lengths, else the reason."""
    for side, keys in NEED.items():
        d = load_features(feature_path(cfg, side, vid))
        miss = [k for k in keys if k not in d]
        if miss:
            return f"{side} missing {miss}"
        T1 = d["meta"]["T1"]
        bad = [k for k in keys if k.endswith("_1hz") and len(d[k]) != T1]
        if bad:
            return f"{side} {bad} length != T1 {T1}"
    return None


# ---------------------------------------------------------------- stages
def separate(src: Path, bdir: Path, ids: list[str], device: str) -> dict[str, dict]:
    """Demucs no_vocals stems -> <bdir>/bgm/<id>.flac; Chromaprint of the mix. {id: record}."""
    bgm_dir = bdir / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)
    files = [audio_file(src, i) for i in ids]
    a = SimpleNamespace(model="htdemucs", device=device, flac=True, segment=0)
    recs = {i: {"id": i} for i in ids}
    tmp = bdir / "demucs_tmp"
    try:
        run_demucs(files, tmp, a)
    except subprocess.CalledProcessError as e:
        log.warning("demucs failed on %s: %s", bdir.name, e)
    for i, f in zip(ids, files):
        nv, voc = tmp / "htdemucs" / f.stem / "no_vocals.flac", tmp / "htdemucs" / f.stem / "vocals.flac"
        if not nv.exists():
            recs[i].update(status="failed", error="demucs produced no stem")
            continue
        recs[i].update(bgm_rms=rms(nv), vocals_rms=rms(voc) if voc.exists() else None)
        shutil.move(str(nv), bgm_dir / f"{i}.flac")
        if recs[i]["bgm_rms"] < SILENT_RMS:
            recs[i].update(status="silent")
    shutil.rmtree(tmp, ignore_errors=True)
    return recs


def fingerprints(src: Path, out: Path, ids: list[str]) -> dict[str, str]:
    """Chromaprint of each original mix -> <out>/fingerprints/<id>.npy. {id: error} for failures."""
    fdir = out / "fingerprints"
    fdir.mkdir(parents=True, exist_ok=True)

    def one(i):
        try:
            np.save(fdir / f"{i}.npy", fingerprint(str(audio_file(src, i))).values)
            return i, None
        except Exception as e:  # noqa: BLE001
            return i, str(e)[:300]
    with ThreadPoolExecutor(8) as ex:
        return {i: e for i, e in ex.map(one, ids) if e}


def batch_cfg(out: Path, bdir: Path):
    return load_config(overrides=OVERRIDES + [f"paths.manifests={bdir / 'manifests'}", f"paths.interim={bdir / 'interim'}",
                                              f"paths.features={out / 'features'}", f"paths.reports={bdir / 'reports'}"])


def run_groups(cfg, df: pd.DataFrame, ids: list[str], groups: list[str], bank: EncoderBank | None, workers: int) -> dict[str, str]:
    """Extract `groups` for `ids`; if the batch call fails, redo one video at a time. {id: error}."""
    def extract(sub: list[str]) -> None:
        manifest.write_pairs(df[df["pair_id"].isin(sub)], cfg)
        ex = Extractor(cfg, workers=workers)
        if bank is not None:
            ex.bank = bank
        ex.run(groups)
    errors = {}
    try:
        extract(ids)
    except Exception as e:  # noqa: BLE001 -- one bad video must not sink the batch
        log.warning("batch %s failed (%s); retrying per video", groups, e)
        for i in ids:
            try:
                extract([i])
            except Exception as e1:  # noqa: BLE001
                errors[i] = f"{groups}: {str(e1)[:300]}"
    return errors


def gpu_stage(out: Path, bdir: Path, rows: list[dict], bank: EncoderBank, workers: int) -> tuple[list[str], dict[str, str]]:
    """prep, shots and the GPU feature groups for one batch. (ids still in the running, {id: error})."""
    cfg = batch_cfg(out, bdir)
    df = pd.DataFrame(rows)
    (bdir / "manifests").mkdir(parents=True, exist_ok=True)
    df.to_csv(bdir / "manifests" / "pairs_all.csv", index=False)
    manifest.write_pairs(df, cfg)
    prepped = run_prep(cfg)
    ok = sorted(set(prepped["pair_id"])) if len(prepped) else []
    errors = {r["pair_id"]: "prep failed (batch reports/prep_failures.md)" for r in rows if r["pair_id"] not in ok}
    if not ok:
        return [], errors
    manifest.write_pairs(df[df["pair_id"].isin(ok)], cfg)
    run_shots(cfg, workers=workers)
    errors |= run_groups(cfg, df, ok, GPU_GROUPS, bank, workers)
    return [i for i in ok if i not in errors], errors


def cpu_stage(out: Path, bdir: Path, ids: list[str], workers: int) -> dict[str, str]:
    """Dynamics and key groups, ImageBind compaction and the feature check. {id: error}."""
    if not ids:
        return {}
    cfg = batch_cfg(out, bdir)
    df = pd.read_csv(bdir / "manifests" / "pairs_all.csv", dtype=str, keep_default_na=False)
    errors = run_groups(cfg, df, ids, CPU_GROUPS, None, workers)
    manifest.write_pairs(df[df["pair_id"].isin(ids)], cfg)
    for i in ids:
        if i in errors:
            continue
        for side in ("video", "music"):
            p = feature_path(cfg, side, i)
            if p.exists():
                save_features(p, compact_imagebind(load_features(p)))
        if (why := check_item(cfg, i)):
            errors[i] = why
    return errors


def delete_media(src: Path, bdir: Path, ids: list[str]) -> None:
    for i in ids:
        (src / "video" / f"{i}.mp4").unlink(missing_ok=True)
        if (a := audio_file(src, i)) is not None:
            a.unlink(missing_ok=True)
    for d in ("bgm", "interim", "demucs_tmp"):
        shutil.rmtree(bdir / d, ignore_errors=True)


# ---------------------------------------------------------------- loop
class DiskGuard:
    """SIGSTOP the downloader below `low_gb` free, SIGCONT it again above `high_gb`. The downloader is found by its
    command line on every tick (`pgrep -f <match>`), so a restarted downloader stays guarded."""
    def __init__(self, path: Path, match: str | None, low_gb: float, high_gb: float):
        self.path, self.match, self.low, self.high, self.stopped = path, match, low_gb, high_gb, set()

    def free_gb(self) -> float:
        return shutil.disk_usage(self.path).free / 1e9

    def pids(self) -> list[int]:
        r = subprocess.run(["pgrep", "-f", self.match], capture_output=True, text=True)
        return [int(p) for p in r.stdout.split() if int(p) != os.getpid()]

    def _signal(self, pids, sig) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def tick(self) -> None:
        if not self.match:
            return
        f = self.free_gb()
        if f < self.low:
            new = set(self.pids()) - self.stopped
            if new:
                self._signal(new, signal.SIGSTOP)
                self.stopped |= new
                log.warning("disk free %.0f GB < %.0f: downloader %s paused", f, self.low, sorted(new))
        elif f > self.high and self.stopped:
            self.release()
            log.info("disk free %.0f GB: downloader resumed", f)

    def release(self) -> None:
        self._signal(self.stopped, signal.SIGCONT)
        self.stopped = set()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("bgm.features.extract", "bgm.data.prep", "bgm.data.shots", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="folder written by download_yt_dataset.py")
    p.add_argument("--out", default="dataset_himv_stream")
    p.add_argument("--batch", type=int, default=112, help="videos per batch; a multiple of 4 x --workers keeps the dynamics pool full")
    p.add_argument("--max-batches", type=int, default=0, help="stop after this many batches (0 = no limit)")
    p.add_argument("--follow", action="store_true", help="keep waiting for new downloads instead of exiting")
    p.add_argument("--keep-media", action="store_true", help="do not delete the media (debugging)")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=14, help="CPU processes for shots and dynamics")
    p.add_argument("--pause-match", default="", help="pgrep -f pattern of the downloader to pause when the disk runs low")
    p.add_argument("--low-gb", type=float, default=40.0)
    p.add_argument("--high-gb", type=float, default=80.0)
    a = p.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(out, a.pause_match or None, a.low_gb, a.high_gb)
    bank = EncoderBank(load_config(overrides=OVERRIDES), fake=False)
    meta = last_status(src / "manifest.jsonl")
    man = open(out / "stream_manifest.jsonl", "a", encoding="utf-8")
    nb = len(list((out / "batches").glob("b*"))) if (out / "batches").exists() else 0
    stats = {"ok": 0, "failed": 0, "silent": 0}
    t0 = time.time()

    sep_pool, cpu_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    claimed: set[str] = set()

    def launch(ids):
        nonlocal nb
        bdir = out / "batches" / f"b{nb:05d}"
        nb += 1
        claimed.update(ids)
        return ids, bdir, sep_pool.submit(lambda: (separate(src, bdir, ids, a.device), fingerprints(src, out, ids)))

    def pick() -> list[str]:
        if a.max_batches and launched[0] >= a.max_batches:
            return []
        ids = pending_ids(src, out, a.retry_failed)
        ids = [i for i in ids if i not in claimed]
        if len(ids) < a.batch and a.follow:
            return []
        launched[0] += bool(ids)
        return ids[:a.batch]

    def finish(ids, bdir, recs, fp_err, gpu_ok, errors, tb):
        """CPU stage of one batch, then its manifest lines and media deletion (runs on cpu_pool)."""
        try:
            errors = errors | cpu_stage(out, bdir, gpu_ok, a.workers)
        except Exception as e:  # noqa: BLE001
            log.exception("cpu stage of %s failed", bdir.name)
            errors = errors | {i: f"cpu stage: {str(e)[:300]}" for i in gpu_ok}
        finished = []
        for i in ids:
            r = recs[i] | {k: meta.get(i, {}).get(k) for k in ("title", "channel", "duration", "upload_date")}
            r.update(batch=bdir.name, time=time.strftime("%Y-%m-%d %H:%M:%S"), fp_error=fp_err.get(i))
            if "status" not in r:
                r["status"] = "failed" if i in errors else "ok"
            if i in errors:
                r["error"] = errors[i]
            stats[r["status"]] += 1
            man.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["status"] != "failed":
                finished.append(i)
        man.flush()
        if not a.keep_media:
            delete_media(src, bdir, finished)
            for d in ("interim", "demucs_tmp"):
                shutil.rmtree(bdir / d, ignore_errors=True)
        n = sum(stats.values())
        log.info("%s: %d videos in %.0f s | total %s | %.1f videos/min | disk free %.0f GB", bdir.name, len(ids),
                 time.time() - tb, stats, n / (time.time() - t0) * 60, guard.free_gb())

    # demucs(k + 1) on sep_pool  ||  prep, shots, GPU features(k) here  ||  CPU features(k - 1) on cpu_pool
    launched, cur, cpu_fut = [0], None, None
    try:
        while True:
            guard.tick()
            if cur is None:
                ids = pick()
                if not ids:
                    if not a.follow or (a.max_batches and launched[0] >= a.max_batches):
                        break
                    time.sleep(60)
                    continue
                cur = launch(ids)
            ids, bdir, fut = cur
            tb = time.time()
            recs, fp_err = fut.result()
            nxt_ids = pick()
            nxt = launch(nxt_ids) if nxt_ids else None
            rows, errors = [], {}
            for i in ids:
                if "status" in recs[i]:
                    continue
                try:
                    rows.append(pair_row(i, src / "video" / f"{i}.mp4", bdir / "bgm" / f"{i}.flac"))
                except Exception as e:  # noqa: BLE001
                    errors[i] = f"ffprobe: {str(e)[:300]}"
            gpu_ok = []
            if rows:
                try:
                    gpu_ok, e2 = gpu_stage(out, bdir, rows, bank, a.workers)
                    errors |= e2
                except Exception as e:  # noqa: BLE001
                    log.exception("gpu stage of %s failed", bdir.name)
                    errors |= {r["pair_id"]: f"gpu stage: {str(e)[:300]}" for r in rows}
            if cpu_fut is not None:
                cpu_fut.result()
            cpu_fut = cpu_pool.submit(finish, ids, bdir, recs, fp_err, gpu_ok, errors, tb)
            cur = nxt
    finally:
        if cpu_fut is not None:
            cpu_fut.result()
        guard.release()
        sep_pool.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)
        man.close()
    log.info("done: %s", stats)


if __name__ == "__main__":
    main()
