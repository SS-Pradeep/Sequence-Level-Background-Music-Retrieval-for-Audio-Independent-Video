#!/usr/bin/env python3
"""
Keep only the BGM of a download_yt_dataset.py output folder, using Demucs.

For every <out>/audio/<id>.{m4a,wav,...} without a finished BGM file it runs
    demucs -n htdemucs --two-stems vocals
and saves
    <out>/bgm/<id>.wav      the no_vocals stem (44.1 kHz stereo, as Demucs writes it)
    <out>/vocals/<id>.wav   only with --keep-vocals
One JSON line per track goes to <out>/bgm_manifest.jsonl with the stem RMS, so
near-silent outputs (a known Demucs failure mode) are flagged as "silent".
Re-running the same command skips finished tracks.

Usage:
    python tools/separate_bgm.py yt_dataset
    python tools/separate_bgm.py yt_dataset --batch 10 --device cpu --keep-vocals
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXT = {".m4a", ".wav", ".mp3", ".opus", ".webm", ".flac", ".ogg"}
SILENT_RMS = 1e-4  # stems below this RMS are almost certainly a failed separation


def rms(path):
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def run_demucs(files, tmp, a):
    cmd = [sys.executable, "-m", "demucs", "-n", a.model, "--two-stems", "vocals",
           "-d", a.device, "-o", str(tmp), "--filename", "{track}/{stem}.{ext}"] + (["--flac"] if a.flac else [])
    if a.segment:
        cmd += ["--segment", str(a.segment)]
    subprocess.run(cmd + [str(f) for f in files], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="folder written by download_yt_dataset.py (has audio/)")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=20, help="tracks per demucs call")
    p.add_argument("--segment", type=int, default=0, help="demucs --segment (lower it on GPU OOM)")
    p.add_argument("--keep-vocals", action="store_true")
    p.add_argument("--flac", action="store_true", help="save stems as lossless FLAC (about half the size of wav)")
    a = p.parse_args()

    out = Path(a.out)
    audio_dir, bgm_dir, voc_dir = out / "audio", out / "bgm", out / "vocals"
    bgm_dir.mkdir(exist_ok=True)
    if a.keep_vocals:
        voc_dir.mkdir(exist_ok=True)

    srcs = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in AUDIO_EXT and ".part" not in f.name)
    ext = "flac" if a.flac else "wav"
    todo = [f for f in srcs if not any((bgm_dir / f"{f.stem}.{e}").exists() for e in ("wav", "flac"))]
    print(f"{len(srcs):,} audio files, {len(srcs) - len(todo):,} already separated, {len(todo):,} to do")

    stats = {"ok": 0, "silent": 0, "failed": 0}
    t0 = time.time()
    with open(out / "bgm_manifest.jsonl", "a", encoding="utf-8") as man:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                try:
                    run_demucs(chunk, tmp, a)
                except subprocess.CalledProcessError as e:
                    print(f"demucs failed on batch {i // a.batch}: {e}", flush=True)
                for f in chunk:
                    stem_dir = Path(tmp) / a.model / f.stem
                    nv, voc = stem_dir / f"no_vocals.{ext}", stem_dir / f"vocals.{ext}"
                    rec = {"id": f.stem, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                    if not nv.exists():
                        rec["status"] = "failed"
                    else:
                        rec["bgm_rms"], rec["vocals_rms"] = rms(nv), rms(voc) if voc.exists() else None
                        rec["status"] = "silent" if rec["bgm_rms"] < SILENT_RMS else "ok"
                        shutil.move(str(nv), bgm_dir / f"{f.stem}.{ext}")
                        if a.keep_vocals and voc.exists():
                            shutil.move(str(voc), voc_dir / f"{f.stem}.{ext}")
                    stats[rec["status"]] += 1
                    man.write(json.dumps(rec) + "\n")
                    man.flush()
            n = min(i + a.batch, len(todo))
            rate = n / (time.time() - t0)
            print(f"[{n:,}/{len(todo):,}] {stats} | {rate * 60:.1f}/min", flush=True)
    print(f"done: {stats}")


if __name__ == "__main__":
    main()
