"""Step 4: render every media/inputs video with P0 (D75, ./render_best.sh) and with the best person-branch system.

Run: python -m bgm.person.outputs --person-run runs/filmsplit/person/eval/<system>/fold0/seed0

outputs/person_branch/
  baseline/<input>.mp4   P0
  person/<input>.mp4     the best run
  index.csv              input, duration, and per system every piece: video start / end, music file, track, file offset
  _work/<system>/        render work folders, plans, native-resolution renders (git-ignored)
Upscaling (user request 2026-10-20): the render stream-copies the source picture; outputs under 1080 lines are
re-encoded to 1080 lines (Lanczos, aspect kept, libx264 CRF 16 preset slow, yuv420p, source timestamps kept), audio
stream-copied. Sync is
checked on every final file: stream start times and frame count equal to the native render's, and the audio's lag
against the render's own soundtrack (bgm.wav) by cross-correlation (D86's check): music sample 0 must meet the first
frame within 1 ms.
Inputs that fail to render are skipped and listed. Nothing outside outputs/person_branch/ is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)
INPUTS = Path("media/inputs")
OUT = Path("outputs/person_branch")
WORK = OUT / "_work"
COMMON = ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", "pipeline.dyn_variant=null",
          "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.audio_bitrate=320k"]
TARGET_LINES = 1080


def probe(path: Path) -> dict:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                                   "stream=codec_type,width,height,start_time,nb_read_packets,r_frame_rate:format=duration",
                                   "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    s = {x["codec_type"]: x for x in j["streams"]}
    return {"width": int(s["video"]["width"]), "height": int(s["video"]["height"]), "v_start": float(s["video"]["start_time"]),
            "a_start": float(s["audio"]["start_time"]), "frames": int(s["video"]["nb_read_packets"]), "fps": s["video"]["r_frame_rate"],
            "duration": float(j["format"]["duration"])}


def render(system: str, video: Path, person_run: str | None) -> tuple[Path, Path]:
    """-> (native render, its plan). system 'baseline' = ./render_best.sh; 'person' = bgm render with the hook on."""
    renders = WORK / system
    native = renders / "native" / video.name
    native.parent.mkdir(parents=True, exist_ok=True)
    if system == "baseline":
        cmd = ["./render_best.sh", str(video), str(native), f"paths.renders={renders}", "render.audio_bitrate=320k"]
    else:
        cmd = [".venv/bin/bgm", "render", "--video", str(video), "--out", str(native), "--run", person_run, *COMMON,
               f"paths.renders={renders}", "person.enabled=true"]
    if not native.exists():
        with open(renders / f"{video.stem}.log", "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    return native, renders / f"{video.stem}.plan.json"


def upscale(native: Path, final: Path) -> str:
    final.parent.mkdir(parents=True, exist_ok=True)
    info = probe(native)
    if final.exists() and final.stat().st_mtime > native.stat().st_mtime:
        return f"done already ({probe(final)['width']}x{probe(final)['height']})"
    if info["height"] >= TARGET_LINES:
        subprocess.run(["cp", "--reflink=auto", str(native), str(final)], check=True)
        return f"kept {info['width']}x{info['height']}"
    # -copyts keeps the source's video start time: with_audio.mp4's first frame is at 0.041 s and the music is padded to
    # meet it (D86); a plain re-encode moved the frame to 0 and the music 41 ms late (caught by check_sync)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(native), "-map", "0:v:0", "-map", "0:a:0",
                    "-vf", f"scale=-2:{TARGET_LINES}:flags=lanczos", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(final)], check=True)
    out = probe(final)
    return f"upscaled {info['width']}x{info['height']} -> {out['width']}x{out['height']}"


def audio_lag_ms(final: Path, bgm_wav: Path, seconds: float = 30.0) -> float:
    """Lag of the final file's audio against the render's soundtrack, ms (positive = the file's audio is late)."""
    ref, sr = sf.read(bgm_wav, dtype="float32", always_2d=True)
    ref = ref.mean(1)[: int(seconds * sr)]
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(final), "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                          "-t", str(seconds), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32)
    n = min(len(y), len(ref))
    y, ref = y[:n], ref[:n]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y, m) * np.conj(np.fft.rfft(ref, m)), m)
    lag = int(np.argmax(np.r_[c[-int(0.5 * sr):], c[: int(0.5 * sr)]])) - int(0.5 * sr)
    return 1000.0 * lag / sr


def check_sync(native: Path, final: Path, bgm_wav: Path) -> dict:
    a, b = probe(native), probe(final)
    lag = audio_lag_ms(final, bgm_wav)
    # music sample 0 must meet the first frame: the lag inside the audio stream equals the first frame's time after
    # the audio stream's start (0 for most inputs; 40 ms of leading silence for 12.mp4, D86)
    res = {"native": a, "final": b, "lag_ms": lag, "sync_error_ms": lag - 1000.0 * (b["v_start"] - b["a_start"])}
    assert abs(a["v_start"] - b["v_start"]) < 1e-3 and abs(a["a_start"] - b["a_start"]) < 1e-3, res
    assert a["frames"] == b["frames"], res
    assert abs(res["sync_error_ms"]) < 1.0, res
    return res


def index_rows(video: Path, system: str, plan: dict, store_track: dict) -> list[dict]:
    return [{"input": video.name, "duration_s": round(plan["duration_s"], 3), "system": system, "piece": i,
             "video_start_s": round(p["start"], 3), "video_end_s": round(p["end"], 3), "music_id": p["music_id"],
             "track_id": store_track.get(p["music_id"], ""), "music_offset_s": round(p["offset"], 3)} for i, p in enumerate(plan["pieces"])]


def main() -> None:
    from bgm.data import manifest
    from bgm.person.diagnose import film_cfg
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-run", default=None)
    ap.add_argument("--systems", default="baseline,person")
    a = ap.parse_args()
    systems = a.systems.split(",")
    assert "person" not in systems or a.person_run, "--person-run is needed for the person system"
    pairs = manifest.read_pairs(film_cfg([]))
    track = dict(zip(pairs.music_id, pairs.track_id))
    rows, report, failed = [], {}, {}
    for video in sorted(INPUTS.glob("*.mp4")):
        for system in systems:
            try:
                native, plan_path = render(system, video, a.person_run)
                plan = json.loads(plan_path.read_text())
                final = OUT / system / video.name
                how = upscale(native, final)
                sync = check_sync(native, final, Path(plan["bgm_wav"]))
                n_sw = sum(x["music_id"] != y["music_id"] for x, y in zip(plan["segments"], plan["segments"][1:]))
                report[f"{system}/{video.name}"] = {"how": how, "segments": len(plan["segments"]), "pieces": len(plan["pieces"]),
                                                     "switches": n_sw, "files": len({p["music_id"] for p in plan["pieces"]}),
                                                     "lufs": plan["output_lufs"], **sync}
                rows += index_rows(video, system, plan, track)
                log.info("%s %s: %s; %d segments, %d switches; music sample 0 vs first frame %.2f ms", system, video.name, how,
                         len(plan["segments"]), n_sw, sync["sync_error_ms"])
            except Exception as e:                        # noqa: BLE001  skip and list (task instruction)
                failed[f"{system}/{video.name}"] = repr(e)
                log.error("%s %s failed: %r", system, video.name, e)
    if systems != ["baseline", "person"]:
        log.info("partial run (%s): index.csv not written", systems)
        return
    with open(OUT / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["input", "duration_s", "system", "piece", "video_start_s", "video_end_s", "music_id", "track_id", "music_offset_s"])
        w.writeheader()
        w.writerows(rows)
    (WORK / "outputs_report.json").write_text(json.dumps({"rendered": report, "failed": failed, "person_run": a.person_run}, indent=1, default=float))
    log.info("done: %d rendered, %d failed %s", len(report), len(failed), list(failed))


if __name__ == "__main__":
    main()
