"""ffprobe helpers. All durations and start times are float seconds."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def ffprobe(path: str | Path) -> dict:
    """Stream and format info for a media file.

    Returns {"duration_s", "video": {fps, width, height, start_s, duration_s} | None,
    "audio": {sr, channels, start_s, duration_s} | None}. Raises RuntimeError if ffprobe fails.
    """
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    j = json.loads(proc.stdout)
    out: dict = {"duration_s": float(j.get("format", {}).get("duration", "nan")), "video": None, "audio": None}
    for s in j.get("streams", []):
        start = float(s.get("start_time", "nan"))
        dur = float(s.get("duration", "nan"))
        if s["codec_type"] == "video" and out["video"] is None and not s.get("disposition", {}).get("attached_pic"):
            num, den = s.get("avg_frame_rate", "0/1").split("/")
            out["video"] = {"fps": float(num) / float(den) if float(den) else float("nan"),
                            "width": int(s["width"]), "height": int(s["height"]), "start_s": start, "duration_s": dur}
        elif s["codec_type"] == "audio" and out["audio"] is None:
            out["audio"] = {"sr": int(s["sample_rate"]), "channels": int(s.get("channels", 0)),
                            "start_s": start, "duration_s": dur}
    return out
