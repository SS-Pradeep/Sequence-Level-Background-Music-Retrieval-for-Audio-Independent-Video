"""Pair trimmer: a local web page for cutting unwanted parts out of dataset pairs, video and music together.

Run from the repo root, then open http://127.0.0.1:8765 in a browser:

    .venv/bin/python tools/pair_trimmer.py --root dataset_1248

`<root>/pairs.csv` lists the pairs (paths relative to `<root>`). The originals are never modified. Every cut is
made from the original files, so re-trimming a pair never compounds. Outputs, all under `<root>/trimmed/`:

* `video/<name>.mp4`, `audio/<name>.wav`: trimmed copies, same file names as the originals.
* `edits.json`: the state (pair_id -> {"action": "trim", "ranges": [[start_s, end_s], ...]} or {"action": "exclude"}).
* `edits.csv`: the same, one row per kept part.
* `pairs.csv`: the edited dataset. Excluded pairs are dropped, trimmed pairs point to `trimmed/...` with new
  durations and `trim_start_s`/`trim_end_s`, the rest point to the originals. Paths stay relative to `<root>`,
  so `bgm prep` runs from `<root>` with `paths.manifests=trimmed`.

A pair can keep several parts (say 0-30 s, 35-45 s and 53-59 s of a 1 min clip). Each part becomes its own pair,
never a joined clip, so no clip has a hard music jump inside it: pair_id, music_id and file names get a `_p1`,
`_p2`, ... suffix, `trim_source_id` names the original pair, and track_id and film stay the same (so the parts
stay in one fold). A single kept part keeps the original pair_id and file names.

Cuts can also be given as a list instead of in the browser, as a CSV with columns `pair_id,start_s,end_s`
(a part to KEEP, in seconds of the original clip; several rows for one pair keep several parts; `exclude` in
start_s drops the pair):

    .venv/bin/python tools/pair_trimmer.py --root dataset_1248 --apply my_cuts.csv

Timing: cut points snap to the source video's frame grid (1/fps), and the music is cut at exactly the same
instants (sample-accurate), so music sample 0 stays the first video frame (SPEC §3.4).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import soundfile as sf

log = logging.getLogger("pair_trimmer")

HTML = Path(__file__).with_name("pair_trimmer.html")
MIN_LEN_S = 0.5
TRIM_COLS = ["trim_start_s", "trim_end_s", "trim_source_id"]


# ---------------------------------------------------------------- pure timing helpers

def snap_to_frame(t: float, fps: float) -> float:
    """Nearest frame boundary (s) on a 0-based grid of step 1/fps."""
    return round(t * fps) / fps


def clamp_cut(start_s: float, end_s: float, duration_s: float, fps: float) -> tuple[float, float]:
    """Snap [start_s, end_s] to frames inside [0, duration_s]; raises ValueError if under MIN_LEN_S."""
    s = max(0.0, snap_to_frame(start_s, fps))
    e = snap_to_frame(end_s, fps)
    if e > duration_s + 1e-6:                               # the last frame boundary that fits
        e = np.floor(duration_s * fps + 1e-6) / fps
    if e - s < MIN_LEN_S - 1e-9:
        raise ValueError(f"cut {s:.3f}-{e:.3f} s is shorter than {MIN_LEN_S} s")
    return float(s), float(e)


def clamp_ranges(ranges: list, duration_s: float, fps: float) -> list[tuple[float, float]]:
    """clamp_cut each [start_s, end_s] part, sorted by start; raises ValueError if none or if parts overlap."""
    out = sorted(clamp_cut(float(a), float(b), duration_s, fps) for a, b in ranges)
    if not out:
        raise ValueError("no part to keep")
    for (_, e0), (s1, _) in zip(out, out[1:]):
        if s1 < e0 - 1e-6:
            raise ValueError(f"parts overlap: one ends at {e0:.3f} s, the next starts at {s1:.3f} s")
    return out


def part_name(name: str, k: int | None) -> str:
    """File name or id of part k (1-based) of a multi-part trim: `x.mp4` -> `x_p2.mp4`; k=None keeps it as is."""
    if k is None:
        return name
    p = Path(name)
    return f"{p.stem}_p{k}{p.suffix}" if p.suffix else f"{name}_p{k}"


def edit_ranges(ed: dict) -> list[tuple[float, float]]:
    """The kept parts of a trim edit; also reads the older single-cut form {"start_s", "end_s"}."""
    if "ranges" in ed:
        return [(float(a), float(b)) for a, b in ed["ranges"]]
    return [(float(ed["start_s"]), float(ed["end_s"]))]


def part_ids(ed: dict) -> list[int | None]:
    """Part numbers of a trim edit's outputs: [None] for one part (original names), else [1, 2, ...]."""
    n = len(edit_ranges(ed))
    return [None] if n == 1 else list(range(1, n + 1))


def sample_range(start_s: float, end_s: float, sr: int, n_samples: int) -> tuple[int, int]:
    """Sample indices [a, b) of a cut, clipped to the file (music may be a little shorter than the video)."""
    a = min(int(round(start_s * sr)), n_samples)
    b = min(int(round(end_s * sr)), n_samples)
    return a, b


def shift_film_times(row: dict, start_s: float, new_dur_s: float) -> dict:
    """Move meta_movie_start_s/end_s (where the clip sits in its film) to the trimmed clip."""
    out = dict(row)
    st = row.get("meta_movie_start_s", "")
    if st not in ("", None):
        out["meta_movie_start_s"] = f"{float(st) + start_s:.3f}"
        out["meta_movie_end_s"] = f"{float(st) + start_s + new_dur_s:.3f}"
    return out


def build_pairs(rows: list[dict], edits: dict, trimmed_info: dict) -> list[dict]:
    """The edited manifest: drop excluded pairs, repoint trimmed ones, one row per kept part.

    trimmed_info: pair_id -> list of {"video_duration_s", "music_duration_s"} (s), one per part, in part order.
    """
    out = []
    for r in rows:
        ed = edits.get(r["pair_id"])
        r = {**r, **{c: "" for c in TRIM_COLS}}
        if ed and ed["action"] == "exclude":
            continue
        if not (ed and ed["action"] == "trim"):
            out.append(r)
            continue
        for (s, e), k, info in zip(edit_ranges(ed), part_ids(ed), trimmed_info[r["pair_id"]]):
            q = shift_film_times(r, s, info["video_duration_s"])
            q["pair_id"] = part_name(r["pair_id"], k)
            if q.get("music_id"):
                q["music_id"] = part_name(r["music_id"], k)
            q["video_path"] = f"trimmed/video/{part_name(Path(r['video_path']).name, k)}"
            q["music_path"] = f"trimmed/audio/{part_name(Path(r['music_path']).name, k)}"
            q["video_duration_s"] = f"{info['video_duration_s']:.6f}"
            q["music_duration_s"] = f"{info['music_duration_s']:.6f}"
            q["trim_start_s"], q["trim_end_s"], q["trim_source_id"] = f"{s:.6f}", f"{e:.6f}", r["pair_id"]
            out.append(q)
    return out


# ---------------------------------------------------------------- media I/O

def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=duration:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True).stdout
    j = json.loads(out)
    d = (j.get("streams") or [{}])[0].get("duration") or j["format"]["duration"]
    return float(d)


def trim_video(src: Path, dst: Path, start_s: float, end_s: float) -> float:
    """Frame-accurate cut (decode + re-encode, H.264 CRF 18, original frame timing); returns the duration (s)."""
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ss", f"{start_s:.6f}",
                    "-t", f"{end_s - start_s:.6f}", "-map", "0:v:0", "-map", "0:a:0?", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(tmp)], check=True, capture_output=True)
    os.replace(tmp, dst)
    return probe_duration(dst)


def trim_music(src: Path, dst: Path, start_s: float, end_s: float) -> float:
    """Sample-accurate cut in the source's own format and sample rate; returns the duration (s)."""
    info = sf.info(str(src))
    a, b = sample_range(start_s, end_s, info.samplerate, info.frames)
    y, sr = sf.read(str(src), start=a, stop=b, dtype="float32", always_2d=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    sf.write(str(tmp), y, sr, subtype=info.subtype, format=info.format)
    os.replace(tmp, dst)
    return len(y) / sr


def peaks(path: Path, n_bins: int = 1200) -> list[float]:
    """Max |amplitude| per bin of the mono mix, for drawing a waveform."""
    y, _ = sf.read(str(path), dtype="float32", always_2d=True)
    y = np.abs(y).max(axis=1)
    if len(y) == 0:
        return []
    edges = np.linspace(0, len(y), min(n_bins, len(y)) + 1).astype(int)
    return [round(float(y[a:b].max()), 3) for a, b in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------- the dataset being edited

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.out = root / "trimmed"
        with open(root / "pairs.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.fields = list(self.rows[0]) + [c for c in TRIM_COLS if c not in self.rows[0]]
        self.by_id = {r["pair_id"]: r for r in self.rows}
        self.lock = threading.Lock()
        p = self.out / "edits.json"
        self.edits: dict = json.loads(p.read_text()) if p.exists() else {}
        for ed in self.edits.values():                      # older single-cut edits -> the parts form
            if ed["action"] == "trim" and "ranges" not in ed:
                ed["ranges"] = [list(r) for r in edit_ranges(ed)]
                ed.pop("start_s", None), ed.pop("end_s", None)

    def src(self, pair_id: str, kind: str) -> Path:
        r = self.by_id[pair_id]
        return self.root / r["video_path" if kind == "video" else "music_path"]

    def dst(self, pair_id: str, kind: str, part: int | None = None) -> Path:
        """Trimmed file of a pair; part = 1, 2, ... for the parts of a multi-part trim."""
        name = part_name(self.src(pair_id, kind).name, part)
        return self.out / ("video" if kind == "video" else "audio") / name

    def listing(self) -> list[dict]:
        out = []
        for r in self.rows:
            ed = self.edits.get(r["pair_id"], {})
            out.append({"pair_id": r["pair_id"], "film": r.get("meta_film", ""), "source": r.get("meta_source", ""),
                        "fps": float(r["fps"]), "video_duration_s": float(r["video_duration_s"]),
                        "music_duration_s": float(r["music_duration_s"]), "status": ed.get("action", "untouched"),
                        "ranges": ed.get("ranges"), "edited_at": ed.get("edited_at")})
        return out

    def trim(self, pair_id: str, ranges: list) -> dict:
        """Keep the parts `ranges` = [[start_s, end_s], ...] (s, original clip time), each as its own pair."""
        r = self.by_id[pair_id]
        fps = float(r["fps"])
        ranges = clamp_ranges(ranges, float(r["video_duration_s"]), fps)
        ed = {"action": "trim", "ranges": [list(x) for x in ranges], "edited_at": _now()}
        with self.lock:
            self._remove_files(pair_id)                     # the old cut may have had other parts
            for k in ("video", "music"):
                self.dst(pair_id, k).parent.mkdir(parents=True, exist_ok=True)
            try:
                for (s, e), part in zip(ranges, part_ids(ed)):
                    dv = trim_video(self.src(pair_id, "video"), self.dst(pair_id, "video", part), s, e)
                    dm = trim_music(self.src(pair_id, "music"), self.dst(pair_id, "music", part), s, e)
                    if abs(dv - (e - s)) > 1.5 / fps:
                        raise RuntimeError(f"trimmed video is {dv:.3f} s, expected {e - s:.3f} s")
                    log.info("trim %s part %s %.3f-%.3f s (video %.3f s, music %.3f s)", pair_id, part or 1, s, e, dv, dm)
            except Exception:
                self.edits[pair_id] = ed                    # so the half-written parts get removed
                self._remove_files(pair_id)
                self.edits.pop(pair_id)
                self._save()
                raise
            self.edits[pair_id] = ed
            self._save()
        return ed

    def exclude(self, pair_id: str) -> dict:
        with self.lock:
            self._remove_files(pair_id)
            self.edits[pair_id] = {"action": "exclude", "edited_at": _now()}
            self._save()
        log.info("exclude %s", pair_id)
        return self.edits[pair_id]

    def reset(self, pair_id: str) -> dict:
        with self.lock:
            self._remove_files(pair_id)
            self.edits.pop(pair_id, None)
            self._save()
        log.info("reset %s", pair_id)
        return {"action": "untouched"}

    def _remove_files(self, pair_id: str) -> None:
        """Delete the trimmed outputs of the pair's current edit (all its parts) and the single-part names."""
        ed = self.edits.get(pair_id, {})
        parts = set(part_ids(ed)) if ed.get("action") == "trim" else set()
        for part in parts | {None}:
            for k in ("video", "music"):
                self.dst(pair_id, k, part).unlink(missing_ok=True)

    def _trimmed_info(self) -> dict:
        info = {}
        for pid, ed in self.edits.items():
            if ed["action"] == "trim":
                info[pid] = [{"video_duration_s": probe_duration(self.dst(pid, "video", k)),
                              "music_duration_s": sf.info(str(self.dst(pid, "music", k))).duration}
                             for k in part_ids(ed)]
        return info

    def _save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.out / "edits.json", json.dumps(self.edits, indent=1, sort_keys=True))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["pair_id", "action", "part", "start_s", "end_s", "edited_at"])
        for pid, ed in sorted(self.edits.items()):
            if ed["action"] == "trim":
                for i, (s, e) in enumerate(edit_ranges(ed), 1):
                    w.writerow([pid, "trim", i, s, e, ed["edited_at"]])
            else:
                w.writerow([pid, ed["action"], "", "", "", ed["edited_at"]])
        _atomic_write(self.out / "edits.csv", buf.getvalue())
        rows = build_pairs(self.rows, self.edits, self._trimmed_info())
        tmp = self.out / "pairs.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.out / "pairs.csv")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- HTTP

CTYPES = {".mp4": "video/mp4", ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet: media range requests are many
            pass

        def do_GET(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/")]
            try:
                if parts == [""]:
                    return self._send(HTML.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "pairs"]:
                    return self._json(store.listing())
                if len(parts) == 3 and parts[0] == "api" and parts[1] == "peaks" and parts[2] in store.by_id:
                    return self._json(peaks(store.src(parts[2], "music")))
                if len(parts) == 4 and parts[0] == "media" and parts[3] in store.by_id and parts[2] in ("video", "music"):
                    part = parse_qs(url.query).get("part", [""])[0]
                    part = int(part) if part.isdigit() else None
                    path = store.src(parts[3], parts[2]) if parts[1] == "original" else store.dst(parts[3], parts[2], part)
                    if parts[1] in ("original", "trimmed") and path.exists():
                        return self._file(path)
                self.send_error(HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                pid = body.get("pair_id")
                if pid not in store.by_id or len(parts) != 2 or parts[0] != "api":
                    return self._json({"error": "unknown pair or action"}, HTTPStatus.BAD_REQUEST)
                if parts[1] == "trim":
                    ranges = body["ranges"] if "ranges" in body else [[body["start_s"], body["end_s"]]]
                    return self._json(store.trim(pid, ranges))
                if parts[1] == "exclude":
                    return self._json(store.exclude(pid))
                if parts[1] == "reset":
                    return self._json(store.reset(pid))
                self._json({"error": "unknown action"}, HTTPStatus.BAD_REQUEST)
            except subprocess.CalledProcessError as e:
                self._json({"error": e.stderr.decode(errors="replace")[-500:]}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except (ValueError, RuntimeError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj).encode(), "application/json", status)

        def _send(self, data: bytes, ctype: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path: Path):
            """Serve with HTTP Range support, which browsers need to seek in media."""
            size = path.stat().st_size
            start, end, status = 0, size - 1, HTTPStatus.OK
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].strip().partition("-")
                if a:
                    start, end = int(a), (int(b) if b else size - 1)
                else:
                    start = max(0, size - int(b))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    return self.end_headers()
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", CTYPES.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def apply_cuts(store: Store, cuts_csv: Path) -> list[tuple[str, str]]:
    """Apply a `pair_id,start_s,end_s` list; several rows for one pair keep several parts, start_s = 'exclude'
    drops the pair. A bad row is reported and skipped, the pair's other rows still apply. Returns (pair_id, error)."""
    failures, plan = [], {}                                 # pair_id -> "exclude" or [[start_s, end_s], ...]

    def fail(pid: str, e: Exception) -> None:
        msg = e.stderr.decode(errors="replace")[-300:] if isinstance(e, subprocess.CalledProcessError) else str(e)
        failures.append((pid, f"{type(e).__name__}: {msg}"))
        log.error("%s: %s", pid, failures[-1][1])

    with open(cuts_csv, newline="") as f:
        for r in csv.DictReader(f):
            pid = (r.get("pair_id") or "").strip()
            try:
                if pid not in store.by_id:
                    raise KeyError("unknown pair_id")
                if (r.get("start_s") or "").strip().lower() == "exclude":
                    plan[pid] = "exclude"
                    continue
                row = store.by_id[pid]
                cut = list(clamp_cut(float(r["start_s"]), float(r["end_s"]), float(row["video_duration_s"]), float(row["fps"])))
                if plan.get(pid) != "exclude":
                    plan.setdefault(pid, []).append(cut)
            except Exception as e:  # noqa: BLE001 -- report every bad row, keep going
                fail(pid, e)
    for pid, what in plan.items():
        try:
            store.exclude(pid) if what == "exclude" else store.trim(pid, what)
        except Exception as e:  # noqa: BLE001
            fail(pid, e)
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="dataset_1248", help="folder holding pairs.csv, video/ and audio/")
    ap.add_argument("--apply", type=Path, help="CSV of cuts (pair_id,start_s,end_s) to apply, then exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    store = Store(Path(a.root))
    if a.apply:
        failures = apply_cuts(store, a.apply)
        log.info("applied %s: %d failed", a.apply, len(failures))
        raise SystemExit(1 if failures else 0)
    log.info("%d pairs, %d edited; writing to %s", len(store.rows), len(store.edits), store.out)
    log.info("open http://%s:%d", a.host, a.port)
    ThreadingHTTPServer((a.host, a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
