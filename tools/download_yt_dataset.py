                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #!/usr/bin/env python3
"""
Resumable, parallel downloader for a YouTube-URL dataset (e.g. HIMV-200K style
video-music pairs) for video-to-BGM retrieval.

For every video it saves
    <out>/video/<id>.mp4   video-only stream (silent), capped at --max-height
    <out>/audio/<id>.wav   audio track (the BGM), mono, --sr Hz   (or .m4a with --keep-m4a)
and logs one JSON line per attempt to <out>/manifest.jsonl, so re-running the
same command skips finished / permanently-dead videos and retries the rest.

Install:
    pip install -U yt-dlp        # keep it updated, YouTube changes often
    sudo apt install ffmpeg

Typical usage:
    # 1) try a random 2,000-video subset first
    python download_yt_dataset.py Youtube_ID.txt --out himv --sample 2000 --workers 4

    # 2) only the first 30 s of each video (much less bandwidth/disk)
    python download_yt_dataset.py Youtube_ID.txt --out himv --sample 2000 --clip-seconds 30

    # 3) split the full list across 4 machines / terminals
    python download_yt_dataset.py Youtube_ID.txt --out himv --shard 0 --num-shards 4
    ...
    python download_yt_dataset.py Youtube_ID.txt --out himv --shard 3 --num-shards 4

    # 4) if YouTube starts saying "Sign in to confirm you're not a bot"
    python download_yt_dataset.py ... --cookies-from-browser firefox --sleep 3
"""
import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yt_dlp
    from yt_dlp.utils import DownloadError, download_range_func
except ImportError:
    sys.exit("yt-dlp missing:  pip install -U yt-dlp")

ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|^)([A-Za-z0-9_-]{11})(?:[&?#\s]|$)")

# Errors that will never succeed on retry -> mark as dead and skip next time
PERMANENT = (
    "video unavailable", "video is unavailable", "private video", "has been removed", "account associated",
    "copyright", "not available in your country", "members-only", "join this channel",
    "this video has been terminated", "confirm your age", "does not exist", "too long",
)


# --------------------------------------------------------------------------- utils
def read_ids(path):
    ids, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = ID_RE.search(line.strip())
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                ids.append(m.group(1))
    return ids


def load_done(manifest):
    """IDs that are finished (ok) or permanently dead."""
    done = {}
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") in ("ok", "dead"):
                    done[r["id"]] = r["status"]
    return done


class Manifest:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()


def to_wav(src, dst, sr):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- core
def base_opts(a):
    o = {
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "fragment_retries": 3, "socket_timeout": 30,
        "concurrent_fragment_downloads": 2,
        "sleep_interval": a.sleep, "max_sleep_interval": a.sleep * 2 if a.sleep else 0,
        "noplaylist": True, "overwrites": False, "continuedl": True,
    }
    if a.max_duration:
        o["match_filter"] = yt_dlp.utils.match_filter_func(f"duration <= {a.max_duration}")
    if a.clip_seconds:
        o["download_ranges"] = download_range_func(None, [(a.clip_start, a.clip_start + a.clip_seconds)])
        o["force_keyframes_at_cuts"] = True
    if a.cookies:
        o["cookiefile"] = a.cookies
    if a.cookies_from_browser:
        o["cookiesfrombrowser"] = (a.cookies_from_browser,)
    return o


def download_one(vid, a, dirs):
    url = f"https://www.youtube.com/watch?v={vid}"
    rec = {"id": vid, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    vpath = dirs["video"] / f"{vid}.mp4"
    wav = dirs["audio"] / f"{vid}.wav"
    try:
        info = None
        # ---- video-only stream (silent video)
        if a.mode in ("both", "video") and not vpath.exists():
            h = a.max_height
            o = base_opts(a) | {
                "format": f"bv*[height<={h}][ext=mp4]/bv*[height<={h}]/bv*",
                "outtmpl": str(dirs["video"] / f"{vid}.%(ext)s"),
                "merge_output_format": "mp4",
                "remux_video": "mp4",
            }
            with yt_dlp.YoutubeDL(o) as y:
                info = y.extract_info(url, download=True)
            if not vpath.exists():
                dur = (info or {}).get("duration") or 0
                if a.max_duration and dur > a.max_duration:
                    raise DownloadError(f"too long ({dur}s)")
                raise DownloadError("video file missing after download")

        # ---- audio stream (the BGM)
        have_audio = wav.exists() or (a.keep_m4a and any(dirs["audio"].glob(f"{vid}.*")))
        if a.mode in ("both", "audio") and not have_audio:
            o = base_opts(a) | {
                "format": "ba[ext=m4a]/ba",
                "outtmpl": str(dirs["audio"] / f"{vid}.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(o) as y:
                ainfo = y.extract_info(url, download=True)
                src = Path(y.prepare_filename(ainfo)) if ainfo else None
            if not src or not src.exists():
                dur = (ainfo or {}).get("duration") or 0
                if a.max_duration and dur > a.max_duration:
                    raise DownloadError(f"too long ({dur}s)")
                raise DownloadError("audio file missing after download")
            info = info or ainfo
            if not a.keep_m4a:
                to_wav(src, wav, a.sr)
                src.unlink(missing_ok=True)

        if info:
            rec.update(title=info.get("title"), duration=info.get("duration"),
                       channel=info.get("channel"), upload_date=info.get("upload_date"))
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001  (log everything, never crash the pool)
        msg = str(e).lower()
        rec["error"] = str(e)[:300]
        rec["status"] = "dead" if any(k in msg for k in PERMANENT) else "retry"
        # remove half-written files so the next run starts clean
        for p in list(dirs["video"].glob(f"{vid}.*part*")) + list(dirs["audio"].glob(f"{vid}.*part*")):
            p.unlink(missing_ok=True)
    return rec


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("id_file", help="txt with one YouTube URL or ID per line")
    p.add_argument("--out", default="yt_dataset")
    p.add_argument("--mode", choices=["both", "video", "audio"], default="both")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads (keep 2-8 to avoid bans)")
    p.add_argument("--sample", type=int, default=0, help="random subset size (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=int, default=0, help="slice start in the (sampled) list")
    p.add_argument("--limit", type=int, default=0, help="max videos this run (0 = no limit)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-height", type=int, default=360, help="video resolution cap (360 is plenty for CLIP/ViT features)")
    p.add_argument("--max-duration", type=int, default=900, help="skip videos longer than N s (0 = no cap)")
    p.add_argument("--clip-seconds", type=float, default=0, help="download only this many seconds (0 = full)")
    p.add_argument("--clip-start", type=float, default=0)
    p.add_argument("--sr", type=int, default=22050, help="wav sample rate")
    p.add_argument("--keep-m4a", action="store_true", help="keep compressed audio instead of wav")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds to sleep between requests per worker")
    p.add_argument("--cookies", help="cookies.txt (Netscape format)")
    p.add_argument("--cookies-from-browser", help="e.g. firefox, chrome")
    p.add_argument("--dry-run", action="store_true", help="just show what would be downloaded")
    a = p.parse_args()

    out = Path(a.out)
    dirs = {"video": out / "video", "audio": out / "audio"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    ids = read_ids(a.id_file)
    print(f"{len(ids):,} unique IDs in {a.id_file}")
    if a.sample:
        ids = random.Random(a.seed).sample(ids, min(a.sample, len(ids)))
        (out / f"subset_seed{a.seed}_n{len(ids)}.txt").write_text("\n".join(ids))
    ids = ids[a.shard::a.num_shards][a.start:]
    done = load_done(manifest_path)
    todo = [i for i in ids if i not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"shard {a.shard}/{a.num_shards}: {len(ids):,} ids, {len(ids) - len(todo):,} already done/dead, {len(todo):,} to download")
    if a.dry_run or not todo:
        return

    man = Manifest(manifest_path)
    stats = {"ok": 0, "dead": 0, "retry": 0}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(download_one, v, a, dirs): v for v in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            man.write(rec)
            stats[rec["status"]] += 1
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / rate / 3600 if rate else 0
                print(f"[{n:,}/{len(todo):,}] ok={stats['ok']} dead={stats['dead']} retry={stats['retry']} "
                      f"| {rate*60:.1f}/min | ETA {eta:.1f} h", flush=True)
                if stats["retry"] > 50 and stats["ok"] == 0:
                    print("!! Everything is failing - probably bot-blocked. Stop, add --cookies-from-browser, "
                          "lower --workers, raise --sleep, and `pip install -U yt-dlp`.", flush=True)
    print(f"done: {stats}.  Re-run the same command to retry the 'retry' ones.")


if __name__ == "__main__":
    main()
