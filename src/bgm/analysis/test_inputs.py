"""Held-out demo inputs (D106): join one film's TEST clips (of a fold) in upload order into a silent video, plus the same
video with the clips' own music as a reference.

    python -m bgm.analysis.test_inputs --fold 0 --films "Pizza,Kabali" --max-s 90
        -> outputs/emotion_variants/<film>/input_silent.mp4, original.mp4, clips.csv
Video from the silent 25 fps interim files (rule 8), scaled / padded to 1280x720; each clip's music trimmed or padded
to its video length so music sample 0 stays at the clip's first frame (rule 6).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def pick(fold: int, film: str, max_s: float) -> pd.DataFrame:
    f = json.load(open("data/manifests/folds_title.json"))
    test = set(f["folds"][str(fold)]["test"])
    p = pd.read_csv("data/manifests/pairs.csv")
    src = pd.read_csv("outputs/film_titles.csv")
    title = dict(zip(src.source + ":" + src.source_id.astype(str), src.title))
    p["title"] = (p.meta_source + ":" + p.meta_film.astype(str)).map(title).fillna(p.meta_film)
    p = p[p.pair_id.isin(test) & (p.title == film)].copy()
    main = p.groupby("meta_film").video_duration_s.sum().idxmax()            # the source upload with the most test footage
    p = p[p.meta_film == main].sort_values("pair_id")
    keep, tot = [], 0.0
    for r in p.itertuples():
        if tot + r.video_duration_s > max_s and keep:
            break
        keep.append(r.Index)
        tot += r.video_duration_s
    return p.loc[keep]


def build(fold: int, film: str, max_s: float) -> Path:
    clips = pick(fold, film, max_s)
    out = Path("outputs/emotion_variants") / slug(film)
    out.mkdir(parents=True, exist_ok=True)
    vids = [f"data/interim/video/{p}.mp4" for p in clips.pair_id]
    auds = [f"data/interim/music/{m}.flac" for m in clips.music_id]
    n = len(vids)
    vf = "".join(f"[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v{i}];"
                 for i in range(n)) + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *sum([["-i", v] for v in vids], []), "-filter_complex", vf, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out / "input_silent.mp4")], check=True)
    durs = clips.video_duration_s.tolist()
    af = "".join(f"[{i + 1}:a]aresample=48000,apad,atrim=0:{d:.3f},asetpts=PTS-STARTPTS[a{i}];" for i, d in enumerate(durs)) + \
         "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "input_silent.mp4"), *sum([["-i", a] for a in auds], []), "-filter_complex", af, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "320k", str(out / "original.mp4")], check=True)
    clips[["pair_id", "music_id", "title", "video_duration_s"]].to_csv(out / "clips.csv", index=False)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--films", required=True, help="film titles separated by |")
    a.add_argument("--max-s", type=float, default=90.0)
    x = a.parse_args()
    for film in x.films.split("|"):
        print(build(x.fold, film, x.max_s))
