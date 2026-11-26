"""Emotion variants of one render (user request, D106): the same video scored once per emotion of the scene brief.

Every segment's brief gets the same emotion and a stock music description for it; pace match is off and the brief
weight is doubled so the emotion, not the video's pace, drives the music. Each render is upscaled to 1080 lines
(Lanczos, CRF 16, -copyts) and summarised in index.md.

    python -m bgm.analysis.emotion_variants --video sarpatta.mp4 --name sarpatta
        -> outputs/emotion_variants/<name>/<emotion>.mp4, index.md, index.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)
MUSIC = {
    "heroic": "heroic mass hero theme, powerful brass and big drums, rising and triumphant",
    "action": "fast aggressive action music, heavy drums, electric guitar, driving pulse",
    "tension": "tense suspense music, pulsing low strings, ticking percussion, dark",
    "fear": "horror music, eerie dissonant strings, creepy ambient drones",
    "sad": "sad emotional music, slow soft strings and flute, melancholic",
    "romantic": "romantic gentle love melody, soft piano and strings, warm",
    "happy": "happy festive celebration music, folk dance beat, bright and cheerful",
    "comic": "playful comedy music, bouncy pizzicato, quirky and light",
    "peaceful": "calm peaceful music, soft flute and veena, serene ambient",
    "wonder": "magical wonder music, shimmering bells and choir, a sense of awe",
}
BASE = ["--run", "runs/titlesplit/pipeline_ens10/fold0/seed0", "splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit",
        "pipeline.dyn_variant=null", "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.audio_bitrate=320k",
        "render.pace_match.weight=0.0", "render.scene_brief.weight=2.0", "render.length_fit.weight=1.0",
        "render.replay_penalty=3.0", "render.max_plays=2", "render.max_windows_per_file=4"]


def library_energy() -> tuple[np.ndarray, np.ndarray]:
    mids = pd.read_csv("data/manifests/pairs.csv").music_id.unique()
    on, rms = [], []
    for m in mids:
        x = np.load(f"data/features/music/{m}.npz")["dyn_25hz"]
        on.append(x[:, 0].mean()), rms.append(x[:, 2].mean())
    return np.sort(on), np.sort(rms)


def piece_energy(p: dict, lo: np.ndarray, lr: np.ndarray) -> float:
    tot = 0.0
    for x in p["pieces"]:
        d = np.load(f"data/features/music/{x['music_id']}.npz")["dyn_25hz"]
        w = d[int(x["offset"] * 25):int((x["offset"] + x["end"] - x["start"]) * 25)]
        e = (np.searchsorted(lo, w[:, 0].mean()) / len(lo) + np.searchsorted(lr, w[:, 2].mean()) / len(lr)) / 2
        tot += e * (x["end"] - x["start"])
    return 100 * tot / p["duration_s"]


def main(video: str, name: str, source_work: str | None, emotions: list[str]) -> Path:
    out = Path("outputs/emotion_variants") / name
    out.mkdir(parents=True, exist_ok=True)
    segs = None
    titles = pd.read_csv("outputs/film_titles.csv")
    title = dict(zip(titles.source + ":" + titles.source_id.astype(str), titles.title))
    pairs = pd.read_csv("data/manifests/pairs.csv")
    film_of = dict(zip(pairs.music_id, (pairs.meta_source + ":" + pairs.meta_film.astype(str)).map(title).fillna(pairs.meta_film)))
    lo, lr = library_energy()
    rows = []
    for emo in emotions:
        tag = f"{name}_emo_{emo}"
        work = Path("renders") / tag
        work.mkdir(parents=True, exist_ok=True)
        if source_work and not (work / "query.npz").exists():
            for f in ("query.npz", "silent25.mp4"):
                shutil.copy(Path(source_work) / f, work / f)
        if segs is None:
            with np.load(work / "query.npz") as z:
                segs = [(float(a), float(b)) for a, b in z["segments"]] if "segments" in z.files else None
        brief = [{"start_s": round(a, 2), "end_s": round(b, 2), "scene": f"(emotion variant: {emo})", "emotion": emo, "music": MUSIC[emo]}
                 for a, b in segs]
        bfile = Path("renders") / f"{tag}.brief.yaml"
        bfile.write_text(f"# emotion variant '{emo}' for {video}\n" + yaml.safe_dump(brief, sort_keys=False))
        raw = Path("renders") / name / f"{tag}.mp4"
        raw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [".venv/bin/bgm", "render", "--video", video, "--out", str(raw), *BASE, f"render.scene_brief.file={bfile}"]
        with open(f"reports/render_{tag}.log", "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
        final = out / f"{emo}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(raw), "-vf", "scale=-2:1080:flags=lanczos", "-c:v", "libx264",
                        "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(final)], check=True)
        p = json.loads((Path("renders") / f"{tag}.plan.json").read_text())
        films = []
        for x in p["pieces"]:
            f = film_of.get(x["music_id"], "?")
            if not films or films[-1] != f:
                films.append(f)
        switches = sum(a["music_id"] != b["music_id"] for a, b in zip(p["segments"], p["segments"][1:]))
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(final)],
                                          capture_output=True, text=True).stdout)
        starts = {s["codec_type"]: float(s["start_time"]) for s in probe["streams"]}
        rows.append({"emotion": emo, "file": final.name, "music_from": " → ".join(films), "switches": switches,
                     "energy_pct": round(piece_energy(p, lo, lr)), "av_start": f"{starts.get('video', 0):.3f}/{starts.get('audio', 0):.3f}"})
        log.info("%s: %s", emo, rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "index.csv", index=False)
    L = [f"# Emotion variants: {video}", "",
         "The same video scored once per emotion. Every segment of the scene brief gets that emotion and the music description "
         "below; pace match off and brief weight 2 so the emotion drives the choice (title-split final system, length fit, reprise "
         "rule). Energy = time-weighted library percentile of the chosen music's onsets + loudness. Videos at 1080 lines, 320k audio.", "",
         "| emotion | file | brief music text | music taken from (films) | switches | energy pct |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['emotion']} | {r['file']} | {MUSIC[r['emotion']]} | {r['music_from']} | {r['switches']} | {r['energy_pct']} |")
    (out / "index.md").write_text("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    a = argparse.ArgumentParser()
    a.add_argument("--video", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--source-work", default=None, help="a renders/<name>/ dir with cached query.npz + silent25.mp4")
    a.add_argument("--emotions", default=",".join(MUSIC))
    x = a.parse_args()
    print(main(x.video, x.name, x.source_work, x.emotions.split(",")))
