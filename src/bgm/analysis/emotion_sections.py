"""Section figures + labelled renders for one held-out TEST clip per emotion (like outputs/godfather/sections).

Pick rule (no retrieval rank used, so the set is not cherry-picked): among clips in outputs/emotion_showcase/scores.csv
(each a test clip of its title-split fold, emotion = zero-shot SigLIP + CLAP estimate, `both_agree`), at least 25 s and
3 dataset segments, under 60 s (the long-video shortlist pools every fold's heads), no Konjum file, one film per emotion, the highest emotion score.

Per clip, with the clip's own fold's final system (10-head ensemble; never trained on the clip's film):
  viterbi.mp4 / greedy.mp4 / fixed.mp4   the three decoders, labelled (segment, music file, SWITCH / JUMP / continues)
  original.mp4                           the clip's own music
  compare.mp4                            original -> greedy -> fixed -> cue Viterbi, back to back
  sec1_training_coarse_to_fine.png, sec2_transition_viterbi.png, sections.json   (bgm.analysis.godfather_sections)

    python -m bgm.analysis.emotion_sections            -> outputs/emotion_sections/
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
OUT = Path("outputs/emotion_sections")
RENDERS = Path("renders/emotion_sections")
EMOS = ["heroic", "action", "tension", "fear", "sad", "romantic", "happy", "comic", "peaceful", "wonder"]
BASE = ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "pipeline.dyn_variant=null",
        "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", "render.length_fit.weight=1.0", "render.max_plays=2",
        "render.max_windows_per_file=4", "render.exclude_music=[Konjum]", f"paths.renders={RENDERS}"]
KINDS = {"viterbi": ["render.replay_penalty=3.0"],                                  # the final system (cue decoder on)
         "greedy": ["cue.enabled=false", "render.replay_penalty=0", "render.penalty_overrides.lam_switch=0",
                    "render.penalty_overrides.lam_jump=0", "render.penalty_overrides.mu=0"],
         "fixed": ["cue.enabled=false", "render.replay_penalty=0"]}               # the run's val-tuned fixed penalties
TITLES = {"original": "ORIGINAL MUSIC (the clip's own soundtrack)",
          "greedy": "GREEDY (best window per segment, no transition cost)",
          "fixed": "FIXED-PENALTY VITERBI", "viterbi": "CUE VITERBI (learned transition costs, final system)"}
BIN = Path(sys.executable).parent


def picks() -> pd.DataFrame:
    s = pd.read_csv("outputs/emotion_showcase/scores.csv")
    s = s.join(pd.read_csv("data/manifests/segments.csv").groupby("pair_id").size().rename("n_seg"), on="pair_id")
    # < 60 s: longer clips trigger render.run's long-video shortlist, which pools heads of ALL folds (test-fold leak)
    c = s[s.both_agree & (s.duration_s >= 25) & (s.duration_s < 60) & (s.n_seg >= 3) & ~s.pair_id.str.contains("Konjum", case=False)
          & ~s.film.astype(str).str.contains("Konjum", case=False)]
    used, rows = set(), []
    for e in EMOS:
        d = c[(c.emotion == e) & ~c.film.isin(used)].sort_values("emo_score", ascending=False)
        if len(d):
            used.add(d.iloc[0].film)
            rows.append(d.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def ts(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Head,Noto Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,3,10,0,7,40,40,36,1
Style: Info,Noto Sans,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,3,10,0,1,40,40,40,1
Style: Sw,Noto Sans,52,&H0000A5FF,&H00FFFFFF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,3,12,0,9,40,40,36,1
Style: Cont,Noto Sans,40,&H0090EE90,&H00FFFFFF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,3,10,0,9,40,40,36,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def overlay(plan: dict | None, title: str, dur: float, film, true_track: str, track_of: dict, emo: str) -> str:
    ev = []
    if plan is None:                                                    # the original soundtrack
        ev.append(f"Dialogue: 0,{ts(0)},{ts(dur)},Head,,0,0,0,,{title}  ·  {emo}")
        return ASS_HEAD + "\n".join(ev) + "\n"
    segs, n = plan["segments"], len(plan["segments"])
    sw = sum(a["music_id"] != b["music_id"] for a, b in zip(segs, segs[1:]))
    ev.append(f"Dialogue: 0,{ts(0)},{ts(dur)},Head,,0,0,0,,{title}  ·  {sw} music switches  ·  {emo}")
    for i, s in enumerate(segs):
        a, b = s["start_s"], s["end_s"]
        hit = "  (= TRUE music)" if track_of[s["music_id"]] == true_track else ""
        ev.append(f"Dialogue: 0,{ts(a)},{ts(b)},Info,,0,0,0,,segment {i + 1}/{n}  ·  {a:.1f}–{b:.1f} s\\N"
                  f"music: {film(s['music_id'])}  from {s['offset_s']:.1f} s in the file{hit}")
        if i:
            p = segs[i - 1]
            end = ts(min(a + 1.8, b))
            if p["music_id"] != s["music_id"]:
                ev.append(f"Dialogue: 0,{ts(a)},{end},Sw,,0,0,0,,♪ SWITCH")
            elif abs(s["offset_s"] - (p["offset_s"] + p["end_s"] - p["start_s"])) > 0.05:
                ev.append(f"Dialogue: 0,{ts(a)},{end},Sw,,0,0,0,,↷ JUMP (same file)")
            else:
                ev.append(f"Dialogue: 0,{ts(a)},{end},Cont,,0,0,0,,→ music continues")
    return ASS_HEAD + "\n".join(ev) + "\n"


def burn(src: Path, ass: Path, dst: Path) -> None:
    """1080 lines (lanczos), labels burned in, AAC 320k 48 kHz (re-encoded so the concat of all four is uniform)."""
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vf",
                    f"scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,ass={ass}", "-r", "25",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
                    "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(dst)], check=True)


def run_clip(i: int, r, store, cfg) -> dict:
    from bgm.analysis.cue_demo import original_soundtrack
    from bgm.analysis.poster_pipeline import film_of
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(r.film)).strip("_").lower()
    name = f"{i + 1:02d}_{r.emotion}_{slug}"
    d = OUT / name
    (d / "logs").mkdir(parents=True, exist_ok=True)
    video = Path(f"data/interim/video/{r.pair_id}.mp4")
    run = f"runs/titlesplit/pipeline_ens10/fold{int(r.fold)}/seed0"
    plans = {}
    for k, extra in KINDS.items():
        stem = f"{name}_{k}"
        if k != "viterbi":                                                 # reuse the features cached by the first render
            (RENDERS / stem).mkdir(parents=True, exist_ok=True)
            for f in ("query.npz", "silent25.mp4"):
                src = RENDERS / f"{name}_viterbi" / f
                if src.exists() and not (RENDERS / stem / f).exists():
                    shutil.copy(src, RENDERS / stem / f)
        plan_p = RENDERS / f"{stem}.plan.json"
        if not plan_p.exists():
            with open(d / "logs" / f"render_{k}.log", "w") as lf:
                subprocess.run([str(BIN / "bgm"), "render", "--video", str(video), "--out", str(RENDERS / f"{stem}.mp4"),
                                "--run", run, *BASE, *extra], check=True, stdout=lf, stderr=subprocess.STDOUT)
        plans[k] = json.loads(plan_p.read_text())
    true_mid = store.music_of[r.pair_id]
    if not (d / "sections.json").exists():
      with open(d / "logs" / "figures.log", "w") as lf:
        subprocess.run([str(BIN / "python"), "-m", "bgm.analysis.godfather_sections", "--video", str(video), "--name",
                        f"{name}_viterbi", "--out", str(d), "--run", run, "--label", f"{r.emotion} · {r.film} (test clip)",
                        "--true-music", true_mid, "--note",
                        "this clip's film is in the test fold: the heads never saw it", *BASE, *KINDS["viterbi"]],
                       check=True, stdout=lf, stderr=subprocess.STDOUT)
    frames = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                 "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True,
                                check=True).stdout.strip())
    orig = RENDERS / f"{name}_original.mp4"
    original_soundtrack(cfg, store, [r.pair_id], [frames], video, orig)
    dur = frames / 25
    film = lambda m: film_of(store, m)  # noqa: E731
    srcs = {"original": orig, **{k: RENDERS / f"{name}_{k}.mp4" for k in KINDS}}
    fixed_lam = plans["fixed"]["penalties"]["lam_switch"]
    for k, src in srcs.items():
        if (d / "compare.mp4").exists():
            break
        title = TITLES[k] + (f" (λ = {fixed_lam:g})" if k == "fixed" else "")
        ass = d / "logs" / f"overlay_{k}.ass"
        ass.write_text(overlay(plans.get(k), title, dur, film, store.track[r.pair_id], store.music_track, r.emotion))
        burn(src, ass, d / f"{k}.mp4")
    lst = d / "logs" / "concat.txt"
    lst.write_text("".join(f"file '{(d / f'{k}.mp4').resolve()}'\n" for k in ("original", "greedy", "fixed", "viterbi")))
    if not (d / "compare.mp4").exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        "-movflags", "+faststart", str(d / "compare.mp4")], check=True)
    sec = json.loads((d / "sections.json").read_text())["section2"]["costs"]
    hit = lambda k: any(store.music_track[s["music_id"]] == store.track[r.pair_id] for s in plans[k]["segments"])  # noqa: E731
    sw = lambda k: sum(a["music_id"] != b["music_id"] for a, b in zip(plans[k]["segments"], plans[k]["segments"][1:]))  # noqa: E731
    return {"folder": name, "emotion": r.emotion, "film": r.film, "fold": int(r.fold), "pair_id": r.pair_id,
            "duration_s": round(dur, 1), "segments": len(plans["viterbi"]["segments"]), "emo_score": round(float(r.emo_score), 2),
            "rank_all": int(r.rank_all), "rank_test": int(r.rank_test), "n_test": int(r.n_test),
            "switches_greedy": sw("greedy"), "switches_fixed": sw("fixed"), "switches_viterbi": sw("viterbi"),
            "fixed_lam": fixed_lam, "true_music_in_viterbi": hit("viterbi"),
            "cost_greedy": round(sec["greedy"]["total"], 2), "cost_fixed": round(sec["fixed"]["total"], 2),
            "cost_viterbi": round(sec["cue"]["total"], 2),
            "music_viterbi": " → ".join(dict.fromkeys(film(s["music_id"]) for s in plans["viterbi"]["segments"]))}


def main() -> None:
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    cfg = load_config("configs/default.yaml", BASE)
    store = FeatureStore(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    RENDERS.mkdir(parents=True, exist_ok=True)
    pk = picks()
    pk.to_csv(OUT / "picks.csv", index=False)
    rows = []
    for i, r in pk.iterrows():
        log.info("[%d/%d] %s %s", i + 1, len(pk), r.emotion, r.pair_id)
        rows.append(run_clip(i, r, store, cfg))
        pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    log.info("done: %d clips", len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    main()
