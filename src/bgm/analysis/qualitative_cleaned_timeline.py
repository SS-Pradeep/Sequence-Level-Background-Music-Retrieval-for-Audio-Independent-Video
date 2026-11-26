"""Timeline figure (worked example) on the cleaned 1,011 pairs, using bgm.analysis.worked_example unchanged.

python -m bgm.analysis.qualitative_cleaned_timeline -> outputs/qualitative_cleaned/timeline/

Case rule, fixed before any decoder was run on it: among the frozen blocked test sequences of one film with >= 3 clips and
<= 125 s, the one whose worst clip-level true-music rank is lowest. Result: fold 1, sequence 4 (Sivappu Manjal Pachai).
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.axes import Axes

from bgm.analysis import worked_example as we

FILM = "Sivappu Manjal Pachai"
we.OUT = Path("outputs/qualitative_cleaned/timeline")
we.FOLD, we.CASE_IDX = 1, 4
we.RUN = Path("runs/cleaned/pipeline_ens10/fold1/seed0")
we.OVERRIDES = ["splits.folds_file=folds_title.json", "eval.concat_sets_file=concat_sets_title.json",
                "train.semantic.music_feature=both", "train.semantic.max_clips_per_film_per_epoch=30",
                "paths.runs=runs/cleaned", "paths.manifests=cleaned_dataset/manifests", "paths.interim=cleaned_dataset/interim",
                "paths.features=cleaned_dataset/features", "paths.results=results/cleaned", "paths.reports=reports/cleaned",
                "pipeline.dyn_variant=null", "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]",
                "cue.sequences_file=cue_sequences_title.json", "cue.enabled=true"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0", "#d6457f"]


def load_case() -> dict:
    e = json.loads(Path("runs/cleaned/cue/eval/fold1/eval.json").read_text())
    return next(c for c in e["cases"] if c["split"] == "test" and c["mode"] == "blocked" and c["idx"] == we.CASE_IDX)


def tags_for(ex: dict) -> tuple[dict, dict]:
    """Track -> (label, colour): true songs S1.. in order of first appearance (a song may return), others a, b, .."""
    songs = list(dict.fromkeys(c["track_id"] for c in ex["case"]["clips"]))
    tag = {t: (f"S{i + 1}", COLORS[i]) for i, t in enumerate(songs)}
    n, files = 0, {}
    for name in ("greedy", "viterbi_fixed", "final"):
        for s in ex["paths"][name]["segments"]:
            if s["track_id"] not in tag:
                tag[s["track_id"]] = (chr(ord("a") + n), we.OTHER)
                n += 1
            files.setdefault(tag[s["track_id"]][0], s["music_id"])
    return tag, files


def build_silent_input(parts: list[str]) -> None:
    """Join the clips at 640x360 (the two sources are 640 and 638 px wide; concat needs one size). Frame counts are kept."""
    inp = we.OUT / "input_silent.mp4"
    if inp.exists():
        return
    we.OUT.mkdir(parents=True, exist_ok=True)
    ins = sum((["-i", f"cleaned_dataset/interim/video/{p}.mp4"] for p in parts), [])
    chain = "".join(f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
                    for i in range(len(parts)))
    graph = chain + "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    we.sh(*ins, "-filter_complex", graph, "-map", "[v]", "-r", str(we.FPS), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(inp))
    assert we.n_frames(inp) == sum(we.n_frames(Path(f"cleaned_dataset/interim/video/{p}.mp4")) for p in parts)


def main() -> None:
    draw, we.load_case, we.tags_for = we.figure, load_case, tags_for
    build_silent_input(load_case()["parts"])
    we.figure = lambda ex=None: None
    we.main()
    p = we.OUT / "example.json"
    ex = json.loads(p.read_text())
    ex["case"]["film"] = f"{FILM} (title group full_bgm_dataset_youtube:128)"
    p.write_text(json.dumps(ex, indent=1))
    n, k = len(ex["case"]["clips"]), len({c["track_id"] for c in ex["case"]["clips"]})
    title, text = Axes.set_title, Axes.text

    def set_title(self, label, *a, **kw):
        if str(label).startswith("Held-out"):
            label = f"Held-out test case: {n} clips of one film ({FILM}), {k} different songs, joined into one {ex['case']['duration_s']:.0f} s silent video"
        return title(self, label, *a, **kw)

    def new_text(self, x, y, s, *a, **kw):
        return text(self, x, y, str(s).replace("S1–S4", f"S1–S{k}"), *a, **kw)

    Axes.set_title, Axes.text = set_title, new_text
    try:
        draw(ex)
    finally:
        Axes.set_title, Axes.text = title, text


if __name__ == "__main__":
    main()
