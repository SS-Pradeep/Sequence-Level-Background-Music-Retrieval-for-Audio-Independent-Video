"""Emotion variants of the long transition-showcase films (user request after D108): each held-out test film of
outputs/transition_showcase/ rendered once per emotion, with the learned transitions (R3s) on.

Same render as the showcase's learned version (cue_splice_demo: film split, the sequence's OWN fold runs, leak-free
long-video shortlist, R3s seeds 0-2), plus an emotion scene brief as emotion_variants (D106): every segment gets the
emotion and its stock music text, brief weight 2, pace match off. Features are reused from the showcase render.
Scores per render: switches / min and boundary F1 (+-1 s) against the film's true music changes (cue_demo.score),
energy = time-weighted library percentile of the chosen music (emotion_variants.piece_energy).

    python -m bgm.analysis.emotion_transitions
        -> outputs/transition_showcase/emotions/<NN_name>/<emotion>.mp4, index.md;  emotions/index.md
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bgm.analysis import cue_demo
from bgm.analysis.cue_splice_demo import BASE, R3S, finish
from bgm.analysis.emotion_variants import MUSIC, library_energy, piece_energy
from bgm.config import load_config
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
SRC = Path("outputs/cue_splice_demo/_work")
OUT = Path("outputs/transition_showcase/emotions")
WORK = Path("outputs/transition_showcase/_work")
FILMS = ["f2_same_film4_280", "f4_same_film2_padagotti", "f0_same_film17_370", "f0_same_film4_42", "f4_timeline2_139"]
EMO = ["render.scene_brief.weight=2.0", "render.pace_match.weight=0.0"]


def render(name: str, fold: int, emo: str, segs: list[tuple[float, float]]) -> dict:
    """One emotion render of showcase film `name` with its own fold's runs (leak-free shortlist, as cue_splice_demo)."""
    import bgm.analysis.long_video as lv
    from bgm.render.run import render_video
    tag = f"{name}_{emo}"
    out = WORK / f"{tag}.mp4"
    if out.exists() and (WORK / f"{tag}.plan.json").exists():
        return json.loads((WORK / f"{tag}.plan.json").read_text())
    work = WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    for f in ("query.npz", "silent25.mp4"):                                    # reuse the showcase render's features
        if not (work / f).exists():
            os.link(SRC / f"{name}_r3s" / f, work / f)
    brief = [{"start_s": round(a, 2), "end_s": round(b, 2), "scene": f"(emotion variant: {emo})", "emotion": emo, "music": MUSIC[emo]}
             for a, b in segs]
    bfile = WORK / f"{tag}.brief.yaml"
    bfile.write_text(f"# emotion variant '{emo}' for {name}\n" + yaml.safe_dump(brief, sort_keys=False))
    cfg = load_config("configs/default.yaml", BASE + R3S + EMO + [f"paths.renders={WORK}", f"render.scene_brief.file={bfile}"])
    orig = lv.load_models
    lv.load_models = lambda c, fold_=None, _k=fold: orig(c, fold=_k)          # leak-free long-video shortlist
    try:
        plan = render_video(cfg, str(SRC / f"{name}_input.mp4"), str(out), f"runs/filmsplit/pipeline_ens10/fold{fold}/seed0")
    finally:
        lv.load_models = orig
    return json.loads(Path(plan).read_text())


def main() -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config("configs/default.yaml", BASE)
    store = FeatureStore(cfg)
    rows = {r["name"]: r for r in json.loads((SRC / "demo.json").read_text())["rows"]}
    titles = pd.read_csv("outputs/film_titles.csv")
    title = dict(zip(titles.source + ":" + titles.source_id.astype(str), titles.title))
    pairs = pd.read_csv("data/manifests/pairs.csv")
    film_of = dict(zip(pairs.music_id, (pairs.meta_source + ":" + pairs.meta_film.astype(str)).map(title).fillna(pairs.meta_film)))
    lo, lr = library_energy()
    WORK.mkdir(parents=True, exist_ok=True)
    table = []
    for n, name in enumerate(FILMS, 1):
        r = rows[name]
        counts = cue_demo.build_input(cfg, r["parts"], SRC / f"{name}_input.mp4")
        starts = np.r_[0, np.cumsum(counts)[:-1]] / 25.0
        tracks = [store.track[p] for p in r["parts"]]
        truth = lambda t, s=starts, tr=tracks: tr[int(np.searchsorted(s, t, side="right") - 1)]  # noqa: E731
        with np.load(SRC / f"{name}_r3s" / "query.npz") as z:
            segs = [(float(a), float(b)) for a, b in z["segments"]]
        film = pairs.set_index("pair_id").loc[r["parts"][0]]
        film = title.get(f"{film.meta_source}:{film.meta_film}", str(film.meta_film))
        out = OUT / f"{n:02d}_{film.replace(' ', '_')}"
        out.mkdir(parents=True, exist_ok=True)
        for src, dst in ((Path("outputs/cue_splice_demo/original") / f"{name}.mp4", out / "00_original.mp4"),
                         (Path("outputs/cue_splice_demo/r3s") / f"{name}.mp4", out / "00_learned_no_emotion.mp4")):
            if not dst.exists():
                os.link(src, dst)
        lines = []
        for emo in MUSIC:
            final = out / f"{emo}.mp4"
            try:
                plan = render(name, r["fold"], emo, segs)
                if not final.exists():
                    finish(WORK / f"{name}_{emo}.mp4", final, Path(plan["bgm_wav"]))
                s = cue_demo.score(plan, r["true_changes"], truth)
                films = []
                for x in plan["pieces"]:
                    f = film_of.get(x["music_id"], "?")
                    if not films or films[-1] != f:
                        films.append(f)
                row = {"n": n, "film": film, "emotion": emo, "seconds": round(plan["duration_s"]), "switches": len(s["switches"]),
                       "switches_per_min": round(s["switches_per_min"], 1), "f1_1s": round(s["f1_1"], 2),
                       "energy_pct": round(piece_energy(plan, lo, lr)), "music_from": " → ".join(films)}
            except Exception as e:                                             # noqa: BLE001  skip and list
                log.error("%s %s failed: %r", name, emo, e)
                row = {"n": n, "film": film, "emotion": emo, "failed": repr(e)}
            table.append(row)
            lines.append(row)
            log.info("%s", row)
        df = pd.DataFrame(lines)
        md = [f"# {film}: one render per emotion", "",
              f"Held-out test sequence {name} (fold {r['fold']}, {round(r['duration_s'])} s, {len(r['true_changes'])} true music changes). "
              "Learned transitions (R3s) on; every segment's brief = the emotion + its music text (brief weight 2, pace off). "
              "`00_original.mp4` = the film's own music, `00_learned_no_emotion.mp4` = the showcase render without a brief. "
              "F1 = switches within ±1 s of a true music change; energy = library percentile of the chosen music.", "",
              df.to_markdown(index=False)]
        (out / "index.md").write_text("\n".join(md) + "\n")
    df = pd.DataFrame(table)
    df.to_csv(OUT / "index.csv", index=False)
    ok = df[df.get("failed").isna()] if "failed" in df else df
    piv = ok.pivot_table(index="emotion", values=["energy_pct", "switches_per_min", "f1_1s"], aggfunc="mean").round(2)
    piv = piv.loc[[e for e in MUSIC if e in piv.index]]
    md = ["# Emotion variants of the long transition-showcase films", "",
          f"{len(FILMS)} held-out test films (title-checked: no same-title clip in their fold's train / val) × {len(MUSIC)} emotions, "
          "learned transitions on. Per film: emotions/<NN_film>/index.md.", "",
          "## Mean over films", "", piv.to_markdown(), "",
          f"Renders: {len(ok)} done, {len(df) - len(ok)} failed."]
    (OUT / "index.md").write_text("\n".join(md) + "\n")
    return OUT / "index.md"


if __name__ == "__main__":
    print(main())
