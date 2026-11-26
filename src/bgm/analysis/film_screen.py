"""Screen the M8 unlabeled pool for labeled films (PROGRESS.md D39).

With the film split, every labeled film is a test film in some fold, so pretraining must not see any of them.
The pool's source videos have no titles, so the screen uses SigLIP:

* Unit score (source video U vs labeled film group F): the max cosine between the clip-mean SigLIP vectors of U's
  clips and F's clips. A source is flagged when its best film scores >= t.
* Calibration on the labeled set, at the same unit size: each film group with >= 4 clips is split at random into
  half A (<= 6 clips, like a pool source) and half B. Same-film score: A vs B. Null: A vs its best OTHER film group.
  t = the `null_q` quantile of the null, i.e. about (1 - null_q) of pool sources with no labeled film get flagged.
* The same threshold lists labeled film groups that look like one film under two keys (candidates for
  `splits.film.extra_links`).

Writes reports/film_screen.md, reports/film_screen.json and contact sheets in reports/film_screen/.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.data.films import film_groups, film_keys
from bgm.data.prep import interim_video_path
from bgm.features.extract import feature_path, load_features
from bgm.pretrain.unlabeled import unlabeled_config

log = logging.getLogger(__name__)
UNIT_MAX = 6


def clip_means(cfg: DictConfig, ids) -> np.ndarray:
    """[N, 1152] L2-normalized clip means of sem_siglip_1hz."""
    X = np.stack([load_features(feature_path(cfg, "video", i))["sem_siglip_1hz"].astype(np.float32).mean(0) for i in ids])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def calibrate(S: np.ndarray, grp: np.ndarray, rng: np.random.Generator, n_rep: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Same-film and null unit scores from half splits of the labeled film groups (n_rep random splits each).
    S: [N, N] clip cosine; grp: [N] film group per clip."""
    groups = {k: np.flatnonzero(grp == k) for k in np.unique(grp)}
    pos, null = [], []
    for _ in range(n_rep):
        for k, ix in groups.items():
            if len(ix) < 4:
                continue
            ix = rng.permutation(ix)
            A, B = ix[:min(UNIT_MAX, len(ix) // 2)], ix[len(ix) // 2:]
            pos.append(S[np.ix_(A, B)].max())
            other = grp != k
            null.append(S[A][:, other].max())
    return np.array(pos), np.array(null)


def _frame(video: Path, t: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=-2:180", str(out)], check=True)
    return out


def pair_sheet(pool_videos: list[Path], lab_videos: list[Path], title: str, out_png: Path, tmp: Path) -> Path:
    """Two rows of middle frames: pool clips (top) and the matched labeled clips (bottom)."""
    from PIL import Image, ImageDraw
    rows = []
    for vids in (pool_videos, lab_videos):
        ims = []
        for j, v in enumerate(vids):
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                                       capture_output=True, text=True, check=True).stdout.strip() or 0)
            ims.append(Image.open(_frame(v, dur / 2, tmp / f"{out_png.stem}_{len(rows)}_{j}.png")).convert("RGB"))
        rows.append(ims)
    w = max(sum(i.width for i in r) for r in rows)
    sheet = Image.new("RGB", (w, 24 + sum(max(i.height for i in r) for r in rows)), "white")
    ImageDraw.Draw(sheet).text((4, 4), title, fill="black")
    y = 24
    for r in rows:
        x = 0
        for im in r:
            sheet.paste(im, (x, y))
            x += im.width
        y += max(i.height for i in r)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def run_film_screen(cfg: DictConfig, null_q: float = 0.95, n_examples: int = 20, seed: int = 0) -> dict:
    pairs = manifest.read_pairs(cfg)
    groups = film_groups(pairs, [tuple(x) for x in cfg.splits.film.extra_links])
    grp = np.array([groups[p] for p in pairs.pair_id])
    L = clip_means(cfg, pairs.pair_id)
    S = L @ L.T
    pos, null = calibrate(S, grp, np.random.default_rng(seed))
    t = float(np.quantile(null, null_q))
    sens = float(np.mean(pos >= t))
    log.info("calibration: %d same-film / %d null units; t = %.3f (null q %.2f), same-film detected %.2f", len(pos), len(null), t, null_q, sens)

    # labeled film groups that look like one film (clip pairs across groups >= t)
    keys = film_keys(pairs)
    iu = np.triu_indices(len(grp), 1)
    cross = (grp[iu[0]] != grp[iu[1]]) & (S[iu] >= t)
    lab_links = pd.DataFrame({"a": pairs.pair_id.to_numpy()[iu[0][cross]], "b": pairs.pair_id.to_numpy()[iu[1][cross]], "sim": S[iu][cross]})
    lab_links["group_a"], lab_links["group_b"] = lab_links.a.map(groups), lab_links.b.map(groups)
    lab_group_pairs = (lab_links.groupby(["group_a", "group_b"]).agg(n_clip_pairs=("sim", "size"), max_sim=("sim", "max"))
                       .reset_index().sort_values("max_sim", ascending=False))

    # the pool
    ucfg = unlabeled_config(cfg)
    upairs = manifest.read_pairs(ucfg)
    U = clip_means(ucfg, upairs.pair_id)
    SU = U @ L.T                                                   # [pool clips, labeled clips]
    g_names = np.unique(grp)
    gidx = np.searchsorted(g_names, grp)
    best_per_group = np.full((len(U), len(g_names)), -1.0, np.float32)
    np.maximum.at(best_per_group.T, gidx, SU.T)                     # [pool clips, groups]: max over the group's clips
    src = upairs.track_id.to_numpy()
    rows = []
    for s in np.unique(src):
        ix = np.flatnonzero(src == s)
        per_group = best_per_group[ix].max(0)
        g = int(np.argmax(per_group))
        ci = ix[int(np.argmax(best_per_group[ix, g]))]
        lj = np.flatnonzero(gidx == g)[int(np.argmax(SU[ci, gidx == g]))]
        rows.append({"source": s, "n_clips": len(ix), "score": float(per_group[g]), "film_group": g_names[g],
                     "n_clips_ge_t": int((best_per_group[ix].max(1) >= t).sum()), "pool_clip": upairs.pair_id.iloc[ci],
                     "labeled_clip": pairs.pair_id.iloc[lj], "second_film_score": float(np.sort(per_group)[-2])})
    src_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    src_df["flag"] = src_df.score >= t
    flagged = src_df[src_df.flag]

    # contact sheets: examples spread over the flagged score range, then the closest unflagged sources (>= 4)
    out_dir = Path(cfg.paths.reports) / "film_screen"
    tmp = out_dir / "_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    n_flag = min(n_examples - 4, len(flagged))
    pick = flagged.iloc[np.unique(np.linspace(0, len(flagged) - 1, n_flag).astype(int))] if n_flag else flagged.iloc[:0]
    below = src_df[~src_df.flag].head(n_examples - len(pick))
    examples = []
    for tag, part in (("flagged", pick), ("below_t", below)):
        for r in part.itertuples():
            pool_ids = upairs.pair_id[src == r.source].tolist()[:4]
            lab_ids = pairs.pair_id[grp == r.film_group].tolist()
            lab_ids = [r.labeled_clip] + [x for x in lab_ids if x != r.labeled_clip][:3]
            png = pair_sheet([interim_video_path(ucfg, x) for x in pool_ids], [interim_video_path(cfg, x) for x in lab_ids],
                             f"{tag}: pool {r.source} vs {r.film_group}  score {r.score:.3f} (t = {t:.3f})",
                             out_dir / f"{tag}_{r.source}.png", tmp)
            examples.append({"tag": tag, "source": r.source, "film_group": r.film_group, "score": r.score, "png": str(png)})
    if examples:
        from PIL import Image
        ims = [Image.open(e["png"]) for e in examples]
        sheet = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
        y = 0
        for im in ims:
            sheet.paste(im, (0, y))
            y += im.height
        sheet.save(out_dir / "examples.png")
    res = {"threshold": t, "null_q": null_q, "calibration": {"n_same_film_units": len(pos), "n_null_units": len(null),
                                                            "same_film_detected": sens, "pos_quantiles": np.quantile(pos, [.1, .25, .5, .75, .9]).tolist(),
                                                            "null_quantiles": np.quantile(null, [.5, .9, .95, .99]).tolist()},
           "pool_sources": len(src_df), "flagged_sources": int(len(flagged)), "flagged_clips": int(flagged.n_clips.sum()),
           "flagged_film_groups": int(flagged.film_group.nunique()), "flagged_list": flagged.to_dict("records"),
           "labeled_group_pairs_ge_t": lab_group_pairs.to_dict("records"), "examples": examples}
    Path(cfg.paths.reports, "film_screen.json").write_text(json.dumps(res, indent=1, default=float))
    src_df.to_csv(Path(cfg.paths.reports) / "film_screen_sources.csv", index=False)
    _write_md(cfg, res, src_df, keys)
    return res


def _write_md(cfg, res, src_df, keys) -> None:
    c = res["calibration"]
    L = ["# Screening the M8 pool for labeled films (SigLIP)", "",
         f"Threshold t = {res['threshold']:.3f}: the {res['null_q']:.2f} quantile of the null (a ≤ 6-clip half of a labeled film vs "
         f"its best other film group; {c['n_null_units']} units). At t, {c['same_film_detected']:.0%} of same-film halves are detected "
         "(these include adjacent clips of one scene, so real song-vs-BGM matches will be detected less often).", "",
         f"- Pool: {res['pool_sources']} source videos. **Flagged: {res['flagged_sources']} sources ({res['flagged_clips']} clips), "
         f"matching {res['flagged_film_groups']} labeled film groups.**",
         f"- Expected false flags at this threshold: about {(1 - res['null_q']) * res['pool_sources']:.0f} sources, if the pool held no labeled film.",
         "", "| score quantile | 10% | 25% | 50% | 75% | 90% |", "| --- | --- | --- | --- | --- | --- |",
         "| same-film halves | " + " | ".join(f"{x:.3f}" for x in c["pos_quantiles"]) + " |",
         "", "| null quantile | 50% | 90% | 95% | 99% |", "| --- | --- | --- | --- | --- |",
         "| best other film | " + " | ".join(f"{x:.3f}" for x in c["null_quantiles"]) + " |", "",
         "## Flagged sources by labeled film group", "", "| film group | sources | best score |", "| --- | --- | --- |"]
    f = src_df[src_df.flag].groupby("film_group").agg(n=("source", "size"), best=("score", "max")).sort_values("n", ascending=False)
    L += [f"| {g} | {r.n} | {r.best:.3f} |" for g, r in f.iterrows()]
    L += ["", "## Labeled film groups with a clip pair ≥ t (possible same film under two keys)", "",
          "| group A | group B | clip pairs ≥ t | max |", "| --- | --- | --- | --- |"]
    L += [f"| {r['group_a']} | {r['group_b']} | {r['n_clip_pairs']} | {r['max_sim']:.3f} |" for r in res["labeled_group_pairs_ge_t"][:40]]
    L += ["", "## Example contact sheets (top row: pool clips; bottom row: the matched labeled film's clips)", ""]
    L += [f"- {e['tag']}: {e['source']} vs {e['film_group']} ({e['score']:.3f}): {e['png']}" for e in res["examples"]]
    Path(cfg.paths.reports, "film_screen.md").write_text("\n".join(L) + "\n")
