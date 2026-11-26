"""Same film under two names (PROGRESS.md D49): old clips (numeric SunNXT / YouTube keys) that show one of the 32 new
films (named keys), before the extended film split is frozen.

Run: python -m bgm.analysis.full_film_screen   -> reports/full/film_screen.md, reports/full/film_screen/*.png

Score s(c, F) = max over the new film F's clips of the SigLIP clip-mean cosine with old clip c. The new films hold
~95 clips each, so D40's threshold (calibrated on <= 6-clip units) does not transfer. Calibration instead uses the
old clips that shared audio with a new film (fingerprint links, already one film group): their score for the true
film vs for the 31 others. A flag needs s(c, F) >= t and F to be c's best new film by a margin; every flag gets a
contact sheet (the old clip's middle frame above the new film's three nearest clips) to be checked by eye.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bgm.analysis.film_screen import clip_means, pair_sheet
from bgm.analysis.full_dataset import BASE_FOLDS, full_cfg
from bgm.data import manifest
from bgm.data.films import film_groups
from bgm.data.prep import interim_video_path

log = logging.getLogger(__name__)


def scores(cfg: DictConfig) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """[old clips x new films] s(c, F), the old clip ids, the new film keys and {film key: new clip ids}."""
    pairs = manifest.read_pairs(cfg)
    old = set(json.loads(BASE_FOLDS.read_text())["film_group"])
    key = dict(zip(pairs.pair_id, pairs.meta_source + ":" + pairs.meta_film))
    old_ids = sorted(p for p in pairs.pair_id if p in old)
    new_by_film: dict[str, list[str]] = {}
    for p in sorted(set(pairs.pair_id) - old):
        new_by_film.setdefault(key[p], []).append(p)
    films = sorted(new_by_film)
    Xo = clip_means(cfg, old_ids)
    S = np.zeros((len(old_ids), len(films)), np.float32)
    for j, f in enumerate(films):
        S[:, j] = (Xo @ clip_means(cfg, new_by_film[f]).T).max(1)
    return pairs, S, old_ids, {"films": films, "new_by_film": new_by_film, "key": key}


def run(cfg: DictConfig, margin: float = 0.01) -> dict:
    pairs, S, old_ids, d = scores(cfg)
    films, key = d["films"], d["key"]
    g = film_groups(pairs)
    # positives: old clips whose film group (by shared audio) contains a new film
    film_of_group = {}
    for f in films:
        film_of_group.setdefault(g[d["new_by_film"][f][0]], set()).add(f)
    pos = [(i, films.index(next(iter(film_of_group[g[c]])))) for i, c in enumerate(old_ids)
           if g[c] in film_of_group and key[c] != "full_bgm_dataset:Konjum salangai"]
    pos_true = np.array([S[i, j] for i, j in pos])
    pos_other = np.array([np.delete(S[i], j).max() for i, j in pos])
    known = {i for i, _ in pos}
    rest = np.array([i for i in range(len(old_ids)) if i not in known])
    best = S.argmax(1)
    srt = np.sort(S, 1)
    gap = srt[:, -1] - srt[:, -2]
    # threshold: the lowest score that no non-positive old clip reaches more than 1% of the time for its best film
    null_best = S[rest, best[rest]]
    t = float(max(np.quantile(null_best, 0.99), np.median(pos_true) - 0.02 if len(pos_true) else 0.0))
    flags = [i for i in rest if S[i, best[i]] >= t and gap[i] >= margin]
    rows = [{"old_clip": old_ids[i], "old_key": key[old_ids[i]], "new_film": films[best[i]], "score": float(S[i, best[i]]),
             "gap_to_next_film": float(gap[i])} for i in flags]
    res = {"threshold": t, "margin": margin, "positives": {"n": len(pos), "true_film_score_quantiles": np.quantile(pos_true, [0, .25, .5, .75, 1]).tolist() if len(pos) else [],
                                                         "best_other_film_quantiles": np.quantile(pos_other, [0, .25, .5, .75, 1]).tolist() if len(pos) else [],
                                                         "true_film_ranked_first": float(np.mean(pos_true > pos_other)) if len(pos) else None},
           "null_best_quantiles": np.quantile(null_best, [.5, .9, .99, .999, 1]).tolist(), "flags": rows}
    out = Path(cfg.paths.reports) / "film_screen"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            nf = d["new_by_film"][r["new_film"]]
            c = old_ids.index(r["old_clip"])
            Xc = clip_means(cfg, [r["old_clip"]])[0]
            near = [nf[k] for k in np.argsort(-(clip_means(cfg, nf) @ Xc))[:3]]
            r["sheet"] = str(pair_sheet([interim_video_path(cfg, r["old_clip"])], [interim_video_path(cfg, p) for p in near],
                                        f"{r['old_key']} vs {r['new_film']}: {r['score']:.3f} (gap {r['gap_to_next_film']:.3f})",
                                        out / f"{c:04d}_{r['new_film'].split(':')[1]}.png", Path(tmp)))
    _write_md(cfg, res, len(old_ids), len(films))
    (Path(cfg.paths.reports) / "film_screen.json").write_text(json.dumps(res, indent=1))
    return res


def _write_md(cfg: DictConfig, res: dict, n_old: int, n_films: int) -> None:
    p = res["positives"]
    L = ["# Same film under two names: old clips vs the new films (SigLIP)", "",
         f"s(c, F) = max over film F's clips of the SigLIP clip-mean cosine with old clip c; {n_old} old clips × {n_films} new films.", "",
         f"- Calibration: {p['n']} old clips already joined to a new film by shared audio. Score for the true film "
         f"(min / 25% / median / 75% / max): {', '.join(f'{x:.3f}' for x in p['true_film_score_quantiles'])}; best other film: "
         f"{', '.join(f'{x:.3f}' for x in p['best_other_film_quantiles'])}; true film ranked first: {p['true_film_ranked_first']:.0%}.",
         f"- Other old clips, score for their best new film (50 / 90 / 99 / 99.9% / max): {', '.join(f'{x:.3f}' for x in res['null_best_quantiles'])}.",
         f"- Flag: score ≥ t = {res['threshold']:.3f} and ≥ {res['margin']:.2f} above the next film. **{len(res['flags'])} flagged** "
         "(each with a contact sheet; confirmed ones become `splits.film.extra_links`).", "",
         "| old clip | old key | new film | score | gap | sheet |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r['old_clip']} | {r['old_key']} | {r['new_film']} | {r['score']:.3f} | {r['gap_to_next_film']:.3f} | {Path(r['sheet']).name} |"
          for r in sorted(res["flags"], key=lambda r: -r["score"])]
    (Path(cfg.paths.reports) / "film_screen.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps({k: v for k, v in run(full_cfg()).items() if k != "flags"}, indent=1))
