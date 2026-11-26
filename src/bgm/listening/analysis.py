"""M12 analysis: win rates and 1-5 ratings from the completed rater sheets, with a two-way bootstrap over raters and
videos (SPEC §5.12). Run: python -m bgm.listening.analysis [--ratings kit/ratings.csv] -> kit/analysis.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.listening.kit import COMPARISONS, DIMENSIONS, KIT


def long_form(ratings: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pairwise rows: rater, video, comparison, ours_won), (rating rows: rater, video, condition, dimension, score)."""
    cond = {f"{r.video}_{r.letter}.mp4": r.condition for r in key.itertuples()}
    pw, rt = [], []
    for r in ratings.itertuples():
        cl, cr = cond[r.left], cond[r.right]
        pair = tuple(sorted((cl, cr), key=lambda c: c != "ours"))           # ("ours", other)
        winner = cl if str(r.preferred).strip().upper() == "L" else cr
        pw.append({"rater": r.rater, "video": r.video, "comparison": f"{pair[0]} vs {pair[1]}", "ours_won": float(winner == "ours")})
        for side, c in (("left", cl), ("right", cr)):
            for d in DIMENSIONS:
                v = getattr(r, f"{side}_{d}")
                if pd.notna(v) and str(v).strip() != "":
                    rt.append({"rater": r.rater, "video": r.video, "condition": c, "dimension": d, "score": float(v)})
    return pd.DataFrame(pw), pd.DataFrame(rt)


def cell_sums(df: pd.DataFrame, value: str, raters, videos) -> tuple[np.ndarray, np.ndarray]:
    """[R, V] sums of `value` and counts per (rater, video) cell."""
    ri = {r: i for i, r in enumerate(raters)}
    vi = {v: i for i, v in enumerate(videos)}
    S, N = np.zeros((len(raters), len(videos))), np.zeros((len(raters), len(videos)))
    for r, v, x in zip(df.rater, df.video, df[value]):
        S[ri[r], vi[v]] += x
        N[ri[r], vi[v]] += 1
    return S, N


def two_way_bootstrap(cells: list[tuple[np.ndarray, np.ndarray]], combine, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Crossed design: resample raters and videos independently with replacement. `cells` are (sums, counts) matrices;
    each statistic is combine([mean_1, mean_2, ...]) of cell-weighted means. Returns (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    R, V = cells[0][0].shape
    est = combine([S.sum() / max(N.sum(), 1e-12) for S, N in cells])
    boots = np.empty(n)
    for b in range(n):
        wr = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        wv = np.bincount(rng.integers(0, V, V), minlength=V).astype(float)
        boots[b] = combine([(wr @ S @ wv) / max(wr @ N @ wv, 1e-12) for S, N in cells])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def analyze(ratings_path: Path, key_path: Path = KIT / "key.csv", n_boot: int = 10000) -> str:
    ratings = pd.read_csv(ratings_path)
    ratings = ratings[ratings.preferred.astype(str).str.strip().str.upper().isin(["L", "R"])]
    key = pd.read_csv(key_path)
    pw, rt = long_form(ratings, key)
    L = ["# Listening study results", "", f"{ratings.rater.nunique()} raters, {ratings.video.nunique()} videos, {len(ratings)} completed "
         "trials. CIs: 95% two-way bootstrap over raters and videos (10,000 resamples).", "",
         "## Pairwise preference", "", "| comparison | ours preferred | 95% CI | trials |", "| --- | --- | --- | --- |"]
    raters, videos = sorted(ratings.rater.unique()), sorted(ratings.video.unique())
    for a, b in COMPARISONS:
        sub = pw[pw.comparison == f"{a} vs {b}"]
        if len(sub):
            e, lo, hi = two_way_bootstrap([cell_sums(sub, "ours_won", raters, videos)], lambda m: m[0], n_boot)
            L.append(f"| {a} vs {b} | {e:.1%} | [{lo:.1%}, {hi:.1%}] | {len(sub)} |")
    L += ["", "Above 50% with a CI excluding 50% means listeners prefer ours.", "", "## Ratings (1–5)", "",
          "| dimension | " + " | ".join(c for c in ("ours", "greedy", "baseline", "original")) + " | ours − greedy [95% CI] |",
          "| --- | --- | --- | --- | --- | --- |"]
    for d in DIMENSIONS:
        sub = rt[rt.dimension == d]
        means = [f"{sub[sub.condition == c].score.mean():.2f}" for c in ("ours", "greedy", "baseline", "original")]
        e, lo, hi = two_way_bootstrap([cell_sums(sub[sub.condition == "ours"], "score", raters, videos),
                                       cell_sums(sub[sub.condition == "greedy"], "score", raters, videos)], lambda m: m[0] - m[1], n_boot)
        L.append(f"| {d} | " + " | ".join(means) + f" | {e:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    text = "\n".join(L) + "\n"
    (KIT / "analysis.md").write_text(text)
    return text


def fake_ratings(key_path: Path = KIT / "key.csv", seed: int = 0) -> pd.DataFrame:
    """Dry-run data: every rater sheet filled with a known effect (ours better than greedy / baseline, worse than original)."""
    key = pd.read_csv(key_path)
    cond = {f"{r.video}_{r.letter}.mp4": r.condition for r in key.itertuples()}
    quality = {"original": 3.8, "ours": 3.4, "greedy": 3.0, "baseline": 2.7}
    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted((KIT / "sessions").glob("rater_*.csv")):
        s = pd.read_csv(f)
        bias = rng.normal(0, 0.3)
        for r in s.to_dict("records"):
            ql, qr = quality[cond[r["left"]]] + bias, quality[cond[r["right"]]] + bias
            r["preferred"] = "L" if ql + rng.normal(0, 0.6) > qr + rng.normal(0, 0.6) else "R"
            for d in DIMENSIONS:
                r[f"left_{d}"] = int(np.clip(round(ql + rng.normal(0, 0.8)), 1, 5))
                r[f"right_{d}"] = int(np.clip(round(qr + rng.normal(0, 0.8)), 1, 5))
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default=str(KIT / "ratings.csv"))
    ap.add_argument("--dry-run", action="store_true", help="fill the sheets with fake ratings first (tests the analysis)")
    a = ap.parse_args()
    if a.dry_run:
        fake_ratings().to_csv(KIT / "fake_ratings.csv", index=False)
        a.ratings = str(KIT / "fake_ratings.csv")
    print(analyze(Path(a.ratings)))
