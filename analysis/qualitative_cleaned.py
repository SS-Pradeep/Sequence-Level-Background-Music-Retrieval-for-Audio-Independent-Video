"""Failure table of the final system on the cleaned 1,011 pairs (title split, test folds, 10-head ensemble, run seed 0).

python -m bgm.analysis.qualitative_cleaned table -> outputs/qualitative_cleaned/{failures.csv,failure_stats.json}

Scores are the saved per-checkpoint cosine matrices (runs/cleaned/similarity/both); ranks are asserted equal to the saved
ensemble run (runs/cleaned/semantic_ens/siglip-both-ens10). No model is trained or tuned here.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ["BGM_SWEEP_DATA"] = "cleaned"

import numpy as np
import pandas as pd

from bgm.sweep import similarity as S

OUT = Path("outputs/qualitative_cleaned")
SEEDS = range(10)


def build_table() -> pd.DataFrame:
    """One row per test query (894): true-music rank in its fold's gallery, top-1 file, hub count, audit verdicts."""
    _, store, split, _ = S._ctx()
    pairs = pd.read_csv("cleaned_dataset/manifests/pairs.csv", dtype=str, keep_default_na=False)
    aud = pd.read_csv("results/cleaned/bgm_presence_files.csv").set_index("music_id")
    titles = pd.read_csv("outputs/film_titles.csv")
    key2title = dict(zip(titles.source + ":" + titles.source_id.astype(str), titles.title))   # per source film
    group2title = titles.groupby("film_group").title.first().to_dict()                        # fallback: its title group
    grp = split["film_group"]
    m2p, p2m = dict(zip(pairs.music_id, pairs.pair_id)), dict(zip(pairs.pair_id, pairs.music_id))
    trk = dict(zip(pairs.pair_id, pairs.track_id))
    dur = dict(zip(pairs.pair_id, pairs.video_duration_s.astype(float)))
    film = {p: key2title.get(f"{s}:{f}", group2title.get(grp.get(p), f)) for p, s, f in zip(pairs.pair_id, pairs.meta_source, pairs.meta_film)}
    rows = []
    for k in range(5):
        ids = split["folds"][str(k)]["test"]
        _, _, files = S._items(store, ids)
        g = S.gallery_masks(store, split, k, "test")
        M = np.mean([np.load(f"runs/cleaned/similarity/both/fold{k}/seed{s}.npz")["test_trk_mean"] for s in SEEDS], 0)
        r = S.ranks_track(M, g["trk_hits"])
        saved = np.load(f"runs/cleaned/semantic_ens/siglip-both-ens10/fold{k}/seed0/perquery.npz")["track_test_fold"]
        assert (r == saved).all(), k
        top = [files[i] for i in M.argmax(1)]
        hub = Counter(top)
        own = [sum(grp[x] == grp[q] for x in ids) / len(ids) for q in ids]
        for i, q in enumerate(ids):
            t = m2p[top[i]]
            rows.append(dict(fold=k, pair_id=q, film=film[q], film_group=grp[q], dur=dur[q], rank=int(r[i]), N=len(files),
                             top1=t, top1_film=film[t], top1_own_film=grp[t] == grp[q],
                             same_film=grp[t] == grp[q] and trk[t] != trk[q], hub_n=hub[top[i]], chance_own_film=own[i],
                             verdict=aud.loc[p2m[q], "verdict"], bgm_share=aud.loc[p2m[q], "bgm_share"],
                             top1_verdict=aud.loc[p2m[t], "verdict"], top1_share=aud.loc[p2m[t], "bgm_share"]))
    df = pd.DataFrame(rows)
    df["hit10"], df["hit1"] = df["rank"] <= 10, df["rank"] == 1
    return df


def stats(df: pd.DataFrame) -> dict:
    miss = df[~df.hit10]
    out = {"n": len(df), "R1": df.hit1.mean(), "R10": df.hit10.mean(), "MedR": df["rank"].median(), "misses": len(miss),
           "top1_own_film": df.top1_own_film.mean(), "chance_own_film": df.chance_own_film.mean(),
           "miss_same_film_n": int(miss.same_film.sum()), "miss_same_film": miss.same_film.mean(),
           "hub5": (df.hub_n >= 5).mean(), "hub10": (df.hub_n >= 10).mean(),
           "gallery_used": df.groupby("fold").top1.nunique().sum() / df.groupby("fold").N.first().sum(),
           "top5_share": np.mean([d.top1.value_counts().head(5).sum() / len(d) for _, d in df.groupby("fold")]),
           "r10_by_verdict": {v: [len(g), g.hit10.mean()] for v, g in df.groupby("verdict")},
           "top1_verdict_share": df.top1_verdict.value_counts(normalize=True).to_dict(),
           "gallery_verdict_share": df.verdict.value_counts(normalize=True).to_dict(),
           "r10_by_len": {f"{a}-{b}": [int(((df.dur >= a) & (df.dur < b)).sum()), df[(df.dur >= a) & (df.dur < b)].hit10.mean()]
                          for a, b in ((0, 12), (12, 20), (20, 10**6))}}
    return json.loads(json.dumps(out, default=float))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUT / "failures.csv", index=False)
    (OUT / "failure_stats.json").write_text(json.dumps(stats(table), indent=1))
