"""Film groups for the film-level split (PROGRESS.md D36).

Folds grouped by track_id leak by film: neighbouring clips of one scene carry different music files (so different
track_ids) but near-identical pictures, and land in train and test. A film key is `meta_source:meta_film`; film keys
that share a track_id (the same music under two source names) merge into one film group, so every track group lies
inside one film group.
"""
from __future__ import annotations

import pandas as pd

from bgm.data.tracks import union_find


def film_keys(pairs: pd.DataFrame) -> pd.Series:
    """pair_id-indexed `meta_source:meta_film`."""
    return pd.Series((pairs["meta_source"].astype(str) + ":" + pairs["meta_film"].astype(str)).to_numpy(), index=pairs["pair_id"].to_numpy())


def film_groups(pairs: pd.DataFrame, extra_links: list[tuple[str, str]] = ()) -> dict[str, str]:
    """{pair_id: film group}. Film keys are merged when they share a track_id, and along `extra_links` (pairs of film
    keys judged to be one film). A group is named after its alphabetically first film key."""
    keys = film_keys(pairs)
    names = sorted(set(keys))
    idx = {k: i for i, k in enumerate(names)}
    links = []
    for _, g in pd.DataFrame({"key": keys.to_numpy(), "track": pairs["track_id"].to_numpy()}).groupby("track"):
        ks = sorted(set(g["key"]))
        links += [(idx[ks[0]], idx[k]) for k in ks[1:]]
    links += [(idx[a], idx[b]) for a, b in extra_links]
    comp = union_find(len(names), links)
    return {p: names[comp[idx[k]]] for p, k in keys.items()}
