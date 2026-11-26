"""`bgm folds` (SPEC §3.3): 5 folds over pairs, grouped by track_id, stratified by duration quartile."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def make_folds(pairs: pd.DataFrame, n_splits: int = 5, val_fraction: float = 0.1, seed: int = 0) -> dict[str, dict[str, list[str]]]:
    """pairs needs pair_id, track_id, video_duration_s. Test = fold k; the rest splits into train / val with
    GroupShuffleSplit(test_size=val_fraction, random_state=k), grouped by track_id."""
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    ids, groups = pairs["pair_id"].to_numpy(), pairs["track_id"].to_numpy()
    y = pd.qcut(pairs["video_duration_s"], 4, labels=False, duplicates="drop").to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for k, (rest, test) in enumerate(sgkf.split(ids, y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=k)
        tr, va = next(gss.split(rest, groups=groups[rest]))
        out[str(k)] = {"train": sorted(ids[rest[tr]]), "val": sorted(ids[rest[va]]), "test": sorted(ids[test])}
    return out


def make_film_folds(pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str], n_splits: int = 5,
                    val_fraction: float = 0.1, seed: int = 0) -> dict[str, dict[str, list[str]]]:
    """Film-level split (PROGRESS.md D36). Pairs in `train_only` film groups go to every fold's train set and never to
    val or test. The rest: StratifiedGroupKFold over film groups, y = duration quartile (of those pairs); val =
    GroupShuffleSplit(test_size=val_fraction, random_state=k) of the non-test film groups."""
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    fixed = pairs["pair_id"].map(film_group).isin(train_only).to_numpy()
    always = sorted(pairs.loc[fixed, "pair_id"])
    rest = pairs.loc[~fixed].reset_index(drop=True)
    ids, groups = rest["pair_id"].to_numpy(), rest["pair_id"].map(film_group).to_numpy()
    y = pd.qcut(rest["video_duration_s"], 4, labels=False, duplicates="drop").to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for k, (tv, test) in enumerate(sgkf.split(ids, y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=k)
        tr, va = next(gss.split(tv, groups=groups[tv]))
        out[str(k)] = {"train": sorted(list(ids[tv[tr]]) + always), "val": sorted(ids[tv[va]]), "test": sorted(ids[test])}
    return out


def extend_film_folds(base: dict[str, dict[str, list[str]]], pairs: pd.DataFrame, film_group: dict[str, str],
                      train_only: set[str], n_splits: int = 5, seed: int = 0, moved: dict | None = None) -> dict[str, dict[str, list[str]]]:
    """Add pairs to an existing film split, moving base pairs only where the film rule forces it (PROGRESS.md D49).

    A film group holding pairs of `base` keeps their assignment in every fold, and its new pairs follow it. When new
    shared tracks join base pairs that sat in different splits (the same film under several source keys, revealed
    by the new clips), the merged group is tested in one fold, the one holding most of its base test pairs (ties: the
    lowest fold), and is in train in every other fold; the base pairs this moves are recorded in `moved`
    ({pair_id: (old assignment, new assignment)}) so comparisons with the base split can leave them out.
    New-only film groups: StratifiedGroupKFold(n_splits, shuffle, seed) over those groups, y = duration quartile of
    their pairs -> the test set of one fold and the train set of the others; they never go to val (new pairs of a group
    with base pairs in val follow them there). New pairs of `train_only` groups go to every train set."""
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    keys = sorted(base, key=int)
    base_ids = set().union(*(base[keys[0]][s] for s in ("train", "val", "test")))
    assert base_ids <= set(pairs["pair_id"]), "every pair of the base split must be in the new manifest"
    where = {k: {p: s for s in ("train", "val", "test") for p in base[k][s]} for k in keys}
    patterns: dict[str, dict[str, tuple[str, ...]]] = {}
    for p in sorted(base_ids):
        patterns.setdefault(film_group[p], {})[p] = tuple(where[k][p] for k in keys)
    inherit: dict[str, tuple[str, ...]] = {}
    moved = {} if moved is None else moved
    for g, pats in patterns.items():
        if len(set(pats.values())) == 1:
            inherit[g] = next(iter(pats.values()))
            continue
        assert g not in train_only, f"film group {g} joins a train-only film with base pairs outside train"
        n_test = [sum(a[i] == "test" for a in pats.values()) for i in range(len(keys))]
        t = int(np.argmax(n_test))
        inherit[g] = tuple("test" if i == t else "train" for i in range(len(keys)))
        moved.update({p: (a, inherit[g]) for p, a in pats.items() if a != inherit[g]})
    out = {k: {s: [p for p in base[k][s] if p not in moved] for s in ("train", "val", "test")} for k in keys}
    for p, (_, a) in moved.items():
        for k, s in zip(keys, a):
            out[k][s].append(p)
    new = pairs[~pairs["pair_id"].isin(base_ids)].reset_index(drop=True)
    grp = new["pair_id"].map(film_group)
    for p, g in zip(new["pair_id"], grp):
        if g in inherit:
            for k, s in zip(keys, inherit[g]):
                out[k][s].append(p)
        elif g in train_only:
            for k in keys:
                out[k]["train"].append(p)
    rest = new[~grp.isin(set(inherit) | set(train_only))].reset_index(drop=True)
    if len(rest):
        ids, groups = rest["pair_id"].to_numpy(), rest["pair_id"].map(film_group).to_numpy()
        y = pd.qcut(rest["video_duration_s"], 4, labels=False, duplicates="drop").to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, test) in zip(keys, sgkf.split(ids, y, groups)):
            in_test = set(ids[test])
            out[k]["test"] += sorted(in_test)
            for j in keys:
                if j != k:
                    out[j]["train"] += sorted(in_test)
    return {k: {s: sorted(f[s]) for s in ("train", "val", "test")} for k, f in out.items()}


def check_folds(folds: dict[str, dict[str, list[str]]], pairs: pd.DataFrame, film_group: dict[str, str] | None = None,
                train_only: set[str] = frozenset()) -> None:
    """SPEC M1 acceptance: every pair in exactly one test fold; no track_id in more than one fold's test set;
    within each fold, train / val / test share no track_id and cover every pair.
    Film split (film_group given): the same checks for film groups, which contain track groups; pairs of
    `train_only` film groups are in every fold's train set and in no test fold."""
    all_ids = set(pairs["pair_id"])
    fixed = {p for p in all_ids if film_group is not None and film_group[p] in train_only}
    tests = [set(f["test"]) for f in folds.values()]
    assert sum(len(t) for t in tests) == len(all_ids - fixed) and set().union(*tests) == all_ids - fixed, \
        "test folds must partition the pairs (minus train-only films)"
    groupings = {"track_id": dict(zip(pairs["pair_id"], pairs["track_id"]))}
    if film_group is not None:
        groupings["film group"] = film_group
    for name, grp in groupings.items():
        test_groups = [{grp[p] for p in t} for t in tests]
        for a in range(len(test_groups)):
            for b in range(a + 1, len(test_groups)):
                assert not test_groups[a] & test_groups[b], f"a {name} is in the test sets of folds {a} and {b}"
        for k, f in folds.items():
            parts = {s: {grp[p] for p in f[s]} for s in ("train", "val", "test")}
            assert not parts["train"] & parts["val"] and not parts["train"] & parts["test"] and not parts["val"] & parts["test"], \
                f"fold {k}: splits share a {name}"
    for k, f in folds.items():
        assert set(f["train"]) | set(f["val"]) | set(f["test"]) == all_ids, f"fold {k} does not cover every pair"
        assert fixed <= set(f["train"]), f"fold {k}: a train-only pair is outside train"


def folds_document(folds: dict, manifest_hash: str, **extra) -> dict:
    """extra (film split): film_group {pair_id: group}, train_only [groups], split = "film"."""
    return {"folds": folds, "manifest_hash": manifest_hash, "created": dt.datetime.now().isoformat(timespec="seconds"), **extra}


def fold_summary(folds: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    dur = dict(zip(pairs["pair_id"], pairs["video_duration_s"]))
    return pd.DataFrame([{"fold": k, **{s: len(f[s]) for s in ("train", "val", "test")},
                          "test_median_dur_s": float(np.median([dur[p] for p in f["test"]]))} for k, f in folds.items()])
