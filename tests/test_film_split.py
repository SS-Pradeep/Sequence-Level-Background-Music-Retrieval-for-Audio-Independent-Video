"""Film-level split (PROGRESS.md D36): film groups, folds, the per-film epoch cap and nested film subsets.
Synthetic manifests only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bgm.data.films import film_groups
from bgm.data.folds import check_folds, make_film_folds
from bgm.train.semantic import cap_per_film, epoch_batches, film_subset


def synthetic_film_pairs(n_films=120, seed=0) -> pd.DataFrame:
    """Films of 1-15 clips; one big film (the Konjum stand-in); some tracks reused across two film keys."""
    rng = np.random.default_rng(seed)
    rows, t = [], 0
    for f in range(n_films):
        n = 60 if f == 0 else int(rng.integers(1, 16))
        src = ["yt", "yt2", "sunnxt"][f % 3]
        for i in range(n):
            rows.append({"pair_id": f"{src}_{f:03d}_{i:03d}", "meta_source": src, "meta_film": f"film{f}",
                         "track_id": f"trk_{t:05d}", "video_duration_s": float(rng.uniform(10, 60))})
            t += 1
    df = pd.DataFrame(rows)
    for a, b in ((5, 6), (6, 7), (20, 41)):                   # a reused track links film keys 5-6-7 and 20-41
        ia = df.index[df.meta_film == f"film{a}"][0]
        ib = df.index[df.meta_film == f"film{b}"][0]
        df.loc[ib, "track_id"] = df.loc[ia, "track_id"]
    return df


def test_film_groups_merge_keys_that_share_a_track():
    df = synthetic_film_pairs()
    g = film_groups(df)
    key = dict(zip(df.pair_id, df.meta_source + ":" + df.meta_film))
    by_key = {key[p]: v for p, v in g.items()}
    assert by_key["sunnxt:film5"] == by_key["yt:film6"] == by_key["yt2:film7"]
    assert by_key["sunnxt:film20"] == by_key["sunnxt:film41"]
    assert len(set(g.values())) == 120 - 3
    for _, grp in df.groupby("track_id"):                     # every track group lies inside one film group
        assert len({g[p] for p in grp.pair_id}) == 1
    extra = film_groups(df, [("yt:film0", "yt:film3")])
    assert extra[df.pair_id[df.meta_film == "film0"].iloc[0]] == extra[df.pair_id[df.meta_film == "film3"].iloc[0]]


@pytest.mark.parametrize("seed", [0, 1])
def test_film_folds_satisfy_invariants(seed):
    df = synthetic_film_pairs(seed=seed)
    g = film_groups(df)
    big = {g[df.pair_id.iloc[0]]}
    folds = make_film_folds(df, g, big)
    check_folds(folds, df, g, big)
    big_ids = {p for p in df.pair_id if g[p] in big}
    for f in folds.values():
        assert big_ids <= set(f["train"]) and not big_ids & (set(f["val"]) | set(f["test"]))
        rest = {g[p] for p in f["train"] + f["val"]} - big
        assert abs(len({g[p] for p in f["val"]}) / len(rest) - 0.1) < 0.03
    assert make_film_folds(df, g, big) == make_film_folds(df.sample(frac=1, random_state=3), g, big)


def test_check_folds_catches_a_film_across_train_and_test():
    df = synthetic_film_pairs()
    g = film_groups(df)
    big = {g[df.pair_id.iloc[0]]}
    folds = make_film_folds(df, g, big)
    test = folds["0"]["test"]
    moved = next(p for p in test if sum(g[q] == g[p] for q in test) > 1)       # a clip with a same-film test neighbour
    test.remove(moved)                                   # fold 0: train; fold 1: test (the pairs stay partitioned)
    folds["0"]["train"].append(moved)
    folds["1"]["train" if moved in folds["1"]["train"] else "val"].remove(moved)
    folds["1"]["test"].append(moved)
    tracks = df.set_index("pair_id").track_id
    assert (tracks == tracks[moved]).sum() == 1          # its track is its own, so only the film check can catch it
    with pytest.raises(AssertionError, match="film group"):
        check_folds(folds, df, g, big)
    folds = make_film_folds(df, g, big)
    folds["2"]["val"].append(next(iter(p for p in df.pair_id if g[p] in big)))    # a train-only clip in val
    with pytest.raises(AssertionError):
        check_folds(folds, df, g, big)


def test_cap_per_film_redraws_and_keeps_small_films():
    film = ["k"] * 100 + ["a"] * 5 + ["b"] * 30 + ["c"] * 31
    s0, s1 = cap_per_film(film, 30, np.random.default_rng(0)), cap_per_film(film, 30, np.random.default_rng(1))
    for s in (s0, s1):
        f = np.array(film)[s]
        assert (f == "k").sum() == 30 and (f == "a").sum() == 5 and (f == "b").sum() == 30 and (f == "c").sum() == 30
        assert len(set(s)) == len(s) == 95
    assert set(s0) != set(s1)                                  # a different subset each epoch
    batches = epoch_batches([f"p{i}" for i in range(len(film))], s0, 16)      # still one segment per video
    flat = [p for b in batches for p in b]
    assert len(flat) == len(set(flat)) == 95


def test_film_subsets_are_nested_and_keep_train_only_films():
    df = synthetic_film_pairs()
    g = film_groups(df)
    big = {g[df.pair_id.iloc[0]]}
    train = make_film_folds(df, g, big)["0"]["train"]
    films = {f: {g[p] for p in film_subset(train, g, f, big, [0, 0])} for f in (0.25, 0.5, 0.75, 1.0)}
    assert films[0.25] < films[0.5] < films[0.75] < films[1.0] == {g[p] for p in train}
    n = len({g[p] for p in train} - big)
    for f, s in films.items():
        assert big <= s and len(s - big) == int(np.ceil(f * n))
    assert film_subset(train, g, 0.5, big, [0, 1]) != film_subset(train, g, 0.5, big, [0, 0])   # per-fold order


def test_cluster_ci_resamples_whole_clusters():
    from bgm.eval.report_filmsplit import _cluster_ci
    v = np.repeat([1.0, 5.0, 9.0], 50)                        # three clusters, constant inside
    est, lo, hi = _cluster_ci(v, np.repeat(["a", "b", "c"], 50), np.median, 500)
    assert est == 5.0 and lo == 1.0 and hi == 9.0             # only whole-cluster medians are reachable
    est, lo, hi = _cluster_ci(v, np.arange(150), np.median, 500)
    assert lo == hi == 5.0                                    # independent queries: the median barely moves


def test_extend_film_folds_keeps_the_base_split_and_places_new_films():
    from bgm.data.folds import extend_film_folds
    df = synthetic_film_pairs()
    g0 = film_groups(df)
    big = {g0[df.pair_id.iloc[0]]}
    base = make_film_folds(df, g0, big)
    rng = np.random.default_rng(1)
    rows = [{"pair_id": f"new_{f:03d}_{i:03d}", "meta_source": "new", "meta_film": f"nf{f}", "track_id": f"trk_n{f:03d}_{i:03d}",
             "video_duration_s": float(rng.uniform(10, 60))} for f in range(30) for i in range(int(rng.integers(5, 40)))]
    rows.append({"pair_id": "new_zlink", "meta_source": "new", "meta_film": "nz", "track_id": df.track_id[df.meta_film == "film9"].iloc[0],
                 "video_duration_s": 20.0})                       # reuses a base film's track -> follows that film
    full = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    g = film_groups(full)
    out = extend_film_folds(base, full, g, big)
    check_folds(out, full, g, big)
    for k in base:                                            # every base pair keeps its split in every fold
        for s in ("train", "val", "test"):
            assert set(base[k][s]) <= set(out[k][s])
    film9 = df.pair_id[df.meta_film == "film9"].iloc[0]
    for k in base:
        assert ("new_zlink" in out[k]["test"]) == (film9 in base[k]["test"])
        assert not {p for p in out[k]["val"] if p.startswith("new_") and p != "new_zlink"}   # new-only films never go to val
    tests = [{g[p] for p in out[k]["test"] if p.startswith("new_") and p != "new_zlink"} for k in base]
    assert all(tests) and sum(map(len, tests)) == 30          # each new film is tested in exactly one fold
    bad = full.copy()                                         # a new clip sharing tracks with two base films in different test folds
    a = next(p for p in base["0"]["test"] if g0[p] not in big)
    b = next(p for p in base["1"]["test"] if g0[p] not in big)
    tr = dict(zip(bad.pair_id, bad.track_id))
    bad.loc[bad.pair_id == "new_000_000", "track_id"] = tr[a]
    bad.loc[bad.pair_id == "new_000_001", "track_id"] = tr[b]
    gb, moved = film_groups(bad), {}
    out2 = extend_film_folds(base, bad, gb, big, moved=moved)
    check_folds(out2, bad, gb, big)
    merged = gb["new_000_000"]
    assert gb[a] == gb[b] == merged and moved                 # the two base films are one group now
    assert sum(any(gb[p] == merged for p in out2[k]["test"]) for k in base) == 1
    assert set(moved) <= {p for p in df.pair_id if gb[p] == merged}           # only that group's base pairs moved
    for k in base:                                            # everything else is where the base split put it
        for s in ("train", "val", "test"):
            assert set(base[k][s]) - set(moved) <= set(out2[k][s])
