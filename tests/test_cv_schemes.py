"""D133 cross-validation schemes: k-fold, leave-one-film-out and holdout folds respect film groups."""
import numpy as np
import pandas as pd
import pytest

from bgm.analysis.cv_schemes import check_scheme, holdout_folds, lofo_folds
from bgm.data.folds import make_film_folds


def _pairs(n_films: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, film = [], {}
    for f in range(n_films):
        for c in range(int(rng.integers(1, 6))):
            p = f"f{f:02d}_c{c}"
            rows.append({"pair_id": p, "track_id": f"t{f:02d}_{c % 2}", "video_duration_s": float(rng.uniform(5, 60))})
            film[p] = f"film{f:02d}"
    film.update({f"big_c{c}": "big" for c in range(8)})
    rows += [{"pair_id": f"big_c{c}", "track_id": "t_big", "video_duration_s": 20.0} for c in range(8)]
    return pd.DataFrame(rows), film, {"big"}


def test_lofo_one_fold_per_testable_film():
    pairs, film, only = _pairs()
    folds = lofo_folds(pairs, film, only)
    assert len(folds) == len(set(film.values())) - 1
    for f in folds.values():
        assert len({film[p] for p in f["test"]}) == 1
        assert {f"big_c{c}" for c in range(8)} <= set(f["train"])
    check_scheme(folds, pairs, film, only, partition=True)


def test_holdout_draws_differ_and_hold_out_a_fifth_of_films():
    pairs, film, only = _pairs()
    folds = holdout_folds(pairs, film, only, draws=5)
    check_scheme(folds, pairs, film, only, partition=False)
    tests = [frozenset(film[p] for p in f["test"]) for f in folds.values()]
    assert len(set(tests)) == 5
    assert all(len(t) == 6 for t in tests)                   # ceil(0.2 * 30 testable films)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_kfold_split_seeds_change_the_partition(k):
    pairs, film, only = _pairs()
    a, b = (make_film_folds(pairs, film, only, k, 0.1, s) for s in (0, 1))
    check_scheme(a, pairs, film, only, partition=True)
    check_scheme(b, pairs, film, only, partition=True)
    assert [sorted(f["test"]) for f in a.values()] != [sorted(f["test"]) for f in b.values()]


def test_check_scheme_catches_a_film_leak():
    pairs, film, only = _pairs()
    folds = lofo_folds(pairs, film, only)
    f = folds["0"]
    leaked = next(p for p in f["train"] if film[p] != "big")
    f["train"].remove(leaked)
    f["test"].append(leaked)
    with pytest.raises(AssertionError):
        check_scheme({"0": f}, pairs, film, only, partition=False)
