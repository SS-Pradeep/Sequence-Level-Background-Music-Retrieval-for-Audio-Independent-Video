"""1c (D57): film-stratified batch order (train/semantic.py film_stratified_order)."""
import numpy as np

from bgm.train.semantic import epoch_batches, film_stratified_order


def _films(counts: dict[str, int]) -> list[str]:
    return [f for f, n in counts.items() for _ in range(n)]


def test_order_is_a_permutation_of_the_selection():
    film = _films({"A": 7, "B": 3, "C": 1, "D": 4})
    idx = np.array([0, 2, 3, 5, 6, 8, 9, 11, 12, 14])        # a capped subset, not all indices
    out = film_stratified_order(idx, film, 3, np.random.default_rng(0))
    assert sorted(out.tolist()) == sorted(idx.tolist())


def test_exact_chunks_make_single_film_batches():
    # every film has exactly k clips, so each batch of size k is one film
    k, film = 3, _films({chr(65 + i): 3 for i in range(8)})
    out = film_stratified_order(np.arange(len(film)), film, k, np.random.default_rng(1))
    fa = np.asarray(film)
    for b in range(0, len(out), k):
        assert len(set(fa[out[b:b + k]])) == 1


def test_stratification_raises_same_film_cooccurrence():
    rng = np.random.default_rng(2)
    film = _films({f"f{i}": 8 for i in range(16)})           # 128 clips, 16 films
    idx = np.arange(len(film))
    fa = np.asarray(film)

    def pairs(order, batch=8):
        tot = 0
        for ids in epoch_batches(list(range(len(film))), order, batch):
            g = fa[np.array(ids)]
            tot += sum((g == f).sum() * ((g == f).sum() - 1) // 2 for f in set(g))
        return tot

    plain = pairs(rng.permutation(idx))
    strat = pairs(film_stratified_order(idx, film, 4, rng))
    assert strat > 3 * plain                                 # chunks of 4 pack >= 3 same-film pairs each


def test_single_film_degenerates_to_permutation():
    film = _films({"only": 9})
    out = film_stratified_order(np.arange(9), film, 4, np.random.default_rng(3))
    assert sorted(out.tolist()) == list(range(9))
