"""bgm.rerank: a planted hub window must lose its grip after dual softmax / inverted softmax; metric conventions."""
from __future__ import annotations

import numpy as np

from bgm.rerank.evaluate import m2v_ranks, summary, v2m_ranks
from bgm.rerank.hubness import k_occurrence
from bgm.rerank.normalize import activation_set, dsl, inverted_softmax, mutual_nn, qbnorm

HUB = 0


def planted(n: int = 40, seed: int = 0, boost: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """[n, n] scores with a diagonal margin of 0.3 and window HUB raised by `boost` for every scene; a bank with the same hub."""
    rng = np.random.default_rng(seed)
    S = rng.normal(0.0, 0.05, (n, n)) + 0.3 * np.eye(n)
    S[:, HUB] += boost
    bank = rng.normal(0.0, 0.05, (3 * n, n))
    bank[:, HUB] += boost
    return S, bank


def test_hub_dominates_baseline():
    S, _ = planted()
    assert k_occurrence(S, 1)[HUB] == len(S)                  # the hub is everybody's top-1
    assert (v2m_ranks(S)[1:] == 2).all()


def test_dsl_removes_hub():
    S, _ = planted(boost=0.4)                                  # hub above the margin: most rows' top-1 at baseline
    assert k_occurrence(S, 1)[HUB] > len(S) // 2
    out = dsl(S, S, 0.05)
    assert k_occurrence(out, 1)[HUB] <= 2                      # its own scene plus at most one noisy row
    assert (v2m_ranks(out) == 1).mean() > 0.9


def test_inverted_softmax_removes_hub():
    S, bank = planted()
    out = inverted_softmax(S, bank, 0.05, 1.0)
    assert k_occurrence(out, 1)[HUB] <= 1
    assert (v2m_ranks(out) == 1).mean() > 0.9


def test_qbnorm_only_touches_activated_rows():
    S, bank = planted()
    act = activation_set(bank)
    assert act[HUB] and act.sum() == 1
    out = qbnorm(S, bank, 0.05, 1.0, k=1)
    assert (v2m_ranks(out) == 1).mean() > 0.9
    S2 = S.copy()
    S2[:, HUB] -= 1.0                                          # nobody's top-1 is the hub any more -> rows unchanged
    np.testing.assert_array_equal(qbnorm(S2, bank, 0.05, 1.0, k=1), S2)


def test_dsl_ties_an_extreme_hub():
    """A hub boosted beyond twice the margin is only cancelled to first order by DSL: it drops, but does not vanish."""
    S, _ = planted(boost=0.6)
    assert k_occurrence(dsl(S, S, 0.05), 1)[HUB] < k_occurrence(S, 1)[HUB]


def test_mutual_nn_bonus_and_mask():
    S, _ = planted()
    S[0, 1] = -np.inf
    out = mutual_nn(S, S, m=1, eta=1.0)
    assert np.isneginf(out[0, 1])
    fin = np.isfinite(S)
    assert np.isclose((out[fin] - S[fin]).sum(), S.shape[1])   # m = 1: one bonus per window, never on a masked entry


def test_masked_entries_never_count():
    S = np.array([[0.5, 0.9, 0.1], [0.2, 0.6, 0.1], [0.3, 0.2, 0.4]])
    assert list(v2m_ranks(S)) == [2, 1, 1]
    S[0, 1] = -np.inf
    assert list(v2m_ranks(S)) == [1, 1, 1]
    assert list(m2v_ranks(S)) == [1, 1, 1]
    assert np.isneginf(inverted_softmax(S, np.zeros((2, 3)), 0.1, 1.0)[0, 1])
    assert np.isneginf(dsl(S, S, 0.1)[0, 1])


def test_m2v_is_column_ranking():
    S = np.array([[0.9, 0.1], [0.95, 0.5]])
    assert list(v2m_ranks(S)) == [1, 2]
    assert list(m2v_ranks(S)) == [2, 1]
    s = summary(np.array([1, 2, 4, 20]))
    assert s["R@1"] == 0.25 and s["R@3"] == 0.5 and s["MedR"] == 3.0
