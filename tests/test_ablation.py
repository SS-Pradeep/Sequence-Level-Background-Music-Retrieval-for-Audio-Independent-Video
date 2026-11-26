"""D127 ablation report: pairing and readings on synthetic per-query runs (no dataset)."""
import numpy as np
import pytest

from bgm.analysis.ablation import paired_seed_mean, verdict


def _runs(vals: dict[int, list[np.ndarray]], pairs: dict[int, np.ndarray]) -> dict:
    return {k: [{"x": v, "pair": pairs[k]} for v in vs] for k, vs in vals.items()}


def test_verdict():
    assert verdict({"mean": -0.05, "lo": -0.08, "hi": -0.02}) == "worse"
    assert verdict({"mean": 0.05, "lo": 0.01, "hi": 0.09}) == "better"
    assert verdict({"mean": 0.0, "lo": -0.02, "hi": 0.02}) == "no significant difference"


def test_paired_seed_mean_averages_seeds_then_pairs():
    pairs = {0: np.array(["a", "b"]), 1: np.array(["c", "d"])}
    a = _runs({0: [np.array([1.0, 0.0]), np.array([0.0, 0.0])], 1: [np.array([1.0, 1.0]), np.array([1.0, 1.0])]}, pairs)
    b = _runs({0: [np.array([0.0, 0.0])], 1: [np.array([0.0, 1.0])]}, pairs)
    film = {"a": "f1", "b": "f1", "c": "f2", "d": "f3"}
    ma, mb, ci = paired_seed_mean(a, "x", b, "x", "pair", film, 200)
    assert ma == pytest.approx((0.5 + 0 + 1 + 1) / 4)
    assert mb == pytest.approx(0.25)
    assert ci["mean"] == pytest.approx(ma - mb)
    assert ci["n_clusters"] == 3


def test_paired_seed_mean_identical_is_zero():
    pairs = {0: np.array(["a", "b", "c"])}
    a = _runs({0: [np.array([1.0, 0.0, 1.0])]}, pairs)
    _, _, ci = paired_seed_mean(a, "x", a, "x", "pair", None, 200)
    assert ci["mean"] == ci["lo"] == ci["hi"] == 0.0


def test_paired_seed_mean_rejects_different_queries():
    a = _runs({0: [np.array([1.0, 0.0])]}, {0: np.array(["a", "b"])})
    b = _runs({0: [np.array([1.0, 0.0])]}, {0: np.array(["a", "c"])})
    with pytest.raises(AssertionError):
        paired_seed_mean(a, "x", b, "x", "pair", None, 200)
