"""D55: pinned-dyn fusion grid (fuse.fine_grid) and the mood-fusion tuning helpers (arch_mood_fusion)."""
import numpy as np
import pytest

from bgm.analysis.arch_mood_fusion import rank_at, tune_wa
from bgm.retrieval.fuse import coarse_grid, fine_grid


def test_fine_grid_pinned_zero():
    g = fine_grid(0.1, 0.0)
    assert len(g) == 11
    assert all(w["dyn"] == 0.0 for w in g)
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in g)


def test_fine_grid_unpinned_and_partial():
    assert len(fine_grid(0.1, None)) == 66                  # 3 branches, 11-point simplex
    g = fine_grid(0.1, 0.2)
    assert all(w["dyn"] == pytest.approx(0.2) for w in g)
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in g)


def test_fine_grid_both_pinned_is_semantic_only():
    assert fine_grid(0.1, 0.0, 0.0) == [{"sem": 1.0, "dyn": 0.0, "attr": 0.0}]   # D56: nothing left to tune
    g = fine_grid(0.1, 0.1, 0.2)
    assert g == [{"sem": pytest.approx(0.7), "dyn": 0.1, "attr": 0.2}]


def test_coarse_grid_pinned_is_semantic_only():
    assert coarse_grid(0.1, 0.0) == [{"sem": 1.0, "attr": 0.0}]                  # D56: shortlist semantic-only
    assert len(coarse_grid(0.1, None)) == 11


def _toy_queries(flip: bool):
    """Two queries where attr fixes the sem ranking iff wa > 0 (scores pre-built, 3 candidates, hit = index 0)."""
    hits = np.array([True, False, False])
    sem = np.array([0.1, 0.9, 0.5])                         # sem alone ranks the hit 3rd
    attr = np.array([0.9, 0.1, 0.5]) if flip else np.array([0.1, 0.9, 0.5])
    return [({"sem": sem, "attr": attr}, hits)]


def test_rank_at_weights():
    q = _toy_queries(flip=True)
    assert rank_at(*q[0], wa=0.0) == 3                      # semantic only
    assert rank_at(*q[0], wa=1.0) == 1                      # attr only


def test_tune_wa_prefers_smaller_on_ties():
    wa, r10 = tune_wa(_toy_queries(flip=False))             # attr identical to sem: every wa ties
    assert wa == 0.0 and r10 == 1.0                         # R@10 saturates with 3 candidates; smallest wa kept
    wa, _ = tune_wa([({"sem": np.r_[0.1, np.ones(20)], "attr": np.r_[0.9, np.zeros(20)]},
                      np.r_[True, np.zeros(20, bool)])])    # 21 candidates: only attr puts the hit at rank 1
    assert wa > 0.0
