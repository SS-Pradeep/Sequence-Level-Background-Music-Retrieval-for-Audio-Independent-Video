"""Constant duration prior p_i = 1 - exp(-d_i / L) for the cue decoder (bgm.cue.evaluate.hazard_logits)."""
import numpy as np

from bgm.cue.evaluate import hazard_logits


def test_hazard_logits_match_formula_and_drop_last_segment():
    w = np.array([2.0, 5.0, 10.0, 3.0])
    z = hazard_logits(w, 8.0)
    assert z.shape == (3,)
    p = 1 / (1 + np.exp(-z))
    np.testing.assert_allclose(p, 1 - np.exp(-w[:-1] / 8.0), rtol=1e-6)


def test_hazard_logits_monotone_in_duration_and_finite():
    z = hazard_logits(np.array([0.0, 1.0, 4.0, 1e6, 1.0]), 4.0)
    assert np.all(np.isfinite(z)) and np.all(np.diff(z) > 0)
