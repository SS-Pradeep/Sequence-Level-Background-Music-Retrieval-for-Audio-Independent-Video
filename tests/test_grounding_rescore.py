"""Grounding with chance levels: rule hits, the re-scoring of saved predictions, seed-averaging and paired CIs."""
from types import SimpleNamespace

import numpy as np
import pytest

from bgm.config import load_config
from bgm.eval.metrics import GROUNDING_RULES, rule_hits, top_offsets
from bgm.eval.protocol import grounding, grounding_legacy, grounding_queries, score_predictions
from bgm.eval.rescore_grounding import check_alignment, paired, summarize
from bgm.features.dynamics import M_CHANNELS


def test_rule_hits_is_array_at_prediction_and_mean():
    rules = {"exact": np.array([0, 1, 0, 0], bool), "strict": np.array([1, 1, 1, 0], bool), "sync": np.array([1, 1, 0, 1], bool)}
    h = rule_hits(rules, 2)
    assert h == {"exact": (0.0, 0.25), "strict": (1.0, 0.75), "sync": (0.0, 0.75)}


class FakeStore:
    """Three videos on two tracks; beat-grid music with a noisy onset envelope (25 Hz)."""
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.track = {"A": "t1", "B": "t1", "C": "t2"}
        self.music_of = {"A": "mA", "B": "mB", "C": "mC"}
        self.segments = {"A": [(0.0, 3.0), (3.0, 7.5)], "B": [(0.0, 4.0), (4.0, 30.0)], "C": [(1.0, 5.0), (5.0, 9.96)]}
        self._m = {}
        for m, n, period in (("mA", 400, 0.5), ("mB", 300, 0.46), ("mC", 250, 0.61)):
            beats = np.arange(0.2, n / 25, period)
            onset = 0.3 * rng.standard_normal(n)
            onset[np.round(beats * 25).astype(int).clip(0, n - 1)] += 3.0
            x = np.zeros((n, len(M_CHANNELS)), np.float32)
            x[:, M_CHANNELS.index("log_onset")] = onset
            self._m[m] = {"dyn_25hz": x, "beat_times_s": beats}

    def music(self, m):
        return self._m[m]

    def dyn(self, side, m):
        return self._m[m]["dyn_25hz"]


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.yaml", [])


def test_rescoring_saved_predictions_reproduces_old_hits(cfg):
    """Predictions recovered from saved errors reproduce the old strict / sync; the new rules score them with
    strict unchanged, exact = |err| <= 0.1 s, sync containing exact, chance = the rule arrays' means."""
    st = FakeStore()
    rng = np.random.default_rng(1)
    scores = {}

    def offset_score(p, a, b, m):
        n = len(st.dyn("music", m)) - int(round((b - a) * 25)) + 1
        return scores.setdefault((p, a), rng.standard_normal(n))
    ids = ["A", "B", "C"]
    old = grounding_legacy(st, ids, offset_score, cfg)
    new = grounding(st, ids, offset_score, cfg)
    qs = grounding_queries(st, ids)
    assert len(qs) == len(old.err_s) == 5                      # B's 26 s segment does not fit its 12 s file: skipped
    pred = np.array([q.true for q in qs]) + np.round(old.err_s * 25).astype(int)
    check_alignment(st, qs, {"strict": old.strict, "sync": old.sync}, pred, cfg, "fake")
    assert np.array_equal(pred, new.pred) and np.array_equal(new.strict, old.strict)
    assert np.array_equal(new.exact, np.abs(new.err_s) <= 0.1 + 1e-9)
    assert (new.sync >= new.exact).all()
    for i, q in enumerate(qs):
        n_off = len(st.dyn("music", q.music_id)) - q.L + 1
        within = np.abs(np.arange(n_off) - q.true) <= 12.5
        assert new.chance["strict"][i] == pytest.approx(within.mean())
    with pytest.raises(AssertionError, match="strict not reproduced"):     # saved hits from other queries are caught
        check_alignment(st, qs, {"strict": ~old.strict, "sync": old.sync}, pred, cfg, "fake")
    with pytest.raises(AssertionError, match="sync not reproduced"):
        check_alignment(st, qs, {"strict": old.strict, "sync": ~old.sync}, pred, cfg, "fake")


def fake_result(hits, chance, clusters):
    return SimpleNamespace(hits={r: np.asarray(hits[r], bool) for r in GROUNDING_RULES},
                           chance={r: np.asarray(chance[r], float) for r in GROUNDING_RULES})


def test_summarize_averages_seeds_then_pools_folds_and_pairs():
    ch = {r: [0.2, 0.5, 0.1] for r in GROUNDING_RULES}
    s0 = fake_result({r: [1, 0, 1] for r in GROUNDING_RULES}, ch, None)
    s1 = fake_result({r: [1, 1, 0] for r in GROUNDING_RULES}, ch, None)
    f1 = fake_result({r: [0, 0] for r in GROUNDING_RULES}, {r: [0.3, 0.3] for r in GROUNDING_RULES}, None)
    cl = {0: np.array(["a", "a", "b"]), 1: np.array(["c", "d"])}
    out = summarize({0: [s0, s1], 1: [f1, f1]}, cl, 500)
    hit = np.array([1, 0.5, 0.5, 0, 0])
    chance = np.array([0.2, 0.5, 0.1, 0.3, 0.3])
    for r in GROUNDING_RULES:
        assert out[r]["accuracy"]["mean"] == pytest.approx(hit.mean())
        assert out[r]["chance"]["mean"] == pytest.approx(chance.mean())
        assert out[r]["excess"]["mean"] == pytest.approx((hit - chance).mean())
        assert out[r]["excess"]["n_clusters"] == 4
    assert out["n_queries"] == 5 and out["n_runs"] == 4
    base = summarize({0: [fake_result({r: [0, 0, 0] for r in GROUNDING_RULES}, ch, None)], 1: [f1]}, cl, 500)
    d = paired(out, base, 500)
    for r in GROUNDING_RULES:
        assert d[r]["mean"] == pytest.approx(hit.mean())       # same chance cancels: learned - baseline hits


def test_fused_offsets_hit_every_integer_frame():
    """The fused scorer's window starts arange(n) / 25 map back to exactly the integer offsets (no rounding slips)."""
    n = 20000
    st = np.arange(n) / 25
    assert np.array_equal(np.clip(np.round(st * 25).astype(int), 0, None), np.arange(n))


def test_ties_share_the_hit_and_a_flat_scorer_scores_chance(cfg):
    """Offsets tied at the top score share the hit (the expected hit of a random tie-break): a scorer that gives every
    offset the same score lands exactly on the chance level, whatever offset argmax would pick first."""
    rules = {"exact": np.array([1, 0, 0, 0], bool), "strict": np.array([1, 1, 0, 0], bool), "sync": np.array([1, 0, 1, 1], bool)}
    assert rule_hits(rules, np.array([0, 1])) == {"exact": (0.5, 0.25), "strict": (1.0, 0.5), "sync": (0.5, 0.75)}
    assert list(top_offsets(np.array([0.2, 1.0, 1.0 - 1e-9, 0.3, 1.0]))) == [1, 2, 4]
    assert list(top_offsets(np.zeros(5))) == [0, 1, 2, 3, 4]
    st = FakeStore()
    flat = grounding(st, ["A", "C"], lambda p, a, b, m: np.zeros(len(st.dyn("music", m)) - int(round((b - a) * 25)) + 1), cfg)
    assert (flat.pred == 0).all() and (flat.n_tied > 1).all()
    for r in GROUNDING_RULES:
        assert np.allclose(flat.hits[r], flat.chance[r])


def test_score_predictions_rejects_out_of_range(cfg):
    st = FakeStore()
    qs = grounding_queries(st, ["A"])
    with pytest.raises(AssertionError):
        score_predictions(st, qs, [0, 10 ** 6], cfg)
