"""D60: offset-precision rules, the pipeline without a dynamics model, the concat-sets file key."""
from dataclasses import dataclass

import numpy as np
import pytest
from omegaconf import OmegaConf

from bgm.eval.attributes import Calibration
from bgm.eval.metrics import cluster_bootstrap_ratio
from bgm.retrieval.index import FoldModels, MusicIndex, VideoQuery
from bgm.retrieval.pipeline import fuse, offset_rules, rule_chance, run_video, segment_offset_rules, top1_hits
from bgm.retrieval.stitch_eval import concat_sets_path


# ---------------------------------------------------------------- pure rules
def _rules():
    #                    own file (track A)        same group, other file     other track
    start = np.array([0.0, 1.0, 2.0, 3.0, 9.0,    2.0, 8.0,                  2.0, 3.0, 4.0])
    in_group = np.array([1, 1, 1, 1, 1,            1, 1,                      0, 0, 0], bool)
    same_file = np.array([1, 1, 1, 1, 1,           0, 0,                      0, 0, 0], bool)
    return offset_rules(in_group, same_file, start, true_off=2.0), start


def test_offset_rules_need_track_and_tolerance():
    r, _ = _rules()
    assert r["track"].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert r["1"].tolist() == [0, 1, 1, 1, 0, 1, 0, 0, 0, 0]          # other-track windows at the same offset miss
    assert r["3"].tolist() == [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert r["5"].tolist() == [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert r["window"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]     # own file only, within 0.5 s
    for a, b in (("1", "3"), ("3", "5"), ("5", "track")):
        assert not (r[a] & ~r[b]).any()                              # a wider rule accepts everything a tighter one does


def test_chance_is_the_rule_mean_and_cond_is_within_the_track_group():
    r, _ = _rules()
    c = rule_chance(r)
    assert c["track"] == pytest.approx(0.7) and c["1"] == pytest.approx(0.4) and c["window"] == pytest.approx(0.1)
    assert c["cond/1"] == pytest.approx(4 / 7) and c["cond/5"] == pytest.approx(5 / 7)
    assert "cond/track" not in c and "cond/window" not in c
    none = offset_rules(np.zeros(3, bool), np.zeros(3, bool), np.arange(3.0), 0.0)
    assert np.isnan(rule_chance(none)["cond/1"])


def test_top1_hits_share_ties():
    r, _ = _rules()
    s = np.array([0.1, 0.9, 0.2, 0.3, 0.9, 0.0, 0.0, 0.5, 0.5, 0.5])   # windows 1 (start 1.0) and 4 (start 9.0) tie
    h = top1_hits(r, s)
    assert h["track"] == 1.0 and h["1"] == 0.5 and h["5"] == 0.5 and h["window"] == 0.0
    assert top1_hits(r, np.arange(10.0))["track"] == 0.0              # a clear winner in the other track


def test_cluster_bootstrap_ratio():
    num, den = np.array([1, 0, 1, 0.5, 0, 0]), np.array([1, 1, 1, 1, 0, 0])
    b = cluster_bootstrap_ratio(num, den, np.array(list("aabbcc")), 2000)
    assert b["mean"] == pytest.approx(2.5 / 4) and b["lo"] <= b["mean"] <= b["hi"] and b["n_clusters"] == 3
    one = cluster_bootstrap_ratio(num, den, np.zeros(6), 200)         # one cluster: every resample is the full set
    assert one["lo"] == pytest.approx(one["hi"]) == pytest.approx(2.5 / 4)


# ---------------------------------------------------------------- pipeline without a dynamics model
D = 8


class _Sem:
    """Fake head: the music side keeps the CLAP half of [CLAP | MERT], both sides are unit-normalized."""

    def embed(self, side, x):
        x = np.asarray(x, np.float32)[:, :D]
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


@dataclass
class _Store:
    files: dict
    music_track: dict
    music_of: dict
    track: dict
    min_span: float = 5.0

    def music(self, m):
        return self.files[m]

    def duration(self, side, m):
        return float(len(self.files[m]["sem_clap_1hz"]))


def _music(rng, T):
    return {"sem_clap_1hz": rng.standard_normal((T, D)).astype(np.float32), "sem_mert_1hz": rng.standard_normal((T, D)).astype(np.float32),
            "attr_logits_1hz": rng.standard_normal((T, 4)).astype(np.float32), "beat_times_s": np.arange(0, T, 0.5, dtype=np.float32),
            "tempo_bpm": np.float32(120), "key_pc": np.int8(0), "key_mode": np.int8(0), "tonic_pc": np.int8(-1)}


def _setup():
    rng = np.random.default_rng(0)
    files = {"own": _music(rng, 20), "same_group": _music(rng, 20), "other": _music(rng, 20)}
    store = _Store(files, {"own": "A", "same_group": "A", "other": "B"}, {"q": "own"}, {"q": "A"})
    cal = Calibration("attr_logits_1hz", {s: np.zeros(4) for s in ("video", "music")}, {s: np.ones(4) for s in ("video", "music")},
                      {"video": 1.0, "music": 1.0})
    models = FoldModels(0, 0, _Sem(), None, cal, "cpu")
    # the video's 1 Hz semantics are its own music's CLAP vectors from t = 0: the aligned window scores cosine 1
    vq = VideoQuery(files["own"]["sem_clap_1hz"][:12].copy(), rng.standard_normal((12, 4)).astype(np.float32), np.zeros((300, 5), np.float32),
                    np.zeros(0), [(4.0, 10.0)], 12.0, {"parts": [("q", 0.0, 12.0)]})
    vq.prepare(models, store.min_span)
    return store, models, MusicIndex(store, models, list(files)), vq


def test_pipeline_runs_without_a_dynamics_model():
    store, models, index, vq = _setup()
    assert index.dyn is None and vq.segment(4.0, 10.0).hv is None
    (r,) = run_video(index, vq, {"sem": 1.0, "attr": 0.0}, K=3, hop=0.5)
    assert set(r.raw) == {"sem", "dyn", "attr"} and not r.raw["dyn"].any()        # the branch scores 0 everywhere
    S = fuse(r.raw, r.stats, {"sem": 1.0, "dyn": 0.0, "attr": 0.0})
    assert np.isfinite(S).all()
    best = np.flatnonzero(S >= S.max() - 1e-6)
    assert all(index.ids[r.fi[i]] == "own" for i in best) and 4.0 in r.st[best]   # the aligned window is at the top
    gtrack = np.array([store.music_track[m] for m in index.ids])
    rules = segment_offset_rules(index, store, vq, r, gtrack)
    h = top1_hits(rules, S)
    assert h["track"] == 1.0 and h["1"] == 1.0 and h["window"] == 1.0
    c = rule_chance(rules)
    n = len(r.st)                                                                 # 3 files x 29 starts (0 .. 14 s, every 0.5 s)
    assert n == 87 and c["track"] == pytest.approx(58 / 87)
    assert c["1"] == pytest.approx(2 * 5 / 87) and c["window"] == pytest.approx(3 / 87) and c["cond/1"] == pytest.approx(10 / 58)


def test_semantic_only_scores_tie_between_neighbouring_windows():
    """Why the metric shares ties: with 1 Hz features, [4.0, 10.0) and [4.5, 10.5) pool the same seconds."""
    _, _, index, vq = _setup()
    (r,) = run_video(index, vq, {"sem": 1.0, "attr": 0.0}, K=3, hop=0.5)
    own = np.flatnonzero((r.fi == index.pos["own"]) & np.isin(r.st, (4.0, 4.5)))
    assert len(own) == 2 and r.raw["sem"][own[0]] == r.raw["sem"][own[1]]


# ---------------------------------------------------------------- concat sets file
def test_concat_sets_path_follows_the_config(tmp_path):
    cfg = OmegaConf.create({"paths": {"manifests": str(tmp_path)}, "eval": {}})
    assert concat_sets_path(cfg).name == "concat_sets.json"
    cfg.eval.concat_sets_file = "concat_sets_film.json"
    assert concat_sets_path(cfg) == tmp_path / "concat_sets_film.json"


def test_ensemble_semantic_scores_are_the_mean_of_member_cosines():
    from bgm.retrieval.index import EnsembleSemantic

    class _Proj:
        def __init__(self, seed):
            self.W = np.random.default_rng(seed).standard_normal((6, 4)).astype(np.float32)

        def embed(self, side, x):
            z = np.asarray(x, np.float32) @ self.W
            return z / np.linalg.norm(z, axis=1, keepdims=True)

        def eval(self):
            pass

    members = [_Proj(s) for s in range(3)]
    ens = EnsembleSemantic(members)
    rng = np.random.default_rng(9)
    v, m = rng.standard_normal((2, 6)), rng.standard_normal((5, 6))
    got = ens.embed("music", m) @ ens.embed("video", v).T
    want = np.mean([x.embed("music", m) @ x.embed("video", v).T for x in members], 0)
    assert got.shape == (5, 2) and np.allclose(got, want, atol=1e-6)


def test_context_weight_adds_the_whole_video_file_score():
    store, models, index, vq = _setup()
    (r0,) = run_video(index, vq, {"sem": 1.0, "attr": 0.0}, K=3, hop=0.5)
    vq.set_context(index, 1.0)
    q = vq.segment(4.0, 10.0)
    raw = index.window_scores(q, r0.fi, r0.st, 6.0)
    whole = index.g @ vq.pooled(0.0, vq.duration)[0]
    assert np.allclose(raw["sem"], r0.raw["sem"] + whole[r0.fi], atol=1e-6)
    vq.set_context(index, 0.0)
    assert vq.segment(4.0, 10.0).ctx is None
