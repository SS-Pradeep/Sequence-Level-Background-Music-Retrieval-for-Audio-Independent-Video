"""Per-query metrics for any system, so every run (P0-P4) is scored by one code path (docs/person_branch_plan.md).

Retrieval (M5 protocol, as the heads are evaluated): track ranks against the test-fold gallery and the whole
1,248-file library, segment ranks against every test window ('segment_all').
Pipeline (D60): per test segment, after the coarse shortlist and the fine windows,
  track@1, window@1, window@J        as `pipeline.segment_hits`;
  track@10                           the query's track group among the 10 best files, each file scored by its best
                                     fine window (the R@10 form of track@1), with its exact chance;
  offset/<rule>, offset/chance/...   as `fuse_one` (top-1 ties shared, chance over every window of every file).
"""
from __future__ import annotations

import logging

import numpy as np

from bgm.eval.metrics import chance_recall, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.retrieval.pipeline import fuse, rule_chance, run_video, segment_hits, segment_offset_rules, top1_hits

log = logging.getLogger(__name__)


def zq(x: np.ndarray) -> np.ndarray:
    """z-score over one query's candidates (D77's fusion rule)."""
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-3)


class MeanScorer:
    """Members' scores averaged per candidate (the D65 ensemble: mean cosine)."""

    def __init__(self, members: list):
        self.members = list(members)

    def track(self, q, gallery):
        return np.mean([m.track(q, gallery) for m in self.members], 0)

    def window(self, q, a, b, windows):
        return np.mean([np.asarray(m.window(q, a, b, windows)) for m in self.members], 0)


class LateFusionScorer:
    """z(frame) + w * z(branch) over each query's candidates (score = s_frame + w * s_person, D77 normalisation)."""

    def __init__(self, frame, branch, w: float):
        self.frame, self.branch, self.w = frame, branch, float(w)

    def track(self, q, gallery):
        f = zq(self.frame.track(q, gallery))
        return f if self.w == 0 else f + self.w * zq(self.branch.track(q, gallery))

    def window(self, q, a, b, windows):
        f = zq(self.frame.window(q, a, b, windows))
        return f if self.w == 0 else f + self.w * zq(self.branch.window(q, a, b, windows))


def retrieval_scores(store, sc, ids: list[str]) -> dict:
    """Raw scores of one system on one fold: track [Q, G_test], track_all [Q, G_all], segment [list of [W]]."""
    gal = list(dict.fromkeys(store.music_of[p] for p in ids))
    lib = list(dict.fromkeys(store.pairs.music_id))
    segs = test_segments(store, ids)
    return {"track": np.stack([sc.track(q, gal) for q in ids]), "track_all": np.stack([sc.track(q, lib) for q in ids]),
            "segment": [np.asarray(sc.window(q[0], q[2], q[3], segment_gallery(store, segs, q, "all")[0])) for q in segs]}


def retrieval_truth(store, ids: list[str]) -> dict:
    gal = list(dict.fromkeys(store.music_of[p] for p in ids))
    lib = list(dict.fromkeys(store.pairs.music_id))
    gt, ga = np.array([store.music_track[m] for m in gal]), np.array([store.music_track[m] for m in lib])
    segs = test_segments(store, ids)
    seg_hits = []
    for q in segs:
        w, hit = segment_gallery(store, segs, q, "all")
        h = np.zeros(len(w), bool)
        h[hit] = True
        seg_hits.append(h)
    return {"track": [gt == store.track[p] for p in ids], "track_all": [ga == store.track[p] for p in ids], "segment": seg_hits,
            "pair": np.array(ids), "seg_pair": np.array([q[0] for q in segs]), "seg_idx": np.array([q[1] for q in segs])}


def retrieval_ranks(scores: dict, truth: dict) -> dict:
    """{'track_test_fold', 'track_all', 'segment_all': ranks}, the section names of the saved perquery.npz files."""
    return {"track_test_fold": np.array([rank_of_first_hit(s, h) for s, h in zip(scores["track"], truth["track"])]),
            "track_all": np.array([rank_of_first_hit(s, h) for s, h in zip(scores["track_all"], truth["track_all"])]),
            "segment_all": np.array([rank_of_first_hit(s, h) for s, h in zip(scores["segment"], truth["segment"])])}


def retrieval_chance(truth: dict, k: int = 10) -> dict:
    return {sec: np.array([chance_recall(len(h), int(h.sum()), k) for h in truth[key]])
            for sec, key in (("track_test_fold", "track"), ("track_all", "track_all"), ("segment_all", "segment"))}


def track_at_k(index, store, rr, w: dict, true_track, k: int = 10) -> tuple[float, float]:
    """(hit, chance): the query's track group among the k best index files, each scored by its best fine window.
    Files outside the coarse shortlist rank below every shortlisted file (they are never offered)."""
    S = fuse(rr.raw, rr.stats, w)
    G = len(index.ids)
    best = np.full(G, -np.inf)
    np.maximum.at(best, rr.fi, S)
    best[~np.isfinite(best)] = S.min() - 1.0
    gtrack = np.array([store.music_track[m] for m in index.ids])
    hits = gtrack == true_track
    return float(rank_of_first_hit(best, hits) <= k), chance_recall(G, int(hits.sum()), k)


def pipeline_fold(store, index, queries: dict, fusion: dict, chance: bool = True) -> dict:
    """Per test segment metrics of one system on one fold. queries: {pair_id: prepared VideoQuery}; index: the
    test-fold MusicIndex (or a subclass that adds branches); fusion: {'w_coarse', 'w_fine', 'K', 'J', 'window_hop_s'}.
    chance=False skips the every-window pass (offset chance levels depend only on the windows, not the system)."""
    w, K, J, hop = fusion["w_fine"], fusion["K"], fusion["J"], fusion["window_hop_s"]
    gtrack = np.array([store.music_track[m] for m in index.ids])
    per: dict[str, list] = {}
    for p, vq in queries.items():
        res = run_video(index, vq, fusion["w_coarse"], K, hop)
        res_nc = run_video(index, vq, fusion["w_coarse"], K, hop, use_coarse=False) if chance else res
        for i, (rr, rn) in enumerate(zip(res, res_nc)):
            h = segment_hits(index, store, vq, rr, w, J)
            t10, c10 = track_at_k(index, store, rr, w, store.track[p])
            rows = {"track@1": h["track@1"], "window@1": h["window@1"], "window@J": h["window@J"], "track@10": t10,
                    "chance/track@10": c10, "dur": rr.b - rr.a, "pair": p, "seg_idx": i}
            rows.update({f"offset/{k}": v for k, v in top1_hits(segment_offset_rules(index, store, vq, rr, gtrack), fuse(rr.raw, rr.stats, w)).items()})
            if chance:
                rows.update({f"offset/chance/{k}": v for k, v in rule_chance(segment_offset_rules(index, store, vq, rn, gtrack)).items()})
            for k, v in rows.items():
                per.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in per.items()}


def assert_reproduces(saved, pipe: dict, tag) -> int:
    """Recomputed per-segment pipeline hits must equal the saved run's. The one allowed difference: first-index
    window@1 / window@J where the top windows tie (the D34 tie-shared hit strictly between 0 and 1), because GPU batch
    arithmetic can flip the order of exactly tied windows. Returns the number of such tie flips."""
    assert (saved["pair"] == pipe["pair"]).all(), tag
    for a, b in (("fused/track@1", "track@1"), ("offset/fused/1", "offset/1"), ("offset/fused/window", "offset/window")):
        assert np.allclose(saved[a], pipe[b], equal_nan=True), (tag, a)
    flips = 0
    tied = (pipe["offset/window"] > 0) & (pipe["offset/window"] < 1)
    for a, b in (("fused/window@1", "window@1"), ("fused/window@J", "window@J")):
        bad = ~np.isclose(saved[a], pipe[b])
        assert not (bad & ~tied).any(), (tag, a, np.where(bad & ~tied)[0])
        flips += int(bad.sum())
        if bad.any():
            log.warning("%s %s: %d tie flip(s) at segments %s", tag, a, bad.sum(), np.where(bad)[0].tolist())
    return flips
