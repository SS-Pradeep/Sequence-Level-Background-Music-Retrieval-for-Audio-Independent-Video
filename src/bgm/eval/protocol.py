"""Evaluation protocols (SPEC §5.12): track retrieval, segment retrieval, grounding.

Systems plug in as score functions:
  track_score(pair_id, music_ids) -> [G] scores for the whole clip against whole music files
  window_score(pair_id, a, b, windows) -> [W] scores for the segment [a, b) against windows [(music_id, start, dur)]
  offset_score(pair_id, a, b, music_id) -> [n_offsets] scores for every 25 Hz offset of the whole music file
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgm.eval.metrics import GROUNDING_RULES, grounding_rules, rank_of_first_hit, rule_hits, top_offsets
from bgm.eval.store import FeatureStore
from bgm.features.dynamics import M_CHANNELS
from bgm.train.shifts import shift_in_sync


@dataclass
class TrackResult:
    ranks: np.ndarray       # [Q]
    N: np.ndarray           # gallery size per query
    h: np.ndarray           # relevant files per query
    clusters: np.ndarray    # query track_id


def track_retrieval(store: FeatureStore, queries: list[str], gallery: list[str], track_score) -> TrackResult:
    gtrack = np.array([store.music_track[m] for m in gallery])
    ranks, N, h = [], [], []
    for q in queries:
        hits = gtrack == store.track[q]
        ranks.append(rank_of_first_hit(track_score(q, gallery), hits))
        N.append(len(gallery))
        h.append(int(hits.sum()))
    return TrackResult(np.array(ranks), np.array(N), np.array(h), np.array([store.track[q] for q in queries]))


def test_segments(store: FeatureStore, ids: list[str]) -> list[tuple[str, int, float, float]]:
    return [(p, k, a, b) for p in ids for k, (a, b) in enumerate(store.segments[p])]


def segment_gallery(store: FeatureStore, segs, query: tuple, variant: str = "all"):
    """Windows [(music_id, start, dur)] and the index of the hit for one query segment (SPEC §5.12).
    For every test segment: its music window at the aligned start, cut to the query's duration and shifted back
    to fit the file. Windows of OTHER videos sharing the query's track_id are left out. variant 'other': also leave
    out the other windows of the query's own video."""
    qp, qk, qa, qb = query
    d = qb - qa
    windows, hit = [], None
    for p, k, a, b in segs:
        own = (p == qp and k == qk)
        if not own and p != qp and store.track[p] == store.track[qp]:
            continue
        if variant == "other" and p == qp and not own:
            continue
        m = store.music_of[p]
        start = min(a, max(0.0, store.duration("music", m) - d))
        if own:
            hit = len(windows)
        windows.append((m, start, d))
    return windows, hit


def segment_retrieval(store: FeatureStore, ids: list[str], window_score, variant: str = "all") -> TrackResult:
    segs = test_segments(store, ids)
    ranks, N = [], []
    for q in segs:
        windows, hit = segment_gallery(store, segs, q, variant)
        s = window_score(q[0], q[2], q[3], windows)
        hits = np.zeros(len(windows), bool)
        hits[hit] = True
        ranks.append(rank_of_first_hit(s, hits))
        N.append(len(windows))
    return TrackResult(np.array(ranks), np.array(N), np.ones(len(segs), int), np.array([store.track[q[0]] for q in segs]))


@dataclass(frozen=True)
class GroundingQuery:
    pair_id: str
    seg_idx: int
    a: float                # segment start, s (on the shared timeline, so the true music offset too)
    b: float
    music_id: str
    true: int               # true start frame (25 Hz)
    L: int                  # segment length, frames
    track_id: str


def grounding_queries(store: FeatureStore, ids: list[str]) -> list[GroundingQuery]:
    """The grounding queries, in the order and with the skip rule every grounding evaluation uses: each test segment
    at its exact duration; skipped when shorter than one frame or when its aligned window does not fit the file."""
    out = []
    for p, k, a, b in test_segments(store, ids):
        m = store.music_of[p]
        L = int(round((b - a) * 25))
        true = int(round(a * 25))
        if L < 1 or true + L > len(store.music(m)["dyn_25hz"]):
            continue
        out.append(GroundingQuery(p, k, a, b, m, true, L, store.track[p]))
    return out


_RULES_CACHE: dict = {}


def query_rules(store: FeatureStore, q: GroundingQuery, cfg) -> dict[str, np.ndarray]:
    """grounding_rules for one query (model-independent, cached): boolean arrays over offsets 0 .. n - L."""
    d = cfg.train.dynamics
    key = (id(store), q.music_id, q.true, q.L, cfg.eval.sync_tol_s, cfg.eval.grounding_tol_s, d.guard_s, d.max_ac, d.ac_ctx_steps)
    if key not in _RULES_CACHE:
        onset = store.dyn("music", q.music_id)[:, M_CHANNELS.index("log_onset")]
        beats = np.asarray(store.music(q.music_id)["beat_times_s"], float)
        _RULES_CACHE[key] = grounding_rules(onset, beats, q.true, q.L, 25, exact_s=cfg.eval.sync_tol_s,
                                            strict_s=cfg.eval.grounding_tol_s, guard_s=d.guard_s, max_ac=d.max_ac, ctx=d.ac_ctx_steps)
    return _RULES_CACHE[key]


@dataclass
class GroundingResult:
    """Per query: signed error, true / predicted offsets, and for each rule in GROUNDING_RULES the hit (the rule's
    array at the predicted offset; its mean over the offsets tied at the top score when there is a tie) and the chance
    level (the array's mean). `strict` and `sync` are the hits. n_tied: offsets sharing the top score (1 = no tie)."""
    err_s: np.ndarray
    true: np.ndarray
    pred: np.ndarray
    hits: dict
    chance: dict
    clusters: np.ndarray
    queries: list
    n_tied: np.ndarray

    @property
    def strict(self) -> np.ndarray:
        return self.hits["strict"]

    @property
    def sync(self) -> np.ndarray:
        return self.hits["sync"]

    @property
    def exact(self) -> np.ndarray:
        return self.hits["exact"]


def score_predictions(store: FeatureStore, queries: list[GroundingQuery], preds, cfg, tied=None) -> GroundingResult:
    """Hits and chance levels of predicted start frames under every grounding rule (no model needed).
    tied: optional per-query offset arrays sharing the top score (each containing its pred); the hit is then the
    rule's mean over them. Without it every prediction counts as untied."""
    preds = np.asarray(preds, int)
    tied = [np.array([p]) for p in preds] if tied is None else [np.asarray(t, int) for t in tied]
    hits = {r: np.zeros(len(queries)) for r in GROUNDING_RULES}
    chance = {r: np.zeros(len(queries)) for r in GROUNDING_RULES}
    for i, (q, p, t) in enumerate(zip(queries, preds, tied)):
        rules = query_rules(store, q, cfg)
        n = len(rules["exact"])
        assert 0 <= p < n and p in t and t.min() >= 0 and t.max() < n, (q, p, n)
        for r, (h, c) in rule_hits(rules, t).items():
            hits[r][i], chance[r][i] = h, c
    true = np.array([q.true for q in queries])
    return GroundingResult((preds - true) / 25, true, preds, hits, chance, np.array([q.track_id for q in queries]), list(queries),
                           np.array([len(t) for t in tied]))


def grounding(store: FeatureStore, ids: list[str], offset_score, cfg) -> GroundingResult:
    """Each test segment at its exact duration against every 25 Hz offset of its own music file; scored with
    grounding_rules (exact, strict = within 0.5 s, sync) and their chance levels. pred = the first argmax; offsets
    tied at the top score share the hit (top_offsets)."""
    queries = grounding_queries(store, ids)
    scores = [offset_score(q.pair_id, q.a, q.b, q.music_id) for q in queries]
    return score_predictions(store, queries, [int(np.argmax(s)) for s in scores], cfg, [top_offsets(s) for s in scores])


@dataclass
class LegacyGroundingResult:
    err_s: np.ndarray       # signed error of the argmax offset (s)
    strict: np.ndarray      # |err| <= tol
    sync: np.ndarray        # |err| <= sync_tol or the shift stays in sync (sampler guards reject it)
    clusters: np.ndarray


def grounding_legacy(store: FeatureStore, ids: list[str], offset_score, cfg) -> LegacyGroundingResult:
    """The pre-chance-level grounding, kept byte-identical for the dynamics trainer's early stopping only, so models
    trained before and after the metric change select epochs the same way (PROGRESS.md D31). Reports use `grounding`.
    Each test segment at its exact duration against every 25 Hz offset of its own music file."""
    tol, sync_tol = cfg.eval.grounding_tol_s, cfg.eval.sync_tol_s
    dcfg = cfg.train.dynamics
    errs, strict, sync, cl = [], [], [], []
    for p, k, a, b in test_segments(store, ids):
        m = store.music_of[p]
        L = int(round((b - a) * 25))
        n_m = len(store.music(m)["dyn_25hz"])
        true = int(round(a * 25))
        if L < 1 or true + L > n_m:
            continue
        s = offset_score(p, a, b, m)
        pred = int(np.argmax(s))
        e = (pred - true) / 25
        onset = store.dyn("music", m)[:, M_CHANNELS.index("log_onset")]
        in_sync = abs(e) <= sync_tol or shift_in_sync(onset, store.music(m)["beat_times_s"], true, L, pred - true,
                                                      25, dcfg.guard_s, dcfg.max_ac, dcfg.ac_ctx_steps)
        errs.append(e)
        strict.append(abs(e) <= tol)
        sync.append(in_sync)
        cl.append(store.track[p])
    return LegacyGroundingResult(np.array(errs), np.array(strict), np.array(sync), np.array(cl))
