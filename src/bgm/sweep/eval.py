"""Segment retrieval for the loss sweep, both directions.

v->m (primary) is the existing protocol, unchanged: `segment_retrieval(store, ids, Scorer.window, "all")` (the one
`bgm train semantic` early-stops on). Gallery = every segment of `ids`, each as its aligned music window cut to the
query's duration; windows of other videos sharing the query's track_id are left out; the hit is the query's own window.

m->v mirrors it: query = a segment's aligned music window (its own duration, shifted back to fit the file, as in
`segment_gallery`); gallery = every segment of `ids` as a video span; other videos sharing the query's track_id are
left out; the hit is the query's own segment.

Chance = exact random-ranking recall / MRR per query (gallery size N, one relevant item), averaged.
"""
from __future__ import annotations

import numpy as np

from bgm.eval.metrics import chance_summary, rank_of_first_hit, recall_summary
from bgm.eval.protocol import segment_retrieval, test_segments
from bgm.train.semantic import Scorer

KS = (1, 5, 10)


def _summ(ranks, N) -> dict:
    s = recall_summary(ranks, KS)
    ch = chance_summary(list(N), [1] * len(N), KS)
    return {"R1": s["R@1"], "R5": s["R@5"], "R10": s["R@10"], "MedR": s["MedR"], "MRR": s["MRR"], "n": s["n"],
            "chance_R10": ch["R@10"], "chance_MRR": ch["MRR"]}


def v2m(store, model, inp, ids) -> tuple[dict, np.ndarray]:
    r = segment_retrieval(store, ids, Scorer(model, inp).window, "all")
    return _summ(r.ranks, r.N), r.ranks


def m2v(store, model, inp, ids) -> tuple[dict, np.ndarray]:
    segs = test_segments(store, ids)
    V = model.embed("video", np.stack([inp.video(p, a, b) for p, _, a, b in segs]))
    Mq = []
    for p, _, a, b in segs:
        m, d = store.music_of[p], b - a
        s = min(a, max(0.0, store.duration("music", m) - d))
        Mq.append(inp.music_window(m, s, d))
    M = model.embed("music", np.stack(Mq))
    S = M @ V.T                                       # [query music, gallery video]
    tracks = np.array([store.track[p] for p, *_ in segs])
    pids = np.array([p for p, *_ in segs])
    ranks, N = [], []
    for i in range(len(segs)):
        keep = ~((tracks == tracks[i]) & (pids != pids[i]))
        hits = np.zeros(len(segs), bool)
        hits[i] = True
        ranks.append(rank_of_first_hit(S[i][keep], hits[keep]))
        N.append(int(keep.sum()))
    return _summ(ranks, N), np.array(ranks)


def evaluate(store, model, inp, ids) -> dict:
    model.eval()
    a, ra = v2m(store, model, inp, ids)
    b, rb = m2v(store, model, inp, ids)
    return {"v2m": a, "m2v": b, "ranks_v2m": ra, "ranks_m2v": rb}


def r10_v2m(store, model, inp, ids) -> float:
    """The early-stopping score: v->m segment R@10 (the existing trainer's criterion)."""
    return recall_summary(segment_retrieval(store, ids, Scorer(model, inp).window, "all").ranks)["R@10"]
