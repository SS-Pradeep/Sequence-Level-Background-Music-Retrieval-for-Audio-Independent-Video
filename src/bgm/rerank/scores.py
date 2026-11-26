"""Segment score matrices of the 10-head ensembles for both directions (D140).

Per fold and split s (test, val), with window j = segment j's aligned music window (start shifted back to fit the file,
as segment_gallery):

    v2m_s   [E, Q, Q]  scene i x window j cut to scene i's duration: exactly the segment_all protocol (tab:segret)
    m2v_s   [E, Q, Q]  scene i x window j at its own duration: window j as an M2V query, ranked over column j
    bank_s  [E, B, Q]  the fold's TRAINING segments x window j cut to the bank scene's duration (inductive query bank)
    mask_s  [Q, Q]     True = scene i and window j come from different videos of the same track (left out, -inf)

E = 3 disjoint 10-head ensembles (head seeds 10e .. 10e + 9), each the mean cosine of its members (MeanScorer, D65).

Run: python -m bgm.rerank.scores [original|cleaned ...]   -> runs/<dataset>/bidir/scores_fold<k>.npz
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from bgm.eval.protocol import test_segments

log = logging.getLogger(__name__)
ENS_RUNS, ENS_SIZE, N_FOLDS = (0, 1, 2), 10, 5
CHUNK = 24          # query rows pooled and embedded together


def window_start(store, m: str, a: float, d: float) -> float:
    return min(a, max(0.0, store.duration("music", m) - d))


def same_track_mask(store, q_segs: list, g_segs: list) -> np.ndarray:
    """[Q, G] True where the window's video differs from the scene's video but shares its track_id."""
    qp = np.array([s[0] for s in q_segs])
    gp = np.array([s[0] for s in g_segs])
    qt = np.array([store.track[p] for p in qp])
    gt = np.array([store.track[p] for p in gp])
    return (qt[:, None] == gt[None, :]) & (qp[:, None] != gp[None, :])


def _video_emb(members, inp, segs) -> list[np.ndarray]:
    X = np.stack([inp.video(p, a, b) for p, _, a, b in segs])
    return [sc.model.embed("video", X) for sc in members]


def _ens_mean(per_member: list[np.ndarray]) -> np.ndarray:
    """[n_members, ...] -> [E, ...] mean of each disjoint ensemble."""
    A = np.stack(per_member)
    return np.stack([A[ENS_SIZE * e: ENS_SIZE * (e + 1)].mean(0) for e in ENS_RUNS]).astype(np.float32)


def cut_to_query(store, members, q_segs: list, g_segs: list) -> np.ndarray:
    """[E, Q, G] scene i x window j cut to scene i's duration."""
    inp = members[0].inp
    V = _video_emb(members, inp, q_segs)
    gm = [(store.music_of[p], a) for p, _, a, _ in g_segs]
    out = [np.empty((len(q_segs), len(g_segs)), np.float32) for _ in members]
    for c in range(0, len(q_segs), CHUNK):
        rows = q_segs[c:c + CHUNK]
        X = np.stack([inp.music_window(m, window_start(store, m, a, b - qa), b - qa)
                      for _, _, qa, b in rows for m, a in gm])
        for i, sc in enumerate(members):
            E = sc.model.embed("music", X).reshape(len(rows), len(g_segs), -1)
            out[i][c:c + len(rows)] = np.einsum("qgd,qd->qg", E, V[i][c:c + len(rows)])
        if c // CHUNK % 20 == 0:
            log.info("  rows %d / %d", c + len(rows), len(q_segs))
    return _ens_mean(out)


def own_duration(store, members, q_segs: list, g_segs: list) -> np.ndarray:
    """[E, Q, G] scene i x window j at window j's own duration."""
    inp = members[0].inp
    V = _video_emb(members, inp, q_segs)
    X = np.stack([inp.music_window(store.music_of[p], window_start(store, store.music_of[p], a, b - a), b - a)
                  for p, _, a, b in g_segs])
    return _ens_mean([V[i] @ sc.model.embed("music", X).T for i, sc in enumerate(members)])


def build_fold(cfg, store, fold: int, device: str = "cuda") -> dict[str, np.ndarray]:
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.eval.store import fold_ids
    f = fold_ids(cfg, fold)
    heads = Path(cfg.paths.runs) / "semantic" / "siglip-both" / f"fold{fold}"
    members = [load_scorer(heads / f"seed{s}", store, device) for s in range(ENS_SIZE * len(ENS_RUNS))]
    segs = {s: test_segments(store, f[s]) for s in ("train", "val", "test")}
    out = {}
    with torch.no_grad():
        for s in ("test", "val"):
            g = segs[s]
            log.info("fold %d %s: %d scenes, bank %d training scenes", fold, s, len(g), len(segs["train"]))
            out[f"mask_{s}"] = same_track_mask(store, g, g)
            out[f"v2m_{s}"] = cut_to_query(store, members, g, g)
            out[f"m2v_{s}"] = own_duration(store, members, g, g)
            out[f"bank_{s}"] = cut_to_query(store, members, segs["train"], g)
            out[f"pairs_{s}"] = np.array([p for p, _, _, _ in g])
            out[f"seg_{s}"] = np.array([k for _, k, _, _ in g])
    return out


def run(dataset: str, device: str = "cuda") -> Path:
    from bgm.analysis.cleaned_compare import CLEAN, ORIG, make_cfg
    from bgm.eval.store import FeatureStore
    from bgm.runs import run_dir, write_run
    cfg = make_cfg({"original": ORIG, "cleaned": CLEAN}[dataset])
    store = FeatureStore(cfg)
    root = Path(cfg.paths.runs) / "bidir"
    root.mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        p = root / f"scores_fold{k}.npz"
        if p.exists():
            continue
        np.savez_compressed(p, **build_fold(cfg, store, k, device))
        log.info("wrote %s", p)
    write_run(run_dir(cfg, "bidir", "scores"), cfg, {"heads": "semantic/siglip-both", "ensembles":
              [list(range(ENS_SIZE * e, ENS_SIZE * (e + 1))) for e in ENS_RUNS]}, 0)
    return root


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for d in sys.argv[1:] or ("cleaned", "original"):
        run(d)
