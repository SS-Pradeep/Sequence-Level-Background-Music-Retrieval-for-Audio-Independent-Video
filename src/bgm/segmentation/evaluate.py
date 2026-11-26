"""Boundary metrics against true music changes, with a cluster bootstrap (D140).

Per sequence a variant gives
  * candidate boundaries with a score (AP), and
  * predicted cut times WITHOUT the oracle clip joins (F1): the joins are forced only into the segments handed to
    retrieval, otherwise every variant would recall every change.
True changes are the clip joins whose track groups differ. Groups:
  all      every change; every prediction
  in_film  changes at same-film joins; predictions within tol of a cross-film join are dropped (that join is
           neither a hit nor a miss); AP over the 'clip' and 'film' candidates
  edge     changes at boundaries whose BaSSL window is clamped at a sequence edge; predictions inside the edge
           region; AP over the edge candidates
F1 matching: greedy one-to-one within +-tol (bgm.cue.evaluate.match_counts).
Bootstrap: clusters = test sequences joined by shared film groups (union-find, per fold); 10,000 resamples of
clusters, one draw matrix shared by every variant, so differences are paired.
"""
from __future__ import annotations

import numpy as np

from bgm.cue.evaluate import match_counts

TOLS = (0.5, 1.0)
GROUPS = ("all", "in_film", "edge")


def average_precision(scores: np.ndarray, labels: np.ndarray, w: np.ndarray | None = None) -> float:
    """Step-wise AP (sklearn's definition up to tie handling): mean precision at each positive, by decreasing score."""
    o = np.argsort(-scores, kind="stable")
    y = labels[o].astype(float)
    w = np.ones_like(y) if w is None else w[o]
    tp = np.cumsum(w * y)
    n = np.cumsum(w)
    P = (w * y).sum()
    return float((w * y * tp / np.maximum(n, 1e-12)).sum() / P) if P > 0 else float("nan")


def seq_counts(rec: dict) -> dict:
    """rec: preds, gt (all change times), gt_film (same-film change times), cross (cross-film join times),
    gt_edge (change times at edge boundaries), edge_lo / edge_hi. -> {f'{group}/{tol}': (tp, fp, fn)}."""
    out = {}
    preds = np.asarray(rec["preds"], float)
    lo, hi = rec["edge_lo"], rec["edge_hi"]
    for tol in TOLS:
        out[f"all/{tol:g}"] = match_counts(list(preds), list(rec["gt"]), tol)
        cross = np.asarray(rec["cross"], float)
        keep = [p for p in preds if not (len(cross) and np.min(np.abs(cross - p)) <= tol)]
        out[f"in_film/{tol:g}"] = match_counts(keep, list(rec["gt_film"]), tol)
        out[f"edge/{tol:g}"] = match_counts([p for p in preds if p <= lo or p >= hi], list(rec["gt_edge"]), tol)
    return out


def f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), np.nan)


def seq_clusters(seq_films: list[list[str]], prefix: str) -> list[str]:
    parent = list(range(len(seq_films)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, fs in enumerate(seq_films):
        for f in set(fs):
            if f in seen:
                parent[find(i)] = find(seen[f])
            else:
                seen[f] = i
    return [f"{prefix}:{find(i)}" for i in range(len(seq_films))]


class Bootstrap:
    def __init__(self, clusters: list[str], n: int = 10000, seed: int = 0):
        self.uniq, self.inv = np.unique(np.asarray(clusters), return_inverse=True)
        C = len(self.uniq)
        draws = np.random.default_rng(seed).integers(0, C, (n, C))
        self.M = np.zeros((n, C), np.float32)
        np.add.at(self.M, (np.repeat(np.arange(n), C), draws.reshape(-1)), 1.0)

    def f1(self, counts: np.ndarray) -> np.ndarray:
        """counts [n_seq, 3] (tp, fp, fn) -> F1 per resample [n]."""
        per = np.zeros((len(self.uniq), 3))
        np.add.at(per, self.inv, counts)
        s = self.M @ per
        return f1(s[:, 0], s[:, 1], s[:, 2])

    def ap(self, scores: np.ndarray, labels: np.ndarray, seq_idx: np.ndarray, chunk: int = 250) -> np.ndarray:
        """Pooled AP per resample [n]; seq_idx [n_cand] maps each candidate to its sequence."""
        o = np.argsort(-scores, kind="stable")
        y = labels[o].astype(np.float32)
        cl = self.inv[seq_idx[o]]
        out = []
        for i in range(0, len(self.M), chunk):
            w = self.M[i:i + chunk][:, cl]                       # [chunk, n_cand]
            tp = np.cumsum(w * y, 1)
            n = np.cumsum(w, 1)
            P = (w * y).sum(1)
            out.append((w * y * tp / np.maximum(n, 1e-12)).sum(1) / np.maximum(P, 1e-12))
        return np.concatenate(out)


def ci(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


# ---------------------------------------------------------------- segment-level metrics (D142)
SEG_KEYS = ("purity", "coverage", "miou", "pk", "windowdiff")


def _segs(cuts, T: float) -> list[tuple[float, float]]:
    b = [0.0] + sorted(float(c) for c in cuts) + [float(T)]
    return [(a, e) for a, e in zip(b[:-1], b[1:]) if e > a]


def _ov(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def segment_metrics(pred_cuts, true_cuts, T: float, dt: float = 0.1) -> dict:
    """Predicted vs true segments of one sequence (true segments = runs of one music cue).
    purity: share of time where a predicted segment lies in its best-matching cue (low = segments mix cues);
    coverage: share of time where a cue lies in its best-matching predicted segment (low = cues are chopped up);
    miou: mean over cues of the best IoU with a predicted segment (MovieNet / BaSSL);
    pk / windowdiff: two points k s apart (k = half the mean cue length, sampled every dt) disagree on 'same segment'
    (pk) or on the number of boundaries between them (windowdiff); lower is better."""
    pred, true = _segs(pred_cuts, T), _segs(true_cuts, T)
    pur = sum(max(_ov(p, t) for t in true) for p in pred) / T
    cov = sum(max(_ov(t, p) for p in pred) for t in true) / T
    miou = float(np.mean([max(_ov(t, p) / (max(t[1], p[1]) - min(t[0], p[0])) for p in pred) for t in true]))
    k = max(dt, 0.5 * T / len(true))
    x = np.arange(0, max(T - k, dt), dt)
    ends = lambda S: np.array([s[1] for s in S])  # noqa: E731
    nb = lambda S: np.searchsorted(ends(S), x + k, side="right") - np.searchsorted(ends(S), x, side="right")  # noqa: E731
    tb, pb = nb(true), nb(pred)
    return {"purity": pur, "coverage": cov, "miou": miou, "pk": float(np.mean((tb > 0) != (pb > 0))),
            "windowdiff": float(np.mean(tb != pb))}
