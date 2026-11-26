"""Stitching evaluation (SPEC §5.10, §5.12; M10 / H5).

* Concatenation sets: per fold, 50 val and 50 test videos, each 3-4 clips of that split with different track_ids,
  joined end to end (data/manifests/concat_sets.json).
* Lattice: per video, each segment's top J windows plus the contiguous continuation of every candidate of the previous
  segment, all scored with the segment's saved z-statistics. `decode_lattice` runs the beam + Viterbi of §5.10 for one
  penalty set: continuations are generated only from the previous segment's survivors, and a list longer than
  beam_factor x J keeps the lowest ACCUMULATED cost. Transition costs are vectorized equivalents of the spec's
  `transition_cost` (tests/test_m9_m10.py checks equality).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from bgm.data import manifest
from bgm.eval.store import fold_ids
from bgm.retrieval.index import MusicIndex, VideoQuery
from bgm.retrieval.pipeline import SegmentResult, fuse, truth_for
from bgm.retrieval.stitch import Candidate


# ---------------------------------------------------------------- concatenation sets
def concat_sets_path(cfg):
    """`eval.concat_sets_file` under the manifests dir: concat_sets.json (track_id split, the default) or a
    split-specific file such as concat_sets_film.json (film split, PROGRESS.md D60). The sets are built from the
    configured folds, so each folds file needs its own."""
    return manifest.manifests_dir(cfg) / cfg.eval.get("concat_sets_file", "concat_sets.json")


def make_concat_sets(cfg, store, n_per_split: int = 50, seed: int = 0) -> dict:
    out = {}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        out[str(k)] = {}
        for split in ("val", "test"):
            ids = list(f[split])
            rng = np.random.default_rng([seed, k, 0 if split == "val" else 1])
            vids = []
            for _ in range(n_per_split):
                n = int(rng.choice(cfg.eval.concat_parts))
                parts, tracks = [], set()
                for p in rng.permutation(ids):
                    if store.track[p] not in tracks:
                        parts.append(str(p))
                        tracks.add(store.track[p])
                    if len(parts) == n:
                        break
                vids.append(parts)
            out[str(k)][split] = vids
    concat_sets_path(cfg).write_text(json.dumps(out, indent=1))
    return out


def read_concat_sets(cfg) -> dict:
    return json.loads(concat_sets_path(cfg).read_text())


# ---------------------------------------------------------------- vectorized costs
def fifths_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs((7 * a[:, None]) % 12 - (7 * b[None, :]) % 12)
    out = np.minimum(d, 12 - d) / 6.0
    return np.where((a[:, None] < 0) | (b[None, :] < 0), 0.5, out)


def tempo_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(np.log2(b[None, :] / a[:, None]))
        d = 2.0 * np.abs(r - np.round(r))
    return np.where((a[:, None] > 0) & (b[None, :] > 0), d, 0.5)


def transition_matrix(P: "Layer", C: "Layer", pen: dict, tol_s: float = 0.04) -> np.ndarray:
    same = P.mid[:, None] == C.mid[None, :]
    contiguous = np.abs(C.off[None, :] - (P.off + P.dur)[:, None]) <= tol_s
    k_out, t_out = (P.key, P.tempo) if P.key_out is None else (P.key_out, P.tempo_out)
    k_in, t_in = (C.key, C.tempo) if C.key_in is None else (C.key_in, C.tempo_in)
    switch = pen["lam_switch"] + pen["mu"] * (fifths_matrix(k_out, k_in) + tempo_matrix(t_out, t_in))
    return np.where(same, np.where(contiguous, 0.0, pen["lam_jump"]), switch)


@dataclass
class Layer:
    """All candidates of one segment (top J first, then continuations), as arrays."""
    mid: np.ndarray            # music-file code
    off: np.ndarray
    dur: np.ndarray
    key: np.ndarray
    tempo: np.ndarray
    cost: np.ndarray           # -S_fine
    n_top: int                 # the first n_top are the segment's top-J windows
    cont_of: np.ndarray = field(default=None)   # [n_prev] -> index of that candidate's continuation here, -1 if none
    # D137: key / tempo of the window's first and last seconds; a switch compares the outgoing candidate's *_out with
    # the incoming one's *_in. None = the per-file key / tempo above
    key_in: np.ndarray = field(default=None)
    tempo_in: np.ndarray = field(default=None)
    key_out: np.ndarray = field(default=None)
    tempo_out: np.ndarray = field(default=None)

    def subset(self, idx):
        sub = lambda a: None if a is None else a[idx]  # noqa: E731
        return Layer(self.mid[idx], self.off[idx], self.dur[idx], self.key[idx], self.tempo[idx], self.cost[idx], 0, None,
                     sub(self.key_in), sub(self.tempo_in), sub(self.key_out), sub(self.tempo_out))


def top_windows(S: np.ndarray, fi: np.ndarray, J: int, max_per_file: int | None = None) -> np.ndarray:
    """Indices of the J best windows, at most max_per_file from one music file (None = no cap)."""
    order = np.argsort(-S)
    if max_per_file is None:
        return order[:J]
    keep, n = [], {}
    for k in order:
        f = int(fi[k])
        if n.get(f, 0) < max_per_file:
            keep.append(k)
            n[f] = n.get(f, 0) + 1
            if len(keep) == J:
                break
    return np.array(keep, int)


def build_lattice(index: MusicIndex, vq: VideoQuery, results: list[SegmentResult], w: dict, J: int,
                  max_per_file: int | None = None) -> list[Layer]:
    layers = []
    for i, r in enumerate(results):
        S = fuse(r.raw, r.stats, w)
        d = r.b - r.a
        top = top_windows(S, r.fi, J, max_per_file)
        mid, off, cost = list(r.fi[top]), list(r.st[top]), list(-S[top])
        cont_of = None
        if i:
            P = layers[-1]
            seen = {(int(m), round(o / 0.04)): j for j, (m, o) in enumerate(zip(mid, off))}
            cont_of = np.full(len(P.mid), -1)
            q = vq.segment(r.a, r.b)
            new_m, new_o = [], []
            for j, (m, o, pd) in enumerate(zip(P.mid, P.off, P.dur)):
                o2 = o + pd
                if o2 + d > index.items[m].duration:
                    continue
                key = (int(m), round(o2 / 0.04))
                if key not in seen:
                    seen[key] = len(mid) + len(new_m)
                    new_m.append(int(m)), new_o.append(o2)
                cont_of[j] = seen[key]
            if new_m:
                raw = index.window_scores(q, np.array(new_m), np.array(new_o), d)
                mid += new_m
                off += new_o
                cost += list(-fuse(raw, r.stats, w))
        mid = np.array(mid, int)
        layers.append(Layer(mid, np.array(off, float), np.full(len(mid), d), np.array([index.items[m].key for m in mid]),
                            np.array([index.items[m].tempo for m in mid]), np.array(cost, float), len(top), cont_of))
    return layers


def decode_lattice(layers: list[Layer], pen: dict, beam: int | None, tol_s: float = 0.04) -> list[int]:
    """Beam + Viterbi; returns the chosen candidate index (into each full Layer) per segment."""
    act = np.arange(len(layers[0].mid))
    D = layers[0].cost[act].copy()
    acts, backs = [act], []
    for i in range(1, len(layers)):
        L = layers[i]
        conts = L.cont_of[acts[-1]]
        cur = np.unique(np.r_[np.arange(L.n_top), conts[conts >= 0]])
        T = transition_matrix(layers[i - 1].subset(acts[-1]), L.subset(cur), pen, tol_s)
        tot = D[:, None] + T
        bk = tot.argmin(0)
        Dc = tot.min(0) + L.cost[cur]
        if beam is not None and len(cur) > beam:
            keep = np.sort(np.argsort(Dc)[:beam])
            cur, Dc, bk = cur[keep], Dc[keep], bk[keep]
        acts.append(cur), backs.append(bk)
        D = Dc
    assert np.isfinite(D.min()), "path cost is not finite: use large finite penalties, not inf"
    j = int(np.argmin(D))
    path = [j]
    for bk in reversed(backs):
        j = int(bk[j])
        path.append(j)
    path = path[::-1]
    return [int(acts[i][path[i]]) for i in range(len(layers))]


def greedy_lattice(layers: list[Layer]) -> list[int]:
    return [int(np.argmin(L.cost[:L.n_top])) for L in layers]


def chosen_candidates(index: MusicIndex, layers: list[Layer], picks: list[int]) -> list[Candidate]:
    return [Candidate(index.ids[L.mid[j]], float(L.off[j]), float(L.dur[j]), int(L.key[j]), float(L.tempo[j]), float(L.cost[j]))
            for L, j in zip(layers, picks)]


# ---------------------------------------------------------------- metrics
def switch_times(chosen: list[Candidate], results: list[SegmentResult]) -> tuple[list[float], list[float]]:
    """(switch times, jump times): boundaries where the file changes / where it stays but is not contiguous."""
    sw, jp = [], []
    for c0, c1, r in zip(chosen, chosen[1:], results[1:]):
        if c1.music_id != c0.music_id:
            sw.append(r.a)
        elif abs(c1.offset - (c0.offset + c0.dur)) > 0.04:
            jp.append(r.a)
    return sw, jp


def switch_f1(pred: list[float], true: list[float], tol: float = 1.0) -> float:
    """Greedy one-to-one matching within +/- tol s."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    used, tp = set(), 0
    for t in true:
        cands = [(abs(p - t), k) for k, p in enumerate(pred) if k not in used and abs(p - t) <= tol]
        if cands:
            used.add(min(cands)[1])
            tp += 1
    prec, rec = tp / len(pred), tp / len(true)
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


def part_track_accuracy(store, vq: VideoQuery, results, chosen) -> float:
    w = np.array([r.b - r.a for r in results])
    ok = [store.music_track[c.music_id] == store.music_track[truth_for(vq, store, r.a, r.b)[0]] for r, c in zip(results, chosen)]
    return float(np.average(ok, weights=w))


def soundtrack_beats(chosen: list[Candidate], results, beats_of) -> np.ndarray:
    """The rendered soundtrack's beats on the video timeline (the chosen windows' beat_times mapped through)."""
    out = []
    for c, r in zip(chosen, results):
        b = beats_of(c.music_id)
        b = b[(b >= c.offset) & (b < c.offset + c.dur)]
        out.append(r.a + (b - c.offset))
    return np.concatenate(out) if out else np.zeros(0)


def within_piece_cuts(cuts: np.ndarray, duration: float, piece_starts: list[float], margin: float = 0.5, tol: float = 0.05) -> np.ndarray:
    c = cuts[(cuts > margin) & (cuts < duration - margin)]
    if piece_starts:
        c = c[np.abs(c[:, None] - np.array(piece_starts)[None, :]).min(1) > tol]
    return c
