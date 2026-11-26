"""Sequence lattices for end-to-end training and evaluation (D141).

Candidates always come from the shipped pipeline (`run_video` + `build_lattice`, the frozen 10-head ensemble of the
run, semantic-only fusion), so the starting point is the shipped decoder:
  * training sequences (the fold's training clips only): concatenations of 3-4 clips with different track groups
    (as concat_sets) and short in-film blocks (cue.sequences.sample_sequence, 30-75 s). The target of a segment is
    every candidate of its gold track group (gold_mask; offsets latent, so same-song windows are positives, never
    negatives, rule 4); the gold window (own file at the aligned offset, truth_for) is added only when the layer holds
    no window of the gold track group. Each layer keeps the top J, up to GOLD_KEEP gold-track candidates and at most
    CAP candidates in all, lowest frozen cost first.
  * val / test sequences: the frozen concat_sets_title.json concatenations and the frozen in-film blocked sequences
    (cue_sequences_title.json), lattices exactly as build_lattice makes them (no gold, no pruning); decoded with the
    shipped beam.
Music inputs of a candidate are pooled on demand (FoldMusic) with the index's own rule, so a lattice stores only
(file, offset, duration).
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bgm.e2e.bounds import at_times, music_frame_features, video_boundary_features
from bgm.eval.store import apply_span_rule, l2
from bgm.features.keys import stitch_key

log = logging.getLogger(__name__)
CAP = 100
N_CONCAT, N_BLOCKED = 240, 160
GOLD_KEEP = 30


@dataclass
class Seq:
    kind: str                          # train | concat | infilm
    split: str                         # train | val | test
    idx: int
    parts: list[str]
    mid: list[np.ndarray]              # per layer: fold-level music code of each candidate
    off: list[np.ndarray]
    dur: list[np.ndarray]
    n_top: list[int]
    cont_of: list                      # per layer [n_prev] -> continuation index, -1 none (layer 0: None)
    zmu: np.ndarray                    # [N] frozen z-statistics of the semantic branch (the emission's mu, sigma)
    zsd: np.ndarray
    frozen: list[np.ndarray]           # per layer: frozen raw semantic score of each candidate
    gold: np.ndarray                   # [N] gold candidate index, -1 if not in the layer
    seg: np.ndarray                    # [N, 2] segment start / end, s
    xv: np.ndarray                     # [N, dv] pooled video head inputs (unit-norm span means)
    vfeat: np.ndarray                  # [N - 1, 9] video boundary features
    labels: np.ndarray                 # [N - 1] 1 = the track group changes
    types: list[str]                   # [N - 1] clip | film | cross
    seg_track: np.ndarray              # [N] gold track group
    cand_track: list[np.ndarray] = field(default_factory=list)
    gold_mask: list[np.ndarray] = field(default_factory=list)   # per layer: candidate in the gold track group
    gold_win: np.ndarray = None                                  # [N, 3] gold window (music code, offset, duration)

    @property
    def btimes(self) -> np.ndarray:
        return self.seg[1:, 0]

    @property
    def w(self) -> np.ndarray:
        return self.seg[:, 1] - self.seg[:, 0]


class FoldMusic:
    """Every music file a fold's sequences can use: pooled head inputs, boundary feature tables, key, tempo."""

    def __init__(self, store, music_ids: list[str]):
        from bgm.retrieval.index import MUSIC_KEYS, Prefix
        self.store, self.ids = store, list(music_ids)
        self.code = {m: i for i, m in enumerate(self.ids)}
        self.prefix, self.table, key, tempo, self.duration = [], [], [], [], []
        for m in self.ids:
            d = store.music(m)
            dur = store.duration("music", m)
            self.prefix.append([Prefix(d[k]) for k in MUSIC_KEYS])
            self.table.append(music_frame_features(np.asarray(d["beat_times_s"], float), store.dyn("music", m), dur))
            key.append(stitch_key(int(d["key_pc"]), int(d["key_mode"]), int(d["tonic_pc"]), False))
            tempo.append(float(d["tempo_bpm"]))
            self.duration.append(dur)
        self.key, self.tempo, self.duration = np.array(key), np.array(tempo), np.array(self.duration)
        self.track = np.array([store.music_track[m] for m in self.ids])

    def pooled(self, mid: np.ndarray, off: np.ndarray, dur: np.ndarray) -> np.ndarray:
        """[n, dm] head inputs of windows, MusicIndex.window_scores' rule (span rule, unit-norm CLAP | MERT means)."""
        out = None
        for i in np.unique(mid):
            sel = np.flatnonzero(mid == i)
            spans = np.array([apply_span_rule(s, s + d, self.duration[i], self.store.min_span) for s, d in zip(off[sel], dur[sel])])
            x = np.concatenate([l2(p.means(spans[:, 0], spans[:, 1])) for p in self.prefix[i]], 1).astype(np.float32)
            if out is None:
                out = np.zeros((len(mid), x.shape[1]), np.float32)
            out[sel] = x
        return out

    def exit_feats(self, mid, off, dur) -> np.ndarray:
        return self._feats(mid, np.asarray(off) + np.asarray(dur))

    def entry_feats(self, mid, off) -> np.ndarray:
        return self._feats(mid, np.asarray(off))

    def _feats(self, mid, t) -> np.ndarray:
        out = np.zeros((len(mid), self.table[0].shape[1]), np.float32)
        for i in np.unique(mid):
            sel = mid == i
            out[sel] = at_times(self.table[i], t[sel])
        return out


# ---------------------------------------------------------------- one sequence
def build_seq(store, index, models, fusion: dict, cfg, music: FoldMusic, film_group: dict, parts: list[str],
              kind: str, split: str, idx: int, force_gold: bool) -> Seq:
    from bgm.cue.sequences import boundary_labels
    from bgm.retrieval.index import concat_query
    from bgm.retrieval.pipeline import run_video, truth_for
    from bgm.retrieval.stitch_eval import build_lattice
    r = cfg.retrieval
    vq = concat_query(store, parts)
    vq.prepare(models, store.min_span)
    res = run_video(index, vq, fusion["w_coarse"], r.coarse_top_k, r.window_hop_s)
    layers = build_lattice(index, vq, res, fusion["w_fine"], r.fine_top_j)
    code = np.array([music.code[m] for m in index.ids])
    mids, offs, durs, ntops, conts, frozen, gold, zmu, zsd, xv, gwin = [], [], [], [], [], [], [], [], [], [], []
    seg_part, seg_track = [], []
    for i, (L, x) in enumerate(zip(layers, res)):
        mu, sd = x.stats["sem"]
        mid, off, dur, raw = code[L.mid], L.off.copy(), L.dur.copy(), mu - L.cost * sd
        m_true, o_true = truth_for(vq, store, x.a, x.b)
        g_code = music.code[m_true]
        hit = np.flatnonzero((mid == g_code) & (np.abs(off - o_true) <= 0.04))
        g = int(hit[0]) if len(hit) else -1
        ntop, cont = L.n_top, L.cont_of
        if force_gold:
            in_gold = music.track[mid] == music.track[g_code]
            if not in_gold.any():
                q = vq.segment(x.a, x.b)
                gr = index.window_scores(q, np.array([index.pos[m_true]]), np.array([o_true]), x.b - x.a)["sem"]
                mid, off, dur, raw = np.r_[mid, g_code], np.r_[off, o_true], np.r_[dur, x.b - x.a], np.r_[raw, gr]
                g, in_gold = len(mid) - 1, np.r_[in_gold, True]
            cost = -(raw - mu) / sd
            top = np.arange(len(mid)) < L.n_top
            by_cost = np.argsort(cost, kind="stable")
            keep = list(np.flatnonzero(top)) + [k for k in by_cost if in_gold[k] and not top[k]][:GOLD_KEEP]
            chosen = set(keep)
            keep += [k for k in by_cost if k not in chosen][:max(0, CAP - len(keep))]
            keep = np.array(sorted(keep))
            ntop = int((keep < L.n_top).sum())
            mid, off, dur, raw = mid[keep], off[keep], dur[keep], raw[keep]
            g = int(np.flatnonzero(keep == g)[0]) if g >= 0 and g in keep else -1
            cont = None
        mids.append(mid), offs.append(off), durs.append(dur), ntops.append(ntop), conts.append(cont), frozen.append(raw)
        gold.append(g), zmu.append(mu), zsd.append(sd), gwin.append((g_code, o_true, x.b - x.a))
        a2, b2 = apply_span_rule(x.a, x.b, vq.duration, store.min_span)
        xv.append(l2(vq._sem.means(np.array([a2]), np.array([b2]))[0]).astype(np.float32))
        mid_t = (x.a + x.b) / 2
        seg_part.append(next((j for j, (_, t0, d) in enumerate(vq.meta["parts"]) if t0 <= mid_t < t0 + d), len(parts) - 1))
        seg_track.append(store.music_track[m_true])
    y, typ = boundary_labels(seg_part, parts, store.track, film_group)
    return Seq(kind, split, idx, list(parts), mids, offs, durs, ntops, conts, np.array(zmu), np.array(zsd), frozen,
               np.array(gold), np.array([[x.a, x.b] for x in res]), np.stack(xv),
               video_boundary_features(vq.sem_1hz, vq.dyn_25hz, vq.cut_times, vq.segments), y, typ,
               np.array(seg_track), [music.track[m] for m in mids],
               [music.track[m] == t for m, t in zip(mids, seg_track)], np.array(gwin, float))


# ---------------------------------------------------------------- training sequences
def training_parts(store, cfg, train_ids: list[str], film_group: dict, fold: int, e: int) -> list[tuple[str, list[str]]]:
    """[(kind, parts)]: N_CONCAT concatenations of 3-4 different-track clips, N_BLOCKED short in-film blocks."""
    from bgm.cue.sequences import by_film, clip_table, sample_sequence
    from bgm.data import manifest
    out = []
    ids = list(train_ids)
    for i in range(N_CONCAT):
        rng = np.random.default_rng([41, fold, e, 0, i])
        n = int(rng.choice(list(cfg.eval.concat_parts)))
        parts, tracks = [], set()
        for p in rng.permutation(ids):
            if store.track[p] not in tracks:
                parts.append(str(p))
                tracks.add(store.track[p])
            if len(parts) == n:
                break
        out.append(("concat", parts))
    pairs = manifest.read_pairs(cfg).set_index("pair_id", drop=False)
    films = by_film(clip_table(pairs, film_group, ids))
    limited = frozenset(film_group[p] for p in manifest.read_folds(cfg).get("train_only", []) if p in film_group)
    for i in range(N_BLOCKED):
        rng = np.random.default_rng([41, fold, e, 1, i])
        parts = sample_sequence(films, rng, 30.0, 75.0, "blocked", 4, limited)
        if len(parts) >= 2:
            out.append(("blocked", parts))
    return out


# ---------------------------------------------------------------- one fold and ensemble
def fold_models(cfg, store, fold: int, e: int, ens_size: int = 10):
    from bgm.retrieval.index import FoldModels
    cfg.pipeline.sem_seeds = list(range(ens_size * e, ens_size * (e + 1)))
    return FoldModels.load(cfg, store, fold, e, cfg.pipeline.sem_variant, None, train_ids=None)


def build_fold(cfg, store, fold: int, e: int, out: Path) -> dict:
    """All sequences of one (fold, ensemble), cached as a pickle."""
    from bgm.cue.sequences import read_sets
    from bgm.data import manifest
    from bgm.eval.store import fold_ids
    from bgm.retrieval.index import MusicIndex
    from bgm.retrieval.stitch_eval import read_concat_sets
    if out.exists():
        with open(out, "rb") as fh:
            return pickle.load(fh)
    f = fold_ids(cfg, fold)
    fg = manifest.read_folds(cfg)["film_group"]
    run = Path(cfg.paths.runs) / cfg.pipeline.exp / f"fold{fold}" / f"seed{e}"
    fusion = json.loads((run / "fusion.json").read_text())
    stitch = json.loads((run / "stitch.json").read_text())
    models = fold_models(cfg, store, fold, e)
    gal = {s: list(dict.fromkeys(store.music_of[p] for p in f[s])) for s in ("train", "val", "test")}
    music = FoldMusic(store, list(dict.fromkeys(gal["train"] + gal["val"] + gal["test"])))
    index = {s: MusicIndex(store, models, gal[s]) for s in gal}
    concat = read_concat_sets(cfg)[str(fold)]
    cfg.cue.sequences_file = "cue_sequences_title.json"
    infilm = read_sets(cfg)["sets"][str(fold)]["test"]["blocked"]
    data = {"fold": fold, "e": e, "music": music, "fusion": fusion, "stitch": stitch, "train": [], "val": [], "test": [], "infilm": []}
    for i, (kind, parts) in enumerate(training_parts(store, cfg, f["train"], fg, fold, e)):
        data["train"].append(build_seq(store, index["train"], models, fusion, cfg, music, fg, parts, kind, "train", i, True))
    for split in ("val", "test"):
        for i, parts in enumerate(concat[split]):
            data[split].append(build_seq(store, index[split], models, fusion, cfg, music, fg, parts, "concat", split, i, False))
    for i, parts in enumerate(infilm):
        data["infilm"].append(build_seq(store, index["test"], models, fusion, cfg, music, fg, parts, "infilm", "test", i, False))
    log.info("fold %d e %d: %d training sequences (%d segments), val %d, test %d, in-film test %d", fold, e, len(data["train"]),
             sum(len(s.mid) for s in data["train"]), len(data["val"]), len(data["test"]), len(data["infilm"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    data["music"].store = None                       # not picklable; re-attached on load
    with open(out, "wb") as fh:
        pickle.dump(data, fh)
    data["music"].store = store
    return data
