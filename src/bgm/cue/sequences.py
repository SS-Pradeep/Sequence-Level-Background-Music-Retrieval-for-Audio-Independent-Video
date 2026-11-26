"""Sequences of joined clips and their boundary labels (cue-switch Step 1; PROGRESS.md D78).

A sequence is an ordered list of dataset clips joined end to end with `concat_query`, so its segments are exactly the
clips' own segments.csv segments, and the D75 emission lattice applies unchanged.

Boundary between segment i and i + 1:
  * inside one clip (shot cuts included):        label 0, type 'clip'
  * two clips of one film group:                 label 1 if their track groups differ, else 0; type 'film'
  * clips of two film groups:                    label 1 (a track group never spans film groups, D36); type 'cross'
Types 'clip' and 'film' are the boundaries inside one film (headline metrics); 'cross' are between films.

Modes:
  * blocked    (main sets) a chain of film blocks; a block is 1+ clips of one film, in film-time order when the film's
               clips all carry timestamps (sunnxt, one source film), else in random order
  * same_film  one film's clips in random order (secondary test set)
  * timeline   one timestamped film's clips in film-time order (secondary test set)
  * cross_film every clip from a different film (R5 training only: the shortcut check)

Val / test sets are built once and frozen (data/manifests/<cue.sequences_file>); training sequences are drawn on the
fly from per-item seeds. Every builder only sees the clips of the split it is given, so a sequence never mixes splits.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

TYPES = ("clip", "film", "cross")
IN_FILM = ("clip", "film")


@dataclass(frozen=True)
class Clip:
    pair_id: str
    film: str            # film group (folds_film.json)
    track: str           # track group (track_id, D17)
    dur: float           # s
    t_film: float        # start within the source film, s (nan if unknown)
    timeline: str        # source film key of the timestamp ("" if none)


def clip_table(pairs, film_group: dict[str, str], ids) -> dict[str, Clip]:
    """pairs: the pairs.csv frame indexed by pair_id; ids: the clips of one split."""
    out = {}
    for p in ids:
        r = pairs.loc[p]
        t = _num(r.get("meta_movie_start_s", math.nan))
        tl = f"{r['meta_source']}:{r['meta_film']}" if np.isfinite(t) else ""
        out[p] = Clip(p, film_group[p], str(r["track_id"]), float(r["video_duration_s"]), t, tl)
    return out


def with_timeline(pairs, cfg):
    """pairs with `meta_movie_start_s` filled in from `cue.timeline_file` (a CSV with clip_id, start_sec; read only) for
    clips that have no timestamp in the manifest. The full dataset's 2,998 Demucs clips carry their start in their
    source film only in dataset_merged/merged_manifest.csv; each of their films is one source video, so the timeline key
    meta_source:meta_film stays one per film. `cue.timeline_file: null` (default) returns `pairs` unchanged."""
    path = cfg.cue.get("timeline_file", None)
    if not path:
        return pairs
    import pandas as pd
    t = pd.read_csv(path, usecols=["clip_id", "start_sec"]).dropna().set_index("clip_id")["start_sec"]
    out = pairs.copy()
    have = pd.to_numeric(out["meta_movie_start_s"], errors="coerce")       # read_pairs keeps missing meta values as ""
    miss = have.isna() & out.index.isin(t.index)
    col = out["meta_movie_start_s"].astype(object)
    col[miss] = t.reindex(out.index[miss]).to_numpy(float)
    out["meta_movie_start_s"] = col
    return out


def _num(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x if np.isfinite(x) else math.nan


def by_film(clips: dict[str, Clip]) -> dict[str, list[Clip]]:
    out: dict[str, list[Clip]] = {}
    for c in sorted(clips.values(), key=lambda c: c.pair_id):
        out.setdefault(c.film, []).append(c)
    return out


def is_timeline(clips: list[Clip]) -> bool:
    """All clips timestamped, from one source film (so film time orders them)."""
    return len(clips) > 0 and all(np.isfinite(c.t_film) for c in clips) and len({c.timeline for c in clips}) == 1


def order_film(clips: list[Clip], rng: np.random.Generator, timeline: bool | None = None) -> list[Clip]:
    """A film's clips in film-time order (timeline films) or a random order that avoids two clips of one track group
    side by side when possible (up to 20 tries)."""
    if (is_timeline(clips) if timeline is None else timeline):
        return sorted(clips, key=lambda c: (c.t_film, c.pair_id))
    best = None
    for _ in range(20):
        order = [clips[i] for i in rng.permutation(len(clips))]
        if all(a.track != b.track for a, b in zip(order, order[1:])):
            return order
        best = best or order
    return best


def chunk(block: list[Clip], max_s: float) -> list[list[Clip]]:
    """Consecutive pieces of a block, each <= max_s long (a single longer clip stays alone)."""
    out, cur, tot = [], [], 0.0
    for c in block:
        if cur and tot + c.dur > max_s:
            out.append(cur)
            cur, tot = [], 0.0
        cur.append(c)
        tot += c.dur
    if cur:
        out.append(cur)
    return out


def _dur(xs: list[Clip]) -> float:
    return float(sum(c.dur for c in xs))


def partition_sequences(clips: dict[str, Clip], rng: np.random.Generator, lo: float, hi: float,
                        order=order_film) -> list[list[str]]:
    """Blocked sequences that use every clip exactly once (each film's clips form one block, split into <= hi s
    pieces). Packing: for each sequence draw a target length in [lo, hi]; take the first queued block that keeps the
    sequence <= hi and does not repeat the previous block's film; close the sequence at the target, or when nothing
    fits and it is already >= lo; if nothing fits below lo, take the next block anyway (the sequence may exceed hi).
    A short remainder (< lo) joins the previous sequence. order(clips, rng): a film's clip order (probes.py passes one
    that puts same-track clips side by side)."""
    films = by_film(clips)
    blocks = []
    for f in [sorted(films)[i] for i in rng.permutation(len(films))]:
        blocks += chunk(order(films[f], rng), hi)
    queue = [blocks[i] for i in rng.permutation(len(blocks))]
    seqs: list[list[Clip]] = []
    cur, last_film, target = [], None, float(rng.uniform(lo, hi))
    while queue:
        tot = _dur(cur)
        pick = next((k for k, b in enumerate(queue) if tot + _dur(b) <= hi and b[0].film != last_film), None)
        if pick is None:
            if cur and tot >= lo:
                seqs.append(cur)
                cur, last_film, target = [], None, float(rng.uniform(lo, hi))
                continue
            pick = next((k for k, b in enumerate(queue) if b[0].film != last_film), 0)
        b = queue.pop(pick)
        cur += b
        last_film = b[0].film
        if _dur(cur) >= target:
            seqs.append(cur)
            cur, last_film, target = [], None, float(rng.uniform(lo, hi))
    if cur:
        if seqs and _dur(cur) < lo:
            seqs[-1] += cur
        else:
            seqs.append(cur)
    return [[c.pair_id for c in s] for s in seqs]


def single_film_sequences(clips: dict[str, Clip], rng: np.random.Generator, min_s: float, hi: float,
                          timeline: bool) -> list[list[str]]:
    """One sequence per film with >= 2 clips and >= min_s of clips (split into <= hi s pieces; a piece under min_s
    joins the previous one). timeline: only timestamped films, in film-time order; else random order."""
    out = []
    for f, cs in sorted(by_film(clips).items()):
        if len(cs) < 2 or _dur(cs) < min_s or (timeline and not is_timeline(cs)):
            continue
        pieces = chunk(order_film(cs, rng, timeline=timeline), hi)
        merged = [pieces[0]]
        for p in pieces[1:]:
            if _dur(p) < min_s:
                merged[-1] = merged[-1] + p
            else:
                merged.append(p)
        out += [[c.pair_id for c in p] for p in merged if len(p) >= 2]
    return out


def sample_sequence(films: dict[str, list[Clip]], rng: np.random.Generator, lo: float, hi: float, mode: str = "blocked",
                    block_max: int = 8, limited: frozenset = frozenset()) -> list[str]:
    """One training sequence. Films are drawn uniformly; never the same film twice in a row; a film in `limited`
    (Konjum, 266 clips of one film) at most once per sequence; no clip twice. blocked: a block of 2..block_max clips
    (a contiguous run in film time for timeline films, else a random subset in random order). cross_film: one clip
    per film, no film twice."""
    assert mode in ("blocked", "cross_film"), mode
    names = sorted(films)
    target = float(rng.uniform(lo, hi))
    seq: list[Clip] = []
    used, used_films, last = set(), set(), None
    for _ in range(10000):
        if _dur(seq) >= target:
            break
        f = names[int(rng.integers(len(names)))]
        if f == last or (f in limited and f in used_films) or (mode == "cross_film" and f in used_films):
            continue
        cs = [c for c in films[f] if c.pair_id not in used]
        if not cs:
            continue
        if mode == "cross_film":
            block = [cs[int(rng.integers(len(cs)))]]
        else:
            order = order_film(cs, rng)
            m = int(rng.integers(min(2, len(order)), min(len(order), block_max) + 1))
            if is_timeline(cs):
                s0 = int(rng.integers(0, len(order) - m + 1))
                block = order[s0:s0 + m]
            else:
                block = order[:m]
        added = False
        for c in block:
            tot = _dur(seq)
            if tot >= target or (seq and tot + c.dur > hi and tot >= lo):
                break
            seq.append(c)
            used.add(c.pair_id)
            added = True
        if added:
            used_films.add(f)
            last = f
        elif seq and _dur(seq) >= lo:
            break
    return [c.pair_id for c in seq]


# ---------------------------------------------------------------- labels
def boundary_labels(seg_part: list[int], parts: list[str], track: dict[str, str], film: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """seg_part: [N] index into `parts` of each segment's clip. Returns labels [N - 1] (int8) and types [N - 1]."""
    lab, typ = [], []
    for i, j in zip(seg_part, seg_part[1:]):
        if i == j:
            lab.append(0)
            typ.append("clip")
            continue
        p, q = parts[i], parts[j]
        lab.append(int(track[p] != track[q]))
        typ.append("film" if film[p] == film[q] else "cross")
    return np.array(lab, np.int8), typ


# ---------------------------------------------------------------- frozen val / test sets
def build_sets(pairs, folds: dict, n_folds: int, lo: float, hi: float, single_min_s: float, val_partitions: int,
               seed: int = 0) -> dict:
    """{fold: {'val': {'blocked': [...]}, 'test': {'blocked': [...], 'same_film': [...], 'timeline': [...]}}}; each
    sequence a list of pair_ids. Seeds: [seed, fold, split, mode, partition]."""
    fg = folds["film_group"]
    out = {}
    for k in range(n_folds):
        f = folds["folds"][str(k)]
        val, test = clip_table(pairs, fg, f["val"]), clip_table(pairs, fg, f["test"])
        out[str(k)] = {
            "val": {"blocked": [s for r in range(val_partitions)
                                for s in partition_sequences(val, np.random.default_rng([seed, k, 0, 0, r]), lo, hi)]},
            "test": {"blocked": partition_sequences(test, np.random.default_rng([seed, k, 1, 0, 0]), lo, hi),
                     "same_film": single_film_sequences(test, np.random.default_rng([seed, k, 1, 1, 0]), single_min_s, hi, False),
                     "timeline": single_film_sequences(test, np.random.default_rng([seed, k, 1, 2, 0]), single_min_s, hi, True)},
        }
    return out


def sequences_path(cfg):
    from bgm.data import manifest
    return manifest.manifests_dir(cfg) / cfg.cue.sequences_file


def read_sets(cfg) -> dict:
    return json.loads(sequences_path(cfg).read_text())


def write_sets(cfg) -> dict:
    from bgm.data import manifest
    pairs = with_timeline(manifest.read_pairs(cfg).set_index("pair_id", drop=False), cfg)
    folds = manifest.read_folds(cfg)
    assert folds.get("split") == "film", "cue sequences are built on the film split (folds_film.json)"
    c = cfg.cue
    sets = build_sets(pairs, folds, cfg.splits.n_folds, c.seq_min_s, c.seq_max_s, c.single_film_min_s, c.val_partitions)
    check_sets(sets, folds)
    sequences_path(cfg).write_text(json.dumps({"folds_file": cfg.splits.folds_file, "manifest_hash": folds["manifest_hash"],
                                               "settings": {k: c[k] for k in ("seq_min_s", "seq_max_s", "single_film_min_s", "val_partitions")},
                                               "sets": sets}, indent=1))
    return sets


def check_sets(sets: dict, folds: dict) -> None:
    """No fold leakage: each set only holds clips of its own split; test sets use each clip once; no train-only film
    (Konjum) in val / test; no clip twice in a sequence."""
    fg, train_only = folds["film_group"], set(folds.get("train_only", []))
    for k, s in sets.items():
        f = folds["folds"][k]
        for split, modes in s.items():
            allowed = set(f[split])
            for mode, seqs in modes.items():
                for q in seqs:
                    assert set(q) <= allowed, (k, split, mode, "clip from another split")
                    assert len(set(q)) == len(q), (k, split, mode, "clip repeated")
                    assert not {fg[p] for p in q} & train_only, (k, split, mode, "train-only film outside train")
                if split == "test":
                    flat = [p for q in seqs for p in q]
                    assert len(flat) == len(set(flat)), (k, mode, "test clip used twice")
            test_films = {fg[p] for p in f["test"]}
            if split == "val":
                assert not {fg[p] for q in modes["blocked"] for p in q} & test_films, (k, "val holds a test film")
