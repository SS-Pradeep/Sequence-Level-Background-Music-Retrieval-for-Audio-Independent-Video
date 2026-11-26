"""Loss-sweep split (user decision 1): data/manifests/folds_losssweep.json.

* Units = the title film groups of folds_title.json (219), merged when a track_id spans two groups (union-find), so no
  track_id can cross any boundary. The train-only groups of cfg.splits.film.train_only (Konjum salangai, 266 clips)
  are always training data: never TEST, never a fold's reported films, never early stopping.
* TEST = 22 of the eligible groups (seed 0). Never read before Stage 5.
* The rest go to 5 folds (seed 0; groups shuffled, then each assigned to the fold with the fewest clips so far).
* Per fold, 10% of the training film groups (eligible ones, seed [0, k]) are held out for early stopping (decision 4).

File layout, kept compatible with `bgm train semantic` (so the reference run uses the current trainer unchanged):
    folds[k].train  training clips
    folds[k].val    early-stopping clips (10% of the training films)
    folds[k].test   the fold's REPORTED films (the "val" of the 5-fold CV in the sweep spec)
plus "heldout_test" (TEST clips), "heldout_groups", "film_group", "train_only", "manifest_hash", "split": "film".
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
N_TEST_GROUPS, N_FOLDS, ES_FRACTION, SEED = 22, 5, 0.10, 0


def merge_groups(film_group: dict[str, str], track: dict[str, str]) -> dict[str, str]:
    """pair -> unit: film groups joined when they share a track_id. Unit name = the smallest member group name."""
    parent = {g: g for g in set(film_group.values())}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    first: dict[str, str] = {}
    for p, g in film_group.items():
        t = track[p]
        if t in first:
            a, b = find(first[t]), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[t] = g
    return {p: find(g) for p, g in film_group.items()}


def make_split(film_group: dict[str, str], track: dict[str, str], train_only: list[str], seed: int = SEED) -> dict:
    unit = merge_groups(film_group, track)
    to_units = {unit[p] for p in film_group if film_group[p] in set(train_only)}
    units = sorted(set(unit.values()))
    size = {u: sum(1 for p in unit if unit[p] == u) for u in units}
    elig = [u for u in units if u not in to_units]
    rng = np.random.default_rng(seed)
    order = [elig[i] for i in rng.permutation(len(elig))]
    test_units, rest = sorted(order[:N_TEST_GROUPS]), order[N_TEST_GROUPS:]
    load, part = [0] * N_FOLDS, {}
    for u in rest:                                  # greedy balance by clip count, in the shuffled order
        k = int(np.argmin(load))
        part[u] = k
        load[k] += size[u]
    pairs = sorted(film_group)
    folds = {}
    for k in range(N_FOLDS):
        rep_u = sorted(u for u in rest if part[u] == k)
        tr_elig = sorted(u for u in rest if part[u] != k)
        r = np.random.default_rng([seed, k])
        n_es = max(1, math.ceil(ES_FRACTION * len(tr_elig)))
        es_u = set(tr_elig[i] for i in r.permutation(len(tr_elig))[:n_es])
        tr_u = (set(tr_elig) - es_u) | to_units
        folds[str(k)] = {"train": [p for p in pairs if unit[p] in tr_u],
                         "val": [p for p in pairs if unit[p] in es_u],
                         "test": [p for p in pairs if unit[p] in set(rep_u)]}
    return {"folds": folds, "heldout_test": [p for p in pairs if unit[p] in set(test_units)],
            "heldout_groups": test_units, "film_group": {p: unit[p] for p in pairs}, "train_only": sorted(to_units),
            "split": "film", "roles": "train = training; val = early stopping (10% of training films); "
                                      "test = the fold's reported films; heldout_test = TEST (Stage 5 only)"}


def check_split(s: dict, track: dict[str, str]) -> list[str]:
    """Problems found (empty = fine): overlaps, missing clips, track_ids crossing a boundary."""
    bad = []
    test = set(s["heldout_test"])
    all_p = set(s["film_group"])
    for k, f in s["folds"].items():
        sets = {"train": set(f["train"]), "es": set(f["val"]), "report": set(f["test"]), "TEST": test}
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if sets[names[i]] & sets[names[j]]:
                    bad.append(f"fold {k}: {names[i]} ∩ {names[j]}")
        if set().union(*sets.values()) != all_p:
            bad.append(f"fold {k}: not every clip assigned")
        role = {p: n for n, ps in sets.items() for p in ps}
        tr_roles: dict[str, set] = {}
        for p, n in role.items():
            tr_roles.setdefault(track[p], set()).add(n)
        cross = [t for t, r in tr_roles.items() if len(r) > 1]
        if cross:
            bad.append(f"fold {k}: {len(cross)} track_ids cross a boundary")
    reported = [p for f in s["folds"].values() for p in f["test"]]
    if len(reported) != len(set(reported)) or set(reported) | test | {p for p in all_p if s["film_group"][p] in s["train_only"]} != all_p:
        bad.append("reported folds do not partition the non-TEST, non-train-only clips")
    return bad


def build(cfg) -> Path:
    from bgm.data import manifest
    title = json.loads((manifest.manifests_dir(cfg) / "folds_title.json").read_text())
    pairs = manifest.read_pairs(cfg)
    track = dict(zip(pairs.pair_id, pairs.track_id))
    s = make_split(title["film_group"], track, list(cfg.splits.film.train_only))
    problems = check_split(s, track)
    assert not problems, problems
    s["manifest_hash"] = title["manifest_hash"]
    s["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = manifest.manifests_dir(cfg) / "folds_losssweep.json"
    path.write_text(json.dumps(s, indent=1))
    return path


def describe(s: dict) -> str:
    units = set(s["film_group"].values())
    lines = [f"units (title groups merged by track_id): {len(units)}; train-only: {s['train_only']}",
             f"TEST: {len(s['heldout_groups'])} groups, {len(s['heldout_test'])} clips"]
    for k, f in s["folds"].items():
        g = lambda ids: len({s['film_group'][p] for p in ids})
        lines.append(f"fold {k}: train {len(f['train'])} clips / {g(f['train'])} groups | early stop {len(f['val'])} / "
                     f"{g(f['val'])} | reported {len(f['test'])} / {g(f['test'])}")
    return "\n".join(lines)
