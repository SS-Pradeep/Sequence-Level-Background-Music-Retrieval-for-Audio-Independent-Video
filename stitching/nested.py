"""Phase 1: nested cross-fitted emissions (user decisions 2026-11-17: 2 train + 1 val inner split, 10-seed ensemble).

Title folds T_0..T_4 = the existing folds' test sets (disjoint, film-disjoint, together all non-train-only clips).
For outer fold k, a training clip in T_i (i != k) is scored by heads that saw neither T_k nor T_i. Those heads depend
only on the unordered pair {k, i}, so there are 10 nested splits n = (a, b), a < b:
    test  = T_a + T_b      (scored by these heads: T_b for outer a, T_a for outer b)
    val   = T_e, e = the smallest of the remaining three (early stopping, exactly as the existing recipe)
    train = the other two + the train-only clips (Konjum), as in every existing fold
Heads: the exact saved recipe of runs/titlesplit/semantic/siglip-both/fold0/seed0/config.yaml, seeds 0-9, under
runs/titlesplit_nested/semantic/siglip-both/fold<n>/seed<s>. data/manifests/folds_title.json is not modified.

`.venv/bin/python -m stitching.nested folds`
`.venv/bin/python -m stitching.nested train --nested n [--seeds 0-9]`
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
NESTED_FILE = "folds_title_nested.json"
PAIRS = list(itertools.combinations(range(5), 2))          # n -> (a, b)
RUNS = "runs/titlesplit_nested"
OVERRIDES = [f"splits.folds_file={NESTED_FILE}", "splits.n_folds=10", f"paths.runs={RUNS}"]
BASE_CFG = Path("runs/titlesplit/semantic/siglip-both/fold0/seed0/config.yaml")


def nested_of(k: int, i: int) -> int:
    return PAIRS.index(tuple(sorted((k, i))))


def make_folds() -> Path:
    src = Path("data/manifests/folds_title.json")
    d = json.loads(src.read_text())
    T = {k: list(d["folds"][str(k)]["test"]) for k in range(5)}
    allt = [p for k in range(5) for p in T[k]]
    assert len(allt) == len(set(allt)), "test sets overlap"
    seen = set(allt)          # train-only clips (Konjum: film names in d["train_only"]) are in every fold's train
    only = sorted({p for k in range(5) for s in ("train", "val") for p in d["folds"][str(k)][s]} - seen)
    folds = {}
    for n, (a, b) in enumerate(PAIRS):
        rest = [c for c in range(5) if c not in (a, b)]
        e = rest[0]
        tr = [p for c in rest[1:] for p in T[c]] + only
        folds[str(n)] = {"train": tr, "val": T[e], "test": T[a] + T[b], "pair": [a, b], "val_fold": e}
        fg = d["film_group"]
        films = lambda ids: {fg[p] for p in ids}  # noqa: E731
        assert not films(T[a] + T[b]) & films(tr + T[e]), f"film leak in nested {n}"
        assert not films(T[e]) & films([p for c in rest[1:] for p in T[c]]), f"film leak val/train in nested {n}"
    out = dict(d)
    out["folds"] = folds
    out["nested_from"] = src.name
    p = src.parent / NESTED_FILE
    p.write_text(json.dumps(out))
    log.info("-> %s: %s", p, {n: {s: len(v) for s, v in f.items() if isinstance(v, list) and s != "pair"} for n, f in folds.items()})
    return p


def nested_cfg():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(BASE_CFG)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(OVERRIDES))


def train(n: int, seeds: list[int]) -> None:
    from bgm.train.semantic import run_train_semantic
    cfg = nested_cfg()
    run_train_semantic(cfg, [n], seeds, "siglip-both", None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("folds", "train"))
    ap.add_argument("--nested", type=int)
    ap.add_argument("--seeds", default="0-9")
    a = ap.parse_args()
    if a.stage == "folds":
        make_folds()
    else:
        lo, hi = map(int, a.seeds.split("-"))
        train(a.nested, list(range(lo, hi + 1)))
