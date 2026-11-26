"""Shortlist-size sweep for the re-rankers (D148c): K in {10, 20, 50, 100}, 10-seed ensemble, cleaned title split.

`.venv/bin/python -m exp_supp.rerank.ksweep` -> runs/exp_supp/rerank/ksweep/fold<k>.pkl

Same protocol as run.py (parameters re-chosen on val for every K, test scored once); only the number of base
candidates the re-ranker may re-order changes (methods.K_SHORT). K = 20 must reproduce run.py's saved test ranks
exactly (asserted). LambdaMART is not swept (trained for K = 20 only).
"""
from __future__ import annotations

import logging
import pickle
import sys
import time

import numpy as np

from exp_supp.common import RUNS
from exp_supp.rerank import methods as M
from exp_supp.rerank.cache import CACHE
from exp_supp.rerank.data import gallery, load_split
from exp_supp.rerank.run import BOTH, EVAL, GRIDS, Inputs, N_FOLDS, select_and_test

log = logging.getLogger("exp_supp.rerank.ksweep")
OUT = RUNS / "rerank" / "ksweep"
KS = (10, 20, 50, 100)


def fold_run(k: int) -> None:
    from bgm.eval.store import FeatureStore
    from exp_supp.rerank.cache import cleaned_cfg
    out = OUT / f"fold{k}.pkl"
    if out.exists():
        return
    t0 = time.time()
    store = FeatureStore(cleaned_cfg())
    split = load_split()
    gv, gt = gallery(store, split, k, "val"), gallery(store, split, k, "test")
    z = dict(np.load(CACHE / f"fold{k}.npz"))
    with open(EVAL / f"fold{k}.pkl", "rb") as fh:
        saved = pickle.load(fh)["ens"]
    Iv, It = Inputs(z, gv, "val", BOTH), Inputs(z, gt, "test", BOTH)
    res = {"fold": k, "K": {}}
    for K in KS:
        M.K_SHORT = K
        res["K"][K] = {}
        for name, (fn, grid, protos) in GRIDS.items():
            for proto in protos:
                r = select_and_test(Iv, It, proto, fn, grid)
                res["K"][K].setdefault(name, {})[proto] = r
                if K == 20:
                    assert np.array_equal(r["ranks"], saved[name][proto][0]["ranks"]), f"fold {k} {name} {proto}: K=20 not reproduced"
        log.warning("fold %d K %d done [%.0f s]", k, K, time.time() - t0)
    M.K_SHORT = 20
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    folds = [int(a) for a in sys.argv[1:]] or list(range(N_FOLDS))
    for k in folds:
        fold_run(k)
