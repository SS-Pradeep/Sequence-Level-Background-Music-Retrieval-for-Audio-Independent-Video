"""Re-ranker study (supplementary "Re-Ranker Algorithms", D148): cleaned title split, D116 siglip-both heads fixed,
only the base top-20 shortlist is re-ranked. Hit rule = track_id (segment_all / track_test_fold, as D145 / M5).

`.venv/bin/python -m exp_supp.rerank.run eval [--workers 5]` -> runs/exp_supp/rerank/eval/fold<k>.pkl, results.csv rows
`.venv/bin/python -m exp_supp.rerank.run table`               -> exp_supp/rerank/rerank_table.{md,tex}

Single head = siglip-both seed s (s = 0..9), 50 runs; ensemble = mean cosine of seeds 0..9 (the D116 ens10 rule),
one per fold. Every parameter is picked per run on the fold's val films (criterion R@10 + 0.1 MRR, first grid entry
wins ties); test is scored once with that pick. Banks for "other queries" = the fold's training clips / segments.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from exp_supp.common import RUNS, log_run
from exp_supp.rerank import methods as M
from exp_supp.rerank.cache import CACHE, HEADS, SEEDS
from exp_supp.rerank.data import Gallery, gallery, load_split

log = logging.getLogger("exp_supp.rerank.run")
EVAL = RUNS / "rerank" / "eval"
OUT = Path("exp_supp/rerank")
N_FOLDS = 5
BOTH = [HEADS.index(("both", s)) for s in SEEDS]

A_LIST = [0.5, 1, 2, 3]
GRIDS: dict[str, tuple[str, list[dict], tuple[str, ...]]] = {
    # name: (function, grid, protocols)
    "No re-ranking": ("none", [{}], ("seg", "trk")),
    "AQE": ("aqe", [{"k": k} for k in (1, 2, 3, 5, 10)], ("seg", "trk")),
    "α-QE": ("alpha_qe", [{"k": k, "alpha": a} for k in (1, 2, 3, 5, 10) for a in A_LIST], ("seg", "trk")),
    "DBA": ("dba", [{"k": k} for k in (2, 3, 5, 10)], ("seg", "trk")),
    "k-reciprocal": ("k_reciprocal", [{"k1": a, "k2": b, "lam": c} for a in (5, 10, 20) for b in (1, 3, 6)
                                      for c in (0.1, 0.3, 0.5, 0.7, 0.9)], ("seg", "trk")),
    "Mutual kNN": ("mutual_knn", [{"k": k} for k in (1, 3, 5, 10, 20)], ("seg", "trk")),
    "Dual softmax": ("dual_softmax", [{"T": t} for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)], ("seg", "trk")),
    "Track agg. (max)": ("track_agg", [{"how": "max"}], ("seg", "trk")),
    "Track agg. (mean)": ("track_agg", [{"how": "mean"}], ("seg", "trk")),
    "Track agg. (top-3 mean)": ("track_agg", [{"how": "top3"}], ("seg", "trk")),
    "Song-group de-dup.": ("dedup", [{}], ("seg", "trk")),
    "Key/tempo re-scoring": ("keytempo", [{"w": w, "n": 1} for w in (0.25, 0.5, 1, 2)], ("seg",)),
    "Sequence context": ("seqctx", [{"w": w, "n": n} for n in (1, 2, 99) for w in (0.25, 0.5, 1, 2)], ("seg",)),
}


# ------------------------------------------------------------------ inputs of one (fold, split, head set)
class Inputs:
    """Matrices of one fold / split for head indices `hs` (several = 10-head mean of every matrix)."""

    def __init__(self, z, g: Gallery, which: str, hs: list[int]):
        mean = lambda f: np.mean([f(h) for h in hs], 0)  # noqa: E731
        p = which
        self.g = g
        self.seg = dict(S=mean(lambda h: z[f"{p}_seg_S"][h]),
                        G=mean(lambda h: z[f"{p}_seg_W"][h] @ z[f"{p}_seg_W"][h].T),
                        B=mean(lambda h: z["bank_seg_V"][h] @ z[f"{p}_seg_W"][h].T))
        self.trk = dict(S=mean(lambda h: z[f"{p}_clip_V"][h] @ z[f"{p}_file_F"][h].T),
                        G=mean(lambda h: z[f"{p}_file_F"][h] @ z[f"{p}_file_F"][h].T),
                        B=mean(lambda h: z["bank_clip_V"][h] @ z[f"{p}_file_F"][h].T),
                        W=mean(lambda h: z[f"{p}_clip_V"][h] @ z[f"{p}_fw_E"][h].T))
        self.w_file = z[f"{p}_fw_file"]
        for d in (self.seg, self.trk):
            d["Bs"] = np.sort(d["B"], 0)
        tid = {t: i for i, t in enumerate(dict.fromkeys(g.file_track))}
        self.trk_group = np.array([tid[t] for t in g.file_track])
        # segment blocks: rows of one clip share the gallery mask
        self.seg_blocks = [np.where(g.seg_pid == p)[0] for p in dict.fromkeys(g.seg_pid)]
        self.nb_cache: dict = {}

    def neighbours(self, n: int) -> list[np.ndarray]:
        if n not in self.nb_cache:
            g = self.g
            self.nb_cache[n] = [np.where((g.seg_pid == g.seg_pid[i]) & (np.abs(g.seg_k - g.seg_k[i]) <= n)
                                         & (np.arange(len(g.seg_pid)) != i))[0] for i in range(len(g.seg_pid))]
        return self.nb_cache[n]

    def rerank(self, proto: str, fn: str, params: dict) -> tuple[np.ndarray, np.ndarray, float]:
        """(base S, re-ranked R, seconds) over all queries of the protocol."""
        g = self.g
        p = dict(params)
        if proto == "trk":
            d = self.trk
            S = d["S"]
            t0 = time.perf_counter()
            if fn == "none":
                R = S
            else:
                b = M.Block(S, np.ones(S.shape[1], bool), d["G"], d["B"], self.trk_group,
                            {"Bs": d["Bs"], "W": d["W"], "w_file": self.w_file})
                R = getattr(M, fn)(b, **p)
            return S, R, time.perf_counter() - t0
        d = self.seg
        S = d["S"]
        n = p.pop("n", None)
        nb = self.neighbours(n) if n is not None else None
        t0 = time.perf_counter()
        if fn == "none":
            return S, S, time.perf_counter() - t0
        R = np.empty_like(S)
        for rows in self.seg_blocks:
            extra = {"Bs": d["Bs"]}
            if nb is not None:
                extra |= {"nb": [nb[r] for r in rows], "S_all": S, "valid_all": g.seg_valid, "col_file": g.seg_file,
                          "key": g.key_pc, "mode": g.key_mode, "tempo": g.tempo}
            b = M.Block(S[rows], g.seg_valid[rows[0]], d["G"], d["B"], g.seg_file, extra)
            R[rows] = getattr(M, fn)(b, **p)
        return S, R, time.perf_counter() - t0

    def ranks(self, proto: str, S, R) -> np.ndarray:
        if proto == "trk":
            return M.final_ranks(S, R, np.ones(S.shape[1], bool), self.g.trk_hits)
        return M.final_ranks(S, R, self.g.seg_valid, np.eye(len(S), dtype=bool))


def crit(r: np.ndarray) -> float:
    return float((r <= 10).mean() + 0.1 * (1 / r).mean())


def select_and_test(Iv: Inputs, It: Inputs, proto: str, fn: str, grid: list[dict]) -> dict:
    best, arg = -1.0, None
    for params in grid:
        S, R, _ = Iv.rerank(proto, fn, params)
        c = crit(Iv.ranks(proto, S, R))
        if c > best + 1e-12:
            best, arg = c, params
    S, R, sec = It.rerank(proto, fn, arg)
    r = It.ranks(proto, S, R)
    return {"ranks": r, "pick": arg, "val_crit": best, "ms_per_query": 1000 * sec / len(S)}


def fold_eval(k: int) -> Path:
    from bgm.eval.store import FeatureStore
    from exp_supp.rerank.cache import cleaned_cfg
    out = EVAL / f"fold{k}.pkl"
    if out.exists():
        return out
    t0 = time.time()
    store = FeatureStore(cleaned_cfg())
    split = load_split()
    gv, gt = gallery(store, split, k, "val"), gallery(store, split, k, "test")
    z = dict(np.load(CACHE / f"fold{k}.npz"))
    res = {"fold": k, "single": {}, "ens": {}, "seg_cluster": gt.seg_cluster, "trk_cluster": gt.trk_cluster}
    sets = [("single", s, [BOTH[s]]) for s in SEEDS] + [("ens", 0, BOTH)]
    for kind, s, hs in sets:
        Iv, It = Inputs(z, gv, "val", hs), Inputs(z, gt, "test", hs)
        for name, (fn, grid, protos) in GRIDS.items():
            for proto in protos:
                r = select_and_test(Iv, It, proto, fn, grid)
                res[kind].setdefault(name, {}).setdefault(proto, {})[s] = r
            m = {proto: M.summary(res[kind][name][proto][s]["ranks"]) | {"ms_per_query": res[kind][name][proto][s]["ms_per_query"],
                                                                         "pick": res[kind][name][proto][s]["pick"]}
                 for proto in protos}
            log_run("rerank", f"{kind}:{name}", k, s, "test",
                    {"heads": "siglip-both " + ("seeds 0-9 mean" if kind == "ens" else f"seed {s}"), "shortlist": M.K_SHORT,
                     "grid": grid, "data": "cleaned_dataset, folds_title.json", "hit": "track_id"}, m)
        log.warning("fold %d %s %d done [%.0f s]", k, kind, s, time.time() - t0)
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    return out


def cmd_eval(workers: int, folds) -> None:
    if workers > 1:
        import os
        import subprocess
        import sys
        env = os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        procs = [subprocess.Popen([sys.executable, "-m", "exp_supp.rerank.run", "eval", "--workers", "1", "--folds", str(k)], env=env)
                 for k in folds]
        codes = [p.wait() for p in procs]
        assert not any(codes), codes
        return
    for k in folds:
        fold_eval(k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "table"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    a = ap.parse_args()
    if a.cmd == "eval":
        cmd_eval(a.workers, a.folds)
    else:
        from exp_supp.rerank.table import write_table
        write_table()
