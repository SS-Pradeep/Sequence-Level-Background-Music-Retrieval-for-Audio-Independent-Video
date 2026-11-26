"""LambdaMART re-ranker on the top-20 shortlist (D148, part C).

`.venv/bin/python -m exp_supp.rerank.ltr` -> runs/exp_supp/rerank/eval/fold<k>_ltr.pkl (merged into the table)

Features per (query, shortlisted candidate): cosine of the siglip-both head(s), CLAP-only head score, MERT-only head
score (D127's siglip-clap / siglip-mert heads, same seed; ensemble = 10-seed means), base rank in the shortlist, and
for the segment protocol the key and tempo distance of the candidate's file to the top-1 files of the neighbouring
segments (±1) of the same clip (0 with a no-neighbour flag otherwise). The track protocol has no neighbouring scenes,
so key / tempo are left out there.

Training data = the fold's VAL queries only (labels: hit = 1). Single head: one ranker per fold and protocol trained on
the 10 seeds' val rows pooled, applied to each seed's test; ensemble: trained on the ensemble's val rows. Hyper-
parameters (trees, leaves) by 2-fold grouped CV over val film groups, criterion = R@10 + 0.1 MRR of the re-ranked
shortlist; then refit on all val. Test scored once.
"""
from __future__ import annotations

import logging
import pickle
import time

import numpy as np

from exp_supp.rerank import methods as M
from exp_supp.rerank.cache import CACHE, HEADS, SEEDS
from exp_supp.rerank.data import gallery, load_split
from exp_supp.rerank.run import EVAL, N_FOLDS

log = logging.getLogger("exp_supp.rerank.ltr")
GRID = [{"n_estimators": n, "num_leaves": l} for n in (25, 50, 100, 200) for l in (3, 7)]
NAME = "LambdaMART (learned, val-trained)"


def mats(z, p: str, proto: str, hs_by_enc: dict) -> dict:
    out = {}
    for enc, hs in hs_by_enc.items():
        if proto == "seg":
            out[enc] = np.mean([z[f"{p}_seg_S"][h] for h in hs], 0)
        else:
            out[enc] = np.mean([z[f"{p}_clip_V"][h] @ z[f"{p}_file_F"][h].T for h in hs], 0)
    return out


def features(S: dict, g, proto: str):
    """Rows over every (query, shortlisted candidate): X [n, f], y [n], qid [n], cols [n], cluster per query."""
    base = S["both"]
    if proto == "seg":
        valid, hits = g.seg_valid, np.eye(len(base), dtype=bool)
    else:
        valid, hits = np.ones(base.shape, bool), g.trk_hits
    X, y, q, c = [], [], [], []
    for i in range(len(base)):
        v = valid[i]
        cols = M.topk_idx(np.where(v, base[i], M.NEG)[None], min(M.K_SHORT, int(v.sum())))[0]
        f = [base[i, cols], S["clap"][i, cols], S["mert"][i, cols], np.arange(1, len(cols) + 1, dtype=float)]
        if proto == "seg":
            nb = np.where((g.seg_pid == g.seg_pid[i]) & (np.abs(g.seg_k - g.seg_k[i]) <= 1) & (np.arange(len(base)) != i))[0]
            fc = g.seg_file[cols]
            if len(nb):
                top = g.seg_file[[int(np.argmax(np.where(g.seg_valid[j], base[j], M.NEG))) for j in nb]]
                kd = M.key_dist(g.key_pc[fc][None], g.key_mode[fc][None], g.key_pc[top][:, None], g.key_mode[top][:, None]).mean(0)
                td = M.tempo_dist(g.tempo[fc][None], g.tempo[top][:, None]).mean(0)
                has = np.ones(len(cols))
            else:
                kd = td = has = np.zeros(len(cols))
            f += [kd, td, has]
        X.append(np.stack(f, 1)), y.append(hits[i, cols].astype(int)), q.append(np.full(len(cols), i)), c.append(cols)
    return np.concatenate(X), np.concatenate(y), np.concatenate(q), np.concatenate(c)


def fit(X, y, q, params, seed=0):
    import lightgbm as lgb
    o = np.argsort(q, kind="stable")
    X, y, q = X[o], y[o], q[o]
    _, sizes = np.unique(q, return_counts=True)
    m = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.05, min_child_samples=10, lambdarank_truncation_level=20,
                       random_state=seed, deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1, **params)
    m.fit(X, y, group=sizes)
    return m


def ranks_from(model, S_base, X, q, cols, g, proto):
    R = np.full(S_base.shape, M.NEG)
    R[q, cols] = model.predict(X)
    if proto == "seg":
        return M.final_ranks(S_base, R, g.seg_valid, np.eye(len(S_base), dtype=bool))
    return M.final_ranks(S_base, R, np.ones(S_base.shape[1], bool), g.trk_hits)


def crit(r):
    return float((r <= 10).mean() + 0.1 * (1 / r).mean())


def choose(train_sets, gv, proto):
    """2-fold grouped CV over val film groups. train_sets: list of (S dict, X, y, q, cols) (one per seed)."""
    film = gv.seg_cluster if proto == "seg" else gv.trk_cluster
    fg = np.array(sorted(set(film)))
    half = set(fg[np.random.default_rng(148).permutation(len(fg))[: len(fg) // 2]])
    in_a = np.array([f in half for f in film])
    best, arg = -1.0, None
    for params in GRID:
        sc = []
        for tr_mask in (in_a, ~in_a):
            Xs, ys, qs = [], [], []
            for si, (S, X, y, q, cols) in enumerate(train_sets):
                m_ = tr_mask[q]
                Xs.append(X[m_]), ys.append(y[m_]), qs.append(q[m_] + si * 100000)
            if not np.concatenate(ys).any():
                continue
            mdl = fit(np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs), params)
            for S, X, y, q, cols in train_sets:
                te = ~tr_mask[q]
                r = ranks_from(mdl, S["both"], X[te], q[te], cols[te], gv, proto)
                keep = ~tr_mask
                sc.append(crit(r[keep]))
        c = float(np.mean(sc)) if sc else -1
        if c > best + 1e-12:
            best, arg = c, params
    return arg, best


def fold_run(k: int) -> None:
    from bgm.eval.store import FeatureStore
    from exp_supp.common import log_run
    from exp_supp.rerank.cache import cleaned_cfg
    t0 = time.time()
    store = FeatureStore(cleaned_cfg())
    split = load_split()
    gv, gt = gallery(store, split, k, "val"), gallery(store, split, k, "test")
    z = dict(np.load(CACHE / f"fold{k}.npz"))
    H = lambda enc, ss: [HEADS.index((enc, s)) for s in ss]  # noqa: E731
    res = {"single": {NAME: {}}, "ens": {NAME: {}}}
    for proto in ("seg", "trk"):
        for kind, sets in (("single", [[s] for s in SEEDS]), ("ens", [list(SEEDS)])):
            tr = []
            for ss in sets:
                S = mats(z, "val", proto, {e: H(e, ss) for e in ("both", "clap", "mert")})
                tr.append((S, *features(S, gv, proto)))
            params, vc = choose(tr, gv, proto)
            Xs = np.concatenate([t[1] for t in tr])
            ys = np.concatenate([t[2] for t in tr])
            qs = np.concatenate([t[3] + i * 100000 for i, t in enumerate(tr)])
            mdl = fit(Xs, ys, qs, params)
            out = {}
            for ss in sets:
                t1 = time.perf_counter()
                S = mats(z, "test", proto, {e: H(e, ss) for e in ("both", "clap", "mert")})
                X, y, q, cols = features(S, gt, proto)
                r = ranks_from(mdl, S["both"], X, q, cols, gt, proto)
                ms = 1000 * (time.perf_counter() - t1) / len(S["both"])
                out[ss[0] if kind == "single" else 0] = {"ranks": r, "pick": params, "val_crit": vc, "ms_per_query": ms}
                log_run("rerank", f"{kind}:{NAME}:{proto}", k, ss[0] if kind == "single" else 0, "test",
                        {"features": "cos,clap,mert,rank" + (",key,tempo,has_nb" if proto == "seg" else ""),
                         "train": "val queries", "grid": GRID, **params}, M.summary(r) | {"ms_per_query": ms})
            res[kind][NAME][proto] = out
            log.warning("fold %d %s %s: %s [%.0f s]", k, proto, kind, params, time.time() - t0)
    with open(EVAL / f"fold{k}_ltr.pkl", "wb") as fh:
        pickle.dump(res, fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    for k in range(N_FOLDS):
        fold_run(k)
