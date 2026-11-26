"""Why the alternative losses of D124 do not beat InfoNCE (PROGRESS.md D127). Cleaned dataset, title split.

python -m bgm.analysis.loss_why   -> reports/cleaned/loss/why.json (read by bgm.analysis.ablation)

1. **Where each loss pushes.** On simulated training batches (one clip per video, film cap 30, batch 128: the D116
   recipe) scored by the shipped InfoNCE heads (seeds 0-2), the gradient each loss puts on each in-batch negative
   logit, video -> music direction, normalised per row:
     InfoNCE, DCL   softmax_j(s_ij / tau)            (DCL changes only the positive's term)
     HN-NCE         softmax_j((1 + beta) s_ij / tau)  (beta 0.5, the trained weights' reweighting)
     sigmoid        sigmoid(s_ij / tau + b), b = -10 (the bias is not saved; -10 is its init)
     VSE++          all of it on the hardest negative, only when margin - s_ii + s_ij > 0
   and the share of that push landing on same-film music and on near-duplicates of the query's own music (same film,
   frozen MERT and CLAP whole-file cosines both above the 95th percentile of different-film pairs: the D52 rule).
   Uniform = the share of negatives that are same-film (what a loss blind to difficulty would push).
2. **Train vs test.** Each loss's own heads (seeds 0-2): track R@10 on training clips against a size-matched random
   gallery of training music (as D52 0.4) and on the test fold. A larger gap = fitting the training set's noise.
3. **Geometry.** Each loss's own heads on the test fold: mean cosine of true pairs, of same-film and of other-film
   negatives, the true pair's margin over its hardest negative, and the uniformity of music embeddings
   (log E exp(-2 |x - y|^2), Wang & Isola 2020; lower = more spread).
4. **Training.** From every run's metrics.json (seeds 0-9): epochs run, best epoch, val segment R@10.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from bgm.analysis.cleaned_compare import CLEAN, make_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.store import FeatureStore, fold_ids
from bgm.train.semantic import cap_per_film

log = logging.getLogger(__name__)
ROOT = Path(CLEAN["paths.runs"]) / "semantic"
OUT = Path("reports/cleaned/loss/why.json")
LOSSES = ("siglip-both", "siglip-both-sigmoid", "siglip-both-hnnce", "siglip-both-dcl", "siglip-both-vsemax")
SEEDS = (0, 1, 2)
BATCH, CAP, EPOCHS, BETA, MARGIN, SIG_BIAS = 128, 30, 5, 0.5, 0.2, -10.0


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def push_weights(s: np.ndarray, neg: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    """Per-row share of each loss's gradient on its negatives. s: [B, B] cosines (row = video, column = music,
    diagonal = true pairs); neg: [B, B] bool, the negatives (off-diagonal, other track). Returns {loss: [B, B]}, each row
    summing to 1 over its negatives (0 for a VSE++ row whose hinge is inactive, or a row without negatives)."""
    def norm(w):
        w = np.where(neg, w, 0.0)
        tot = w.sum(1, keepdims=True)
        return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    lg = np.where(neg, s / tau, -np.inf)
    mx = np.max(lg, 1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = {"info_nce": norm(np.exp(lg - mx)), "hn_nce": norm(np.exp((1 + BETA) * (lg - mx))),
           "sigmoid": norm(1.0 / (1.0 + np.exp(-(s / tau + SIG_BIAS))))}
    out["dcl"] = out["info_nce"]
    hard = np.where(neg, s, -np.inf)
    j = np.argmax(hard, 1)
    v = np.zeros_like(s)
    active = np.isfinite(hard[np.arange(len(s)), j]) & (MARGIN - np.diag(s) + hard[np.arange(len(s)), j] > 0)
    v[np.arange(len(s))[active], j[active]] = 1.0
    out["vse_max"] = v
    return out


def uniformity(x: np.ndarray, t: float = 2.0) -> float:
    """log E exp(-t |x_i - x_j|^2) over pairs i != j of unit vectors x [N, D] (Wang & Isola 2020)."""
    d2 = np.clip(2.0 - 2.0 * x @ x.T, 0.0, None)
    iu = np.triu_indices(len(x), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]))))


def _embed(sc, store, ids, side):
    inp = sc.inp
    if side == "video":
        return np.stack([sc._emb(("tv", q), "video", lambda q=q: inp.video(q, 0.0, store.duration("video", q))) for q in ids])
    return np.stack([sc._emb(("tm", m), "music", lambda m=m: inp.music(m, 0.0, store.duration("music", m))) for m in ids])


def near_dup_rule(store, music: list[str], film_of_music: dict, rng) -> tuple[dict, float, float]:
    """Frozen whole-file MERT / CLAP unit vectors and the two 95th percentiles of different-film pair cosines."""
    emb = {m: (l2(store.music(m)["sem_mert_1hz"].astype(np.float32).mean(0)), l2(store.music(m)["sem_clap_1hz"].astype(np.float32).mean(0)))
           for m in music}
    nm, nc = [], []
    while len(nm) < 20000:
        a, b = music[rng.integers(len(music))], music[rng.integers(len(music))]
        if film_of_music[a] != film_of_music[b]:
            nm.append(float(emb[a][0] @ emb[b][0]))
            nc.append(float(emb[a][1] @ emb[b][1]))
    return emb, float(np.percentile(nm, 95)), float(np.percentile(nc, 95))


def run() -> dict:
    cfg = make_cfg(CLEAN)
    store = FeatureStore(cfg)
    film = manifest.read_folds(cfg)["film_group"]
    film_of_music = {store.music_of[p]: film[p] for p in film if p in store.music_of}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(127)
    emb, p95_m, p95_c = near_dup_rule(store, sorted(film_of_music), film_of_music, rng)
    push = {k: {"same_film": [], "near_dup": []} for k in ("info_nce", "sigmoid", "hn_nce", "dcl", "vse_max", "uniform")}
    vse_active = []
    per_loss = {v: {"train_R@10": [], "test_R@10": [], "pos_cos": [], "neg_cos_same_film": [], "neg_cos_other_film": [],
                    "margin_vs_hardest": [], "uniformity": []} for v in LOSSES}
    for k in range(5):
        f = fold_ids(cfg, k)
        tr, te = list(f["train"]), list(f["test"])
        tr_mus = list(dict.fromkeys(store.music_of[p] for p in tr))
        te_mus = list(dict.fromkeys(store.music_of[p] for p in te))
        n_gal = len(te_mus)
        tr_track = np.array([store.track[p] for p in tr])
        tr_film = [film[p] for p in tr]
        for v in LOSSES:
            for s in SEEDS:
                sc = load_scorer(ROOT / v / f"fold{k}" / f"seed{s}", store, dev)
                V, M = _embed(sc, store, tr, "video"), _embed(sc, store, tr_mus, "music")
                mpos = {m: i for i, m in enumerate(tr_mus)}
                own = np.array([mpos[store.music_of[p]] for p in tr])
                mtrack = np.array([store.music_track[m] for m in tr_mus])
                S = V @ M.T
                # 2. train recall, size-matched gallery
                hits = []
                for qi in range(len(tr)):
                    r = np.random.default_rng([77, k, qi])
                    others = r.choice(np.flatnonzero(mtrack != tr_track[qi]), n_gal - 1, replace=False)
                    hits.append(1 + int((S[qi, others] > S[qi, own[qi]]).sum()) <= 10)
                per_loss[v]["train_R@10"].append(float(np.mean(hits)))
                with np.load(ROOT / v / f"fold{k}" / f"seed{s}" / "perquery.npz") as z:
                    per_loss[v]["test_R@10"].append(float((z["track_test_fold"] <= 10).mean()))
                # 3. test geometry
                Vt, Mt = _embed(sc, store, te, "video"), _embed(sc, store, te_mus, "music")
                tpos = {m: i for i, m in enumerate(te_mus)}
                St = Vt @ Mt.T
                t_own = np.array([tpos[store.music_of[p]] for p in te])
                t_neg = np.array([[store.music_track[m] != store.track[p] for m in te_mus] for p in te])
                t_sf = np.array([[film_of_music[m] == film[p] for m in te_mus] for p in te]) & t_neg
                pos = St[np.arange(len(te)), t_own]
                per_loss[v]["pos_cos"].append(float(pos.mean()))
                per_loss[v]["neg_cos_same_film"].append(float(St[t_sf].mean()) if t_sf.any() else np.nan)
                per_loss[v]["neg_cos_other_film"].append(float(St[t_neg & ~t_sf].mean()))
                per_loss[v]["margin_vs_hardest"].append(float(np.mean(pos - np.where(t_neg, St, -np.inf).max(1))))
                per_loss[v]["uniformity"].append(uniformity(Mt))
                # 1. push shares on simulated batches, with the InfoNCE heads only
                if v != "siglip-both":
                    continue
                tau = float(np.exp(sc.model.log_tau.item()))
                for e in range(EPOCHS):
                    r = np.random.default_rng([127, k, s, e])
                    idx = cap_per_film(tr_film, CAP, r)
                    idx = idx[r.permutation(len(idx))]
                    for b in range(0, len(idx) - BATCH + 1, BATCH):
                        ix = idx[b:b + BATCH]
                        cols = own[ix]
                        sb = V[ix] @ M[cols].T
                        neg = (tr_track[ix][:, None] != tr_track[ix][None, :])
                        sf = np.array([[tr_film[i] == tr_film[j] for j in ix] for i in ix]) & neg
                        mq = [store.music_of[tr[i]] for i in ix]
                        nd = sf & np.array([[emb[a][0] @ emb[c][0] > p95_m and emb[a][1] @ emb[c][1] > p95_c for c in mq] for a in mq])
                        W = push_weights(sb, neg, tau)
                        W["uniform"] = neg / np.maximum(neg.sum(1, keepdims=True), 1)
                        rows = neg.any(1)
                        for name, w in W.items():
                            live = rows & (w.sum(1) > 0)
                            push[name]["same_film"].append((w * sf)[live].sum(1))
                            push[name]["near_dup"].append((w * nd)[live].sum(1))
                        vse_active.append(W["vse_max"].sum(1)[rows])
            log.info("fold %d %s: train R@10 %.3f, test %.3f", k, v, np.mean(per_loss[v]["train_R@10"][-3:]),
                     np.mean(per_loss[v]["test_R@10"][-3:]))
    training = {}
    for v in LOSSES:
        ms = [json.loads(p.read_text()) for p in sorted((ROOT / v).glob("fold*/seed*/metrics.json"))]
        training[v] = {"epochs_run": float(np.mean([m["val"]["epochs_run"] for m in ms])),
                       "best_epoch": float(np.mean([m["val"]["best_epoch"] for m in ms])),
                       "val_segment_R@10": float(np.mean([m["val"]["segment_R@10"] for m in ms])), "runs": len(ms)}
    out = {"near_dup_p95": {"mert": p95_m, "clap": p95_c},
           "push": {n: {k: float(np.mean(np.concatenate(x))) for k, x in d.items()} for n, d in push.items()},
           "vse_active_rows": float(np.mean(np.concatenate(vse_active))),
           "heads": {v: {k: float(np.nanmean(x)) for k, x in d.items()} for v, d in per_loss.items()},
           "heads_std": {v: {k: float(np.nanstd(x, ddof=1)) for k, x in d.items()} for v, d in per_loss.items()},
           "training": training}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    log.info("wrote %s", OUT)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run()
