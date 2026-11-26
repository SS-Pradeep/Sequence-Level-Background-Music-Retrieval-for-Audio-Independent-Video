"""Similarity functions beyond mean-pooled cosine, on the title split (user 2026-11-20, D145).

Three changes to how a video is compared with music, all at test time on trained InfoNCE heads (no new loss):
  1. Hubness correction on the score matrix, with a query bank of the fold's TRAINING clips (never test queries):
     CSLS (Lample et al. 2018), inverted softmax (Smith et al. 2017), QB-Norm with dynamic inverted softmax
     (Bogolin et al. 2022).
  2. Max-sim / late interaction: the trained heads embed every 1 Hz second; score = 0.5 * (mean over video seconds of
     the best music second + mean over music seconds of the best video second). Also mean-cosine + max-sim blends.
  3. Per-encoder similarity: separate CLAP-only and MERT-only heads (configs REF_clap, REF_mert), scores fused.
Every weight / k / beta is chosen per run on the fold's `val` films (rule 3); test is scored once.

Pipeline:
    python -m bgm.sweep.similarity cache  [--workers 4]   # score matrices per checkpoint -> runs/titlesplit/similarity/
    python -m bgm.sweep.similarity table                  # selection on val, test, CIs -> results/titlesplit/similarity.md

Segment rows are exact: each query is scored against every window re-cut to its own duration (as segment_gallery
does). Hubness banks use each window at its own segment's duration (a window's hubness is a property of its music
region); this is the one approximation. Protocols are the M5 ones of `bgm.train.semantic.evaluate`: segment_all (gallery = the aligned window of every test
segment, other videos of the query's track left out, one hit) and track_test_fold (gallery = the test fold's music
files, every file of the query's track is a hit). `cache` asserts that its mean-cosine ranks equal the saved
perquery.npz of each run.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("bgm.sweep.similarity")
# BGM_SWEEP_DATA=cleaned (user 2026-11-22): the cleaned 1,011 pairs (cleaned_dataset/, cleaned folds_title.json), heads =
# the cleaned final-system InfoNCE heads (runs/cleaned/semantic/siglip-both, D116) and the D127 single-encoder heads
# (siglip-clap, siglip-mert); cache runs/cleaned/similarity, table results/cleaned/similarity.md. Unset = D145 as run.
import os  # noqa: E402
CLEANED = os.environ.get("BGM_SWEEP_DATA", "old") == "cleaned"
if CLEANED:
    SOURCES = {"both": "siglip-both", "clap": "siglip-clap", "mert": "siglip-mert"}
    MUSIC_FEATURE = {"siglip-both": "both", "siglip-clap": "sem_clap_1hz", "siglip-mert": "sem_mert_1hz"}
    HEADS_ROOT = Path("runs/cleaned")
    CACHE = Path("runs/cleaned/similarity")
    OUT = Path("results/cleaned")
    FOLDS_JSON = Path("cleaned_dataset/manifests/folds_title.json")
    PERQUERY_ROOT = Path("runs/cleaned/semantic")
else:
    SOURCES = {"both": "REF", "clap": "REF_clap", "mert": "REF_mert"}       # music encoder -> trained config
    CACHE = Path("runs/titlesplit/similarity")
    OUT = Path("results/titlesplit")
    FOLDS_JSON = Path("data/manifests/folds_title.json")
    PERQUERY_ROOT = Path("runs/titlesplit/planned_losses/semantic")
SEEDS = tuple(range(10))
N_FOLDS = 5
BANK_SEGS = 1500                     # training segments in the segment-protocol query bank (random, rng [fold, 145])


# ================================================================ pure scoring functions
def frames(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Rows of a [T, D] 1 Hz array inside [a, b) by the Series1Hz rule (k + 0.5 in [a, b)); the middle row if empty."""
    T = len(x)
    lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T))
    hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T))
    if hi <= lo:
        k = int(np.clip(np.floor((a + b) / 2), 0, T - 1))
        return x[k:k + 1]
    return x[lo:hi]


def maxsim(Q: "torch.Tensor", qm: "torch.Tensor", G: "torch.Tensor", gm: "torch.Tensor", chunk: int = 16) -> np.ndarray:
    """Symmetric max-sim. Q [Nq, Tq, d], G [Ng, Tg, d] unit vectors, qm / gm bool masks [N, T] of valid seconds.
    Returns [Nq, Ng]: 0.5 * (mean_i max_j <q_i, g_j> + mean_j max_i <q_i, g_j>) over valid seconds."""
    import torch
    out = []
    neg = torch.finfo(Q.dtype).min
    for s in range(0, len(Q), chunk):
        q, m = Q[s:s + chunk], qm[s:s + chunk]
        S = torch.einsum("aid,bjd->abij", q, G)                               # [c, Ng, Tq, Tg]
        valid = m[:, None, :, None] & gm[None, :, None, :]
        S = S.masked_fill(~valid, neg)
        r = S.max(3).values.masked_fill(~m[:, None, :], 0).sum(2) / m.sum(1)[:, None]
        c = S.max(2).values.masked_fill(~gm[None], 0).sum(2) / gm.sum(1)[None]
        out.append((0.5 * (r + c)).float().cpu().numpy())
    return np.concatenate(out)


def csls(S: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """CSLS ranking score 2 s(q, g) - r_g, r_g = mean of the k largest bank scores of gallery item g (the query's own
    r_q is constant within its row, so it is dropped). S [Nq, Ng], B [Nb, Ng]."""
    k = min(k, len(B))
    r = np.sort(B, 0)[-k:].mean(0)
    return 2 * S - r[None]


def inv_softmax(S: np.ndarray, B: np.ndarray, beta: float) -> np.ndarray:
    """Inverted softmax (log form): beta s(q, g) - log sum_b exp(beta s(b, g)), the bank normalising each gallery item."""
    z = beta * B
    m = z.max(0)
    lse = m + np.log(np.exp(z - m).sum(0))
    return beta * S - lse[None]


def qb_norm(S: np.ndarray, B: np.ndarray, beta: float) -> np.ndarray:
    """QB-Norm with the dynamic inverted softmax: rows whose raw top-1 is a bank hub (top-1 of some bank query) get
    the inverted softmax, the others keep their raw scores (ranking is within a row, so mixed scales are fine)."""
    hubs = np.zeros(S.shape[1], bool)
    hubs[np.unique(B.argmax(1))] = True
    act = hubs[S.argmax(1)]
    out = S.copy()
    if act.any():
        out[act] = inv_softmax(S[act], B, beta)
    return out


HUB = {"none": [None], "csls": [1, 3, 5, 10, 20, 50], "is": [0.5, 1, 2, 4, 8, 16], "qb": [0.5, 1, 2, 4, 8, 16]}


def apply_hub(S, B, method, p):
    return S if method == "none" else {"csls": csls, "is": inv_softmax, "qb": qb_norm}[method](S, B, p)


def hub_variants(S, B, hubs):
    """Yield (method, p, adjusted S) for every (method, p) in hubs, sharing the bank statistics (one sort, one
    hub set) across parameters. Same results as apply_hub."""
    srt = np.sort(B, 0) if any(m == "csls" for m, _ in hubs) else None
    act = None
    for m, p in hubs:
        if m == "none":
            yield m, p, S
        elif m == "csls":
            k = min(p, len(B))
            yield m, p, 2 * S - srt[-k:].mean(0)[None]
        elif m == "is":
            yield m, p, inv_softmax(S, B, p)
        else:
            if act is None:
                hubset = np.zeros(S.shape[1], bool)
                hubset[np.unique(B.argmax(1))] = True
                act = hubset[S.argmax(1)]
            out = S.copy()
            if act.any():
                out[act] = inv_softmax(S[act], B, p)
            yield m, p, out


def ranks_segment(S: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rank of the diagonal hit per row among the row's valid columns (rank_of_first_hit rule: strict >)."""
    h = np.diag(S)[:, None]
    above = (S > h) & valid
    np.fill_diagonal(above, False)
    return 1 + above.sum(1)


def ranks_track(S: np.ndarray, hits: np.ndarray) -> np.ndarray:
    best = np.where(hits, S, -np.inf).max(1, keepdims=True)
    return 1 + ((S > best) & ~hits).sum(1)


def summary(r: np.ndarray) -> dict:
    return {"R1": float((r <= 1).mean()), "R5": float((r <= 5).mean()), "R10": float((r <= 10).mean()),
            "MRR": float((1 / r).mean()), "MedR": float(np.median(r))}


# ================================================================ caching score matrices per checkpoint
def _ctx():
    import torch
    from bgm.eval.store import FeatureStore
    from bgm.sweep.titlesplit import load_cfg
    cfg = load_cfg()
    store = FeatureStore(cfg)
    split = json.loads(FOLDS_JSON.read_text())
    return cfg, store, split, "cuda" if torch.cuda.is_available() else "cpu"


def _items(store, ids: list[str]):
    """Segments (p, k, a, b) with their aligned windows (m, s, d) and the clips / music files of `ids`."""
    from bgm.eval.protocol import test_segments
    segs = test_segments(store, ids)
    wins = []
    for p, k, a, b in segs:
        m, d = store.music_of[p], b - a
        wins.append((m, min(a, max(0.0, store.duration("music", m) - d)), d))
    files = list(dict.fromkeys(store.music_of[p] for p in ids))
    return segs, wins, files


class Embedder:
    """Mean and per-second embeddings of one checkpoint (unit vectors, the trained heads)."""

    def __init__(self, cfg, store, name, fold, seed, device):
        from bgm.sweep.titlesplit import load_scorer
        if CLEANED:
            import copy
            cfg = copy.deepcopy(cfg)
            cfg.paths.runs = str(HEADS_ROOT)
            cfg.train.semantic.music_feature = MUSIC_FEATURE[name]
        self.sc = load_scorer(cfg, store, name, fold, seed, device)
        self.inp, self.store, self.device = self.sc.inp, store, device
        self.model = self.sc.model

    def _per_sec_video(self, p, a, b):
        from bgm.train.semantic import l2
        a, b = self.store.span("video", p, a, b)
        return np.stack([l2(r) for r in frames(self.store.series("video", p, self.inp.vf).x, a, b)]).astype(np.float32)

    def _per_sec_music(self, m, a, b):
        from bgm.eval.store import apply_span_rule
        from bgm.train.semantic import l2
        a, b = apply_span_rule(a, b, self.store.duration("music", m), self.store.min_span)
        parts = [np.stack([l2(r) for r in frames(self.store.series("music", m, k).x, a, b)]) for k in self.inp.m_keys]
        n = min(len(x) for x in parts)
        return np.concatenate([x[:n] for x in parts], 1).astype(np.float32)

    def embed_mean(self, side, X):
        out = [self.model.embed(side, X[i:i + 512]) for i in range(0, len(X), 512)]
        return np.concatenate(out).astype(np.float32)

    def embed_frames(self, side, seqs):
        """list of [T_i, D] raw -> padded [N, Tmax, d] unit vectors + mask [N, Tmax] (torch, on device)."""
        import torch
        flat = np.concatenate(seqs)
        e = self.embed_mean(side, flat)
        T = max(len(s) for s in seqs)
        E = np.zeros((len(seqs), T, e.shape[1]), np.float32)
        M = np.zeros((len(seqs), T), bool)
        o = 0
        for i, s in enumerate(seqs):
            E[i, :len(s)], M[i, :len(s)] = e[o:o + len(s)], True
            o += len(s)
        return torch.from_numpy(E).to(self.device).half(), torch.from_numpy(M).to(self.device)

    def segments(self, segs, frames_too=True):
        X = np.stack([self.inp.video(p, a, b) for p, _, a, b in segs])
        mean = self.embed_mean("video", X)
        return mean, (self.embed_frames("video", [self._per_sec_video(p, a, b) for p, _, a, b in segs]) if frames_too else None)

    def windows(self, wins):
        mean = self.embed_mean("music", np.stack([self.inp.music_window(m, s, d) for m, s, d in wins]))
        return mean, self.embed_frames("music", [self._per_sec_music(m, s, s + d) for m, s, d in wins])

    def clips(self, ids):
        st = self.store
        mean = self.embed_mean("video", np.stack([self.inp.video(p, 0.0, st.duration("video", p)) for p in ids]))
        return mean, self.embed_frames("video", [self._per_sec_video(p, 0.0, st.duration("video", p)) for p in ids])

    def file_frames(self, m):
        """[T, d] per-second music embeddings of a whole file (cached; torch half on device)."""
        if not hasattr(self, "_ff"):
            self._ff = {}
        if m not in self._ff:
            import torch
            raw = self._per_sec_music(m, 0.0, self.store.duration("music", m))
            self._ff[m] = torch.from_numpy(self.embed_mean("music", raw)).to(self.device).half()
        return self._ff[m]

    def seg_exact(self, segs, wins, qv_mean, qv_fr):
        """Exact segment-protocol scores [Nq, Nw]: query q against every window j re-cut to q's duration
        (segment_gallery rule: start = min(a_j, max(0, dur_j - d_q))), mean cosine and max-sim."""
        import torch
        from bgm.eval.store import apply_span_rule
        st = self.store
        Q, qm = qv_fr
        Sm = np.zeros((len(segs), len(wins)), np.float32)
        Sx = np.zeros_like(Sm)
        a_j = np.array([segs[j][2] for j in range(len(wins))])
        m_j = [w[0] for w in wins]
        dur_j = np.array([st.duration("music", m) for m in m_j])
        for i, (p, _, a, b) in enumerate(segs):
            d = b - a
            starts = np.minimum(a_j, np.maximum(0.0, dur_j - d))
            X = np.stack([self.inp.music_window(m, s0, d) for m, s0 in zip(m_j, starts)])
            Sm[i] = self.embed_mean("music", X) @ qv_mean[i]
            fr = []
            for m, s0, du in zip(m_j, starts, dur_j):
                F = self.file_frames(m)
                a2, b2 = apply_span_rule(s0, s0 + d, du, st.min_span)
                lo = int(np.clip(np.ceil(a2 - 0.5 - 1e-9), 0, len(F)))
                hi = int(np.clip(np.ceil(b2 - 0.5 - 1e-9), 0, len(F)))
                if hi <= lo:
                    k = int(np.clip(np.floor((a2 + b2) / 2), 0, len(F) - 1))
                    lo, hi = k, k + 1
                fr.append(F[lo:hi])
            T = max(len(f) for f in fr)
            G = torch.zeros((len(fr), T, F.shape[1]), dtype=F.dtype, device=F.device)
            gm = torch.zeros((len(fr), T), dtype=torch.bool, device=F.device)
            for j, f in enumerate(fr):
                G[j, :len(f)], gm[j, :len(f)] = f, True
            Sx[i] = maxsim(Q[i:i + 1], qm[i:i + 1], G, gm)[0]
        return Sm, Sx

    def files(self, files):
        st = self.store
        mean = self.embed_mean("music", np.stack([self.inp.music(m, 0.0, st.duration("music", m)) for m in files]))
        return mean, self.embed_frames("music", [self._per_sec_music(m, 0.0, st.duration("music", m)) for m in files])


def bank_ids(split, fold):
    return split["folds"][str(fold)]["train"]


def cache_one(cfg, store, split, enc: str, fold: int, seed: int, device: str) -> Path:
    """Score matrices (mean cosine and max-sim) of one checkpoint for the val and test galleries and their banks."""
    out = CACHE / enc / f"fold{fold}" / f"seed{seed}.npz"
    if out.exists():
        return out
    from bgm.eval.protocol import test_segments
    t0 = time.time()
    E = Embedder(cfg, store, SOURCES[enc], fold, seed, device)
    f = split["folds"][str(fold)]
    bank = bank_ids(split, fold)
    bsegs = test_segments(store, bank)
    sel = np.sort(np.random.default_rng([fold, 145]).choice(len(bsegs), min(BANK_SEGS, len(bsegs)), replace=False))
    bsegs = [bsegs[i] for i in sel]
    bv_mean, bv_fr = E.segments(bsegs)
    bc_mean, bc_fr = E.clips(bank)
    res = {}
    for which in ("val", "test"):
        ids = f[which]
        segs, wins, files = _items(store, ids)
        qv, qv_fr = E.segments(segs)
        wm, wm_fr = E.windows(wins)
        cv, cv_fr = E.clips(ids)
        fm, fm_fr = E.files(files)
        seg_mean, seg_max = E.seg_exact(segs, wins, qv, qv_fr)
        res |= {f"{which}_seg_mean": seg_mean, f"{which}_seg_max": seg_max,
                f"{which}_segbank_mean": bv_mean @ wm.T, f"{which}_segbank_max": maxsim(*bv_fr, *wm_fr),
                f"{which}_trk_mean": cv @ fm.T, f"{which}_trk_max": maxsim(*cv_fr, *fm_fr),
                f"{which}_trkbank_mean": bc_mean @ fm.T, f"{which}_trkbank_max": maxsim(*bc_fr, *fm_fr)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v.astype(np.float32) for k, v in res.items()})
    log.warning("cached %s fold %d seed %d in %.0f s", enc, fold, seed, time.time() - t0)
    return out


def check_reproduces(store, split, fold, seed, enc="both"):
    """Mean-cosine ranks from the cache == the run's saved perquery.npz (segment_all, track_test_fold)."""
    z = np.load(CACHE / enc / f"fold{fold}" / f"seed{seed}.npz")
    g = gallery_masks(store, split, fold, "test")
    per = np.load(PERQUERY_ROOT / SOURCES[enc] / f"fold{fold}" / f"seed{seed}" / "perquery.npz")
    rs = ranks_segment(z["test_seg_mean"], g["seg_valid"])
    rt = ranks_track(z["test_trk_mean"], g["trk_hits"])
    for got, want, what in ((rs, per["segment_all"], "segment"), (rt, per["track_test_fold"], "track")):
        d = np.abs(got - want)
        # float near-ties (different embedding batch sizes) may move a rank by 1 for a handful of queries
        assert d.max() <= 1 and (d > 0).mean() <= 0.01, (enc, fold, seed, what, int((d > 0).sum()), int(d.max()))


_GM: dict = {}


def gallery_masks(store, split, fold, which):
    key = (fold, which)
    if key not in _GM:
        ids = split["folds"][str(fold)][which]
        segs, _, files = _items(store, ids)
        tr = np.array([store.track[p] for p, *_ in segs])
        pid = np.array([p for p, *_ in segs])
        valid = ~((tr[:, None] == tr[None]) & (pid[:, None] != pid[None]))      # other videos of the same track out
        ftr = np.array([store.music_track[m] for m in files])
        ctr = np.array([store.track[p] for p in ids])
        film = split["film_group"]
        _GM[key] = {"seg_valid": valid, "trk_hits": ctr[:, None] == ftr[None],
                    "seg_cluster": np.array([film[p] for p in pid]), "trk_cluster": np.array([film[p] for p in ids])}
    return _GM[key]


def cmd_cache(workers: int, encs=None, folds=None, seeds=None) -> None:
    encs, folds, seeds = encs or list(SOURCES), folds if folds is not None else range(N_FOLDS), seeds or SEEDS
    if workers > 1:
        import os
        import subprocess
        import sys
        jobs = [(e, k) for e in encs for k in folds]
        groups = [jobs[i::workers] for i in range(workers)]
        env = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        procs = []
        for g in groups:
            if not g:
                continue
            args = [sys.executable, "-m", "bgm.sweep.similarity", "cache", "--workers", "1", "--jobs",
                    *[f"{e}:{k}" for e, k in g], "--seeds", *map(str, seeds)]
            procs.append(subprocess.Popen(args, env=env))
        codes = [p.wait() for p in procs]
        assert not any(codes), codes
        return
    cfg, store, split, device = _ctx()
    for e in encs:
        for k in folds:
            for s in seeds:
                cache_one(cfg, store, split, e, k, s, device)
                check_reproduces(store, split, k, s, e)


# ================================================================ methods: fusion + hubness, chosen on val
def load(enc, fold, seed):
    return np.load(CACHE / enc / f"fold{fold}" / f"seed{seed}.npz")


def _std(Q, B):
    mu, sd = B.mean(), B.std() + 1e-8
    return (Q - mu) / sd, (B - mu) / sd


def matrices(Z: dict, which: str, proto: str, enc: str, kind: str):
    """(query x gallery, bank x gallery), standardised by the bank's mean / std (an affine map: rankings unchanged)."""
    return _std(Z[enc][f"{which}_{proto}_{kind}"], Z[enc][f"{which}_{proto}bank_{kind}"])


def fused(Z, which, proto, w_enc: dict, alpha: float):
    """sum_enc w_enc * [(1 - alpha) * mean + alpha * maxsim], each term bank-standardised."""
    Q = B = 0.0
    for enc, w in w_enc.items():
        if w == 0:
            continue
        for kind, a in (("mean", 1 - alpha), ("max", alpha)):
            if a == 0:
                continue
            q, b = matrices(Z, which, proto, enc, kind)
            Q, B = Q + w * a * q, B + w * a * b
    return Q, B


def ranks(proto, S, g):
    return ranks_segment(S, g["seg_valid"]) if proto == "seg" else ranks_track(S, g["trk_hits"])


# method name -> grid of (w_enc, alpha) and the hub methods it may use
W11 = [round(x, 1) for x in np.linspace(0, 1, 11)]
A5 = [0.0, 0.25, 0.5, 0.75, 1.0]
HUB_ALL = [(m, p) for m, ps in HUB.items() for p in ps]
METHODS = {
    "REF (cosine, mean-pooled)": {"w": [{"both": 1}], "a": [0.0], "hub": [("none", None)]},
    "REF + CSLS": {"w": [{"both": 1}], "a": [0.0], "hub": [("csls", p) for p in HUB["csls"]]},
    "REF + inverted softmax": {"w": [{"both": 1}], "a": [0.0], "hub": [("is", p) for p in HUB["is"]]},
    "REF + QB-Norm": {"w": [{"both": 1}], "a": [0.0], "hub": [("qb", p) for p in HUB["qb"]]},
    "REF max-sim": {"w": [{"both": 1}], "a": [1.0], "hub": [("none", None)]},
    "REF mean + max-sim (α on val)": {"w": [{"both": 1}], "a": A5, "hub": [("none", None)]},
    "CLAP head only": {"w": [{"clap": 1}], "a": [0.0], "hub": [("none", None)]},
    "MERT head only": {"w": [{"mert": 1}], "a": [0.0], "hub": [("none", None)]},
    "CLAP + MERT fusion (w on val)": {"w": [{"clap": w, "mert": 1 - w} for w in W11], "a": [0.0], "hub": [("none", None)]},
    "CLAP + MERT fusion + max-sim": {"w": [{"clap": w, "mert": 1 - w} for w in W11], "a": A5, "hub": [("none", None)]},
    "Combination: fusion + max-sim + hubness": {"w": [{"clap": w, "mert": 1 - w} for w in W11], "a": A5, "hub": HUB_ALL},
    "All three heads + max-sim + hubness": {"w": [{"both": 1, "clap": w, "mert": 1 - w} for w in W11], "a": A5, "hub": HUB_ALL},
}


def select_and_test(Z, store, split, fold, proto, spec):
    """Grid over spec on val (criterion R@10 + 0.1 MRR; first in grid wins ties), then score test once."""
    gv, gt = gallery_masks(store, split, fold, "val"), gallery_masks(store, split, fold, "test")
    best, arg = -1.0, None
    for w in spec["w"]:
        for a in spec["a"]:
            Q, B = fused(Z, "val", proto, w, a)
            for m, p, S in hub_variants(Q, B, spec["hub"]):
                r = ranks(proto, S, gv)
                crit = (r <= 10).mean() + 0.1 * (1 / r).mean()
                if crit > best + 1e-12:
                    best, arg = crit, (w, a, m, p)
    w, a, m, p = arg
    Q, B = fused(Z, "test", proto, w, a)
    return ranks(proto, apply_hub(Q, B, m, p), gt), {"w": w, "alpha": a, "hub": m, "p": p}


def ensemble(fold, seeds, encs):
    """10-head ensemble per encoder: mean over seeds of the bank-standardised matrices (stored back as 'mean'/'max')."""
    out = {}
    for e in encs:
        zs = [load(e, fold, s) for s in seeds]
        d = {}
        for which in ("val", "test"):
            for proto in ("seg", "trk"):
                for kind in ("mean", "max"):
                    qs, bs = zip(*[_std(z[f"{which}_{proto}_{kind}"], z[f"{which}_{proto}bank_{kind}"]) for z in zs])
                    d[f"{which}_{proto}_{kind}"], d[f"{which}_{proto}bank_{kind}"] = np.mean(qs, 0), np.mean(bs, 0)
        out[e] = d
    return out


def cmd_table(seeds=SEEDS) -> str:
    from bgm.eval.metrics import cluster_bootstrap
    _, store, split, _ = _ctx()
    encs = list(SOURCES)
    rows = {n: {"seg": {"single": [], "hits": [], "ens": [], "ens_r": [], "picks": []},
                "trk": {"single": [], "hits": [], "ens": [], "ens_r": [], "picks": []}} for n in METHODS}
    clusters = {"seg": [], "trk": []}
    for k in range(N_FOLDS):
        for proto in ("seg", "trk"):
            clusters[proto].append(gallery_masks(store, split, k, "test")[f"{proto}_cluster"])
        per_seed = {n: {"seg": [], "trk": []} for n in METHODS}
        for s in seeds:
            Z = {e: load(e, k, s) for e in encs}
            for n, spec in METHODS.items():
                for proto in ("seg", "trk"):
                    r, pick = select_and_test(Z, store, split, k, proto, spec)
                    rows[n][proto]["single"].append(summary(r))
                    rows[n][proto]["picks"].append(pick)
                    per_seed[n][proto].append(r <= 10)
        Zens = ensemble(k, seeds, encs)
        for n, spec in METHODS.items():
            for proto in ("seg", "trk"):
                rows[n][proto]["hits"].append(np.mean(per_seed[n][proto], 0))
                r, pick = select_and_test(Zens, store, split, k, proto, spec)
                rows[n][proto]["ens"].append(summary(r))
                rows[n][proto]["ens_r"].append(r)
        log.warning("fold %d done", k)
    ref = "REF (cosine, mean-pooled)"
    cl = {p: np.concatenate(c) for p, c in clusters.items()}
    L = ["# Similarity functions on the title split (" + ("cleaned dataset, 1,011 pairs, D149" if CLEANED else "D145") + ")", "",
         ("Cleaned dataset (cleaned_dataset/, 1,011 pairs), title split (cleaned_dataset/manifests/folds_title.json), heads = "
          "runs/cleaned/semantic/siglip-both (D116; the final system's heads) and the D127 single-encoder heads siglip-clap / "
          "siglip-mert. " if CLEANED else "") +
         f"Title split (folds_title.json), 5 folds × {len(seeds)} seeds. Every head is the REF trainer (InfoNCE, siglip-both "
         "settings, early stopping on val); only the test-time similarity changes. Hubness banks = the fold's training "
         "clips (1,500 random training segments for the segment protocol, all training clips for the track protocol); "
         "no test query is used to normalise anything. Every weight (CLAP vs MERT w, max-sim α) and every hubness "
         "parameter (CSLS k, inverted-softmax / QB-Norm β) is chosen per run on the fold's val films (criterion R@10 + "
         "0.1·MRR), then test is scored once. Single = one head per run, mean ± std over 50 runs; Ensemble = mean of the "
         "10 seeds' standardised scores per fold, mean ± std over 5 folds. CI: paired vs REF on per-query hit@10 "
         "(single: averaged over seeds), film-group cluster bootstrap, 10,000 resamples; * = excludes 0.", ""]
    for proto, title in (("seg", "Segment retrieval (segment_all, one hit, gallery ≈ " + ("420" if CLEANED else "480") + " windows)"),
                         ("trk", "Track retrieval (track_test_fold, gallery ≈ " + ("179" if CLEANED else "196") + " files)")):
        L += [f"## {title}", "",
              "| similarity | single R@10 | single R@1 | single MRR | Δ R@10 vs REF [95% CI] | ensemble R@10 | ensemble MRR | Δ ens R@10 [95% CI] | typical val pick |",
              "|---|---|---|---|---|---|---|---|---|"]
        href = np.concatenate(rows[ref][proto]["hits"])
        eref = np.concatenate(rows[ref][proto]["ens_r"]) <= 10
        for n in METHODS:
            R = rows[n][proto]
            sm = lambda xs, c: f"{np.mean([x[c] for x in xs]):.3f} ± {np.std([x[c] for x in xs], ddof=1):.3f}"
            if n == ref:
                d1 = d2 = "—"
            else:
                b1 = cluster_bootstrap(np.concatenate(R["hits"]) - href, cl[proto], 10000, 0)
                b2 = cluster_bootstrap((np.concatenate(R["ens_r"]) <= 10).astype(float) - eref, cl[proto], 10000, 0)
                f = lambda b: f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]" + ("*" if b["lo"] > 0 or b["hi"] < 0 else "")
                d1, d2 = f(b1), f(b2)
            L.append(f"| {n} | {sm(R['single'], 'R10')} | {sm(R['single'], 'R1')} | {sm(R['single'], 'MRR')} | {d1} | "
                     f"{sm(R['ens'], 'R10')} | {sm(R['ens'], 'MRR')} | {d2} | {_pick_summary(R['picks'])} |")
        L.append("")
    md = "\n".join(L) + "\n"
    (OUT / "similarity.md").write_text(md)
    print(md, flush=True)
    return md


def _pick_summary(picks) -> str:
    from collections import Counter
    hubs = Counter(p["hub"] if p["hub"] == "none" else f"{p['hub']} {p['p']}" for p in picks).most_common(1)[0][0]
    a = np.median([p["alpha"] for p in picks])
    wc = [p["w"].get("clap") for p in picks if "clap" in p["w"]]
    s = f"α {a:g}"
    if wc:
        s += f", w_CLAP {np.median(wc):g}"
    if hubs != "none":
        s += f", {hubs}"
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "table"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", nargs="+", default=None, help="enc:fold pairs (internal)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    if a.cmd == "cache":
        if a.jobs:
            cfg_, store_, split_, dev_ = _ctx()
            for j in a.jobs:
                e, k = j.split(":")
                for s in a.seeds:
                    cache_one(cfg_, store_, split_, e, int(k), s, dev_)
                    check_reproduces(store_, split_, int(k), s, e)
        else:
            cmd_cache(a.workers, seeds=a.seeds)
    else:
        cmd_table(tuple(a.seeds))
