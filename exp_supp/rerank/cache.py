"""Embedding / score cache for the re-ranker study (cleaned title split, D116 heads; user 2026-11-21, D148).

`.venv/bin/python -m exp_supp.rerank.cache [--folds 0 1 2 3 4]` -> runs/exp_supp/rerank/cache/fold<k>.npz

Per fold and for every head h of runs/cleaned/semantic/{siglip-both, siglip-clap, siglip-mert}/fold<k>/seed0..9
(frozen; nothing is trained here), for split in (val, test):
  {split}_seg_S     [H, Nq, Nq]  exact segment_all scores: query i vs the aligned window of every test segment j
                                 re-cut to query i's duration (segment_gallery rule); columns are segment indices.
  {split}_seg_V     [H, Nq, d]   query segment embeddings
  {split}_seg_W     [H, Nq, d]   aligned window of each segment at its OWN duration (gallery-side neighbourhoods)
  {split}_clip_V    [H, Nc, d]   whole-clip embeddings (track protocol queries)
  {split}_file_F    [H, Nf, d]   whole-file embeddings (track_test_fold gallery = the split's music files)
  {split}_fw_E      [H, Nw, d]   fixed windows of every file (FW_LEN s, hop FW_HOP s; a file shorter than FW_LEN = one window)
  {split}_fw_file   [Nw]         file index of each fixed window
and the fold's training bank: bank_clip_V [H, Ntr, d] (all training clips), bank_seg_V [H, BANK_SEGS, d].
Head order: HEADS (both 0-9, clap 0-9, mert 0-9). Video inputs are the same for every head; music inputs are built once
per window ([l2 CLAP | l2 MERT]) and sliced per head's music_feature, so the 30 heads cost one feature pass.

Check (asserted): siglip-both seed s ranks from {test}_seg_S / clip_V @ file_F equal runs/cleaned/.../perquery.npz
(segment_all, track_test_fold), and the mean-cosine 10-head ensemble equals runs/cleaned/semantic_ens/siglip-both-ens10
seed 0 (members 0-9), up to float near-ties (|Δrank| <= 1 on <= 1% of queries).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from exp_supp.common import RUNS

log = logging.getLogger("exp_supp.rerank.cache")
CACHE = RUNS / "rerank" / "cache"
HEAD_ROOT = Path("runs/cleaned/semantic")
ENCS = ("both", "clap", "mert")
SEEDS = tuple(range(10))
HEADS = [(e, s) for e in ENCS for s in SEEDS]
FW_LEN, FW_HOP = 10.0, 5.0
BANK_SEGS = 1500
D_CLAP = 512


def cleaned_cfg():
    from bgm.analysis.cleaned_compare import CLEAN, make_cfg
    return make_cfg(CLEAN)


def load_heads(store, k: int, device: str) -> list:
    from bgm.analysis.full_dataset import load_scorer
    out = []
    for e, s in HEADS:
        sc = load_scorer(HEAD_ROOT / f"siglip-{e}" / f"fold{k}" / f"seed{s}", store, device, cached=False)
        want = {"both": ["sem_clap_1hz", "sem_mert_1hz"], "clap": ["sem_clap_1hz"], "mert": ["sem_mert_1hz"]}[e]
        assert sc.inp.m_keys == want, (e, sc.inp.m_keys)
        out.append(sc)
    return out


def slice_music(X: np.ndarray, enc: str) -> np.ndarray:
    """[N, 512 + 1024] [l2 CLAP | l2 MERT] inputs -> the head's own input."""
    return X if enc == "both" else (X[:, :D_CLAP] if enc == "clap" else X[:, D_CLAP:])


def embed_all(heads, side: str, X: np.ndarray, bs: int = 2048) -> np.ndarray:
    """[H, N, d] embeddings of inputs X (music: full [CLAP | MERT] vectors, sliced per head)."""
    out = []
    with torch.no_grad():
        for (e, _), sc in zip(HEADS, heads):
            Xi = slice_music(X, e) if side == "music" else X
            out.append(np.concatenate([sc.model.embed(side, Xi[i:i + bs]) for i in range(0, len(Xi), bs)]))
    return np.stack(out).astype(np.float32)


def fixed_windows(store, files: list[str]) -> list[tuple[int, float, float]]:
    out = []
    for fi, m in enumerate(files):
        dur = store.duration("music", m)
        if dur <= FW_LEN:
            out.append((fi, 0.0, dur))
            continue
        starts = list(np.arange(0.0, dur - FW_LEN + 1e-9, FW_HOP))
        if dur - FW_LEN - starts[-1] > 1e-6:
            starts.append(dur - FW_LEN)
        out += [(fi, float(s), FW_LEN) for s in starts]
    return out


def split_arrays(store, both_inp, heads, ids: list[str], which: str) -> dict:
    from bgm.eval.protocol import test_segments
    segs = test_segments(store, ids)
    files = list(dict.fromkeys(store.music_of[p] for p in ids))
    vid = lambda p, a, b: both_inp.video(p, a, b)  # noqa: E731
    mus = lambda m, s, d: both_inp.music_window(m, s, d)  # noqa: E731
    r = {}
    r["seg_V"] = embed_all(heads, "video", np.stack([vid(p, a, b) for p, _, a, b in segs]))
    own = []
    for p, _, a, b in segs:
        m, d = store.music_of[p], b - a
        own.append(mus(m, min(a, max(0.0, store.duration("music", m) - d)), d))
    r["seg_W"] = embed_all(heads, "music", np.stack(own))
    r["clip_V"] = embed_all(heads, "video", np.stack([vid(p, 0.0, store.duration("video", p)) for p in ids]))
    r["file_F"] = embed_all(heads, "music", np.stack([both_inp.music(m, 0.0, store.duration("music", m)) for m in files]))
    fw = fixed_windows(store, files)
    r["fw_E"] = embed_all(heads, "music", np.stack([mus(files[fi], s, d) for fi, s, d in fw]))
    r["fw_file"] = np.array([fi for fi, _, _ in fw])
    # exact segment_all scores: query i vs every aligned window re-cut to i's duration
    a_j = np.array([a for _, _, a, _ in segs])
    m_j = [store.music_of[p] for p, *_ in segs]
    dur_j = np.array([store.duration("music", m) for m in m_j])
    S = np.zeros((len(HEADS), len(segs), len(segs)), np.float32)
    for i, (p, _, a, b) in enumerate(segs):
        d = b - a
        starts = np.minimum(a_j, np.maximum(0.0, dur_j - d))
        E = embed_all(heads, "music", np.stack([mus(m, s0, d) for m, s0 in zip(m_j, starts)]))
        S[:, i] = np.einsum("hjd,hd->hj", E, r["seg_V"][:, i])
    r["seg_S"] = S
    return {f"{which}_{k}": v for k, v in r.items()}


def ranks_segment(S: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h = np.diag(S)[:, None]
    above = (S > h) & valid
    np.fill_diagonal(above, False)
    return 1 + above.sum(1)


def ranks_track(S: np.ndarray, hits: np.ndarray) -> np.ndarray:
    best = np.where(hits, S, -np.inf).max(1, keepdims=True)
    return 1 + ((S > best) & ~hits).sum(1)


def _close(got, want, what):
    d = np.abs(np.asarray(got) - np.asarray(want))
    assert d.max() <= 1 and (d > 0).mean() <= 0.01, (what, int((d > 0).sum()), int(d.max()))
    return int((d > 0).sum())


def check(store, split: dict, k: int, z: dict) -> None:
    from exp_supp.rerank.data import gallery
    g = gallery(store, split, k, "test")
    n_off = 0
    for s in SEEDS:
        h = HEADS.index(("both", s))
        per = np.load(HEAD_ROOT / "siglip-both" / f"fold{k}" / f"seed{s}" / "perquery.npz")
        n_off += _close(ranks_segment(z["test_seg_S"][h], g.seg_valid), per["segment_all"], (k, s, "seg"))
        n_off += _close(ranks_track(z["test_clip_V"][h] @ z["test_file_F"][h].T, g.trk_hits), per["track_test_fold"], (k, s, "trk"))
    ens = Path("runs/cleaned/semantic_ens/siglip-both-ens10") / f"fold{k}" / "seed0" / "perquery.npz"
    if ens.exists():
        per = np.load(ens)
        hb = [HEADS.index(("both", s)) for s in SEEDS]
        n_off += _close(ranks_segment(z["test_seg_S"][hb].mean(0), g.seg_valid), per["segment_all"], (k, "ens", "seg"))
        St = np.mean([z["test_clip_V"][h] @ z["test_file_F"][h].T for h in hb], 0)
        n_off += _close(ranks_track(St, g.trk_hits), per["track_test_fold"], (k, "ens", "trk"))
    log.warning("fold %d: cache reproduces the saved ranks (%d queries off by one rank over all checks)", k, n_off)


def build(k: int, device: str) -> Path:
    from bgm.eval.protocol import test_segments
    from bgm.eval.store import FeatureStore
    from bgm.train.semantic import SemanticInputs
    out = CACHE / f"fold{k}.npz"
    cfg = cleaned_cfg()
    store = FeatureStore(cfg)
    from exp_supp.rerank.data import load_split
    split = load_split()
    if out.exists():
        check(store, split, k, dict(np.load(out)))
        return out
    t0 = time.time()
    heads = load_heads(store, k, device)
    both_inp = heads[0].inp
    assert isinstance(both_inp, SemanticInputs) and both_inp.m_keys == ["sem_clap_1hz", "sem_mert_1hz"]
    f = split["folds"][str(k)]
    z = {}
    for which in ("val", "test"):
        z |= split_arrays(store, both_inp, heads, f[which], which)
        log.warning("fold %d %s cached [%.0f s]", k, which, time.time() - t0)
    bank = f["train"]
    z["bank_clip_V"] = embed_all(heads, "video", np.stack([both_inp.video(p, 0.0, store.duration("video", p)) for p in bank]))
    bsegs = test_segments(store, bank)
    sel = np.sort(np.random.default_rng([k, 148]).choice(len(bsegs), min(BANK_SEGS, len(bsegs)), replace=False))
    z["bank_seg_V"] = embed_all(heads, "video", np.stack([both_inp.video(bsegs[i][0], bsegs[i][2], bsegs[i][3]) for i in sel]))
    check(store, split, k, z)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **z)
    log.warning("fold %d -> %s [%.0f s]", k, out, time.time() - t0)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for k in a.folds:
        build(k, a.device)
