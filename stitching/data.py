"""Build the shared episodes of one fold (imports the existing pipeline; edits nothing).

`.venv/bin/python -m stitching.data --fold k [--n-train 200] [--device cuda]` -> stitching/cache/fold<k>.pkl

* Segmentation: step 1 = TransNetV2 shots + frozen BaSSL + the D142 "C" boundary head (contrast + BiGRU), its own cuts
  (no forced joins), theta tuned on val per fold (data/interim/segmentation/retrieval_segments.json). Val / test =
  the frozen title-split blocked sequences (D143). Train = 'blocked' sequences drawn from fold k's train clips with
  per-item seeds [0, k, i], cut by the same head and theta. No PySceneDetect anywhere.
* Shortlist / lattice: the D121 title-split emission (10-head ensemble, fusion.json of runs/titlesplit/pipeline_ens10),
  `make_lattice` = coarse top-K files, each scene's top-J windows + contiguous continuations of every previous
  candidate. Identical for every method.
* sim = the raw ensemble semantic score (w_fine is semantic-only, so S_fused = (raw - mu) / sd exactly; inverted here).
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from stitching.base import Cands, Episode

log = logging.getLogger(__name__)
CACHE = Path("stitching/cache")
RATE = 25


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class MusicDesc:
    """Per-window music descriptors from the cached features (MERT 1 Hz, rms_db at 25 Hz, beat times)."""

    def __init__(self, store):
        self.store, self.c = store, {}

    def _get(self, m):
        if m not in self.c:
            d = self.store.music(m)
            b = np.asarray(d["beat_times_s"], float)
            ibi = float(np.median(np.diff(b))) if len(b) > 2 else 0.0
            self.c[m] = (np.asarray(d["sem_mert_1hz"], np.float32), np.asarray(d["dyn_25hz"], np.float32)[:, 2], b, ibi)
        return self.c[m]

    def __call__(self, m: str, off: np.ndarray, d: float) -> dict:
        mert, rms, b, ibi = self._get(m)
        S, N = len(mert), len(rms)
        off = np.asarray(off, float)
        end = off + d
        i0 = np.clip(np.floor(off).astype(int), 0, S - 1)
        i1 = np.clip(np.ceil(end).astype(int), 1, S)
        emb_in = l2(np.stack([mert[a:min(a + 2, S)].mean(0) for a in i0]))
        emb_out = l2(np.stack([mert[max(e - 2, 0):e].mean(0) for e in i1]))
        f0 = np.clip((off * RATE).astype(int), 0, N - 1)
        f1 = np.clip((end * RATE).astype(int), 1, N)
        rms_in = np.array([rms[a:min(a + 2 * RATE, N)].mean() for a in f0])
        rms_out = np.array([rms[max(e - 2 * RATE, 0):e].mean() for e in f1])

        def phase(x):
            if ibi <= 0:
                return np.zeros(len(x))
            j = np.clip(np.searchsorted(b, x, side="right") - 1, 0, len(b) - 1)
            return ((x - b[j]) / ibi) % 1.0
        return {"emb_in": emb_in, "emb_out": emb_out, "rms_in": rms_in, "rms_out": rms_out, "ph_in": phase(off), "ph_out": phase(end)}


def build_episode(key, store, index, models, fusion, cfg, desc: MusicDesc, parts, cuts_rel, with_gold: bool) -> Episode:
    from bgm.cue.evaluate import make_lattice
    from bgm.retrieval.index import concat_query
    from bgm.retrieval.pipeline import truth_for
    from bgm.segmentation.retrieval_eval import from_rel, overlap_pieces, segments_from_cuts, true_changes
    assert fusion["w_fine"].get("dyn", 0) == 0 and fusion["w_fine"].get("attr", 0) == 0, "sim inversion needs sem-only fine fusion"
    vq = concat_query(store, parts)
    spans = [(t0, d) for _, t0, d in vq.meta["parts"]]
    starts, durs = [s[0] for s in spans], [s[1] for s in spans]
    tracks = [store.track[p] for p in parts]
    segs = segments_from_cuts(from_rel(cuts_rel, starts, durs), vq.duration)
    vq.segments = segs
    vq.prepare(models, store.min_span)
    lat, res = make_lattice(store, index, models, vq, fusion, cfg)
    wsem = float(fusion["w_fine"]["sem"])
    # local file table: shortlist union (+ gold files)
    pos = sorted(set(int(x) for r in res for x in np.unique(r.fi)) | set(int(x) for L in lat.layers for x in np.unique(L.mid)))
    golds = [truth_for(vq, store, a, b) for a, b in segs]
    if with_gold:
        pos = sorted(set(pos) | {index.pos[m] for m, _ in golds})
    loc = {p: i for i, p in enumerate(pos)}
    files = [index.ids[p] for p in pos]
    cands, emb, sig, mood, gold_win = [], [], [], [], []
    for t, ((a, b), L, r) in enumerate(zip(segs, lat.layers, res)):
        mu, sd = r.stats["sem"]
        sim = (-L.cost / wsem) * sd + mu
        d = b - a
        mid_loc = np.array([loc[int(m)] for m in L.mid])
        D = {k: np.zeros((len(L.mid),) + ((1024,) if k.startswith("emb") else ())) for k in ("emb_in", "emb_out", "rms_in", "rms_out", "ph_in", "ph_out")}
        for m in np.unique(L.mid):
            sel = L.mid == m
            for k, v in desc(index.ids[m], L.off[sel], d).items():
                D[k][sel] = v
        cands.append(Cands(mid_loc, L.off.astype(float), sim.astype(float), int(L.n_top), **{k: v.astype(np.float32) for k, v in D.items()}))
        gv, pv = vq.pooled(a, b)
        emb.append(l2(gv)), mood.append(pv)
        sig.append(l2(vq._sem.means(np.array([a]), np.array([max(b, a + 1e-3)]))[0]))
        if with_gold:
            m, off = golds[t]
            q = vq.segment(a, b)
            gs = float(index.window_scores(q, np.array([index.pos[m]]), np.array([off]), d)["sem"][0])
            gd = {k: v[0] for k, v in desc(m, np.array([off]), d).items()}
            gold_win.append(dict(mid=loc[index.pos[m]], off=float(off), sim=gs, **gd))
    dense = {}
    for fpos in np.unique(res[0].fi):
        grid = np.arange(0.0, index.items[fpos].duration + 1e-9, 0.5)
        tab = np.full((len(segs), len(grid)), -np.inf, np.float32)
        for t, r in enumerate(res):
            sel = r.fi == fpos
            gi = np.clip(np.round(r.st[sel] / 0.5).astype(int), 0, len(grid) - 1)
            tab[t, gi] = np.maximum(tab[t, gi], r.raw["sem"][sel])
        dense[loc[int(fpos)]] = (grid, tab)
    ep = Episode(key, cands, np.array(segs, float), np.stack(emb), np.stack(sig), np.stack(mood), files,
                 np.array([store.music_track[m] for m in files], object), np.array([index.items[p].duration for p in pos]),
                 np.array([index.items[p].key for p in pos]), np.array([index.items[p].tempo for p in pos]), dense,
                 None, gold_win if with_gold else None, overlap_pieces(segs, spans, tracks), true_changes(spans, tracks),
                 float(vq.duration), [store_fg(p) for p in parts], np.asarray(lat.ranks), lat.layers)
    if with_gold:  # gold index among the plain candidates (-1 if the true window is not shortlisted)
        ep.gold = np.array([next((j for j in np.where((c.mid == g["mid"]) & (np.abs(c.off - g["off"]) <= 0.04))[0]), -1)
                            for c, g in zip(cands, gold_win)])
    return ep


_FG = {}


def store_fg(p):
    return _FG.get(p, "")


def train_sequences(cfg, data, k: int, n: int) -> list[list[str]]:
    from bgm.cue.sequences import by_film, clip_table, sample_sequence
    films = by_film(clip_table(data.pairs, data.fg, data.split_ids(k, "train")))
    limited = frozenset(data.folds.get("train_only", []))
    c = cfg.cue
    return [sample_sequence(films, np.random.default_rng([0, k, i]), float(c.seq_min_s), float(c.seq_max_s), "blocked",
                            int(c.block_max_clips), limited) for i in range(n)]


def train_cuts(cfg_seg, k: int, seqs: list[list[str]], theta: float, device: str) -> list[list[list[float]]]:
    """Cuts of the C head (own cuts, fold k's val-tuned theta) on training sequences."""
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold
    from bgm.segmentation.retrieval_eval import to_rel
    from bgm.segmentation.train_head import SegData
    data = SegData(cfg_seg, need_siglip=True)
    data.frozen = lambda kk, split: seqs          # noqa: E731  (instance override; nothing in the package edited)
    fo = Fold(data, BaSSL(device), k, "train", {"combo": "combo"})
    out = []
    for i in range(len(seqs)):
        c = fo.chains("combo")[i]
        out.append(to_rel(fo.cuts("combo", i, theta, False), [0.0] + [float(x) for x in c.joins]))
    return out


def build_fold(k: int, n_train: int, device: str) -> Path:
    from bgm.config import load_config
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    from bgm.segmentation.train_head import SegData
    t0 = time.time()
    cfg = load_config("configs/default.yaml", CUE_OVERRIDES)
    cfg_seg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    segfile = json.loads(Path("data/interim/segmentation/retrieval_segments.json").read_text())
    theta = float(segfile["thetas"][str(k)]["combo"])
    segk = segfile["folds"][str(k)]
    store = FeatureStore(cfg)
    _FG.update(manifest.read_folds(cfg)["film_group"])
    f = fold_ids(cfg, k)
    em = Path(cfg.paths.runs) / cfg.cue.emission_exp / f"fold{k}" / "seed0"
    fusion = json.loads((em / "fusion.json").read_text())
    d75 = json.loads((em / "stitch.json").read_text())
    models = FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device=device, train_ids=f["train"])
    desc = MusicDesc(store)
    out = {"fold": k, "theta": theta, "fusion": fusion, "d75": d75, "episodes": {}}
    for split in ("train", "val", "test"):
        index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f[split])))
        if split == "train":
            seqs = train_sequences(cfg, SegData(cfg_seg), k, n_train)
            cuts = train_cuts(cfg_seg, k, seqs, theta, device)
        else:
            seqs = [r["parts"] for r in segk[split]]
            cuts = [r["cuts"]["combo/own"] for r in segk[split]]
        eps = []
        for i, (parts, rel) in enumerate(zip(seqs, cuts)):
            eps.append(build_episode(f"{k}/{split}/{i}", store, index, models, fusion, cfg, desc, parts, rel, with_gold=True))
        out["episodes"][split] = eps
        out[f"gallery_{split}"] = len(index.ids)
        log.info("fold %d %s: %d episodes, %d scenes, gallery %d [%.0f s]", k, split, len(eps), sum(e.T for e in eps), len(index.ids), time.time() - t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"fold{k}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(out, fh)
    log.info("-> %s [%.0f s]", p, time.time() - t0)
    return p


def load_fold(k: int) -> dict:
    with open(CACHE / f"fold{k}.pkl", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build_fold(a.fold, a.n_train, a.device)
