"""Per-head scores of the Table 3 lattice candidates, for re-ranking inside the HMM (D148, part 2).

`.venv/bin/python -m exp_supp.rerank.lattice [--clean]` -> runs/exp_supp/rerank/lattice/fold<k>.pkl
(--clean: the cleaned-dataset lattices of exp_supp.rerank.clean_t3 -> runs/exp_supp/rerank/clean_t3/lattice/)

The Table 3 lattices (stitching/cache/fold<k>.pkl, read-only) store each candidate's raw ensemble cosine `sim` only.
For k-reciprocal we need gallery-side (candidate x candidate) and bank-side (training segment x candidate) cosines, so
the 10 heads of the emission (runs/titlesplit/semantic/siglip-both seeds 0-9, mean cosine = EnsembleSemantic) are
re-applied: video side = the episode's stored scene_emb split into its 10 per-head unit vectors (scene_sig skips
the min-span rule for scenes < 5 s, scene_emb does not), music input = [l2 CLAP | l2 MERT]
span means of (file, off, scene duration) with the span rule. Asserted: the 10-head mean reproduces every stored sim.
Per scene t of every val / test episode: G_t [K_t, K_t] and B_t [Nb, K_t] (head-mean cosines). Bank = 1,500 random
segments of the fold's training clips (original data, rng [k, 148]).
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from exp_supp.common import RUNS

log = logging.getLogger("exp_supp.rerank.lattice")
OUT = RUNS / "rerank" / "lattice"
HEADS_ROOT = Path("runs/titlesplit/semantic/siglip-both")
# cleaned dataset (D148b): lattices of exp_supp.rerank.clean_t3, heads = the D116 cleaned siglip-both seeds 0-9
OUT_CLEAN = RUNS / "rerank" / "clean_t3" / "lattice"
HEADS_ROOT_CLEAN = Path("runs/cleaned/semantic/siglip-both")
BANK_SEGS = 1500


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def build(k: int, device: str, data: str = "orig") -> Path:
    from bgm.config import load_config
    from bgm.eval.protocol import test_segments
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    from bgm.train.semantic import SemanticInputs, SemanticModel
    if data == "clean":
        from exp_supp.rerank.clean_t3 import cue_cfg, load_fold
        out_dir, heads_root = OUT_CLEAN, HEADS_ROOT_CLEAN
    else:
        from stitching.data import load_fold
        out_dir, heads_root = OUT, HEADS_ROOT
    out = out_dir / f"fold{k}.pkl"
    if out.exists():
        return out
    t0 = time.time()
    cfg = cue_cfg() if data == "clean" else load_config("configs/default.yaml", CUE_OVERRIDES)
    store = FeatureStore(cfg)
    seeds = list(cfg.pipeline.sem_seeds)
    heads = []
    for s in seeds:
        st = torch.load(heads_root / f"fold{k}" / f"seed{s}" / "ckpt.pt", map_location=device)
        m = SemanticModel(st["dv"], st["dm"], cfg, device)
        m.load(st["state"])
        m.eval()
        heads.append(m)
    inp = SemanticInputs(store, cfg.train.semantic.video_feature, "both")

    def emb(side, X):
        with torch.no_grad():
            return np.stack([h.embed(side, X) for h in heads])          # [H, N, d]
    bsegs = test_segments(store, fold_ids(cfg, k)["train"])
    sel = np.sort(np.random.default_rng([k, 148]).choice(len(bsegs), min(BANK_SEGS, len(bsegs)), replace=False))
    BV = emb("video", np.stack([inp.video(bsegs[i][0], bsegs[i][2], bsegs[i][3]) for i in sel]))
    F = load_fold(k)
    res = {"fold": k, "seeds": seeds, "episodes": {}, "data": data}
    worst = 0.0
    for split in ("val", "test"):
        eps = []
        for e in F["episodes"][split]:
            H = len(heads)                                                  # scene_emb = [head_1 | ... | head_H] / sqrt(H)
            V = (e.scene_emb.reshape(e.T, H, -1) * np.sqrt(H)).transpose(1, 0, 2)   # [H, T, d] per-head unit vectors
            scenes = []
            for t, c in enumerate(e.cands):
                d = float(e.segs[t, 1] - e.segs[t, 0])
                X = np.stack([inp.music(e.files[m], float(o), float(o) + d) for m, o in zip(c.mid, c.off)])
                E = emb("music", X)                                         # [H, K, d]
                sim = np.einsum("hkd,hd->hk", E, V[:, t]).mean(0)
                worst = max(worst, float(np.abs(sim - c.sim).max()))
                scenes.append({"G": np.einsum("hkd,hld->kl", E, E) / len(heads),
                               "B": np.einsum("hbd,hkd->bk", BV, E) / len(heads), "sim_check": sim})
            eps.append(scenes)
        res["episodes"][split] = eps
        log.warning("fold %d %s: %d episodes [%.0f s]", k, split, len(eps), time.time() - t0)
    assert worst < 2e-3, f"fold {k}: lattice sims not reproduced (max |Δ| {worst:.2e})"
    res["max_abs_sim_diff"] = worst
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    log.warning("fold %d -> %s (sims reproduced, max |Δ| %.1e) [%.0f s]", k, out, worst, time.time() - t0)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for k in range(5):
        build(k, dev, "clean" if "--clean" in sys.argv else "orig")
