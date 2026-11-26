"""How often a TRAINING segment's own music window makes the top k, against held-out test segments (fold 0).

python -m bgm.analysis.train_vs_test_rank   -> outputs/train_retrieval/train_vs_test_ranks.md

Final system (D75): 10-head ensemble (run seed 0), title split, fold 0. Train row: 200 random train segments, each
against every train segment's aligned music window (segment protocol, 'all'). Test row: every test segment of fold 0,
the saved ranks of retrieval_figure score-segments (gallery = test windows). The train row shows what the heads fit;
early stopping keeps them at epochs 3-10, so each training segment is seen about once or twice and is not memorised.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bgm.analysis.retrieval_figure import OUT, ROOT, TRAIN_OUT, rank_of

log = logging.getLogger(__name__)
N_SAMPLE = 200


def train_ranks(k: int = 0, n: int = N_SAMPLE, seed: int = 0) -> tuple[np.ndarray, int]:
    """Ranks of `n` random train segments of fold k (gallery = train windows) and the gallery size."""
    import torch
    from bgm.analysis.eval_suite import Scorers
    from bgm.config import load_config
    from bgm.eval.protocol import segment_gallery, test_segments
    from bgm.eval.store import FeatureStore, fold_ids

    cfg = load_config(ROOT / "pipeline_ens10" / f"fold{k}" / "seed0" / "config.yaml")
    store = FeatureStore(cfg)
    f = fold_ids(cfg, k)
    segs = test_segments(store, list(f["train"]))
    sc = Scorers(cfg, store, k, 0, f["train"]).sc
    ranks, size = [], 0
    with torch.no_grad():
        for i in np.random.default_rng(seed).choice(len(segs), n, replace=False):
            windows, hit = segment_gallery(store, segs, segs[i], "all")
            s = np.asarray(sc.window(segs[i][0], segs[i][2], segs[i][3], windows), float)
            hits = np.zeros(len(s), bool)
            hits[hit] = True
            ranks.append(rank_of(s, hits))
            size = len(s)
    return np.array(ranks), size


def test_ranks(k: int = 0) -> tuple[np.ndarray, int]:
    """Saved segment ranks of fold k's test segments (gallery = test windows) and the gallery size."""
    rows = pd.read_pickle(OUT / f"segments_fold{k}.pkl")
    r = [rank_of(np.asarray(x["scores"]), np.arange(len(x["scores"])) == x["hit"]) for x in rows]
    return np.array(r), len(rows[0]["scores"])


def row(name: str, r: np.ndarray, size: int) -> str:
    med = np.median(r)
    return (f"| {name} | {size:,} | {np.mean(r <= 1):.2f} | {np.mean(r <= 10):.2f} | {np.mean(r <= 20):.2f} | "
            f"{med:.0f} (top {100 * med / size:.1f}%) |")


def main() -> None:
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    tr, ntr = train_ranks()
    te, nte = test_ranks()
    md = "\n".join([
        "# Training vs test segment retrieval (fold 0, 10-head ensemble, single fold)", "",
        "| Query segments | Gallery | R@1 | R@10 | R@20 | Median rank |", "|---|---|---|---|---|---|",
        row(f"Train ({N_SAMPLE} sampled)", tr, ntr), row(f"Test (all {len(te)})", te, nte), "",
        "Train row: the heads saw these segments (about once or twice each, early stopping at epochs 3-10), so it shows "
        "what they fit, not generalisation. Single-fold development numbers.", ""])
    (TRAIN_OUT / "train_vs_test_ranks.md").write_text(md)
    log.info("\n%s", md)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
