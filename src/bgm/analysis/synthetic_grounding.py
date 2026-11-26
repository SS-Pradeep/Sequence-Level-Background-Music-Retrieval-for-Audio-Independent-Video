"""The synthetic beat-cut grounding check, with its checkpoint kept (runs/synthetic_grounding/).

Rebuilds exactly the set and training of tests/test_m7.py::test_synthetic_pairs_reach_sync_accuracy (36 synthetic
pairs, seed 7, 16-26 s, cuts on beats by construction; train on the first 28, validate on the last 8; seed 0, 4 shifted
negatives, batch 32, patience 15, at most 80 epochs), which gave val sync 0.979 without saving the model. Saves the
checkpoint, the run files and the val grounding predictions (per query: pair, segment start / end, true and predicted
25 Hz offset) so they can be re-scored later without retraining.
Run: python -m bgm.analysis.synthetic_grounding
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from bgm.tools import REPO_ROOT

log = logging.getLogger(__name__)
ROOT = REPO_ROOT / "runs" / "synthetic_grounding"


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from tests.fixtures.pipeline import run_through_extract
    from tests.fixtures.synth import make_pairs

    from bgm.data.segments import run_segment
    from bgm.eval.protocol import grounding, test_segments
    from bgm.eval.store import FeatureStore
    from bgm.runs import write_run
    from bgm.train.dynamics import offset_scorer, train_dynamics

    ROOT.mkdir(parents=True, exist_ok=True)
    pairs = make_pairs(ROOT / "raw", n=36, seed=7, durations=(16.0, 26.0), embed_every=0)
    cfg, df = run_through_extract(ROOT / "work", pairs, what="sem_video,sem_music,dyn_video,dyn_music")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["train.dynamics.batch=32", "train.dynamics.patience=15", "train.dynamics.max_epochs=80"]))
    run_segment(cfg)
    store = FeatureStore(cfg)
    ids = list(df.pair_id)
    train_ids, val_ids = ids[:28], ids[28:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, val = train_dynamics(cfg, store, train_ids, val_ids, seed=0, device=device, k_shift=4, out_dir=ROOT, max_epochs=cfg.train.dynamics.max_epochs)
    torch.save({"state": model.state()}, ROOT / "ckpt.pt")
    # val grounding predictions, in `grounding`'s query order (same skip rule), for later re-scoring
    g = grounding(store, val_ids, offset_scorer(model, store), cfg)
    rows = []
    for p, k, a, b in test_segments(store, val_ids):
        L, true = int(round((b - a) * 25)), int(round(a * 25))
        if L < 1 or true + L > len(store.music(store.music_of[p])["dyn_25hz"]):
            continue
        rows.append({"pair_id": p, "seg_idx": k, "start_s": a, "end_s": b, "true_frame": true})
    assert len(rows) == len(g.err_s), (len(rows), len(g.err_s))
    for r, e in zip(rows, g.err_s):
        r["pred_frame"] = r["true_frame"] + int(round(e * 25))
    (ROOT / "val_predictions.json").write_text(json.dumps(rows, indent=1))
    np.savez(ROOT / "perquery.npz", strict=g.strict.astype(float), sync=g.sync.astype(float), err_s=g.err_s, clusters=g.clusters)
    metrics = {"val_sync_at_best_epoch": val["val_sync"], "val_sync_rescored": float(g.sync.mean()),
               "val_strict": float(g.strict.mean()), "previous_val_sync": 0.9791666666666666,   # the acceptance test printout (PROGRESS.md M7); that model was not saved
               "previous_val_sync_source": "tests/test_m7.py printout, model not saved", **{k: v for k, v in val.items() if k != "val_sync"},
               "n_val_queries": len(rows), "train_ids": train_ids, "val_ids": val_ids}
    write_run(ROOT, cfg, metrics, seed=0)
    log.info("synthetic grounding: val sync %.4f (previously 0.9792), strict %.4f, best epoch %d; checkpoint %s",
             val["val_sync"], g.strict.mean(), val["best_epoch"], ROOT / "ckpt.pt")


if __name__ == "__main__":
    from bgm.log import setup_logging
    setup_logging()
    main()
