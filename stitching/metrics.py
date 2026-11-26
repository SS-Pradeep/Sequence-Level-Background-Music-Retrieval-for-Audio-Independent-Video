"""Per-episode metrics via the existing evaluation functions (nothing reimplemented beyond glue).

track_acc     bgm.segmentation.retrieval_eval.overlap_accuracy (share of time whose chosen file is in the true track group)
switch_f1     bgm.retrieval.stitch_eval.switch_f1, +-1 s and +-0.5 s (truth = joins where the track group changes)
TBIoU         bgm.analysis.eval_suite.tbiou
dev_T2P       median distance from each true change to the nearest predicted switch (eval_suite.nearest), s
align_success share of true changes with a predicted switch within +-1 s (the matching of switch_f1 = its recall)
smoothness    switches / min and jumps / min (eval_suite's definition, without the render-based seam loudness)
entropy       eval_suite.diversity over every chosen file of a fold's test set, normalised by the gallery size
objective     track_acc + switch_f1 (the existing tuning objective, D121 / D143), used to tune every method on val
"""
from __future__ import annotations

import numpy as np

from bgm.analysis.eval_suite import diversity, nearest, tbiou
from bgm.cue.evaluate import match_counts
from bgm.retrieval.stitch_eval import switch_f1
from bgm.segmentation.retrieval_eval import overlap_accuracy
from stitching.base import TOL_S, Episode, Path


def switches_jumps(ep: Episode, p: Path) -> tuple[list[float], list[float]]:
    d = ep.scene_dur
    sw, jp = [], []
    for t in range(1, ep.T):
        if p.mid[t] != p.mid[t - 1]:
            sw.append(float(ep.segs[t, 0]))
        elif abs(p.off[t] - (p.off[t - 1] + d[t - 1])) > TOL_S:
            jp.append(float(ep.segs[t, 0]))
    return sw, jp


def evaluate_path(ep: Episode, p: Path) -> dict:
    tr = ep.file_track[p.mid]
    sw, jp = switches_jumps(ep, p)
    true = ep.true_sw
    acc = overlap_accuracy(np.asarray(tr, object), ep.pieces)
    f1 = switch_f1(sw, true)
    tp, _, fn = match_counts(sw, true, 1.0)
    mins = ep.duration / 60
    return {"track_acc": acc, "switch_f1": f1, "switch_f1_05": switch_f1(sw, true, 0.5), "objective": acc + f1,
            "TBIoU": tbiou(true, sw, ep.duration),
            "dev_T2P": float(np.median(nearest(np.array(true), np.array(sw)))) if true and sw else np.nan,
            "align_success": tp / (tp + fn) if true else np.nan,
            "switches_per_min": len(sw) / mins, "jumps_per_min": len(jp) / mins, "true_switches_per_min": len(true) / mins,
            "n_scenes": ep.T, "runtime_s": p.runtime_s}


def fold_entropy(eps: list[Episode], paths: list[Path], gallery_size: int) -> float:
    picks = [e.files[m] for e, p in zip(eps, paths) for m in p.mid]
    return diversity(picks, gallery_size)["entropy"]


def val_objective(stitcher, eps: list[Episode]) -> float:
    return float(np.mean([evaluate_path(e, stitcher.decode(e))["objective"] for e in eps]))
