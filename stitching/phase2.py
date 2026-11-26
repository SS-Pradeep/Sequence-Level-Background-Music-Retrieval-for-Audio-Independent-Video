"""Phase 2: semi-Markov (segmental) Viterbi rows 2a / 2b / 2c and the equal-budget re-tune of two-stage.

Selection rule (fixed in advance, user 2026-11-17): for each method, every grid config is decoded on the fold's val
sequences; the val means of track accuracy and switch F1 are z-scored ACROSS that method's configs; the config with
the largest z(track acc) + z(switch F1) is used on test (ties: first in grid order).

Semi-Markov (stitching/semi_markov.py, the user's code, unchanged):
  tracks A = the distinct files in the video's shortlist union (ep.dense); each track's windows = ALL its 10 s windows
  every 0.5 s (stitching/win10.py, same frozen ensemble head); scenes = the projected scene embeddings.
  2a SumMaxEmission, 2b PooledEmission, 2c PooledEmission + w_b [log p(start boundary) + sum log(1 - p(inner))]
  inside each segment the track's windows are laid with layout_segment(hop = 0.5 s).
Grid: scale in {0.25, 0.5, 1, 2} / tau, lam_dur {0, 0.5, 1, 2}, switch_logp {-4, -2, -1, 0}, LogNormal(L, sigma in
{0.4, 0.6, 0.9}), max_len {6, 12}, reprise_logp {-inf, log 0.02} = 768 configs; 2c x w_b {0, 0.5, 1, 2} = 3,072.
Two-stage re-tuned with the same budget: half thresholds (val quantiles of the scene distance), half KTS penalties
(log-spaced 0.01-5): 768 (vs 2a / 2b) and 3,072 (vs 2c).

`.venv/bin/python -m stitching.phase2 --fold k [--rows semi_sum,semi_pool,semi_pool_cp,two_stage_768,two_stage_3072]`
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from stitching.base import Episode, Path as SPath, Stitcher
from stitching.data import load_fold
from stitching.metrics import evaluate_path, fold_entropy
from stitching.semi_markov import LogNormalDuration, PooledEmission, SumMaxEmission, layout_segment, semi_markov_decode
from stitching.two_stage import TwoStage

log = logging.getLogger(__name__)
OUT = Path("stitching/results")
PHASE2_ROWS = [("two_stage_768", "Two-stage, re-tuned (768 configs, z rule)"),
               ("two_stage_3072", "Two-stage, re-tuned (3,072 configs, z rule)"),
               ("semi_sum", "2a Semi-Markov, sum-max emission"), ("semi_pool", "2b Semi-Markov, pooled emission"),
               ("semi_pool_cp", "2c Semi-Markov, pooled + change predictor")]


def z_select(scores: list[tuple]) -> int:
    """scores: [(track_acc, switch_f1)] per config -> index of max z(acc) + z(F1) (first on ties)."""
    a = np.array(scores, float)
    z = (a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), 1.0)
    return int(np.argmax(z.sum(1)))


def tune(st, configs: list[dict], val: list[Episode]) -> tuple[dict, dict]:
    res = []
    for c in configs:
        st.set(c)
        rows = [evaluate_path(e, st.decode(e)) for e in val]
        res.append((float(np.mean([r["track_acc"] for r in rows])), float(np.mean([r["switch_f1"] for r in rows]))))
    i = z_select(res)
    st.set(configs[i])
    return configs[i], {"n_configs": len(configs), "val_track_acc": res[i][0], "val_switch_f1": res[i][1],
                        "val_all": [list(r) for r in res]}


class TwoStageZ(TwoStage):
    """Two-stage with the decode cached by grouping (configs that give the same groups decode identically)."""

    def __init__(self):
        super().__init__()
        self._cache = {}

    def set(self, c):
        self.mode, self.param = c["mode"], c["param"]

    def decode(self, ep):
        from stitching.two_stage import decode_groups, kts_groups, threshold_groups
        g = tuple(threshold_groups(ep.scene_emb, self.param) if self.mode == "threshold" else kts_groups(ep.scene_emb, self.param))
        if (ep.key, g) not in self._cache:
            self._cache[(ep.key, g)] = decode_groups(ep, list(g))
        return self._cache[(ep.key, g)]


def two_stage_configs(val: list[Episode], n: int) -> list[dict]:
    d = np.concatenate([1 - (e.scene_emb[1:] * e.scene_emb[:-1]).sum(1) for e in val if e.T > 1])
    th = np.quantile(d, np.linspace(0.0, 1.0, n // 2 + 2)[1:-1])
    kp = np.exp(np.linspace(np.log(0.01), np.log(5.0), n // 2))
    return [{"mode": "threshold", "param": float(x)} for x in th] + [{"mode": "kts", "param": float(x)} for x in kp]


class SemiMarkov(Stitcher):
    def __init__(self, emission: str, win: dict, tau: float, L: float, use_cp: bool = False):
        self.emission, self.win, self.tau, self.L, self.use_cp = emission, win, tau, L, use_cp
        self.c = None
        self._em = {}

    def set(self, c):
        self.c = c

    def _emission(self, ep):
        if ep.key not in self._em:
            files = sorted(ep.dense)
            W = [self.win[ep.files[f]][1] for f in files]
            E = SumMaxEmission if self.emission == "sum" else PooledEmission
            self._em[ep.key] = (files, E(ep.scene_emb, W, 1.0))
        return self._em[ep.key]

    def decode(self, ep: Episode) -> SPath:
        c = self.c
        files, em = self._emission(ep)
        scale = c["scale"] / self.tau
        if self.use_cp and c["w_b"] > 0:
            p = np.clip(ep.p_change, 1e-4, 1 - 1e-4)          # [T - 1], boundary t between scenes t and t + 1
            lp, l1p = np.log(p), np.r_[0.0, np.cumsum(np.log1p(-p))]

            def emission(s, e):
                start = lp[s - 1] if s > 0 else 0.0
                inner = l1p[e] - l1p[s]                       # boundaries s .. e-1 (inside the segment)
                return scale * em(s, e) + c["w_b"] * (start + inner)
        else:
            def emission(s, e):
                return scale * em(s, e)
        segs, _ = semi_markov_decode(emission, ep.scene_dur, len(files), max_len=c["max_len"],
                                     dur_prior=LogNormalDuration(self.L, c["sigma"]), lam_dur=c["lam_dur"],
                                     switch_logp=c["switch_logp"], reprise_logp=c["reprise_logp"])
        mid, off = np.zeros(ep.T, int), np.zeros(ep.T)
        for s, e, a in segs:
            st, W = self.win[ep.files[files[a]]]
            idx = layout_segment(ep.scene_emb[s:e + 1], ep.scene_dur[s:e + 1], W, 0.5)
            mid[s:e + 1], off[s:e + 1] = files[a], st[idx]
        return SPath(mid, off)


def semi_configs(use_cp: bool) -> list[dict]:
    g = itertools.product((0.25, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0), (-4.0, -2.0, -1.0, 0.0), (0.4, 0.6, 0.9), (6, 12),
                          (-math.inf, math.log(0.02)), (0.0, 0.5, 1.0, 2.0) if use_cp else (0.0,))
    return [dict(scale=a, lam_dur=b, switch_logp=c, sigma=d, max_len=e, reprise_logp=f, w_b=w) for a, b, c, d, e, f, w in g]


def main(k: int, rows: list[str]) -> Path:
    from bgm.config import load_config
    from bgm.segmentation.retrieval_eval import CUE_OVERRIDES
    from stitching.evaluate import ensemble_tau
    from stitching.win10 import load as load_win
    t0 = time.time()
    F = load_fold(k)
    va, te = F["episodes"]["val"], F["episodes"]["test"]
    if "semi_pool_cp" in rows:
        from stitching.cue_bassl import attach_p
        attach_p(k, va, "val"), attach_p(k, te, "test")
    cs = json.loads((OUT / "cue_stats.json").read_text())["folds"][str(k)]
    tau = ensemble_tau(load_config("configs/default.yaml", CUE_OVERRIDES), k)
    path = OUT / f"fold{k}_phase2.json"
    out = json.loads(path.read_text()) if path.exists() else {"fold": k, "methods": {}, "timelines": {}}
    win = load_win(k) if any(r.startswith("semi") for r in rows) else None
    for key in rows:
        t1 = time.time()
        if key.startswith("two_stage"):
            st, cfgs = TwoStageZ(), two_stage_configs(va, int(key.split("_")[-1]))
        else:
            st = SemiMarkov("sum" if key == "semi_sum" else "pool", win, tau, float(cs["L_s"]), key == "semi_pool_cp")
            cfgs = semi_configs(key == "semi_pool_cp")
        best, info = tune(st, cfgs, va)
        paths = [st.timed(e) for e in te]
        rr = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
        out["methods"][key] = {"name": dict(PHASE2_ROWS)[key], "params": {**best, **{a: b for a, b in info.items() if a != "val_all"}},
                               "val_grid": info["val_all"], "rows": rr, "entropy": fold_entropy(te, paths, F["gallery_test"]),
                               "fit_s": time.time() - t1}
        out["timelines"][key] = [[e.files[m] for m in p.mid] for e, p in zip(te, paths)]
        log.info("fold %d %-15s %s [%.0f s] | test acc %.3f F1 %.3f", k, key, best, time.time() - t1,
                 np.mean([r["track_acc"] for r in rr]), np.mean([r["switch_f1"] for r in rr]))
        path.write_text(json.dumps(out, default=float))
    log.info("fold %d phase 2 done [%.0f s]", k, time.time() - t0)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--rows", default="two_stage_768,two_stage_3072,semi_sum,semi_pool")
    a = ap.parse_args()
    main(a.fold, a.rows.split(","))
