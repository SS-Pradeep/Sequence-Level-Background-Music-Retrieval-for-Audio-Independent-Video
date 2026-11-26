"""Join-free test of the cue-switch predictors (PROGRESS.md D98): music changes INSIDE test clips.

Every music change in the joined test sequences is also a clip join, so F1 and the D85 probes can only ask about joins.
Here only boundaries inside one clip are used (continuous video, no join, no clip-length clue): positives are the
boundaries where the audio teacher (audio_teacher.py, the fold's TRAIN-fold calibration, p >= 0.5) says the music
changes, negatives the rest. A predictor that learned music changes ranks the positives higher (AUROC > 0.5).

The truth is the teacher's, not a human's: a model trained on teacher targets (R3a, R3sa) is scored against the same
kind of signal it learned, so this test shows whether the video can predict the music's changes, not that the teacher
is right. Within 8 x 8 bins of log left / right segment length (edges: quantiles of these test boundaries, no labels)
the AUROC shows what is left beyond segment length; Rdur (length only) is the reference.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.cue.audio_teacher import AudioTeacher, Calibration
from bgm.cue.probes import (LABEL, auc_draws, ci, cluster_weights, duration_logits, fmt, load_duration, pair_counts)
from bgm.cue.tokens import build_tokens, sequence_tokens
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
SYSTEMS = ("R3", "R3s", "R3a", "R3sa", "R4", "R2", "Rdur")


def teacher_calibration(cfg: DictConfig, fold: int) -> tuple[int, Calibration]:
    """The fold's calibration, as fit for R3a (identical for every seed and variant, asserted)."""
    root = Path(cfg.paths.runs) / cfg.cue.exp
    got = set()
    for v in ("pair_audio", "pair_splice_audio"):
        for m in sorted((root / v / f"fold{fold}").glob("seed*/metrics.json")):
            t = json.loads(m.read_text())["audio_teacher"]
            got.add((t["half_s"], t["calib"]["a"], t["calib"]["b"]))
    assert len(got) == 1, f"fold {fold}: teacher calibrations differ or are missing: {got}"
    h, a, b = got.pop()
    return int(h), Calibration(a, b)


def collect(cfg: DictConfig, evals: list[dict]) -> dict:
    """Per in-clip boundary of the blocked test sequences: fold, cluster, teacher d and p, left / right log length,
    and every run's logit."""
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    cols = {k: [] for k in ("fold", "cluster", "d", "p", "ll", "lr")}
    z: dict[str, list] = {}
    for e in evals:
        f = e["fold"]
        half, cal = teacher_calibration(cfg, f)
        t = AudioTeacher(store, half, cal)
        dur = load_duration(cfg, f)
        names = [n for n in e["systems"] if n.rsplit("_s", 1)[0] in SYSTEMS]
        for c in e["cases"]:
            if c["split"] != "test" or c["mode"] != "blocked":
                continue
            raw, seg_part = sequence_tokens(tokens, c["parts"])
            d = t.distance(c["parts"], raw, seg_part)
            inside = np.array(c["types"]) == "clip"
            if not inside.any():
                continue
            ld = raw[:, -3].astype(np.float64)
            cols["fold"] += [f] * int(inside.sum())
            cols["cluster"] += [f"{f}:{c['cluster']}"] * int(inside.sum())
            cols["d"] += d[inside].tolist()
            cols["p"] += cal(d[inside]).tolist()
            cols["ll"] += ld[:-1][inside].tolist()
            cols["lr"] += ld[1:][inside].tolist()
            key = f"blocked/{c['idx']}"
            for n in names:
                z.setdefault(n, []).append(np.asarray(e["systems"][n]["logits"][key])[inside])
            z.setdefault("Rdur", []).append(duration_logits(dur, ld)[inside])
    out = {k: np.array(v) for k, v in cols.items()}
    out["z"] = {n: np.concatenate(v) for n, v in z.items()}
    return out


def strata_8x8(ll: np.ndarray, lr: np.ndarray, nb: int = 8) -> np.ndarray:
    q = np.linspace(0, 1, nb + 1)[1:-1]
    return np.digitize(ll, np.quantile(ll, q)) * nb + np.digitize(lr, np.quantile(lr, q))


def report_lines(cfg: DictConfig, evals: list[dict]) -> tuple[list[str], list[str]]:
    """(summary bullets, section lines); empty unless R3a ran."""
    if not any(n.startswith("R3a_s") for n in evals[0]["systems"]):
        return [], []
    B = collect(cfg, evals)
    cl_names = sorted(set(B["cluster"]))
    cl = np.array([cl_names.index(x) for x in B["cluster"]])
    K = len(cl_names)
    W = cluster_weights(K)
    pos, neg = B["p"] >= 0.5, B["p"] < 0.5
    st = {"plain": np.zeros(len(pos), int), "8x8": strata_8x8(B["ll"], B["lr"])}
    systems = [s for s in SYSTEMS if s == "Rdur" or any(n.rsplit("_s", 1)[0] == s for n in B["z"])]
    draws = {}
    for s in systems:
        runs = [n for n in B["z"] if n == s or n.rsplit("_s", 1)[0] == s]
        for k, S in st.items():
            draws[s, k] = auc_draws([pair_counts(B["z"][n], pos, neg, S, cl, K) for n in runs], W)
    dq = np.quantile(B["d"], [0.5, 0.9])
    L = ["", "## Music changes inside test clips (D98, join-free)", "",
         "Only boundaries inside one clip of the blocked test sequences (no join). Positives: the audio teacher says the "
         "music changes there (p ≥ 0.5 with the fold's train-fold calibration; music distance ±3 s on MERT); negatives: "
         "the rest. The truth is the teacher's, not a human's (src/bgm/cue/within_clip.py). AUROC: mean over seeds, 95% "
         "cluster-bootstrap CIs over the test clusters; length bins from these boundaries' own segment lengths.", "",
         f"In-clip boundaries: {len(pos)}, teacher positives {int(pos.sum())} ({pos.mean():.1%}); {K} clusters; music "
         f"distance median {dq[0]:.3f}, 90th percentile {dq[1]:.3f}.", "",
         "| predictor | AUROC | AUROC within 8 × 8 length bins | Δ vs R3, plain | Δ vs R3, 8 × 8 |", "| --- | --- | --- | --- | --- |"]
    for s in systems:
        row = [LABEL.get(s, s), fmt(ci(*draws[s, "plain"])), fmt(ci(*draws[s, "8x8"]))]
        for k in ("plain", "8x8"):
            row.append(fmt(ci(draws[s, k][0] - draws["R3", k][0], draws[s, k][1] - draws["R3", k][1]), True) if s != "R3" else "—")
        L.append("| " + " | ".join(row) + " |")
    res = {s: ci(*draws[s, "8x8"]) for s in systems}
    summary = [f"* Music changes inside test clips (D98, teacher truth, length held fixed): R3 {fmt(res['R3'])}, R3s "
               f"{fmt(res['R3s'])}, R3a {fmt(res['R3a'])}" + (f", R3sa {fmt(res['R3sa'])}" if "R3sa" in res else "") + "."]
    out = Path(cfg.paths.runs) / cfg.cue.exp / "within"
    out.mkdir(parents=True, exist_ok=True)
    (out / "within.json").write_text(json.dumps({"n": int(len(pos)), "pos": int(pos.sum()), "clusters": K,
                                                 "auroc": {f"{s}/{k}": ci(*draws[s, k]) for s in systems for k in st}}, indent=1))
    return summary, L
