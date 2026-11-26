"""Music-change prediction at every shot cut inside a clip (user's boundary-classification method, D103).

Labels come from real, continuous filmmaker scoring: only cuts INSIDE a dataset clip (no clip joins), so the music on
both sides is the film's own. At a cut t the music changes if, within +-`tol_s`:
  * the clip's own 1 Hz MERT jumps:  d = 1 - cos(mean MERT [t'-k, t'), mean MERT [t', t'+k)) is high, calibrated per
    fold on TRAIN clips only (positives: artificial joins of two train clips of different track groups; negatives:
    the train clips' in-clip cuts) and thresholded at p >= 0.5; or
  * the music drops to near-silence (rms_db more than `silence_db` below the clip median for >= `silence_min_s`).
Cuts closer than k s to a clip edge cannot be measured on both sides and are left out.

Features at a cut use only what a renderer knows: the silent video around the cut, and the music playing BEFORE it
(at render time: the library window being played), plus the time since that music started. Nothing after the cut on
the music side, nothing about where the clip (= the original cue) ends.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bgm.cue.audio_teacher import Calibration, music_distance

log = logging.getLogger(__name__)
RATE = 25


@dataclass
class Cut:
    pair: str
    t: float                   # cut time, s on the clip's interim timeline
    idx: int                   # index of the cut within the clip


def clip_cuts(cuts_s: np.ndarray, dur_s: float, edge_s: float) -> list[tuple[int, float]]:
    """(index, time) of the cuts at least edge_s from both clip edges."""
    return [(i, float(t)) for i, t in enumerate(cuts_s) if edge_s <= t <= dur_s - edge_s]


def silence_drop(rms_db_25hz: np.ndarray, t: float, tol_s: float, drop_db: float, min_s: float) -> bool:
    """True if rms falls drop_db below the clip median for >= min_s somewhere within [t - tol_s, t + tol_s]."""
    a, b = max(0, int((t - tol_s) * RATE)), min(len(rms_db_25hz), int((t + tol_s) * RATE) + 1)
    low = rms_db_25hz[a:b] < np.median(rms_db_25hz) - drop_db
    run = best = 0
    for x in low:
        run = run + 1 if x else 0
        best = max(best, run)
    return best >= min_s * RATE


def max_distance_near(e_1hz: np.ndarray, t: float, tol_s: float, half_s: int) -> float:
    """Largest MERT distance d over the whole seconds within [t - tol_s, t + tol_s]."""
    ts = np.arange(np.ceil(t - tol_s), np.floor(t + tol_s) + 1)
    ts = ts[(ts >= half_s) & (ts <= len(e_1hz) - half_s)]
    return float(music_distance(e_1hz, ts, half_s).max()) if len(ts) else 0.0


def join_distance(e_a: np.ndarray, e_b: np.ndarray, half_s: int) -> float:
    """d at an artificial join: the last half_s s of clip a against the first half_s s of clip b."""
    return float(music_distance(np.concatenate([e_a[-half_s:], e_b[:half_s]]), np.array([float(half_s)]), half_s)[0])


def calibrate(store, train_ids: list[str], half_s: int, n_joins: int, seed: int) -> tuple[Calibration, dict]:
    """Logistic p(d) from TRAIN clips: different-track joins (1) vs in-clip cut distances (0)."""
    rng = np.random.default_rng(seed)
    ids = [p for p in train_ids if len(store.music(store.music_of[p])["sem_mert_1hz"]) >= half_s]
    pos = []
    while len(pos) < n_joins:
        a, b = rng.choice(len(ids), 2, replace=False)
        if store.track[ids[a]] != store.track[ids[b]]:
            pos.append(join_distance(_mert(store, ids[a]), _mert(store, ids[b]), half_s))
    neg = []
    for p in ids:
        e, dur = _mert(store, p), store.duration("video", p)
        for _, t in clip_cuts(store.video(p)["cut_times_s"], dur, half_s):
            neg.append(float(music_distance(e, np.array([round(t)]), half_s)[0]))
    d = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    cal = Calibration.fit(d, y)
    return cal, {"n_joins": len(pos), "n_incut": len(neg), "d_join_median": float(np.median(pos)), "d_cut_median": float(np.median(neg)),
                 "d_at_p50": float(np.exp(-cal.b / cal.a))}


def _mert(store, pair: str) -> np.ndarray:
    return np.asarray(store.music(store.music_of[pair])["sem_mert_1hz"], np.float32)


def label_cuts(store, pairs: list[str], cal: Calibration, cfg_c) -> pd.DataFrame:
    """One row per measurable cut: pair, idx, t, d (max MERT distance within tol), p, silence, label."""
    rows = []
    for p in pairs:
        dur = store.duration("video", p)
        e = _mert(store, p)
        rms = store.music(store.music_of[p])["dyn_25hz"][:, 2]
        for i, t in clip_cuts(store.video(p)["cut_times_s"], dur, cfg_c.half_s):
            d = max_distance_near(e, t, cfg_c.tol_s, cfg_c.half_s)
            sil = silence_drop(rms, t, cfg_c.tol_s, cfg_c.silence_db, cfg_c.silence_min_s)
            pr = float(cal(np.array([d]))[0])
            rows.append({"pair": p, "idx": i, "t": t, "d": d, "p": pr, "silence": bool(sil), "label": int(pr >= 0.5 or sil)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- features
def shot_bounds(cuts_s: np.ndarray, dur_s: float) -> np.ndarray:
    return np.r_[0.0, np.asarray(cuts_s, float), dur_s]


def span_mean_1hz(x_1hz: np.ndarray, a: float, b: float) -> np.ndarray:
    """Mean of the 1 Hz samples (sample k centred at k + 0.5 s) inside [a, b); the nearest sample if none."""
    lo, hi = int(np.ceil(a - 0.5)), int(np.ceil(b - 0.5))
    lo, hi = max(0, min(lo, len(x_1hz) - 1)), max(0, min(hi, len(x_1hz)))
    return x_1hz[lo:hi].mean(0) if hi > lo else x_1hz[min(int((a + b) / 2), len(x_1hz) - 1)]


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def video_features(cuts_s: np.ndarray, dur_s: float, sig_1hz: np.ndarray, log_motion_25hz: np.ndarray, idx: int, n_shots: int) -> dict:
    """Video side of the cut cuts_s[idx]: SigLIP of n_shots shots each side (sig_before / sig_after), visual distances,
    motion, shot lengths, cut density. Silent video only."""
    B = shot_bounds(cuts_s, dur_s)
    k, t = idx + 1, float(cuts_s[idx])                        # the cut is B[k]
    a0, a1 = B[max(0, k - n_shots)], B[k]
    b0, b1 = B[k], B[min(len(B) - 1, k + n_shots)]
    sig = np.asarray(sig_1hz, np.float32)
    sb, sa = unit(span_mean_1hz(sig, a0, a1)), unit(span_mean_1hz(sig, b0, b1))
    s1, s2 = unit(span_mean_1hz(sig, B[k - 1], a1)), unit(span_mean_1hz(sig, b0, B[k + 1]))
    lm = log_motion_25hz
    f = lambda a, b: float(lm[int(a * RATE):max(int(b * RATE), int(a * RATE) + 1)].mean())
    lens_b, lens_a = np.diff(B[max(0, k - n_shots):k + 1]), np.diff(B[k:min(len(B), k + n_shots + 1)])
    return {"sig_before": sb, "sig_after": sa, "vis_dist_3": float(1 - sb @ sa), "vis_dist_1": float(1 - s1 @ s2),
            "motion_before": f(a0, a1), "motion_after": f(b0, b1),
            "len_before_1": float(np.log(max(lens_b[-1], 1e-3))), "len_after_1": float(np.log(max(lens_a[0], 1e-3))),
            "len_before_mean": float(np.log(np.maximum(lens_b, 1e-3)).mean()), "len_after_mean": float(np.log(np.maximum(lens_a, 1e-3)).mean()),
            "cuts_per_10s": float(np.sum(np.abs(np.asarray(cuts_s) - t) <= 5))}


def music_features(mert_1hz: np.ndarray, dyn_25hz: np.ndarray, te: float, music_s: int) -> dict:
    """The music playing up to time te (s, on that music's own clock): MERT over [te - music_s, te), loudness of the
    last second, loudness trend, onsets of the last 2 s. Training: the clip's music, te = cut - gap. Render: the
    candidate library window, te = window end - gap."""
    rms, on = dyn_25hz[:, 2], dyn_25hz[:, 0]

    def r(a: float, b: float) -> float:                       # mean rms_db over [a, b) s, clamped, never empty
        i = min(max(0, int(a * RATE)), len(rms) - 1)
        return float(rms[i:max(int(b * RATE), i + 1)].mean())
    i2 = min(max(0, int((te - 2) * RATE)), len(on) - 1)
    return {"mert_before": unit(span_mean_1hz(np.asarray(mert_1hz, np.float32), max(0.0, te - music_s), max(te, 0.5))),
            "rms_last1": r(te - 1, te), "rms_trend": r(te - 1, te) - r(te - 4, te - 1),
            "onset_last2": float(on[i2:max(int(te * RATE), i2 + 1)].mean())}


def cut_features(store, row, n_shots: int, music_s: int, since_s: float, gap_s: float) -> dict:
    """All raw features of one in-clip cut (training / evaluation). Music features end gap_s before the cut (= the label
    tolerance), so they never see the music the label is read from."""
    v = store.video(row.pair)
    m = store.music(store.music_of[row.pair])
    out = video_features(np.asarray(v["cut_times_s"], float), store.duration("video", row.pair), v["sem_siglip_1hz"],
                         v["dyn_25hz"][:, 0], row.idx, n_shots)
    out.update(music_features(m["sem_mert_1hz"], m["dyn_25hz"], row.t - gap_s, music_s))
    out["since_start"] = float(np.log1p(since_s))
    return out


SCALARS = ("vis_dist_3", "vis_dist_1", "motion_before", "motion_after", "len_before_1", "len_after_1", "len_before_mean",
           "len_after_mean", "cuts_per_10s", "rms_last1", "rms_trend", "onset_last2", "since_start")
GROUPS = {"video": ("vis_dist_3", "vis_dist_1", "motion_before", "motion_after", "len_before_1", "len_after_1",
                    "len_before_mean", "len_after_mean", "cuts_per_10s"),
          "time": ("since_start",),
          "music": ("rms_last1", "rms_trend", "onset_last2")}


def since_music_start(df: pd.DataFrame) -> np.ndarray:
    """Seconds since the current music started at each cut: the clip start, or the previous LABELED change in the clip
    (teacher forcing in training; at render time the decoded path supplies it)."""
    out = np.zeros(len(df))
    last = {}
    for j, r in enumerate(df.itertuples()):
        s = last.get(r.pair, 0.0)
        out[j] = r.t - s
        if r.label:
            last[r.pair] = r.t
    return out
