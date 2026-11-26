"""Opt-in pace match for renders: fast video -> energetic music (user request after the 123.mp4 render, D100).

The best system (D75) scores music windows by semantics only, so a fast-cut video can get sparse, slow music. Here the
pace of a span of video is the mean of the library percentiles of its cut rate (cuts / min) and its mean log_motion
(raw dyn_25hz channel 0); the energy of a music window is the mean of the library percentiles of its mean log_onset
and rms_db (raw dyn_25hz channels 0 and 2). Score = -|pace - energy|, added as a standardized branch "pace" with
weight `render.pace_match.weight` to the coarse shortlist and to every fine / continuation window score.
Only `bgm render` reads the flag (off by default), so no benchmark number changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bgm.eval.store import FeatureStore
from bgm.render.branches import BranchIndex
from bgm.retrieval.index import MusicIndex

log = logging.getLogger(__name__)
RATE = 25


def pct(ref_sorted: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    """Library percentile in [0, 1] of x against a sorted reference."""
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


@dataclass
class Reference:
    v_cut: np.ndarray          # sorted cuts / min of the library videos
    v_mot: np.ndarray          # sorted mean raw log_motion of the library videos
    m_on: np.ndarray           # sorted mean raw log_onset of the library music files
    m_rms: np.ndarray          # sorted mean rms_db of the library music files


def build_reference(store: FeatureStore, music_ids: list[str]) -> Reference:
    vc, vm = [], []
    for p in store.pairs.pair_id:
        d = store.video(p)
        T = len(d["dyn_25hz"]) / RATE
        if T >= 3.0:
            vc.append(len(d["cut_times_s"]) / T * 60), vm.append(float(d["dyn_25hz"][:, 0].mean()))
    mo, mr = [], []
    for m in music_ids:
        x = store.music(m)["dyn_25hz"]
        mo.append(float(x[:, 0].mean())), mr.append(float(x[:, 2].mean()))
    return Reference(*(np.sort(np.array(a, float)) for a in (vc, vm, mo, mr)))


def video_pace(dyn_raw: np.ndarray, cuts: np.ndarray, a: float, b: float, ref: Reference, min_ctx_s: float = 10.0) -> float:
    """Pace in [0, 1] of video span [a, b] s (widened to >= min_ctx_s around its middle, so short segments are not noise).
    dyn_raw: [T25, 5] raw video curves; cuts: cut times s."""
    T = len(dyn_raw) / RATE
    half = max(b - a, min_ctx_s) / 2
    mid = (a + b) / 2
    a2, b2 = max(0.0, mid - half), min(T, mid + half)
    rate = np.sum((cuts >= a2) & (cuts < b2)) / max(b2 - a2, 1e-3) * 60
    mot = float(dyn_raw[int(a2 * RATE):max(int(b2 * RATE), int(a2 * RATE) + 1), 0].mean())
    return float((pct(ref.v_cut, rate) + pct(ref.v_mot, mot)) / 2)


class MusicEnergy:
    """Energy percentiles of music windows from 25 Hz prefix sums of raw log_onset and rms_db."""

    def __init__(self, store: FeatureStore, index: MusicIndex, ref: Reference):
        self.ref = ref
        self.cs = []
        for m in index.ids:
            x = store.music(m)["dyn_25hz"][:, [0, 2]].astype(np.float64)
            self.cs.append(np.vstack([np.zeros((1, 2)), np.cumsum(x, 0)]))
        self.file = np.array([(pct(ref.m_on, c[-1, 0] / (len(c) - 1)) + pct(ref.m_rms, c[-1, 1] / (len(c) - 1))) / 2
                              for c in self.cs])

    def windows(self, fi: np.ndarray, st: np.ndarray, d: float) -> np.ndarray:
        out = np.empty(len(fi))
        for i in np.unique(fi):
            sel = fi == i
            c = self.cs[i]
            n = len(c) - 1
            a = np.clip(np.round(st[sel] * RATE).astype(int), 0, n - 1)
            b = np.clip(a + max(int(round(d * RATE)), 1), a + 1, n)
            mean = (c[b] - c[a]) / (b - a)[:, None]
            out[sel] = (pct(self.ref.m_on, mean[:, 0]) + pct(self.ref.m_rms, mean[:, 1])) / 2
        return out


class PaceIndex(BranchIndex):
    """Adds the "pace" branch: -|video pace of the span - music energy|."""
    name = "pace"

    def __init__(self, index, energy: MusicEnergy, pace_of):
        super().__init__(index)
        self.energy, self.pace_of = energy, pace_of

    def window_branch(self, a, b, fi, st, d):
        return -np.abs(self.pace_of(a, b) - self.energy.windows(fi, st, d))

    def file_branch(self, a, b):
        return -np.abs(self.pace_of(a, b) - self.energy.file)


def setup(store: FeatureStore, index, feat: dict) -> PaceIndex:
    ref = build_reference(store, index.ids)
    dyn, cuts = np.asarray(feat["dyn_25hz"], float), np.asarray(feat["cut_times_s"], float)
    whole = video_pace(dyn, cuts, 0.0, len(dyn) / RATE, ref)
    log.info("pace match: video pace %.2f (library percentile); %d videos, %d music files in the reference",
             whole, len(ref.v_cut), len(ref.m_on))
    return PaceIndex(index, MusicEnergy(store, index, ref), lambda a, b: video_pace(dyn, cuts, a, b, ref))
