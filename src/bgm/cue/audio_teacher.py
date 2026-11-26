"""Audio teacher for the cue-switch predictor (PROGRESS.md D98): training targets read from the music itself.

The joined training sequences label every clip join between track groups as a change and every boundary inside a clip
as no change, so the label is the join (D85). The teacher instead asks the music at every boundary:

  d(t) = 1 - cos(mean MERT over [t - k, t), mean MERT over [t, t + k))        (k = `cue.audio.half_s` seconds)

on the sequence's own 1 Hz MERT (`sem_mert_1hz` of each part's music, music sample 0 = the part's first frame, joined
on the sequence clock exactly like `concat_query` joins the video). One quantity covers both ideas of the brief:
inside a clip, a music change the clip cutter did not split (pauses < 1.5 s merge cues) gets a high d; at a join, two
clips with similar music get a low d. Targets are p(d) = sigmoid(a * log d + b), a logistic fit on TRAIN-fold
sequences of (track-group change joins = 1, within-clip boundaries = 0), so p is on the same scale as the old labels.

Settings (k = 3 s) were chosen on fold-0 training sequences only (D98): change joins vs within-clip AUROC 0.90, vs
same-music joins 0.81, vs same-music splices 0.80.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

RATE = 25
SEP = "§"                      # synthetic part ids from train.splice_clip: <pair_id>§<t1>-<t2>A / B
EPS = 1e-4


def part_music_1hz(store, part: str) -> tuple[np.ndarray, int]:
    """1 Hz MERT [T, 1024] of a sequence part and its length in 25 Hz frames. Splice parts (D94) take the music of the
    kept span: A = [0, t1) s, B = [t2, end)."""
    base = part.split(SEP)[0]
    m = store.music(store.music_of[base])["sem_mert_1hz"].astype(np.float32)
    n25 = len(store.video(base)["dyn_25hz"])
    if SEP not in part:
        return m, n25
    t1, t2 = map(int, part.split(SEP)[1][:-1].split("-"))
    return (m[:t1], t1 * RATE) if part.endswith("A") else (m[t2:], n25 - t2 * RATE)


def sequence_music_1hz(parts_music: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """[(mert_1hz, n25)] per part -> [T1, D] on the sequence clock: second k takes the part containing k + 0.5 s, row
    floor(k + 0.5 - part start) (concat_query's rule for the video)."""
    spans, t0 = [], 0.0
    for m, n25 in parts_music:
        spans.append((m, t0, n25 / RATE))
        t0 += n25 / RATE
    out = []
    for k in range(int(np.floor(t0 + 1e-9))):
        c = k + 0.5
        m, s0, _ = next((x for x in spans if x[1] <= c < x[1] + x[2]), spans[-1])
        out.append(m[int(np.clip(np.floor(c - s0), 0, len(m) - 1))])
    return np.stack(out)


def boundary_times(raw_tokens: np.ndarray, seg_part: list[int], part_n25: list[int]) -> np.ndarray:
    """Segment start times [N - 1] of boundaries 1..N-1 on the sequence clock. Segments tile each part from 0 (asserted
    for segments.csv), so a start is the part's offset plus the earlier segments' lengths (raw log-duration column)."""
    dur = np.exp(raw_tokens[:, -3].astype(np.float64))
    offs = np.concatenate([[0.0], np.cumsum(np.asarray(part_n25, float) / RATE)])
    starts, prev, acc = [], -1, 0.0
    for i, p in enumerate(seg_part):
        if p != prev:
            prev, acc = p, 0.0
        starts.append(offs[p] + acc)
        acc += dur[i]
    return np.array(starts[1:])


def music_distance(e_1hz: np.ndarray, times: np.ndarray, half_s: int = 3) -> np.ndarray:
    """e_1hz [T1, D] (sample k centred at k + 0.5 s), times [M] s -> d [M] = 1 - cos(mean of the up to half_s samples
    before round(t), mean of the up to half_s from round(t)); 0 where a side is empty."""
    e = e_1hz.astype(np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    out = np.zeros(len(times))
    for j, t in enumerate(times):
        i = int(round(float(t)))
        a, b = e[max(0, i - half_s):i], e[i:i + half_s]
        if len(a) and len(b):
            a, b = a.mean(0), b.mean(0)
            out[j] = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return out


class Calibration:
    """p(d) = sigmoid(a * log(d + EPS) + b)."""

    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    @classmethod
    def fit(cls, d: np.ndarray, y: np.ndarray) -> "Calibration":
        lr = LogisticRegression().fit(np.log(np.asarray(d, float) + EPS)[:, None], np.asarray(y, int))
        return cls(lr.coef_[0, 0], lr.intercept_[0])

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.log(np.asarray(d, float) + EPS) + self.b)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}


class AudioTeacher:
    """Per-boundary music distance and its calibrated target for a sequence of parts."""

    def __init__(self, store, half_s: int = 3, calib: Calibration | None = None):
        self.store, self.half_s, self.calib = store, int(half_s), calib

    def distance(self, parts: list[str], raw_tokens: np.ndarray, seg_part: list[int]) -> np.ndarray:
        pm = [part_music_1hz(self.store, p) for p in parts]
        bt = boundary_times(raw_tokens, seg_part, [n for _, n in pm])
        return music_distance(sequence_music_1hz(pm), bt, self.half_s)

    def targets(self, parts: list[str], raw_tokens: np.ndarray, seg_part: list[int]) -> np.ndarray:
        return self.calib(self.distance(parts, raw_tokens, seg_part)).astype(np.float32)
