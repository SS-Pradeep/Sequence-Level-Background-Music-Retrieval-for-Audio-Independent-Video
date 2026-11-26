"""Retrieval index and queries (SPEC §5.9) for one trained (fold, seed): semantic heads, dynamics encoders and the
fold's attribute calibration.

Index, per music file: the head applied to the whole-file mean, float64 prefix sums of the raw 1 Hz semantic arrays
and of the calibrated per-second mood distributions (O(1) window means), and the whole file's dynamics encoding
(hm, [T25, 64]); the encodings of all files are concatenated with zero padding so one conv1d scores every 25 Hz
offset of every file for a segment. Query: the whole video's curves are encoded once and segments are sliced from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bgm.eval.attributes import Calibration, fit_calibration, s_attr
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.features.keys import stitch_key
from bgm.train.dynamics import DynModel
from bgm.train.semantic import SemanticModel

RATE = 25
MUSIC_KEYS = ("sem_clap_1hz", "sem_mert_1hz")          # SigLIP + CLAP+MERT (M6 best variant)


class Prefix:
    """[T1, D] 1 Hz array with float64 prefix sums; vectorized means over [a, b) (samples centred at k + 0.5)."""

    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, np.float32)
        self.cs = np.vstack([np.zeros((1,) + self.x.shape[1:]), np.cumsum(self.x, 0, dtype=np.float64)])

    def means(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        T = len(self.x)
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        lo = np.clip(np.ceil(a - 0.5 - 1e-9), 0, T).astype(int)
        hi = np.clip(np.ceil(b - 0.5 - 1e-9), 0, T).astype(int)
        empty = hi <= lo
        near = np.clip(np.floor((a + b) / 2), 0, T - 1).astype(int)
        lo2, hi2 = np.where(empty, near, lo), np.where(empty, near + 1, hi)
        return (self.cs[hi2] - self.cs[lo2]) / (hi2 - lo2)[:, None]


class EnsembleSemantic:
    """Several semantic models used as one (D65). Each side's embedding is the members' unit embeddings concatenated
    and scaled by 1 / sqrt(n), so a dot product between a video and a music embedding is the MEAN of the members'
    cosines: every score in the index and the pipeline becomes the ensemble score with no other change."""

    def __init__(self, models: list):
        self.models = list(models)

    def embed(self, side: str, x: np.ndarray) -> np.ndarray:
        return np.concatenate([m.embed(side, x) for m in self.models], 1) / np.sqrt(len(self.models))

    def eval(self) -> None:
        for m in self.models:
            m.eval()


@dataclass
class FoldModels:
    fold: int
    seed: int
    sem: SemanticModel
    dyn: DynModel | None       # None: no dynamics checkpoint (dyn_variant null); the dyn branch then scores 0
    cal: Calibration
    device: str

    @classmethod
    def load(cls, cfg: DictConfig, store: FeatureStore, fold: int, seed: int, sem_variant="siglip-both",
             dyn_variant="shiftneg", device=None, train_ids=None) -> "FoldModels":
        root = Path(cfg.paths.runs)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")   # CPU fallback when the GPU is busy

        def one(s):
            st = torch.load(root / "semantic" / sem_variant / f"fold{fold}" / f"seed{s}" / "ckpt.pt", map_location=device)
            m = SemanticModel(st["dv"], st["dm"], cfg, device)
            m.load(st["state"])
            m.eval()
            return m
        ens = cfg.pipeline.get("sem_seeds", None)           # D65: a list of seeds -> their ensemble (the run's own seed is ignored)
        sem = EnsembleSemantic([one(s) for s in ens]) if ens else one(seed)
        dyn = None
        if dyn_variant is not None:
            dyn = DynModel(cfg, device)
            dyn.load(torch.load(root / "dynamics" / dyn_variant / f"fold{fold}" / f"seed{seed}" / "ckpt.pt", map_location=device)["state"])
            dyn.eval()
        ids = train_ids if train_ids is not None else fold_ids(cfg, fold)["train"]
        cal = fit_calibration(store, ids, "attr_logits_1hz", list(cfg.features.attributes.temps), tag=f"fold{fold}")
        return cls(fold, seed, sem, dyn, cal, device)


@dataclass
class MusicItem:
    music_id: str
    duration: float
    sem: list[Prefix]          # raw CLAP, raw MERT
    mood: Prefix               # calibrated per-second mood distribution
    beats: np.ndarray
    tempo: float
    key: int


class MusicIndex:
    def __init__(self, store: FeatureStore, models: FoldModels, music_ids: list[str], pad_frames: int = 20 * RATE + 8):
        self.store, self.m, self.ids = store, models, list(music_ids)
        self.pos = {m: i for i, m in enumerate(self.ids)}
        tf = models.cal.transform("music")
        self.items = []
        for m in self.ids:
            d = store.music(m)
            self.items.append(MusicItem(m, store.duration("music", m), [Prefix(d[k]) for k in MUSIC_KEYS],
                                        Prefix(tf(d["attr_logits_1hz"].astype(np.float32))), np.asarray(d["beat_times_s"], float),
                                        float(d["tempo_bpm"]), stitch_key(int(d["key_pc"]), int(d["key_mode"]), int(d["tonic_pc"]), False)))
        whole = np.stack([self._sem_raw(i, np.array([0.0]), np.array([it.duration]))[0] for i, it in enumerate(self.items)])
        self.g = models.sem.embed("music", whole)                                          # [G, 256]
        self.p = np.stack([it.mood.means(np.array([0.0]), np.array([it.duration]))[0] for it in self.items])
        self.dyn = None
        if models.dyn is not None:
            with torch.no_grad():
                encs = [models.dyn.encode("music", store.dyn("music", m)) for m in self.ids]
            self.nfr = np.array([len(h) for h in encs])
            self.base = np.r_[0, np.cumsum(self.nfr + pad_frames)[:-1]]
            dev = models.device
            self.dyn = torch.cat([torch.cat([h, torch.zeros(pad_frames, h.shape[1], device=dev)]) for h in encs]).T[None].contiguous()

    def _sem_raw(self, i: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([l2(p.means(a, b)) for p in self.items[i].sem], 1).astype(np.float32)

    def windows(self, files: list[int], d: float, hop: float) -> tuple[np.ndarray, np.ndarray]:
        """Sliding windows of length d every `hop` s over each file (last one flush with the end; start 0 if shorter)."""
        fi, st = [], []
        for i in files:
            last = max(0.0, self.items[i].duration - d)
            s = np.arange(0.0, last + 1e-9, hop)
            if last - s[-1] > 1e-6:
                s = np.r_[s, last]
            fi.append(np.full(len(s), i)), st.append(s)
        return np.concatenate(fi), np.concatenate(st)

    def window_scores(self, q: "SegmentQuery", fi: np.ndarray, st: np.ndarray, d: float) -> dict[str, np.ndarray]:
        """Raw branch scores of one segment against windows (file index fi, start st, length d)."""
        min_span = self.store.min_span
        sem_dim = sum(p.x.shape[1] for p in self.items[0].sem)
        sem_x, mood = np.zeros((len(fi), sem_dim), np.float32), np.zeros((len(fi), self.p.shape[1]))
        for i in np.unique(fi):
            sel = fi == i
            dur = self.items[i].duration
            spans = np.array([apply_span_rule(s, s + d, dur, min_span) for s in st[sel]])
            sem_x[sel] = self._sem_raw(i, spans[:, 0], spans[:, 1])
            mood[sel] = self.items[i].mood.means(spans[:, 0], spans[:, 1])
        sem = self.m.sem.embed("music", sem_x) @ q.gv
        if q.ctx is not None:                               # D75: whole-video context, per window's file
            sem = sem + q.ctx[fi]
        attr = s_attr(q.pv[None], mood)
        if self.dyn is None:
            return {"sem": sem, "dyn": np.zeros(len(fi), np.float32), "attr": attr}
        f0 = self.base[fi] + np.clip(np.round(st * RATE).astype(int), 0, None)
        with torch.no_grad():
            corr = F.conv1d(self.dyn, q.hv.T[None])[0, 0] / q.hv.shape[0]
        dyn = corr[torch.from_numpy(f0).to(corr.device)].cpu().numpy()
        return {"sem": sem, "dyn": dyn, "attr": attr}

    def file_scores(self, gv: np.ndarray, pv: np.ndarray) -> dict[str, np.ndarray]:
        return {"sem": self.g @ gv, "attr": s_attr(pv[None], self.p)}


@dataclass
class SegmentQuery:
    a: float
    b: float
    gv: np.ndarray             # semantic head output [256]
    pv: np.ndarray             # calibrated mood distribution [M]
    hv: torch.Tensor | None    # [L, 64] slice of the whole-video dynamics encoding; None without a dynamics model
    ctx: np.ndarray | None = None   # [G] context_weight x cos(whole video, each index file) (D75); None = off


@dataclass
class VideoQuery:
    """A video's features (a dataset pair, a concatenation of pairs, or an external video)."""
    sem_1hz: np.ndarray        # [T1, 1152] SigLIP
    attr_1hz: np.ndarray       # [T1, M] raw mood logits
    dyn_25hz: np.ndarray       # [T25, 5] normalized curves
    cut_times: np.ndarray
    segments: list[tuple[float, float]]
    duration: float
    meta: dict = field(default_factory=dict)

    def prepare(self, models: FoldModels, min_span: float) -> None:
        self._m, self._min_span = models, min_span
        self._sem, self._mood = Prefix(self.sem_1hz), Prefix(models.cal.transform("video")(self.attr_1hz.astype(np.float32)))
        self._hv = None
        if models.dyn is not None:
            with torch.no_grad():
                self._hv = models.dyn.encode("video", self.dyn_25hz)

    def set_context(self, index: "MusicIndex", weight: float) -> None:
        """D75: add `weight` x cos(whole video, the window's whole file) to every window's semantic score. Whole-clip
        matching is more reliable than a short segment's, and the whole video is known at query time."""
        radius = index.store.cfg.retrieval.get("context_radius_s", None) if hasattr(index.store, "cfg") else None
        self._ctx_local = (index, float(weight), float(radius)) if (weight and radius) else None   # D76: per-segment context
        self._ctx = None if (not weight or radius) else float(weight) * (index.g @ self.pooled(0.0, self.duration)[0])

    def pooled(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        a2, b2 = apply_span_rule(a, b, self.duration, self._min_span)
        v = l2(self._sem.means(np.array([a2]), np.array([b2]))[0]).astype(np.float32)
        return self._m.sem.embed("video", v[None])[0], self._mood.means(np.array([a2]), np.array([b2]))[0]

    def segment(self, a: float, b: float) -> SegmentQuery:
        gv, pv = self.pooled(a, b)
        ctx = getattr(self, "_ctx", None)
        loc = getattr(self, "_ctx_local", None)
        if loc is not None:                                 # D76: the video within `radius` s of the segment's middle
            index, weight, radius = loc
            mid = (a + b) / 2
            ctx = weight * (index.g @ self.pooled(max(0.0, mid - radius), min(self.duration, mid + radius))[0])
        if self._hv is None:
            return SegmentQuery(a, b, gv, pv, None, ctx)
        f0 = min(int(round(a * RATE)), len(self._hv) - 1)
        L = max(1, min(int(round((b - a) * RATE)), len(self._hv) - f0))
        return SegmentQuery(a, b, gv, pv, self._hv[f0:f0 + L], ctx)


def pair_query(store: FeatureStore, pair_id: str) -> VideoQuery:
    v = store.video(pair_id)
    return VideoQuery(v["sem_siglip_1hz"].astype(np.float32), v["attr_logits_1hz"], store.dyn("video", pair_id),
                      np.asarray(v["cut_times_s"], float), list(store.segments[pair_id]), store.duration("video", pair_id),
                      {"parts": [(pair_id, 0.0, store.duration("video", pair_id))]})


def concat_query(store: FeatureStore, parts: list[str]) -> VideoQuery:
    """Join clips end to end on the 25 Hz grid (part i starts at t_i = sum of earlier T25 / 25). 1 Hz samples are
    taken from the part containing each second's centre."""
    t0, dyn, cuts, segs, spans = 0.0, [], [], [], []
    for p in parts:
        v = store.video(p)
        d = len(v["dyn_25hz"]) / RATE
        dyn.append(store.dyn("video", p))
        cuts += [t0 + c for c in v["cut_times_s"]] + ([t0] if t0 > 0 else [])
        segs += [(t0 + a, min(t0 + b, t0 + d)) for a, b in store.segments[p] if a < d]
        spans.append((p, t0, d))
        t0 += d
    T1 = int(np.floor(t0 + 1e-9))
    sem, attr = [], []
    for k in range(T1):
        c = k + 0.5
        p, s0, d = next((x for x in spans if x[1] <= c < x[1] + x[2]), spans[-1])
        v = store.video(p)
        j = int(np.clip(np.floor(c - s0), 0, len(v["sem_siglip_1hz"]) - 1))
        sem.append(v["sem_siglip_1hz"][j].astype(np.float32))
        attr.append(v["attr_logits_1hz"][j])
    return VideoQuery(np.stack(sem), np.stack(attr), np.concatenate(dyn), np.sort(np.array(cuts)), segs, t0, {"parts": spans})
