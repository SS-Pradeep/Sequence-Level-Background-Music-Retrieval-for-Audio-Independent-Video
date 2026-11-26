"""The person / face branch inside the coarse-to-fine pipeline, stitching and render (Step 3-4).

Nothing in bgm.retrieval is edited. A `PersonVideoQuery` (a VideoQuery subclass) returns, from `pooled`, the frame
head's embedding as a `Tagged` array that also carries the branch head's embedding of the same span; a
`PersonMusicIndex` (a MusicIndex subclass) adds a "person" raw score to `file_scores` and `window_scores`. The existing
`run_video`, `coarse_shortlist`, `fine`, `build_lattice`, `decode_lattice` then z-score and weight it like any other
branch: score = z(s_frame) + w * z(s_person) per segment, with w in fusion.json's w_coarse and w_fine ("person" key).

A system is a spec:
  {"mode": "late",  "frame": {"variant": "siglip-both", "seeds": [...]},
                    "branch": {"kind": "person" | "face", "variant": ..., "seeds": [...]}}     P1, P2, P4
  {"mode": "early", "frame": {"variant": "siglipperson-both", "seeds": [...]}}                 P3 (one head)
  {"mode": "late",  "frame": {...}}  without a branch                                          P0, P0s
Several seeds form an ensemble (mean cosine, D65).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from bgm.eval.attributes import fit_calibration
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.person.train import PersonStore, load_person_model, person_vector, span_index, video_input
from bgm.retrieval.index import EnsembleSemantic, FoldModels, MusicIndex, VideoQuery, concat_query, pair_query
from bgm.train.semantic import SemanticModel

import torch


class Tagged(np.ndarray):
    """The frame embedding [256] with the branch embedding of the same span attached as `.person`."""
    person: np.ndarray


def tag(gv: np.ndarray, person: np.ndarray) -> Tagged:
    t = np.asarray(gv).view(Tagged)
    t.person = person
    return t


def _semantic(run: Path, cfg, device: str) -> SemanticModel:
    st = torch.load(run / "ckpt.pt", map_location=device)
    m = SemanticModel(st["dv"], st["dm"], cfg, device)
    m.load(st["state"])
    m.eval()
    return m


def _models(root: Path, variant: str, seeds: list[int], fold: int, cfg, device: str, person_head: bool):
    load = (lambda r: load_person_model(r, cfg, device)) if person_head else (lambda r: _semantic(r, cfg, device))
    ms = [load(root / "semantic" / variant / f"fold{fold}" / f"seed{s}") for s in seeds]
    return ms[0] if len(ms) == 1 else EnsembleSemantic(ms)


@dataclass
class PersonFoldModels(FoldModels):
    spec: dict | None = None
    person: object | None = None          # branch model (late fusion), or None


def load_models(cfg, store: FeatureStore, fold: int, spec: dict, device: str = "cuda", train_ids=None) -> PersonFoldModels:
    root = Path(cfg.paths.runs)
    fr, br = spec["frame"], spec.get("branch")
    sem = _models(root, fr["variant"], fr["seeds"], fold, cfg, device, person_head=spec["mode"] == "early")
    person = None if br is None else _models(root, br["variant"], br["seeds"], fold, cfg, device, person_head=br["kind"] == "person")
    ids = train_ids if train_ids is not None else fold_ids(cfg, fold)["train"]
    cal = fit_calibration(store, ids, "attr_logits_1hz", list(cfg.features.attributes.temps), tag=f"fold{fold}")
    return PersonFoldModels(fold, -1, sem, None, cal, device, spec, person)


@dataclass
class PersonVideoQuery(VideoQuery):
    """A VideoQuery with the per-second person arrays (and D77 face features for P1) of the same timeline."""
    person_emb: np.ndarray | None = None      # [T1, 1152]
    person_has: np.ndarray | None = None      # [T1] bool
    face_1hz: np.ndarray | None = None        # [T1, 17]

    @classmethod
    def wrap(cls, vq: VideoQuery, **extra) -> "PersonVideoQuery":
        return cls(**{f.name: getattr(vq, f.name) for f in fields(VideoQuery)}, **extra)

    def _branch_input(self, kind: str, idx: np.ndarray) -> np.ndarray:
        if kind == "person":
            return person_vector(self.person_emb, self.person_has, idx)
        return l2(self.face_1hz[idx].astype(np.float64).mean(0)).astype(np.float32)

    def pooled(self, a: float, b: float):
        m: PersonFoldModels = self._m
        a2, b2 = apply_span_rule(a, b, self.duration, self._min_span)
        idx = span_index(len(self.sem_1hz), a2, b2)
        pv = self._mood.means(np.array([a2]), np.array([b2]))[0]
        if m.spec["mode"] == "early":
            x = video_input("early", self.sem_1hz, self.person_emb, self.person_has, idx)
            return m.sem.embed("video", x[None])[0], pv
        gv, pv = super().pooled(a, b)
        if m.person is None:
            return gv, pv
        return tag(gv, m.person.embed("video", self._branch_input(m.spec["branch"]["kind"], idx)[None])[0]), pv


class PersonMusicIndex(MusicIndex):
    """MusicIndex plus a 'person' branch scored by the branch model's music head on the same CLAP+MERT inputs."""

    def __init__(self, store: FeatureStore, models: PersonFoldModels, music_ids: list[str], **kw):
        super().__init__(store, models, music_ids, **kw)
        self.gp = None
        if models.person is not None:
            whole = np.stack([self._sem_raw(i, np.array([0.0]), np.array([it.duration]))[0] for i, it in enumerate(self.items)])
            self.gp = models.person.embed("music", whole)

    def file_scores(self, gv, pv):
        out = super().file_scores(gv, pv)
        if self.gp is not None:
            out["person"] = self.gp @ gv.person
        return out

    def window_scores(self, q, fi, st, d):
        out = super().window_scores(q, fi, st, d)
        if self.gp is not None:
            sem_x = np.zeros((len(fi), sum(p.x.shape[1] for p in self.items[0].sem)), np.float32)
            for i in np.unique(fi):
                sel = fi == i
                dur = self.items[i].duration
                spans = np.array([apply_span_rule(s, s + d, dur, self.store.min_span) for s in st[sel]])
                sem_x[sel] = self._sem_raw(i, spans[:, 0], spans[:, 1])
            out["person"] = self.m.person.embed("music", sem_x) @ q.gv.person
        return out


def _face(store: FeatureStore, p: str) -> np.ndarray | None:
    return store.video(p).get("face_expr_1hz")


def timeline_1hz(store: FeatureStore, spans: list[tuple[str, float, float]], T1: int, get) -> np.ndarray:
    """1 Hz rows of `get(pair_id)` on a joined timeline, by concat_query's rule (the part containing each second's
    centre, clipped to that part's length)."""
    rows = []
    for k in range(T1):
        c = k + 0.5
        p, s0, d = next((x for x in spans if x[1] <= c < x[1] + x[2]), spans[-1])
        arr = get(p)
        rows.append(arr[int(np.clip(np.floor(c - s0), 0, len(arr) - 1))])
    return np.stack(rows)


def person_query(store: FeatureStore, pstore: PersonStore, parts: list[str]) -> PersonVideoQuery:
    """A dataset pair (one part) or a concatenation of pairs, with its person and face arrays."""
    vq = pair_query(store, parts[0]) if len(parts) == 1 else concat_query(store, parts)
    spans, T1 = vq.meta["parts"], len(vq.sem_1hz)
    emb = timeline_1hz(store, spans, T1, lambda p: pstore.get(p)[0])
    has = timeline_1hz(store, spans, T1, lambda p: pstore.get(p)[1])
    face = timeline_1hz(store, spans, T1, lambda p: _face(store, p)) if _face(store, spans[0][0]) is not None else None
    return PersonVideoQuery.wrap(vq, person_emb=emb, person_has=has, face_1hz=face)
