"""Coarse-to-fine retrieval and pipeline accuracy (SPEC §5.8-5.10, §5.12).

Fine windows slide every `retrieval.window_hop_s` over each shortlisted file (user decision D21: sliding windows
instead of beat-start windows). Each branch's raw scores are standardized over the segment's full window set and the
mu / sigma are kept, so continuation windows (§5.10) are scored with the same statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgm.eval.metrics import top_offsets
from bgm.retrieval.index import MusicIndex, VideoQuery
from bgm.retrieval.stitch import Candidate, add_continuations, forward_costs, greedy, viterbi

BRANCHES = ("sem", "dyn", "attr")


def zstats(raw: dict[str, np.ndarray], floor: float = 1e-3) -> dict[str, tuple[float, float]]:
    return {b: (float(s.mean()), max(float(s.std()), floor)) for b, s in raw.items()}


def fuse(raw: dict[str, np.ndarray], stats: dict, w: dict[str, float]) -> np.ndarray:
    """SPEC §5.8: sum_b w_b (s_b - mu_b) / max(sigma_b, 1e-3)."""
    out = 0.0
    for b, wb in w.items():
        if wb > 0:
            mu, sd = stats[b]
            out = out + wb * (raw[b] - mu) / sd
    n = len(next(iter(raw.values())))
    return out if not np.isscalar(out) else np.zeros(n)


@dataclass
class SegmentResult:
    a: float
    b: float
    fi: np.ndarray          # window file index into the MusicIndex
    st: np.ndarray          # window start s
    raw: dict               # branch -> raw scores [W]
    stats: dict             # branch -> (mu, sigma) over this segment's window set


def coarse_shortlist(index: MusicIndex, vq: VideoQuery, w_coarse: dict, K: int) -> list[int]:
    """Union of the whole-video top K and each segment's top K files (semantic + mood, per-query standardized)."""
    keep = set()
    spans = [(0.0, vq.duration)] + list(vq.segments)
    for a, b in spans:
        gv, pv = vq.pooled(a, b)
        raw = index.file_scores(gv, pv)
        s = fuse(raw, zstats(raw), w_coarse)
        keep.update(np.argsort(-s)[:K].tolist())
    return sorted(keep)


def fine(index: MusicIndex, vq: VideoQuery, files: list[int], hop: float) -> list[SegmentResult]:
    out = []
    for a, b in vq.segments:
        d = b - a
        q = vq.segment(a, b)
        fi, st = index.windows(files, d, hop)
        raw = index.window_scores(q, fi, st, d)
        out.append(SegmentResult(a, b, fi, st, raw, zstats(raw)))
    return out


def run_video(index: MusicIndex, vq: VideoQuery, w_coarse: dict, K: int, hop: float, use_coarse: bool = True) -> list[SegmentResult]:
    vq.set_context(index, float(index.store.cfg.retrieval.get("context_weight", 0.0)) if hasattr(index.store, "cfg") else 0.0)
    files = coarse_shortlist(index, vq, w_coarse, K) if use_coarse else list(range(len(index.ids)))
    return fine(index, vq, files, hop)


# ---------------------------------------------------------------- ground truth and pipeline accuracy
def truth_for(vq: VideoQuery, store, a: float, b: float) -> tuple[str, float]:
    """(own music_id, aligned offset) of a segment: the part containing its midpoint, offset = segment start within
    the part, shifted back to fit the file (SPEC §5.12)."""
    mid = (a + b) / 2
    p, t0, d = next((x for x in vq.meta["parts"] if x[1] <= mid < x[1] + x[2]), vq.meta["parts"][-1])
    m = store.music_of[p]
    off = min(a - t0, max(0.0, store.duration("music", m) - (b - a)))
    return m, max(0.0, off)


def segment_hits(index: MusicIndex, store, vq: VideoQuery, r: SegmentResult, w: dict, J: int, tol: float = 0.5) -> dict:
    m_true, off = truth_for(vq, store, r.a, r.b)
    S = fuse(r.raw, r.stats, w)
    order = np.argsort(-S)
    files = np.array(index.ids)[r.fi]
    tr = np.array([store.music_track[m] for m in files])
    win_ok = (files == m_true) & (np.abs(r.st - off) <= tol)
    return {"track@1": float(tr[order[0]] == store.music_track[m_true]), "window@1": float(win_ok[order[0]]),
            "window@J": float(win_ok[order[:J]].any()), "dur": r.b - r.a}


# ---------------------------------------------------------------- offset precision (PROGRESS.md D60)
OFFSET_TOLS_S = (1.0, 3.0, 5.0)


def offset_rules(in_group: np.ndarray, same_file: np.ndarray, start_s: np.ndarray, true_off: float,
                 tols=OFFSET_TOLS_S) -> dict[str, np.ndarray]:
    """One segment's candidate windows -> a boolean array per rule: would choosing that window count as correct?
    in_group / same_file: [W] the window's file is in the query's track group / is the query's own music file;
    start_s: [W] window starts, s; true_off: the aligned offset, s.
      'track'   the file is in the query's track group;
      '<tol>'   'track' AND |start - true offset| <= tol s (one rule per tolerance, e.g. '1', '3', '5');
      'window'  the query's own file AND within 0.5 s (window@1's rule).
    As for grounding_rules (D33): a rule's hit is its array at the chosen window, and its chance level for the query
    is the array's mean over the candidate set a random choice would draw from."""
    err = np.abs(np.asarray(start_s, float) - true_off)
    out = {"track": np.asarray(in_group, bool), "window": np.asarray(same_file, bool) & (err <= 0.5 + 1e-9)}
    out.update({f"{t:g}": out["track"] & (err <= t + 1e-9) for t in tols})
    return out


def segment_offset_rules(index: MusicIndex, store, vq: VideoQuery, r: SegmentResult, gtrack: np.ndarray,
                         tols=OFFSET_TOLS_S) -> dict[str, np.ndarray]:
    """offset_rules over the windows of one SegmentResult. gtrack: [G] track_id of each index file."""
    m_true, off = truth_for(vq, store, r.a, r.b)
    return offset_rules(gtrack[r.fi] == store.music_track[m_true], r.fi == index.pos[m_true], r.st, off, tols)


def top1_hits(rules: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, float]:
    """{rule: hit of the top-1 window}; windows tied at the top score share the hit (D34: the expected hit of a
    uniformly random tie-break, the same assumption as the chance level)."""
    top = top_offsets(scores)
    return {k: float(v[top].mean()) for k, v in rules.items()}


def rule_chance(rules: dict[str, np.ndarray]) -> dict[str, float]:
    """{rule: share of the candidate windows the rule accepts}, plus 'cond/<tol>': the share of the track group's
    windows within tol of the true offset (chance of the offset alone, given the right track; nan if none)."""
    out = {k: float(v.mean()) for k, v in rules.items()}
    n = rules["track"].sum()
    out.update({f"cond/{k}": float(v.sum() / n) if n else float("nan") for k, v in rules.items() if k not in ("track", "window")})
    return out


# ---------------------------------------------------------------- candidates for stitching
def build_candidates(index: MusicIndex, vq: VideoQuery, results: list[SegmentResult], w: dict, J: int,
                     pen: dict, beam_factor: int = 5, tol_s: float = 0.04) -> list[list[Candidate]]:
    """Per segment: top J windows + contiguous continuations of the previous segment's survivors, scored with this
    segment's saved z-statistics; lists longer than beam_factor x J keep the lowest ACCUMULATED Viterbi cost."""
    file_len = {it.music_id: it.duration for it in index.items}
    cands, D = [], None
    for i, r in enumerate(results):
        S = fuse(r.raw, r.stats, w)
        d = r.b - r.a
        top = np.argsort(-S)[:J]
        cur = [Candidate(index.ids[r.fi[k]], float(r.st[k]), d, index.items[r.fi[k]].key, index.items[r.fi[k]].tempo, float(-S[k]))
               for k in top]
        if i > 0:
            q = vq.segment(r.a, r.b)

            def score_fn(m, off, dur, q=q, r=r):
                raw = index.window_scores(q, np.array([index.pos[m]]), np.array([off]), dur)
                return float(-fuse(raw, r.stats, w)[0])
            cur = add_continuations(cands[-1], cur, d, file_len, score_fn)
            D_cur = forward_costs(cands[-1], D, cur, **pen, tol_s=tol_s)
            if len(cur) > beam_factor * J:
                keep = np.argsort(D_cur)[:beam_factor * J]
                cur, D_cur = [cur[k] for k in keep], D_cur[keep]
            D = D_cur
        else:
            D = np.array([c.cost for c in cur])
        cands.append(cur)
    return cands


def decode(cands, pen: dict, method: str = "viterbi", tol_s: float = 0.04) -> list[Candidate]:
    path = viterbi(cands, **pen, tol_s=tol_s) if method == "viterbi" else greedy(cands)
    return [c[j] for c, j in zip(cands, path)]


def path_accuracy(store, vq: VideoQuery, results: list[SegmentResult], chosen: list[Candidate], tol: float = 0.5) -> dict:
    """Duration-weighted track / window accuracy of a decoded path over one video."""
    w = np.array([r.b - r.a for r in results])
    tr_ok, win_ok = [], []
    for r, c in zip(results, chosen):
        m_true, off = truth_for(vq, store, r.a, r.b)
        tr_ok.append(store.music_track[c.music_id] == store.music_track[m_true])
        win_ok.append(c.music_id == m_true and abs(c.offset - off) <= tol)
    return {"track": float(np.average(tr_ok, weights=w)), "window": float(np.average(win_ok, weights=w))}
