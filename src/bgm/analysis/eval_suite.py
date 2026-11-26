"""Evaluation suite for the final system (D111): retrieval, alignment, transitions and diversity on the title split.

The final system (D75, as benchmarked; render-only options off) is rebuilt for each (fold, ensemble seed e) of
runs/titlesplit/pipeline_ens10: the 10-head ensemble (head seeds 10e .. 10e + 9), coarse-to-fine retrieval over the
test fold's music files, Viterbi with the run's val-tuned penalties (stitch.json). Nothing is trained or tuned here.
Sanity checks: track / segment ranks equal the saved ensemble runs, natural-video Viterbi accuracy equals
stitch_perquery.npz.

1. Retrieval (982 test clips, 2,552 test segments; galleries as results/titlesplit/final.md §1): R@1/5/10, MedR,
   MRR, nDCG@10 (binary gain: every file of the query's track group is relevant), exact random baseline;
   ImageBind zero-shot as a reference row.
2. Alignment, natural test videos, per video: the soundtrack is the decoded path's windows.
   LB = share of the video's cuts (> 0.5 s from either end; the same cuts for every system) with a soundtrack beat
   within ±70 ms; reported with its excess over 20 circular shifts of the soundtrack's beats.
   tw-LB = mean over cuts of exp(-d² / 2σ²), d = distance to the nearest beat, σ = 70 ms (near misses earn credit).
   IB = ImageBind cosine between the segment's video and its music window (span means of imagebind_1hz),
   duration-weighted over the video's segments. KL = KL(original window ‖ chosen window) of the PANNs CNN14 AudioSet
   tag distributions (per-second sigmoid outputs averaged over the window, normalized to sum 1), duration-weighted.
3. Transitions, test concatenations (eval.concat_sets_file, 50 per fold): the true switches are the joins.
   F1 @ ±1.5 / ±2.0 s (one-to-one matching), deviation (MIREX style: median distance from each true join to the
   nearest predicted switch, and from each predicted switch to the nearest join), TBIoU (each true piece's best
   temporal IoU with a predicted piece, averaged), smoothness (switches and jumps per minute; median loudness jump at
   the seams of the rendered soundtrack, LU), DTW (cosine-distance DTW between the per-second CLAP embeddings of the
   produced and the original soundtrack, divided by the path length).
4. Diversity over all test segments of a fold (and the top-1 files of track retrieval): normalized entropy of the
   chosen-file distribution, coverage (share of the fold's gallery files used at least once), top-5 share (share of
   picks taken by the 5 most-chosen files).

Systems: original music (the reference), final (Viterbi), greedy (each segment's top window), random (a uniformly
random file and offset per segment, per-item seeded), and for alignment the final and the random paths with
render-time beat sync (D35, what `bgm render` outputs). LB / tw-LB are reported over all cuts and over within-piece
cuts only (cuts > 50 ms from any piece start), because beat sync puts every piece start on a beat by construction.

Run: python -m bgm.analysis.eval_suite [run | report]  -> runs/titlesplit/eval_suite/, results/titlesplit/eval_suite.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from bgm.config import load_config
from bgm.data.prep import interim_music_path
from bgm.eval.metrics import chance_mrr, chance_recall, cluster_bootstrap, cut_on_beat_rate, eligible_cuts, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.features.panns_tags import load_tags
from bgm.render.render import Run, beat_sync_runs, render_soundtrack, runs_from_path, seam_loudness_jumps
from bgm.retrieval.fuse import prepared_queries
from bgm.retrieval.index import FoldModels, MusicIndex, Prefix, concat_query
from bgm.retrieval.pipeline import path_accuracy, run_video, truth_for
from bgm.retrieval.stitch import Candidate
from bgm.retrieval.stitch_eval import (build_lattice, chosen_candidates, decode_lattice, greedy_lattice, read_concat_sets, switch_f1,
                                       within_piece_cuts)
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)
ROOT = Path("runs/titlesplit")
EXP = "eval_suite"
RESULTS = Path("results/titlesplit/eval_suite.md")
FOLDS, SEEDS = range(5), (0, 1, 2)
TOL, SIGMA = 0.07, 0.07
SYSTEMS = ("original", "viterbi", "greedy", "random")


# ---------------------------------------------------------------- small pure metrics
def ndcg_at(scores: np.ndarray, hits: np.ndarray, k: int = 10) -> float:
    """nDCG@k with binary gains; ties ordered hit-first (the rank_of_first_hit convention)."""
    order = np.lexsort((~hits, -scores))
    g = hits[order][:k].astype(float)
    disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = disc[:min(int(hits.sum()), k)].sum()
    return float((g * disc[:len(g)]).sum() / ideal)


def chance_ndcg(N: int, h: int, k: int = 10) -> float:
    disc = 1 / np.log2(np.arange(2, k + 2))
    return float((h / N) * disc[:min(k, N)].sum() / disc[:min(h, k)].sum())


def chance_medr(N: np.ndarray, h: np.ndarray, rng, draws: int = 500) -> float:
    """Expected median (over queries) rank of the first relevant item under random rankings."""
    meds = []
    for _ in range(draws):
        r = [1 + int(np.min(rng.choice(n, hh, replace=False))) for n, hh in zip(N, h)]
        meds.append(np.median(r))
    return float(np.mean(meds))


def nearest(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each x to the nearest element of ref (inf if ref is empty)."""
    if not len(ref):
        return np.full(len(x), np.inf)
    ref = np.sort(ref)
    i = np.clip(np.searchsorted(ref, x), 1, max(1, len(ref) - 1))
    lo, hi = ref[np.clip(i - 1, 0, len(ref) - 1)], ref[np.clip(i, 0, len(ref) - 1)]
    return np.minimum(np.abs(x - lo), np.abs(x - hi))


def beat_alignment(cuts: np.ndarray, beats: np.ndarray, duration: float, rng, draws: int = 20) -> dict:
    """LB and tw-LB of one soundtrack and their circular-shift baselines (NaN without cuts)."""
    if not len(cuts):
        return {k: np.nan for k in ("LB", "LB_shift", "twLB", "twLB_shift")}

    def tw(b):
        return float(np.mean(np.exp(-nearest(cuts, b) ** 2 / (2 * SIGMA ** 2)))) if len(b) else 0.0
    sh = [(beats + rng.uniform(0, duration)) % duration for _ in range(draws)]
    return {"LB": cut_on_beat_rate(cuts, beats, TOL), "LB_shift": float(np.mean([cut_on_beat_rate(cuts, b, TOL) for b in sh])),
            "twLB": tw(beats), "twLB_shift": float(np.mean([tw(b) for b in sh]))}


def pieces(bounds: list[float], duration: float) -> list[tuple[float, float]]:
    e = [0.0] + sorted(bounds) + [duration]
    return [(a, b) for a, b in zip(e, e[1:]) if b > a]


def tbiou(true_b: list[float], pred_b: list[float], duration: float) -> float:
    """Mean over true pieces of the best temporal IoU with a predicted piece."""
    P = pieces(pred_b, duration)
    out = []
    for a, b in pieces(true_b, duration):
        out.append(max(max(0.0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c)) for c, d in P))
    return float(np.mean(out))


def dtw_cost(X: np.ndarray, Y: np.ndarray) -> float:
    """Cosine-distance DTW between [n, D] and [m, D] unit sequences, total cost / path length."""
    C = 1 - X @ Y.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(opts))
            prev = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            D[i, j] = C[i - 1, j - 1] + opts[k]
            L[i, j] = L[prev] + 1
    return float(D[n, m] / L[n, m])


def diversity(picks: list[str], gallery_size: int) -> dict:
    _, c = np.unique(np.asarray(picks), return_counts=True)
    p = c / c.sum()
    return {"entropy": float(-(p * np.log(p)).sum() / np.log(gallery_size)), "coverage": len(c) / gallery_size,
            "top5_share": float(np.sort(c)[::-1][:5].sum() / c.sum())}


# ---------------------------------------------------------------- one (fold, seed)
class Scorers:
    """The 10-head ensemble's track / window scores (the D65 mean cosine)."""

    def __init__(self, cfg, store, fold, e, train_ids):
        from bgm.analysis.full_dataset import load_scorer
        from bgm.person.evaluate import MeanScorer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.sc = MeanScorer([load_scorer(ROOT / "semantic" / "siglip-both" / f"fold{fold}" / f"seed{s}", store, dev)
                              for s in range(10 * e, 10 * e + 10)])


def retrieval(store, test, scorer, music_of_gallery) -> dict:
    """Per-query rank, nDCG@10, top-10 files; for track (test-fold gallery) and segment (all windows)."""
    gal = music_of_gallery
    gtrack = np.array([store.music_track[m] for m in gal])
    tr = {"rank": [], "ndcg": [], "N": [], "h": [], "top10": [], "cl": []}
    for q in test:
        s = np.asarray(scorer.track(q, gal), float)
        hits = gtrack == store.track[q]
        tr["rank"].append(rank_of_first_hit(s, hits)), tr["ndcg"].append(ndcg_at(s, hits))
        tr["N"].append(len(gal)), tr["h"].append(int(hits.sum())), tr["top10"].append([gal[i] for i in np.argsort(-s)[:10]])
        tr["cl"].append(q)
    segs = test_segments(store, test)
    sg = {"rank": [], "ndcg": [], "N": [], "h": [], "cl": []}
    for qs in segs:
        windows, hit = segment_gallery(store, segs, qs, "all")
        s = np.asarray(scorer.window(qs[0], qs[2], qs[3], windows), float)
        hits = np.zeros(len(windows), bool)
        hits[hit] = True
        sg["rank"].append(rank_of_first_hit(s, hits)), sg["ndcg"].append(ndcg_at(s, hits))
        sg["N"].append(len(windows)), sg["h"].append(1), sg["cl"].append(f"{qs[0]}#{qs[1]}")
    return {"track": {k: np.array(v, dtype=object if k == "top10" else None) for k, v in tr.items()},
            "segment": {k: np.array(v) for k, v in sg.items()}}


def random_path(index: MusicIndex, segs, rng) -> list[Candidate]:
    out = []
    for a, b in segs:
        d = b - a
        it = index.items[rng.integers(len(index.items))]
        out.append(Candidate(it.music_id, float(rng.uniform(0, max(0.0, it.duration - d))), d, it.key, it.tempo, 0.0))
    return out


def original_path(store, vq, segs) -> list[Candidate]:
    out = []
    for a, b in segs:
        m, off = truth_for(vq, store, a, b)
        out.append(Candidate(m, off, b - a, -1, float(store.music(m)["tempo_bpm"]), 0.0))
    return out


def run_beats(runs: list[Run], beats_of) -> np.ndarray:
    out = []
    for r in runs:
        b = beats_of(r.music_id)
        b = b[(b >= r.offset) & (b < r.offset + r.end - r.start)]
        out.append(r.start + b - r.offset)
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


class Windows:
    """Span means over a music window: ImageBind, PANNs tags, CLAP (cached Prefix per file)."""

    def __init__(self, cfg, store):
        self.cfg, self.store, self._p = cfg, store, {}

    def _prefix(self, kind, m):
        if (kind, m) not in self._p:
            x = load_tags(self.cfg, m) if kind == "panns" else self.store.music(m)[kind]
            self._p[(kind, m)] = Prefix(np.asarray(x, np.float32))
        return self._p[(kind, m)]

    def mean(self, kind, m, off, d) -> np.ndarray:
        a, b = apply_span_rule(off, off + d, self.store.duration("music", m), self.store.min_span)
        return self._prefix(kind, m).means(np.array([a]), np.array([b]))[0]

    def at_seconds(self, kind, m, t: np.ndarray) -> np.ndarray:
        P = self._prefix(kind, m)
        return P.x[np.clip(np.floor(t).astype(int), 0, len(P.x) - 1)]


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p, q = p / p.sum() + eps, q / q.sum() + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def video_alignment(store, W: Windows, pid, vq, segs, path, orig, beats_of, durs_of, rng, sync=False) -> dict:
    runs = runs_from_path(path, segs)
    cuts = eligible_cuts(vq.cut_times, vq.duration)
    if sync:
        runs, _ = beat_sync_runs(runs, vq.cut_times, beats_of, durs_of, 2.0, TOL)
    beats = run_beats(runs, beats_of)
    out = beat_alignment(cuts, beats, vq.duration, rng)
    # within-piece cuts only (D35's convention): cuts where the music does not start a new piece. Beat sync puts
    # every piece start on a beat by construction, so the all-cuts LB rewards switching at every cut under sync.
    inner = within_piece_cuts(vq.cut_times, vq.duration, [r.start for r in runs[1:]])
    out.update({f"in_{k}": v for k, v in beat_alignment(inner, beats, vq.duration, rng).items()})
    out["n_inner"] = len(inner)
    w = np.array([b - a for a, b in segs])
    v_ib = [l2(store.pooled("video", pid, "imagebind_1hz", a, b)) for a, b in segs]
    out["IB"] = float(np.average([l2(W.mean("imagebind_1hz", c.music_id, c.offset, c.dur)) @ v for c, v in zip(path, v_ib)], weights=w))
    out["KL"] = float(np.average([kl(W.mean("panns", o.music_id, o.offset, o.dur), W.mean("panns", c.music_id, c.offset, c.dur))
                                  for c, o in zip(path, orig)], weights=w))
    out["n_cuts"] = len(cuts)
    return out


def soundtrack_seq(W: Windows, path, segs, duration) -> np.ndarray:
    """[T1, 512] unit CLAP vectors of what plays at each second centre."""
    rows = []
    for k in range(int(np.floor(duration))):
        c = k + 0.5
        i = next((j for j, (a, b) in enumerate(segs) if a <= c < b), len(segs) - 1)
        rows.append(W.at_seconds("sem_clap_1hz", path[i].music_id, np.array([path[i].offset + c - segs[i][0]]))[0])
    return l2(np.stack(rows))


def concat_transitions(cfg, store, W, vq, segs, path, orig_seq, rng) -> dict:
    joins = [t0 for _, t0, _ in vq.meta["parts"][1:]]
    sw = [segs[i + 1][0] for i in range(len(path) - 1) if path[i + 1].music_id != path[i].music_id]
    runs = runs_from_path(path, segs)
    jumps = len(runs) - 1 - len(sw)
    minutes = vq.duration / 60
    out = {"F1@1.5": switch_f1(sw, joins, 1.5), "F1@2.0": switch_f1(sw, joins, 2.0),
           "dev_T2P": float(np.median(nearest(np.array(joins), np.array(sw)))) if joins else np.nan,
           "dev_P2T": float(np.median(nearest(np.array(sw), np.array(joins)))) if sw else np.nan,
           "TBIoU": tbiou(joins, sw, vq.duration), "switches_per_min": len(sw) / minutes, "jumps_per_min": jumps / minutes}
    mix, seams = render_soundtrack(runs, vq.duration, lambda m: interim_music_path(cfg, m), cfg.render)
    j = [x for x in seam_loudness_jumps(mix, seams) if np.isfinite(x)]
    out["seam_LU"] = float(np.median(j)) if j else np.nan
    out["DTW"] = dtw_cost(soundtrack_seq(W, path, segs, vq.duration), orig_seq)
    return out


def random_boundaries(segs, n: int, rng) -> list[float]:
    inner = [a for a, _ in segs[1:]]
    return sorted(rng.choice(inner, min(n, len(inner)), replace=False).tolist()) if inner else []


def evaluate_run(fold: int, e: int, store_cache: dict) -> None:
    t0 = time.time()
    src = ROOT / "pipeline_ens10" / f"fold{fold}" / f"seed{e}"
    cfg = load_config(src / "config.yaml")
    assert list(cfg.pipeline.sem_seeds) == list(range(10 * e, 10 * e + 10)), cfg.pipeline.sem_seeds
    store = store_cache.setdefault("store", FeatureStore(cfg))
    W = store_cache.setdefault("W", Windows(cfg, store))
    f = fold_ids(cfg, fold)
    test = list(f["test"])
    fusion = json.loads((src / "fusion.json").read_text())
    st = json.loads((src / "stitch.json").read_text())
    pen, beam, tol_s = st["penalties"], st["beam"], st["tol_s"]
    per = {}
    # ---- 1. retrieval
    gal = list(dict.fromkeys(store.music_of[p] for p in test))
    sc = Scorers(cfg, store, fold, e, f["train"]).sc
    with torch.no_grad():
        R = retrieval(store, test, sc, gal)
    saved = np.load(ROOT / "semantic_ens" / "siglip-both-ens10" / f"fold{fold}" / f"seed{e}" / "perquery.npz")
    assert np.array_equal(R["track"]["rank"], saved["track_test_fold"]), "track ranks differ from the saved ensemble run"
    assert np.array_equal(R["segment"]["rank"], saved["segment_all"]), "segment ranks differ from the saved ensemble run"
    for side in ("track", "segment"):
        for k in ("rank", "ndcg", "N", "h", "cl"):
            per[f"ret/{side}/{k}"] = R[side][k]
    per["ret/track/top1"] = np.array([t[0] for t in R["track"]["top10"]])
    per["ret/track/top10"] = np.array([list(t) for t in R["track"]["top10"]])
    del sc
    torch.cuda.empty_cache()
    if e == 0:                                          # ImageBind zero-shot reference (deterministic: once per fold)
        from types import SimpleNamespace
        from bgm.eval.zeroshot import ZeroShot
        zs = ZeroShot(store, cfg, f["train"], tag=f"fold{fold}")
        Z = retrieval(store, test, SimpleNamespace(track=zs.imagebind_track, window=zs.imagebind_window), gal)
        for side in ("track", "segment"):
            for k in ("rank", "ndcg"):
                per[f"ret_ib/{side}/{k}"] = Z[side][k]
        per["ret_ib/track/top1"] = np.array([t[0] for t in Z["track"]["top10"]])
    # ---- pipeline
    models = FoldModels.load(cfg, store, fold, e, cfg.pipeline.sem_variant, None, train_ids=f["train"])
    index = MusicIndex(store, models, gal)
    beats_of = lambda m: index.items[index.pos[m]].beats                    # noqa: E731
    durs_of = lambda m: index.items[index.pos[m]].duration                  # noqa: E731
    r = cfg.retrieval
    # ---- 2. alignment + diversity, natural test videos
    A = {s: [] for s in (*SYSTEMS, "viterbi_sync", "random_sync")}
    offs = {s: [] for s in SYSTEMS}
    picks = {s: [] for s in SYSTEMS}
    saved_st = np.load(src / "stitch_perquery.npz")
    for n, (p, vq) in enumerate(prepared_queries(store, test, models).items()):
        res = run_video(index, vq, fusion["w_coarse"], r.coarse_top_k, r.window_hop_s)
        lat = build_lattice(index, vq, res, fusion["w_fine"], r.fine_top_j)
        segs = [(x.a, x.b) for x in res]
        rng = np.random.default_rng([e, fold, zlib.crc32(p.encode())])
        paths = {"viterbi": chosen_candidates(index, lat, decode_lattice(lat, pen, beam, tol_s)),
                 "greedy": chosen_candidates(index, lat, greedy_lattice(lat)),
                 "random": random_path(index, segs, rng), "original": original_path(store, vq, segs)}
        acc = path_accuracy(store, vq, res, paths["viterbi"])["track"]
        assert abs(acc - saved_st["natural/viterbi/track"][n]) < 1e-9 and saved_st["natural/pair"][n] == p, (p, acc)
        for s, path in paths.items():
            A[s].append(video_alignment(store, W, p, vq, segs, path, paths["original"], beats_of, durs_of, rng))
            picks[s] += [c.music_id for c in path]
            offs[s] += [c.offset for c in path]
        for s in ("viterbi", "random"):
            A[f"{s}_sync"].append(video_alignment(store, W, p, vq, segs, paths[s], paths["original"], beats_of, durs_of, rng, sync=True))
    for s, rows in A.items():
        for k in rows[0]:
            per[f"align/{s}/{k}"] = np.array([x[k] for x in rows], float)
    per["align/pair"] = np.array(test)
    for s, pk in picks.items():
        per[f"div/seg/{s}"] = np.array(pk)
        per[f"path/{s}/offset"] = np.array(offs[s], float)
    # ---- 3. transitions, test concatenations
    T = {s: [] for s in (*SYSTEMS, "random_bounds")}
    for i, parts in enumerate(read_concat_sets(cfg)[str(fold)]["test"]):
        vq = concat_query(store, parts)
        vq.prepare(models, store.min_span)
        res = run_video(index, vq, fusion["w_coarse"], r.coarse_top_k, r.window_hop_s)
        lat = build_lattice(index, vq, res, fusion["w_fine"], r.fine_top_j)
        segs = [(x.a, x.b) for x in res]
        rng = np.random.default_rng([e, fold, 1000 + i])
        paths = {"viterbi": chosen_candidates(index, lat, decode_lattice(lat, pen, beam, tol_s)),
                 "greedy": chosen_candidates(index, lat, greedy_lattice(lat)),
                 "random": random_path(index, segs, rng), "original": original_path(store, vq, segs)}
        oseq = soundtrack_seq(W, paths["original"], segs, vq.duration)
        for s, path in paths.items():
            T[s].append(concat_transitions(cfg, store, W, vq, segs, path, oseq, rng))
        # boundaries only: as many switches as Viterbi made, at random segment boundaries
        n_sw = sum(paths["viterbi"][k + 1].music_id != paths["viterbi"][k].music_id for k in range(len(segs) - 1))
        rb = random_boundaries(segs, n_sw, rng)
        joins = [t0 for _, t0, _ in vq.meta["parts"][1:]]
        T["random_bounds"].append({"F1@1.5": switch_f1(rb, joins, 1.5), "F1@2.0": switch_f1(rb, joins, 2.0),
                                   "dev_T2P": float(np.median(nearest(np.array(joins), np.array(rb)))),
                                   "dev_P2T": float(np.median(nearest(np.array(rb), np.array(joins)))) if rb else np.nan,
                                   "TBIoU": tbiou(joins, rb, vq.duration)})
    for s, rows in T.items():
        for k in rows[0]:
            per[f"trans/{s}/{k}"] = np.array([x[k] for x in rows], float)
    d = run_dir(cfg, EXP, fold, e)
    np.savez(d / "perquery.npz", **per)
    summary = {k: float(np.nanmean(v)) for k, v in per.items() if v.dtype.kind == "f"}
    write_run(d, cfg, {"source_run": str(src), "penalties": pen, "summary": summary}, e)
    log.info("eval suite fold %d seed %d done in %.0f s: track R@10 %.3f, LB viterbi %.3f, F1@2 %.3f", fold, e, time.time() - t0,
             float((R["track"]["rank"] <= 10).mean()), np.nanmean(per["align/viterbi/LB"]), np.nanmean(per["trans/viterbi/F1@2.0"]))


def run_all() -> None:
    cache: dict = {}
    for k in FOLDS:
        for e in SEEDS:
            if (ROOT / EXP / f"fold{k}" / f"seed{e}" / "perquery.npz").exists():
                continue
            evaluate_run(k, e, cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what == "run":
        run_all()
    else:
        from bgm.analysis.eval_suite_report import write_report
        write_report()
