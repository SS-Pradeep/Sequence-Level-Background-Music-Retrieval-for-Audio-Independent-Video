"""Where do the stitched soundtrack's errors come from? Cleaned title split (PROGRESS.md D138, Step 1). Analysis only.

The shipped system (runs/cleaned/pipeline_ens10: 10-head ensembles, K = 10, C = J = 20, 0.5 s windows, fixed Viterbi
penalties tuned on val, stitch.json), re-run on the frozen test concatenations (concat_sets_title.json, 50 per fold) of
each fold x run seed. Each run's own config.yaml is loaded, so the lattices are tune-stitch's.

a. Candidate recall: share of segments whose true window (own music file, start within 0.5 s, SPEC §5.12) is among the
   segment's top C windows; also among all lattice candidates (top C + continuations), and the true song (track group).
b. Per sequence: total Viterbi cost (emission -z plus move costs) of the ground-truth path vs the decoded path. The
   ground-truth path's windows are scored with the segment's own z-statistics, like any continuation.
c. Every segment whose decoded song is wrong (the unit of song accuracy) gets one cause:
     not in candidates   no window of the true song is in that segment's candidate list;
     lower fit score     one is, but the best of them scores below the decoded window (emission alone picks wrong);
     transition cost     one scores at least as high, and the move costs made Viterbi pick another.
   Shares by count and by duration.
d. Oracle: the true window is forced into every segment's top list (scored with its z-statistics, its continuation
   follows as usual); Viterbi is re-run with the same penalties. Switch F1, song (part track) accuracy, window accuracy
   and switches / min with vs without.

Run: python -m bgm.analysis.error_sources [run|report|all]  -> runs/cleaned/d138_errors/, results/d138_errors/*.csv, summary.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SRC = Path("runs/cleaned/pipeline_ens10")
RUNS = Path("runs/cleaned/d138_errors")
OUT = Path("results/d138_errors")
FOLDS, SEEDS = range(5), range(3)
WIN_TOL_S = 0.5
CAUSES = ("not in candidates", "lower fit score", "transition cost")


# ---------------------------------------------------------------- pure
def path_cost(layers, picks, pen: dict, tol_s: float) -> float:
    """Emission plus move costs of one path through a list of Layers (picks: one index per layer)."""
    from bgm.retrieval.stitch_eval import transition_matrix
    c = float(sum(L.cost[j] for L, j in zip(layers, picks)))
    for P, C, a, b in zip(layers, layers[1:], picks, picks[1:]):
        c += float(transition_matrix(P.subset(np.array([a])), C.subset(np.array([b])), pen, tol_s)[0, 0])
    return c


def classify(cand_tracks: np.ndarray, cand_cost: np.ndarray, true_track, chosen_track, chosen_cost: float) -> str | None:
    """Cause of one segment's song error (None if the decoded song is right). cand_*: the segment's whole candidate list."""
    if chosen_track == true_track:
        return None
    ok = cand_tracks == true_track
    if not ok.any():
        return CAUSES[0]
    return CAUSES[1] if cand_cost[ok].min() > chosen_cost + 1e-12 else CAUSES[2]


def build_lattice_forced(index, vq, results, w: dict, J: int, forced: list[tuple[int, float]]):
    """stitch_eval.build_lattice with one extra window per segment, forced into the top list: forced[i] = (file index,
    start s). It is scored with the segment's z-statistics; continuations are generated from every candidate as usual."""
    from bgm.retrieval.pipeline import fuse
    from bgm.retrieval.stitch_eval import Layer
    layers = []
    for i, r in enumerate(results):
        S = fuse(r.raw, r.stats, w)
        d = r.b - r.a
        top = np.argsort(-S)[:J]
        mid, off, cost = list(r.fi[top]), list(r.st[top]), list(-S[top])
        q = vq.segment(r.a, r.b)
        fm, fo = forced[i]
        if not any(m == fm and abs(o - fo) <= 0.02 for m, o in zip(mid, off)):
            raw = index.window_scores(q, np.array([fm]), np.array([fo]), d)
            mid.append(int(fm)), off.append(float(fo)), cost.append(float(-fuse(raw, r.stats, w)[0]))
        n_top = len(mid)
        cont_of = None
        if i:
            P = layers[-1]
            seen = {(int(m), round(o / 0.04)): j for j, (m, o) in enumerate(zip(mid, off))}
            cont_of = np.full(len(P.mid), -1)
            new_m, new_o = [], []
            for j, (m, o, pd) in enumerate(zip(P.mid, P.off, P.dur)):
                o2 = o + pd
                if o2 + d > index.items[m].duration:
                    continue
                key = (int(m), round(o2 / 0.04))
                if key not in seen:
                    seen[key] = len(mid) + len(new_m)
                    new_m.append(int(m)), new_o.append(o2)
                cont_of[j] = seen[key]
            if new_m:
                raw = index.window_scores(q, np.array(new_m), np.array(new_o), d)
                mid += new_m
                off += new_o
                cost += list(-fuse(raw, r.stats, w))
        mid = np.array(mid, int)
        layers.append(Layer(mid, np.array(off, float), np.full(len(mid), d), np.array([index.items[m].key for m in mid]),
                            np.array([index.items[m].tempo for m in mid]), np.array(cost, float), n_top, cont_of))
    return layers


# ---------------------------------------------------------------- run
def run_one(fold: int, seed: int, device: str = "cuda") -> None:
    from omegaconf import OmegaConf
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.retrieval.pipeline import fuse, truth_for
    from bgm.retrieval.stitch_eval import (Layer, chosen_candidates, decode_lattice, part_track_accuracy, read_concat_sets,
                                           switch_f1, switch_times)
    from bgm.retrieval.tune_stitch import prepare_concats
    from bgm.runs import write_run
    out_dir = RUNS / f"fold{fold}" / f"seed{seed}"
    if (out_dir / "segments.json").exists():
        return
    t0 = time.time()
    d = SRC / f"fold{fold}" / f"seed{seed}"
    cfg = OmegaConf.load(d / "config.yaml")
    fusion, stitch = json.loads((d / "fusion.json").read_text()), json.loads((d / "stitch.json").read_text())
    pen, beam, tol = stitch["penalties"], stitch["beam"], stitch["tol_s"]
    store = FeatureStore(cfg)
    f = fold_ids(cfg, fold)
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, device=device,
                             train_ids=f["train"])
    index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    test = prepare_concats(store, index, models, read_concat_sets(cfg)[str(fold)]["test"], cfg, fusion)
    w, J = fusion["w_fine"], cfg.retrieval.fine_top_j
    trk = np.array([store.music_track[m] for m in index.ids])
    seqs, segs = [], []
    for si, (vq, res, lat) in enumerate(test):
        truth = [truth_for(vq, store, r.a, r.b) for r in res]
        joins = [t for _, t, _ in vq.meta["parts"][1:]]
        picks = decode_lattice(lat, pen, beam, tol)
        ch = chosen_candidates(index, lat, picks)
        # ground-truth path as one-candidate layers
        gt = []
        for r, (m, off) in zip(res, truth):
            i = index.pos[m]
            raw = index.window_scores(vq.segment(r.a, r.b), np.array([i]), np.array([off]), r.b - r.a)
            gt.append(Layer(np.array([i]), np.array([off]), np.array([r.b - r.a]), np.array([index.items[i].key]),
                            np.array([index.items[i].tempo]), np.array([-fuse(raw, r.stats, w)[0]]), 1))
        cost_gt, cost_ch = path_cost(gt, [0] * len(gt), pen, tol), path_cost(lat, picks, pen, tol)
        # oracle
        forced = [(index.pos[m], off) for m, off in truth]
        olat = build_lattice_forced(index, vq, res, w, J, forced)
        opicks = decode_lattice(olat, pen, beam, tol)
        och = chosen_candidates(index, olat, opicks)
        row = {"fold": fold, "seed": seed, "seq": si, "duration": vq.duration, "n_segments": len(res), "n_joins": len(joins),
               "cost_gt": cost_gt, "cost_chosen": cost_ch,
               "cost_oracle": path_cost(olat, opicks, pen, tol)}
        for name, c, L, pk in (("base", ch, lat, picks), ("oracle", och, olat, opicks)):
            sw, _ = switch_times(c, res)
            win = [cc.music_id == m and abs(cc.offset - off) <= WIN_TOL_S for cc, (m, off) in zip(c, truth)]
            row |= {f"{name}/switch_f1": switch_f1(sw, joins), f"{name}/song_acc": part_track_accuracy(store, vq, res, c),
                    f"{name}/window_acc": float(np.average(win, weights=[r.b - r.a for r in res])),
                    f"{name}/switches_per_min": len(sw) / (vq.duration / 60)}
        seqs.append(row)
        for i, (r, (m, off), L, j) in enumerate(zip(res, truth, lat, picks)):
            p = index.pos[m]
            win = (L.mid == p) & (np.abs(L.off - off) <= WIN_TOL_S)
            tt = store.music_track[m]
            segs.append({"fold": fold, "seed": seed, "seq": si, "seg": i, "dur": r.b - r.a,
                         "win_in_topC": bool(win[:L.n_top].any()), "win_in_lattice": bool(win.any()),
                         "song_in_topC": bool((trk[L.mid[:L.n_top]] == tt).any()), "song_in_lattice": bool((trk[L.mid] == tt).any()),
                         "song_ok": bool(trk[L.mid[j]] == tt), "window_ok": bool(win[j]),
                         "cause": classify(trk[L.mid], L.cost, tt, trk[L.mid[j]], float(L.cost[j])),
                         "gt_cost": float(gt[i].cost[0]), "chosen_cost": float(L.cost[j]),
                         "oracle_song_ok": bool(store.music_track[och[i].music_id] == tt)})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequences.json").write_text(json.dumps(seqs))
    (out_dir / "segments.json").write_text(json.dumps(segs))
    write_run(out_dir, cfg, {"n_sequences": len(seqs), "penalties": pen, "seconds": time.time() - t0}, seed)
    log.info("fold %d seed %d: %d sequences, %d segments [%.0f s]", fold, seed, len(seqs), len(segs), time.time() - t0)


# ---------------------------------------------------------------- report
def report() -> Path:
    import pandas as pd
    from bgm.eval.metrics import cluster_bootstrap
    S = pd.DataFrame([r for k in FOLDS for s in SEEDS for r in json.loads((RUNS / f"fold{k}" / f"seed{s}" / "sequences.json").read_text())])
    G = pd.DataFrame([r for k in FOLDS for s in SEEDS for r in json.loads((RUNS / f"fold{k}" / f"seed{s}" / "segments.json").read_text())])
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "sequences.csv", index=False)
    G.to_csv(OUT / "segments.csv", index=False)

    def fold_ms(df, col, weights=None):
        """mean ± std over the 5 fold means (runs averaged within a fold)."""
        if weights is None:
            v = df.groupby("fold")[col].mean()
        else:
            v = df.groupby("fold").apply(lambda x: np.average(x[col], weights=x[weights]))
        return float(v.mean()), float(v.std(ddof=1))

    rows = []
    for col, lab in (("win_in_topC", "true window in the top C = 20"), ("win_in_lattice", "true window anywhere in the lattice"),
                     ("song_in_topC", "true song in the top C = 20"), ("song_in_lattice", "true song anywhere in the lattice"),
                     ("song_ok", "decoded song right (Viterbi)"), ("window_ok", "decoded window right (Viterbi)")):
        G[col] = G[col].astype(float)
        m, s = fold_ms(G, col)
        dm, ds = fold_ms(G, col, "dur")
        rows.append({"section": "a. candidate recall (segments)", "metric": lab, "mean": m, "std": s, "dur_weighted_mean": dm, "dur_weighted_std": ds})
    G["gt_z"], G["chosen_z"] = -G.gt_cost, -G.chosen_cost
    G["gt_z_pct"] = G.gt_z <= G.chosen_z
    for col, lab in (("gt_z", "fit z of the true window (segment-standardised)"), ("chosen_z", "fit z of the decoded window"),
                     ("gt_z_pct", "share of segments where the true window fits no better than the decoded one")):
        m, s = fold_ms(G, col)
        rows.append({"section": "b. path cost (sequences)", "metric": lab, "mean": m, "std": s})
    S["gt_minus_chosen"] = S.cost_gt - S.cost_chosen
    S["gt_cheaper"] = (S.gt_minus_chosen < -1e-9).astype(float)
    for col, lab in (("cost_gt", "ground-truth path cost"), ("cost_chosen", "decoded path cost"),
                     ("gt_minus_chosen", "ground truth − decoded"), ("gt_cheaper", "share of sequences where the ground truth is cheaper (search / lattice error)")):
        m, s = fold_ms(S, col)
        rows.append({"section": "b. path cost (sequences)", "metric": lab, "mean": m, "std": s})
    err = G[G.song_ok == 0]
    for c in CAUSES:
        G[c] = (G.cause == c).astype(float)
        e = G[G.song_ok == 0]
        m, s = fold_ms(e, c)
        dm, ds = fold_ms(e, c, "dur")
        rows.append({"section": "c. cause of song errors (share of wrong segments)", "metric": c, "mean": m, "std": s,
                     "dur_weighted_mean": dm, "dur_weighted_std": ds, "n": int((err.cause == c).sum())})
    clusters = (S.fold.astype(str) + "_" + S.seq.astype(str)).values
    for met, lab in (("switch_f1", "switch F1"), ("song_acc", "song accuracy"), ("window_acc", "window accuracy"),
                     ("switches_per_min", "switches / min")):
        b, o = fold_ms(S, f"base/{met}"), fold_ms(S, f"oracle/{met}")
        per = S.groupby(["fold", "seq"])[[f"base/{met}", f"oracle/{met}"]].mean()
        ci = cluster_bootstrap((per[f"oracle/{met}"] - per[f"base/{met}"]).values, np.arange(len(per)))
        rows.append({"section": "d. oracle: true window forced into every candidate list", "metric": lab, "base_mean": b[0],
                     "base_std": b[1], "oracle_mean": o[0], "oracle_std": o[1], "delta": ci["mean"], "delta_lo": ci["lo"], "delta_hi": ci["hi"]})
    gt_sw = float((S.n_joins / (S.duration / 60)).mean())
    rows.append({"section": "d. oracle: true window forced into every candidate list", "metric": "ground-truth switches / min", "mean": gt_sw})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "summary.csv", index=False)
    n_seq, n_seg = S.groupby(["fold", "seq"]).ngroups, len(G) // len(SEEDS)
    L = ["# Error sources of the stitched soundtrack (D138, Step 1)", "",
         f"Shipped system (10-head ensembles, K = 10, C = 20, 0.5 s windows, val-tuned fixed penalties) on the {n_seq} frozen "
         f"test concatenations of the cleaned title split ({n_seg} segments), 5 folds × 3 run seeds. ± = std over the 5 fold "
         "means. Analysis only; nothing is tuned. Development numbers.", "",
         "## a. Candidate recall", "", "| | segments | duration-weighted |", "|---|---|---|"]
    for _, r in R[R.section.str.startswith("a.")].iterrows():
        L.append(f"| {r.metric} | {r['mean']:.3f} ± {r['std']:.3f} | {r.dur_weighted_mean:.3f} ± {r.dur_weighted_std:.3f} |")
    L += ["", "## b. Viterbi cost: ground-truth path vs decoded path (lower = preferred)", "", "| | value |", "|---|---|"]
    for _, r in R[R.section.str.startswith("b.")].iterrows():
        L.append(f"| {r.metric} | {r['mean']:.3f} ± {r['std']:.3f} |")
    L += ["", "The decoder minimises this cost, so a ground truth more expensive than the decoded path means the costs, not the "
          "search, prefer the wrong music.", "",
          f"## c. Cause of each song error ({len(err)} wrong segments over all runs)", "",
          "| cause | share of wrong segments | duration-weighted | count |", "|---|---|---|---|"]
    for _, r in R[R.section.str.startswith("c.")].iterrows():
        L.append(f"| {r.metric} | {r['mean']:.3f} ± {r['std']:.3f} | {r.dur_weighted_mean:.3f} ± {r.dur_weighted_std:.3f} | {r.n:.0f} |")
    L += ["", "## d. Oracle candidates (same costs and penalties)", "",
          "| metric | shipped | true window forced in | Δ [95% CI, sequences] |", "|---|---|---|---|"]
    for _, r in R[R.section.str.startswith("d.") & R.base_mean.notna()].iterrows():
        L.append(f"| {r.metric} | {r.base_mean:.3f} ± {r.base_std:.3f} | {r.oracle_mean:.3f} ± {r.oracle_std:.3f} | "
                 f"{r.delta:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}] |")
    L += ["", f"Ground-truth switches / min: {gt_sw:.2f}.", ""]
    (OUT / "summary.md").write_text("\n".join(L))
    log.info("wrote %s", OUT / "summary.md")
    return OUT / "summary.md"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        for k in FOLDS:
            for s in SEEDS:
                run_one(k, s)
    if cmd in ("report", "all"):
        print(report())
