"""Per-window key and tempo in the Viterbi switch cost (D137).

python -m bgm.analysis.window_switch_cost chroma   -> data/features/music_chroma/<music_id>.npz  (CPU)
python -m bgm.analysis.window_switch_cost run      -> runs/titlesplit/window_switch_cost/fold<k>/seed<s>/  (GPU)
python -m bgm.analysis.window_switch_cost report   -> results/titlesplit/window_switch_cost.md

The shipped switch cost compares one key and one tempo per music FILE. Here a switch from candidate c = (k, tau, d) to
c' = (k', tau') compares the last w seconds of c's window, [tau + d - w, tau + d], with the first w seconds of c''s,
[tau', tau' + w]: what the listener hears on either side of the seam. Key = Krumhansl-Kessler on the mean per-second
chroma of the window (non-silent seconds); tempo = 60 / median inter-beat interval of the cached beats in the window.
Where a window gives no estimate (silent, < 4 beats) the file's value is used.

System: the final title-split system (runs/titlesplit/pipeline_ens10, 5 folds x 3 ensembles), the same val / test
concatenations and lattices as `bgm tune-stitch`. Every mode is tuned on val with the tune-stitch objective (part
track accuracy + switch F1) over the shipped grid; a second grid extends mu to {0, 0.5, 1, 2, 4}. The track mode
on the shipped grid must reproduce each run's stitch.json and stitch_perquery.npz (asserted).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
ROOT = Path("runs/titlesplit/pipeline_ens10")
OUT_EXP = "window_switch_cost"
REPORT = Path("results/titlesplit/window_switch_cost.md")
CHROMA = {"v": 1, "sr": 22050, "hop": 512}
WINDOWS = (3.0, 5.0, 8.0)
HEARD_W = 5.0                                      # window of the seam distances reported for every mode
MU_EXT = [0.0, 0.5, 1.0, 2.0, 4.0]
KEYS = ("switch_f1", "part_track_acc", "switches_per_min", "seam_key", "seam_tempo")


def chroma_path(cfg, music_id: str) -> Path:
    return Path(cfg.paths.features) / "music_chroma" / f"{music_id}.npz"


def chroma_hash() -> str:
    return hashlib.sha1(json.dumps(CHROMA, sort_keys=True).encode()).hexdigest()[:12]


def _chroma_job(args):
    src, dst = args
    from bgm.features.extract import load_music, save_features
    from bgm.features.keys import chroma_1hz
    y = load_music(Path(src), CHROMA["sr"])
    c, rms = chroma_1hz(y, CHROMA["sr"])
    save_features(Path(dst), {"chroma_1hz": c, "rms_1hz": rms, "meta": {"hash": chroma_hash(), **CHROMA}})
    return dst


def chroma(workers: int = 6) -> None:
    from bgm.config import load_config
    from bgm.data.prep import interim_music_path
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.features.extract import load_features
    cfg = load_config(ROOT / "fold0" / "seed0" / "config.yaml")
    store = FeatureStore(cfg)
    ids = list(dict.fromkeys(store.music_of[p] for k in range(cfg.splits.n_folds) for s in ("val", "test")
                             for p in fold_ids(cfg, k)[s]))
    todo = [(str(interim_music_path(cfg, m)), str(chroma_path(cfg, m))) for m in ids
            if load_features(chroma_path(cfg, m)).get("meta", {}).get("hash") != chroma_hash()]
    log.info("chroma: %d music files, %d to compute", len(ids), len(todo))
    with ProcessPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(_chroma_job, todo, chunksize=4)):
            if (n + 1) % 200 == 0:
                log.info("  %d / %d", n + 1, len(todo))


# ---------------------------------------------------------------- window values on the lattice
class WindowValues:
    """Key / tempo of [t0, t1] s of one music file (index code), the file's value where the window gives none."""

    def __init__(self, cfg, index):
        from bgm.features.extract import load_features
        self.index, self.ch, self.memo = index, {}, {}
        for m in index.ids:
            d = load_features(chroma_path(cfg, m))
            assert d.get("meta", {}).get("hash") == chroma_hash(), f"{m}: run `window_switch_cost chroma` first"
            self.ch[m] = (d["chroma_1hz"], d["rms_1hz"])

    def __call__(self, code: int, t0: float, t1: float) -> tuple[int, float, bool, bool]:
        """(key, tempo, key from the window?, tempo from the window?)"""
        from bgm.features.keys import window_key, window_tempo
        it = self.index.items[code]
        span = (code, int(np.floor(t0)), int(np.ceil(t1)))      # window_key only sees whole seconds
        if span not in self.memo:
            self.memo[span] = window_key(*self.ch[it.music_id], t0, t1)
        k, t = self.memo[span], window_tempo(it.beats, t0, t1)
        return (k if k >= 0 else it.key), (t if t > 0 else it.tempo), k >= 0, t > 0


def window_arrays(layers, wv: WindowValues, w: float) -> tuple[list[tuple], dict]:
    """Per layer (key_in, tempo_in, key_out, tempo_out) for window length w, and coverage counts."""
    out, n = [], {"ends": 0, "key_win": 0, "tempo_win": 0, "key_differs": 0, "tempo_differs": 0}
    for L in layers:
        vin = [wv(int(m), o, o + w) for m, o in zip(L.mid, L.off)]
        vout = [wv(int(m), max(o + d - w, 0.0), o + d) for m, o, d in zip(L.mid, L.off, L.dur)]
        a = (np.array([v[0] for v in vin]), np.array([v[1] for v in vin], float),
             np.array([v[0] for v in vout]), np.array([v[1] for v in vout], float))
        out.append(a)
        n["ends"] += 2 * len(L.mid)
        n["key_win"] += sum(v[2] for v in vin + vout)
        n["tempo_win"] += sum(v[3] for v in vin + vout)
        n["key_differs"] += int(np.sum(a[0] != L.key) + np.sum(a[2] != L.key))
        n["tempo_differs"] += int(np.sum(~np.isclose(a[1], L.tempo, rtol=0.05)) + np.sum(~np.isclose(a[3], L.tempo, rtol=0.05)))
    return out, n


def set_window(layers, arrays: list[tuple] | None) -> None:
    """Put window values on the layers (None: back to the per-file key / tempo)."""
    for i, L in enumerate(layers):
        L.key_in, L.tempo_in, L.key_out, L.tempo_out = (None,) * 4 if arrays is None else arrays[i]


def seam_distances(layers, picks, heard: list[tuple]) -> tuple[float, float]:
    """Mean fifths / tempo distance over the path's switches with the HEARD_W window values (end of the outgoing
    window vs start of the incoming one); nan without switches."""
    from bgm.retrieval.stitch_eval import fifths_matrix, tempo_matrix
    dk, dt = [], []
    for i in range(1, len(layers)):
        j0, j1 = picks[i - 1], picks[i]
        if layers[i].mid[j1] == layers[i - 1].mid[j0]:
            continue
        dk.append(float(fifths_matrix(heard[i - 1][2][[j0]], heard[i][0][[j1]])[0, 0]))
        dt.append(float(tempo_matrix(heard[i - 1][3][[j0]], heard[i][1][[j1]])[0, 0]))
    return (float(np.mean(dk)), float(np.mean(dt))) if dk else (np.nan, np.nan)


def path_metrics(store, index, vq, res, lat, picks, heard=None) -> dict:
    from bgm.retrieval.stitch_eval import chosen_candidates, part_track_accuracy, switch_f1, switch_times
    ch = chosen_candidates(index, lat, picks)
    sw, _ = switch_times(ch, res)
    out = {"switch_f1": switch_f1(sw, [t for _, t, _ in vq.meta["parts"][1:]]), "part_track_acc": part_track_accuracy(store, vq, res, ch),
           "switches_per_min": len(sw) / (vq.duration / 60)}
    if heard is not None:
        out["seam_key"], out["seam_tempo"] = seam_distances(lat, picks, heard)
    return out


def tune(store, index, concats, grid: dict, beam, tol) -> tuple[dict, float]:
    """The tune-stitch val search: same grid order, objective and tie rule (the first best wins)."""
    from bgm.retrieval.stitch_eval import decode_lattice
    best, best_pen = None, None
    for ls, lj, mu in itertools.product(grid["lam_switch"], grid["lam_jump"], grid["mu"]):
        pen = {"lam_switch": float(ls), "lam_jump": float(lj), "mu": float(mu)}
        s = float(np.mean([sum(path_metrics(store, index, vq, res, lat, decode_lattice(lat, pen, beam, tol))[k] for k in ("part_track_acc", "switch_f1"))
                           for vq, res, lat in concats]))
        if best is None or s > best:
            best, best_pen = s, pen
    return best_pen, best


def run_one(cfg, store, fold: int, seed: int) -> dict:
    import torch
    from bgm.eval.store import fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    from bgm.retrieval.stitch_eval import decode_lattice, greedy_lattice, read_concat_sets
    from bgm.retrieval.tune_stitch import prepare_concats
    from bgm.runs import run_dir, write_run

    d = ROOT / f"fold{fold}" / f"seed{seed}"
    fusion = json.loads((d / "fusion.json").read_text())
    stitch = json.loads((d / "stitch.json").read_text())
    beam, tol = stitch["beam"], stitch["tol_s"]
    grid = {k: list(v) for k, v in cfg.stitch.grid.items()}
    grids = {"": grid, "+mu": {**grid, "mu": MU_EXT}}
    f = fold_ids(cfg, fold)
    sets = read_concat_sets(cfg)[str(fold)]
    models = FoldModels.load(cfg, store, fold, seed, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
    data = {}
    for split in ("val", "test"):
        index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f[split])))
        concats = prepare_concats(store, index, models, sets[split], cfg, fusion)
        wv = WindowValues(cfg, index)
        win = {w: [window_arrays(lat, wv, w) for _, _, lat in concats] for w in WINDOWS}
        data[split] = (index, concats, win)
    out = {"fold": fold, "seed": seed, "modes": {}, "coverage": {}}
    for w in WINDOWS:
        c = {k: sum(x[1][k] for x in data["test"][2][w]) for k in data["test"][2][w][0][1]}
        out["coverage"][f"w{w:g}"] = {k: v / c["ends"] for k, v in c.items() if k != "ends"} | {"ends": c["ends"]}
    t_index, t_concats, t_win = data["test"]
    heard = [x[0] for x in t_win[HEARD_W]]
    modes = [("track", None)] + [(f"w{w:g}", w) for w in WINDOWS]
    for (name, w), (gname, g) in itertools.product(modes, grids.items()):
        v_index, v_concats, v_win = data["val"]
        for q, (_, _, lat) in enumerate(v_concats):
            set_window(lat, None if w is None else v_win[w][q][0])
        pen, val = tune(store, v_index, v_concats, g, beam, tol)
        per = []
        for q, (vq, res, lat) in enumerate(t_concats):
            set_window(lat, None if w is None else t_win[w][q][0])
            per.append(path_metrics(store, t_index, vq, res, lat, decode_lattice(lat, pen, beam, tol), heard[q]))
        out["modes"][name + gname] = {"penalties": pen, "val_objective": val, "per_seq": {k: [p[k] for p in per] for k in KEYS}}
        log.info("fold %d seed %d %-8s pen %s val %.4f | test F1 %.3f part %.3f sw/min %.2f seam key %.3f tempo %.3f", fold, seed,
                 name + gname, pen, val, *[np.nanmean([p[k] for p in per]) for k in KEYS])
    for q, (_, _, lat) in enumerate(t_concats):
        set_window(lat, None)
    out["modes"]["greedy"] = {"per_seq": {k: [x[k] for x in [path_metrics(store, t_index, vq, res, lat, greedy_lattice(lat), heard[q])
                                                         for q, (vq, res, lat) in enumerate(t_concats)]] for k in KEYS}}
    # the shipped penalties (stitch.json) on today's lattices: they must give the saved val objective and the saved
    # per-sequence test numbers (a decoder near-tie may flip on GPU float noise, D130: allow 2 of 50). The re-tuned track
    # mode can still pick other penalties: at mu 0 every switch costs the same, so near-ties are common (fold 0 seed 0:
    # mu 0 now scores 0.8519 on val vs the saved best 0.8503 at mu 0.5, which reproduces exactly)
    v_index, v_concats, _ = data["val"]
    for _, _, lat in v_concats:
        set_window(lat, None)
    sp = stitch["penalties"]
    v_saved = float(np.mean([sum(path_metrics(store, v_index, vq, res, lat, decode_lattice(lat, sp, beam, tol))[k] for k in ("part_track_acc", "switch_f1"))
                             for vq, res, lat in v_concats]))
    assert abs(v_saved - stitch["val_objective"]) < 0.01, (fold, seed, v_saved, stitch["val_objective"])
    per = [path_metrics(store, t_index, vq, res, lat, decode_lattice(lat, sp, beam, tol), heard[q]) for q, (vq, res, lat) in enumerate(t_concats)]
    out["modes"]["track (shipped penalties)"] = {"penalties": sp, "val_objective": v_saved, "per_seq": {k: [p[k] for p in per] for k in KEYS}}
    pq = np.load(d / "stitch_perquery.npz")
    diff = sorted({q for k in ("switch_f1", "part_track_acc", "switches_per_min")
                   for q in np.flatnonzero(~np.isclose(out["modes"]["track (shipped penalties)"]["per_seq"][k], pq[f"concat/viterbi/{k}"]))})
    assert len(diff) <= 2, (fold, seed, diff)
    out["n_seq_differing_from_saved"] = len(diff)
    out["retuned_track_matches_saved"] = out["modes"]["track"]["penalties"] == sp
    rd = run_dir(cfg, OUT_EXP, fold, seed)
    (rd / "result.json").write_text(json.dumps(out, default=float))
    write_run(rd, cfg, {k: {"penalties": m.get("penalties"), "val_objective": m.get("val_objective"),
                            "test": {x: float(np.nanmean(m["per_seq"][x])) for x in KEYS}} for k, m in out["modes"].items()}, seed)
    del models
    torch.cuda.empty_cache()
    return out


def run() -> None:
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    store = None
    for fold in range(5):
        for seed in range(3):
            cfg = load_config(ROOT / f"fold{fold}" / f"seed{seed}" / "config.yaml")
            if (Path(cfg.paths.runs) / OUT_EXP / f"fold{fold}" / f"seed{seed}" / "result.json").exists():
                continue
            store = store or FeatureStore(cfg)
            run_one(cfg, store, fold, seed)


# ---------------------------------------------------------------- report
def report() -> None:
    from bgm.eval.metrics import cluster_bootstrap
    runs = [json.loads(p.read_text()) for p in sorted(Path("runs/titlesplit", OUT_EXP).glob("fold*/seed*/result.json"))]
    assert len(runs) == 15, f"{len(runs)} of 15 runs"
    modes = list(runs[0]["modes"])

    def seq_mean(mode, k):           # [5 folds x 50] per-sequence values, averaged over the 3 ensembles of a fold
        return np.concatenate([np.nanmean([r["modes"][mode]["per_seq"][k] for r in runs if r["fold"] == f], 0) for f in range(5)])

    def run_means(mode, k):
        return np.array([np.nanmean(r["modes"][mode]["per_seq"][k]) for r in runs])

    clusters = np.concatenate([[f"concat{f}_{i}" for i in range(len(runs[0]["modes"]["track"]["per_seq"]["switch_f1"]))] for f in range(5)])
    L = ["# Per-window key and tempo in the switch cost (D137)", "",
         "Final title-split system (pipeline_ens10), 5 folds × 3 ensembles, test concatenations (50 per fold). Each mode is "
         "tuned on the val concatenations (part track accuracy + switch F1). *track* = the shipped per-file key / tempo; "
         "*wN* = key / tempo of the last N s of the outgoing window vs the first N s of the incoming one (file value where "
         "the window gives no estimate). *+mu*: μ grid extended to {0, 0.5, 1, 2, 4}. Values are mean ± std over the 15 runs; "
         f"Δ vs track (same grid) with 95% CIs resampling concatenations. Seam distances use the {HEARD_W:g} s window values "
         "for every mode (0 = same key / tempo; fifths distance and octave-folded tempo distance in [0, 1]); the wN modes "
         "optimise a similar quantity, so their seam rows are expected to drop.", "",
         "| mode | switch F1 | Δ F1 | part track acc | Δ part acc | switches / min | seam key dist | Δ seam key | seam tempo dist | tuned μ (15 runs) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in modes:
        base = "track+mu" if m.endswith("+mu") else "track"
        cells = [m]
        for k in ("switch_f1", "part_track_acc", "switches_per_min", "seam_key", "seam_tempo"):
            v = run_means(m, k)
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}")
            if k in ("switch_f1", "part_track_acc", "seam_key"):
                if not m.startswith("w"):
                    cells.append("—")
                else:
                    b = cluster_bootstrap(seq_mean(m, k) - seq_mean(base, k), clusters)
                    cells.append(f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
        mus = [r["modes"][m]["penalties"]["mu"] for r in runs] if m != "greedy" else []
        cells.append(", ".join(f"{u:g} ×{mus.count(u)}" for u in sorted(set(mus))) if mus else "—")
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "How often a window estimate exists and differs from the file value (test lattices, all candidates, both ends):", "",
          "| window | key from window | key ≠ file key | tempo from window | tempo ≠ file tempo (±5%) |", "| --- | --- | --- | --- | --- |"]
    for w in runs[0]["coverage"]:
        c = {k: np.mean([r["coverage"][w][k] for r in runs]) for k in ("key_win", "key_differs", "tempo_win", "tempo_differs")}
        L.append(f"| {w} | {c['key_win']:.3f} | {c['key_differs']:.3f} | {c['tempo_win']:.3f} | {c['tempo_differs']:.3f} |")
    L += ["", "*track (shipped penalties)* = each run's stitch.json penalties, not re-tuned: they reproduce the saved val objective "
          f"(asserted) and the saved test numbers ({sum(r['n_seq_differing_from_saved'] for r in runs)} of {15 * len(clusters) // 5} "
          "run-sequences differ, decoder near-ties). Re-tuning the track mode today picks the saved penalties in "
          f"{sum(r['retuned_track_matches_saved'] for r in runs)} of 15 runs; elsewhere a GPU-noise near-tie in the μ = 0 decode "
          "changes the val ranking. All Δs compare modes re-tuned on the same lattices."]
    REPORT.write_text("\n".join(L) + "\n")
    log.info("wrote %s", REPORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"chroma": chroma, "run": run, "report": report}[sys.argv[1]]()
