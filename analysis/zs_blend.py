"""Blend zero-shot scores into the trained heads' track scores, cleaned title split (PROGRESS.md D136; follows D134).

The heads overfit the training films (D134: train R@10 0.93 vs test 0.26); ImageBind / LanguageBind never saw these
films, so their scores are a film-independent signal. Per query, over its gallery:

    s = z(heads) + w * z(zero-shot)          z = z-score over the query's candidates (D77's late-fusion rule)

heads = the shipped 10-head siglip-both ensembles (3 disjoint ens10 per fold: seeds 0-9, 10-19, 20-29, as D120);
zero-shot = ImageBind (whole-clip 1 Hz means), LanguageBind (whole-clip vectors of the VM-NET features) or both
(z(IB) + z(LB)). w is chosen per fold on that fold's validation clips only (gallery = their music files), on track
MRR averaged over the 3 ensembles (val folds hold 44-69 clips, too few for R@10 to separate nearby weights); ties go to
the smaller w. The test fold never chooses anything. Test score matrices are D120's saved ones
(runs/cleaned/baselines/{final,imagebind,languagebind}.npz); a fixed-w test curve is reported as sensitivity only.

Run: python -m bgm.analysis.zs_blend [val|report|all]  -> runs/cleaned/zs_blend/, results/cleaned/zs_blend.md
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.analysis.baselines_compare import ENS_RUNS, ENS_SIZE, FOLDS, HEADS, MANIFESTS, OUT_RUNS, _test_ids, _whole, \
    first_hit_ranks

log = logging.getLogger(__name__)

RUNS = Path("runs/cleaned/zs_blend")
VAL = RUNS / "val_scores.npz"
OUT = Path("results/cleaned/zs_blend.md")
GRID = np.round(np.arange(0.0, 2.0001, 0.05), 2)
ZS = ("imagebind", "languagebind", "ib+lb")
LABEL = {"imagebind": "ImageBind", "languagebind": "LanguageBind", "ib+lb": "ImageBind + LanguageBind"}


# ---------------------------------------------------------------- pure
def zrow(S: np.ndarray) -> np.ndarray:
    """Row-wise z-score of S [queries, gallery] (D77's floor on the std)."""
    S = np.asarray(S, np.float64)
    return (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-3)


def zs_matrix(name: str, IB: np.ndarray, LB: np.ndarray) -> np.ndarray:
    """The zero-shot score matrix of one variant, z-scored per row."""
    return {"imagebind": zrow(IB), "languagebind": zrow(LB), "ib+lb": zrow(IB) + zrow(LB)}[name]


def blend(heads: np.ndarray, zs_z: np.ndarray, w: float) -> np.ndarray:
    """z(heads) + w * zs_z; w = 0 keeps the heads' ranking."""
    return zrow(heads) + w * zs_z


def track_ranks(S: np.ndarray, track: np.ndarray) -> np.ndarray:
    """1-based rank of the first right-track file per row; S [n, n] with gallery j = the music of pair j."""
    return first_hit_ranks(S, track[:, None] == track[None, :])


def choose_w(heads_runs: list[np.ndarray], zs_z: np.ndarray, track: np.ndarray, grid=GRID) -> tuple[float, np.ndarray]:
    """w maximising track MRR averaged over the ensembles' matrices; the first (smallest) w wins ties.
    Returns (w, MRR per grid point)."""
    mrr = np.array([np.mean([(1.0 / track_ranks(blend(H, zs_z, w), track)).mean() for H in heads_runs]) for w in grid])
    return float(grid[int(np.argmax(mrr))]), mrr


# ---------------------------------------------------------------- validation score matrices
def _val_ids(k: int) -> list[str]:
    return list(json.loads((MANIFESTS / "folds_title.json").read_text())["folds"][str(k)]["val"])


def dump_val(device: str = "cpu") -> None:
    """Per fold: the val clips' score matrices of the 3 ens10, ImageBind and LanguageBind -> runs/cleaned/zs_blend."""
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.analysis.mema_protocols import ImageBindScorer, make_cfg
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.person.evaluate import MeanScorer
    from bgm.runs import run_dir, write_run
    cfg = make_cfg("clean")
    store = FeatureStore(cfg)
    lpos, lv, lm = _whole("languagebind")
    out = {}
    for k in FOLDS:
        val = fold_ids(cfg, k)["val"]
        assert sorted(val) == sorted(_val_ids(k))
        val = _val_ids(k)
        gal = [store.music_of[p] for p in val]
        assert len(set(gal)) == len(gal), "a music file is shared by two val pairs"
        assert all(p in lpos for p in val), "LanguageBind vector missing for a val clip"
        out[f"{k}_ib"] = np.stack([ImageBindScorer(store).track(q, gal) for q in val])
        i = [lpos[p] for p in val]
        out[f"{k}_lb"] = lv[i] @ lm[i].T
        for e in ENS_RUNS:
            sc = MeanScorer([load_scorer(HEADS / f"fold{k}" / f"seed{m}", store, device)
                             for m in range(ENS_SIZE * e, ENS_SIZE * (e + 1))])
            with torch.no_grad():
                out[f"{k}_heads{e}"] = np.stack([np.asarray(sc.track(q, gal), float) for q in val])
        out[f"{k}_track"] = np.array([store.track[p] for p in val])
        log.info("val fold %d: %d clips", k, len(val))
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VAL, **out)
    d = run_dir(cfg, "zs_blend", "all")
    write_run(d, cfg, {"val_clips": {k: int(len(out[f"{k}_track"])) for k in FOLDS}, "grid": GRID.tolist(),
                       "heads": str(HEADS), "ensembles": [list(range(ENS_SIZE * e, ENS_SIZE * (e + 1))) for e in ENS_RUNS]}, 0)


# ---------------------------------------------------------------- report
def report() -> Path:
    import pandas as pd
    from bgm.eval.metrics import cluster_bootstrap
    folds = json.loads((MANIFESTS / "folds_title.json").read_text())
    film = folds["film_group"]
    track_of = dict(zip(*pd.read_csv(MANIFESTS / "pairs.csv")[["pair_id", "track_id"]].T.values))
    V = np.load(VAL)
    T = {m: np.load(OUT_RUNS / f"{m}.npz") for m in ("final", "imagebind", "languagebind")}
    chosen = {z: {} for z in ZS}
    val_curve = {z: [] for z in ZS}
    hits = {z: {"R@1": [], "R@10": [], "MRR": []} for z in ("heads", *ZS)}
    test_curve = {z: [] for z in ZS}
    clusters = []
    for k in FOLDS:
        t = _test_ids(k)
        tr = np.array([track_of[p] for p in t])
        clusters += [film[p] for p in t]
        Ht = [T["final"][f"{k}_{e}"] for e in ENS_RUNS]
        Hv = [V[f"{k}_heads{e}"] for e in ENS_RUNS]
        trv = V[f"{k}_track"]
        for z in ("heads", *ZS):
            if z == "heads":
                w, Zt = 0.0, np.zeros_like(Ht[0])
            else:
                w, mrr = choose_w(Hv, zs_matrix(z, V[f"{k}_ib"], V[f"{k}_lb"]), trv)
                chosen[z][k] = w
                val_curve[z].append(mrr)
                Zt = zs_matrix(z, T["imagebind"][f"{k}_0"], T["languagebind"][f"{k}_0"])
                test_curve[z].append([np.mean([(track_ranks(blend(H, Zt, g), tr) <= 10).mean() for H in Ht]) for g in GRID])
            ranks = [track_ranks(blend(H, Zt, w), tr) for H in Ht]
            for key, f in (("R@1", lambda r: r <= 1), ("R@10", lambda r: r <= 10), ("MRR", lambda r: 1.0 / r)):
                hits[z][key].append(np.mean([f(r).astype(float) for r in ranks], 0))
    clusters = np.array(clusters)
    P = {z: {key: np.concatenate(v) for key, v in d.items()} for z, d in hits.items()}
    n_q = len(clusters)

    def fm(b, sign=False):
        s = "+" if sign else ""
        return f"{b['mean']:{s}.3f} [{b['lo']:{s}.3f}, {b['hi']:{s}.3f}]"

    out = ["# Zero-shot scores blended into the trained heads (cleaned title split, D136)", "",
           f"{n_q} test videos, 5 title folds, track retrieval against the fold's test music (one file per pair). "
           "Score = z(heads) + w · z(zero-shot) per query; heads = the shipped 10-head ensembles (3 per fold, as D120). "
           "**w is chosen per fold on that fold's validation clips only** (track MRR, mean of the 3 ensembles; grid "
           f"{GRID[0]:g}–{GRID[-1]:g} step 0.05, ties to the smaller w). Δ = per test query, ensembles averaged, 95% "
           "cluster bootstrap over film groups (10,000). Development numbers on the title split.", "",
           "| system | w per fold (val) | test R@1 | test R@10 [95% CI] | test MRR | Δ R@10 vs heads [95% CI] | Δ MRR vs heads |",
           "|---|---|---|---|---|---|---|"]
    for z in ("heads", *ZS):
        ws = "—" if z == "heads" else ", ".join(f"{chosen[z][k]:g}" for k in FOLDS)
        d10 = "—" if z == "heads" else fm(cluster_bootstrap(P[z]["R@10"] - P["heads"]["R@10"], clusters), True)
        dm = "—" if z == "heads" else fm(cluster_bootstrap(P[z]["MRR"] - P["heads"]["MRR"], clusters), True)
        name = "heads only (final ens10)" if z == "heads" else f"heads + {LABEL[z]}"
        out.append(f"| {name} | {ws} | {P[z]['R@1'].mean():.3f} | {fm(cluster_bootstrap(P[z]['R@10'], clusters))} | "
                   f"{P[z]['MRR'].mean():.3f} | {d10} | {dm} |")
    out += ["", "## Sensitivity: one w for every fold (test numbers; not used to choose anything)", "",
            "| w | " + " | ".join(f"{LABEL[z]} R@10" for z in ZS) + " | " + " | ".join(f"{LABEL[z]} val MRR" for z in ZS) + " |",
            "|---|" + "---|" * (2 * len(ZS))]
    for i, g in enumerate(GRID):
        if i % 2:
            continue
        sizes = [len(_test_ids(k)) for k in FOLDS]
        tc = [np.average([c[i] for c in test_curve[z]], weights=sizes) for z in ZS]
        vc = [np.mean([c[i] for c in val_curve[z]]) for z in ZS]
        out.append(f"| {g:g} | " + " | ".join(f"{x:.3f}" for x in tc) + " | " + " | ".join(f"{x:.3f}" for x in vc) + " |")
    OUT.write_text("\n".join(out) + "\n")
    (RUNS / "selection.json").write_text(json.dumps({z: chosen[z] for z in ZS}, indent=1))
    log.info("wrote %s", OUT)
    return OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("val", "all"):
        dump_val()
    if cmd in ("report", "all"):
        print(report())
