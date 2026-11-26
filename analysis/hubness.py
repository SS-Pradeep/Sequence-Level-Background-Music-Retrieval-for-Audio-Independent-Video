"""Test-time hubness reduction on the ensemble's track scores, cleaned title split (PROGRESS.md D138, Step 2).

No retraining: the shipped 10-head ensembles' score matrices (3 disjoint ens10 per fold, as D120) are re-scored per
music file with statistics of a QUERY BANK, i.e. the similarities of many video queries to that file:

    inverted softmax (IS)   s'(q, m) = beta * s(q, m) - logsumexp_{q' in B} beta * s(q', m)
    CSLS                    s'(q, m) = 2 s(q, m) - r_B(m),  r_B(m) = mean of the k largest s(q', m), q' in B

(CSLS's query-side term r(q) is constant over one query's ranking and is left out.) Two banks:

    train    the fold's training clips (non-transductive: nothing about the other test queries is used; main row)
    trans.   the evaluated queries themselves (transductive: the other test videos' scores, never their labels)

beta (IS) and k (the 'tuned k' CSLS row) are chosen per fold on that fold's validation clips (gallery = their music,
bank = the same kind of bank for the val gallery), on track MRR averaged over the 3 ensembles (D136's rule: val folds
hold 44-69 clips, too few for R@10 to separate nearby values); ties go to the weaker setting. CSLS k = 10 is fixed.
The test fold never chooses anything.

Run: python -m bgm.analysis.hubness [bank|report|all]
  -> runs/cleaned/hubness/bank_scores.npz, results/d138_hubness/{summary,per_fold,hubs}.csv and summary.md
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from bgm.analysis.baselines_compare import ENS_RUNS, ENS_SIZE, FOLDS, HEADS, MANIFESTS, OUT_RUNS, _test_ids, first_hit_ranks

log = logging.getLogger(__name__)

RUNS = Path("runs/cleaned/hubness")
BANK = RUNS / "bank_scores.npz"
VAL = Path("runs/cleaned/zs_blend/val_scores.npz")        # D136: val clips x val music, 3 ens10 per fold
OUT = Path("results/d138_hubness")
BETAS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
CSLS_KS = (5, 10, 20, 50)
TOP_HUBS = 10


# ---------------------------------------------------------------- pure
def inverted_softmax(S: np.ndarray, bank: np.ndarray, beta: float) -> np.ndarray:
    """S [Q, G] query scores, bank [B, G] bank-query scores for the same gallery -> [Q, G] log inverted-softmax scores."""
    return beta * np.asarray(S, np.float64) - logsumexp(beta * np.asarray(bank, np.float64), axis=0)[None, :]


def csls(S: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """S [Q, G], bank [B, G] -> [Q, G] = 2 S - r(m), r(m) = mean of the k largest bank scores of gallery file m."""
    k = min(int(k), len(bank))
    r = -np.sort(-np.asarray(bank, np.float64), axis=0)[:k].mean(0)
    return 2.0 * np.asarray(S, np.float64) - r[None, :]


def rescore(S: np.ndarray, bank: np.ndarray, method: str, p: float) -> np.ndarray:
    if method == "base":
        return np.asarray(S, np.float64)
    return inverted_softmax(S, bank, p) if method == "is" else csls(S, bank, int(p))


def track_ranks(S: np.ndarray, track: np.ndarray) -> np.ndarray:
    """1-based rank of the first right-track file per row; S [n, n] with gallery j = the music of pair j."""
    return first_hit_ranks(S, track[:, None] == track[None, :])


def top10_counts(S: np.ndarray, k: int = 10) -> np.ndarray:
    """[G] how many queries have each gallery file in their top k (the k-occurrence N_k)."""
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.bincount(top.ravel(), minlength=S.shape[1])


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0


def choose(method: str, grid, runs: list[np.ndarray], bank_of, track: np.ndarray) -> tuple[float, list[float]]:
    """The grid value maximising track MRR averaged over the ensembles; the first value in grid order wins ties.
    bank_of(e) -> the bank matrix for ensemble e."""
    mrr = [float(np.mean([(1.0 / track_ranks(rescore(S, bank_of(e), method, p), track)).mean() for e, S in enumerate(runs)]))
           for p in grid]
    return float(grid[int(np.argmax(mrr))]), mrr


# ---------------------------------------------------------------- bank scores (training clips x val / test music)
def _ids(k: int, split: str) -> list[str]:
    return list(json.loads((MANIFESTS / "folds_title.json").read_text())["folds"][str(k)][split])


def dump_bank(device: str = "cuda") -> None:
    """Per fold and ensemble: the training clips' scores against the val and the test music files."""
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.analysis.mema_protocols import make_cfg
    from bgm.eval.store import FeatureStore
    from bgm.person.evaluate import MeanScorer
    from bgm.runs import run_dir, write_run
    cfg = make_cfg("clean")
    store = FeatureStore(cfg)
    out = {}
    for k in FOLDS:
        train = _ids(k, "train")
        gals = {s: [store.music_of[p] for p in _ids(k, s)] for s in ("val", "test")}
        for e in ENS_RUNS:
            sc = MeanScorer([load_scorer(HEADS / f"fold{k}" / f"seed{m}", store, device)
                             for m in range(ENS_SIZE * e, ENS_SIZE * (e + 1))])
            with torch.no_grad():
                for s, gal in gals.items():
                    out[f"{k}_{s}_{e}"] = np.stack([np.asarray(sc.track(q, gal), np.float32) for q in train])
        log.info("bank fold %d: %d training clips x %d val / %d test files", k, len(train), len(gals["val"]), len(gals["test"]))
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BANK, **out)
    write_run(run_dir(cfg, "hubness", "bank"), cfg, {"heads": str(HEADS), "ensembles": [list(range(ENS_SIZE * e, ENS_SIZE * (e + 1))) for e in ENS_RUNS]}, 0)


# ---------------------------------------------------------------- report
SYSTEMS = (("base", "base", None), ("is", "train", "val"), ("csls", "train", 10), ("csls", "train", "val"),
           ("is", "trans", "val"), ("csls", "trans", 10))
NAME = {"base": "ensemble (current)", "is": "inverted softmax", "csls": "CSLS"}


def sys_name(method, bank, p) -> str:
    if method == "base":
        return NAME["base"]
    par = ("β tuned on val" if method == "is" else "k tuned on val") if p == "val" else f"k = {p}"
    return f"{NAME[method]} ({par}; bank: {'training clips' if bank == 'train' else 'test queries, transductive'})"


def report() -> Path:
    import pandas as pd
    from bgm.eval.metrics import cluster_bootstrap
    folds = json.loads((MANIFESTS / "folds_title.json").read_text())
    film = folds["film_group"]
    pairs = pd.read_csv(MANIFESTS / "pairs.csv")
    track_of, music_of = dict(zip(pairs.pair_id, pairs.track_id)), dict(zip(pairs.pair_id, pairs.music_id))
    V, T, Bk = np.load(VAL), np.load(OUT_RUNS / "final.npz"), np.load(BANK)
    per_fold, hits, counts, clusters = [], {}, {}, []
    for k in FOLDS:
        t = _test_ids(k)
        tr = np.array([track_of[p] for p in t])
        assert all(x == y for x, y in zip(tr, np.array([track_of[p] for p in _ids(k, "test")])))
        clusters += [film[p] for p in t]
        Ht = [T[f"{k}_{e}"] for e in ENS_RUNS]
        Hv = [V[f"{k}_heads{e}"] for e in ENS_RUNS]
        trv = V[f"{k}_track"]
        for method, bank, p in SYSTEMS:
            name = sys_name(method, bank, p)
            val_bank = (lambda e: Bk[f"{k}_val_{e}"]) if bank == "train" else (lambda e: Hv[e])
            test_bank = (lambda e: Bk[f"{k}_test_{e}"]) if bank == "train" else (lambda e: Ht[e])
            if p == "val":
                p_k, curve = choose(method, BETAS if method == "is" else CSLS_KS, Hv, val_bank, trv)
            else:
                p_k, curve = p, []
            for e, H in enumerate(Ht):
                S = rescore(H, test_bank(e), method, p_k)
                r = track_ranks(S, tr)
                hits.setdefault(name, {}).setdefault(e, []).append(r)
                c = counts.setdefault(name, {})
                for j, n in enumerate(top10_counts(S)):
                    c[music_of[t[j]]] = c.get(music_of[t[j]], 0) + n / len(ENS_RUNS)
                per_fold.append({"system": name, "fold": k, "ensemble": e, "param": p_k, "R@1": float((r <= 1).mean()),
                                 "R@10": float((r <= 10).mean()), "MedR": float(np.median(r)), "MRR": float((1 / r).mean()),
                                 "N10_skew": skewness(top10_counts(S)), "N10_max": int(top10_counts(S).max()),
                                 "val_curve": json.dumps([round(x, 4) for x in curve])})
    clusters = np.array(clusters)
    pf = pd.DataFrame(per_fold)
    OUT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT / "per_fold.csv", index=False)
    base = NAME["base"]
    rows = []
    hit = {n: np.mean([np.concatenate(hits[n][e]) <= 10 for e in ENS_RUNS], 0).astype(float) for n in hits}
    rr = {n: np.mean([1.0 / np.concatenate(hits[n][e]) for e in ENS_RUNS], 0) for n in hits}
    for n in hits:
        g = pf[pf.system == n].groupby("fold")[["R@1", "R@10", "MedR", "MRR", "N10_skew", "N10_max"]].mean()
        params = pf[pf.system == n].groupby("fold")["param"].first()
        ci = cluster_bootstrap(hit[n], clusters)
        d = cluster_bootstrap(hit[n] - hit[base], clusters) if n != base else None
        dm = cluster_bootstrap(rr[n] - rr[base], clusters) if n != base else None
        rows.append({"system": n, "param_per_fold": "" if params.isna().all() else ", ".join(f"{x:g}" for x in params),
                     **{f"{m}_mean": float(g[m].mean()) for m in g}, **{f"{m}_std": float(g[m].std(ddof=1)) for m in ("R@1", "R@10", "MedR", "MRR")},
                     "R@10_lo": ci["lo"], "R@10_hi": ci["hi"],
                     "dR@10": np.nan if d is None else d["mean"], "dR@10_lo": np.nan if d is None else d["lo"],
                     "dR@10_hi": np.nan if d is None else d["hi"], "dMRR": np.nan if dm is None else dm["mean"],
                     "dMRR_lo": np.nan if dm is None else dm["lo"], "dMRR_hi": np.nan if dm is None else dm["hi"]})
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / "summary.csv", index=False)
    hub_rows = []
    for n in hits:
        top = sorted(counts[n].items(), key=lambda x: -x[1])[:TOP_HUBS]
        for i, (m, c) in enumerate(top, 1):
            hub_rows.append({"system": n, "rank": i, "music_id": m, "track_id": dict(zip(pairs.music_id, pairs.track_id))[m],
                             "top10_lists": round(c, 2), "base_top10_lists": round(counts[base].get(m, 0.0), 2)})
    hb = pd.DataFrame(hub_rows)
    hb.to_csv(OUT / "hubs.csv", index=False)
    n_q = len(clusters)
    L = ["# Test-time hubness reduction (D138, Step 2)", "",
         f"{n_q} test videos, cleaned title split, 5 folds × 3 disjoint 10-head ensembles; gallery = the fold's test music "
         "(179 files on average). No retraining. Parameters chosen per fold on validation clips only (track MRR); "
         "± = std over the 5 fold means; [CI] = 95% film-group cluster bootstrap (10,000); Δ = paired, per query. "
         "Development numbers.", "",
         "| system | param per fold | R@1 | R@10 [95% CI] | MedR | Δ R@10 [95% CI] | Δ MRR [95% CI] | N10 skew | max N10 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sm.iterrows():
        dd = "—" if np.isnan(r["dR@10"]) else f"{r['dR@10']:+.3f} [{r['dR@10_lo']:+.3f}, {r['dR@10_hi']:+.3f}]"
        dm = "—" if np.isnan(r["dMRR"]) else f"{r['dMRR']:+.3f} [{r['dMRR_lo']:+.3f}, {r['dMRR_hi']:+.3f}]"
        L.append(f"| {r['system']} | {r['param_per_fold'] or '—'} | {r['R@1_mean']:.3f} ± {r['R@1_std']:.3f} | "
                 f"{r['R@10_mean']:.3f} ± {r['R@10_std']:.3f} [{r['R@10_lo']:.3f}, {r['R@10_hi']:.3f}] | "
                 f"{r['MedR_mean']:.1f} ± {r['MedR_std']:.1f} | {dd} | {dm} | {r['N10_skew_mean']:.2f} | {r['N10_max_mean']:.0f} |")
    L += ["", "N10 = number of test queries with a file in their top 10 (k-occurrence); skew > 0 = a few hub files fill many "
          "lists. Fold mean of the per-ensemble values.", "",
          f"## The {TOP_HUBS} files most often in top-10 lists (summed over folds, ensembles averaged)", ""]
    for n in (base, sys_name("is", "train", "val"), sys_name("csls", "train", 10)):
        L += [f"**{n}**", "", "| rank | music file | track | top-10 lists | before (current) |", "|---|---|---|---|---|"]
        for _, h in hb[hb.system == n].iterrows():
            L.append(f"| {h['rank']} | {h['music_id']} | {h['track_id']} | {h['top10_lists']:.1f} | {h['base_top10_lists']:.1f} |")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT / "summary.md")
    return OUT / "summary.md"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("bank", "all"):
        dump_bank()
    if cmd in ("report", "all"):
        print(report())
