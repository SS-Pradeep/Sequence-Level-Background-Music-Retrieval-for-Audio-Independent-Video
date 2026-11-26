"""D133: cross-validation schemes for the final semantic head, cleaned title split.

The headline protocol is 5 folds over film groups (folds_title.json, carried to the cleaned dataset in D115). This
re-runs the final single head (SigLIP → CLAP+MERT, `siglip-both`, training seeds 0-2) under other schemes, all grouped
by the same film groups and with the same train-only film (never tested):

  k3_s{0,1,2}      3-fold, split seeds 0-2      (make_film_folds)
  k5_title         5-fold, the existing folds    (folds_title.json; retrained here so every scheme uses the same code)
  k5_s{1,2}        repeated 5-fold, split seeds 1-2
  k10_s{0,1,2}     10-fold, split seeds 0-2
  lofo             leave one film group out (one fold per testable film)
  holdout          5 independent 80/20 draws over film groups (fold d = draw d, random_state d)

Inside every fold, val = GroupShuffleSplit(test_size=val_fraction) of the non-test film groups (early stopping only).

The test-fold gallery grows with the test fold (about 300 files at 3-fold, 90 at 10-fold, one film's files for lofo),
so test-fold recall cannot be compared across schemes. The comparison uses track retrieval against every music file
(`track_all`, 1,011 files, the same gallery for every scheme), and the exact expected R@10 at a fixed gallery of 100
files cut at random from it (hypergeometric on the rank, segment_retrieval.subsampled_hit).

Run:
  python -m bgm.analysis.cv_schemes make                    -> cleaned_dataset/manifests/cv/<scheme>.json
  python -m bgm.analysis.cv_schemes train --worker i --of n -> runs/cleaned/cv/<scheme>/semantic/siglip-both/...
  python -m bgm.analysis.cv_schemes report                  -> results/cleaned/cv_schemes.md
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit

from bgm.analysis.cleaned_compare import CLEAN
from bgm.analysis.segment_retrieval import _abs, subsampled_hit
from bgm.data import manifest
from bgm.data.folds import check_folds, folds_document, make_film_folds
from bgm.eval.final import _ci
from bgm.eval.metrics import chance_recall, cluster_bootstrap

log = logging.getLogger(__name__)
B = 10000
SEEDS = (0, 1, 2)
VARIANT = "siglip-both"
GALLERY_N = 100
RUNS = Path("runs/cleaned/cv")
OUT = Path("results/cleaned/cv_schemes.md")
TRAIN_OVER = ["train.semantic.music_feature=both", "train.semantic.max_clips_per_film_per_epoch=30"]
# scheme -> (label, kind, n_folds or draws, split seed)
SCHEMES = {
    "k3_s0": ("3-fold", "kfold", 3, 0), "k3_s1": ("3-fold", "kfold", 3, 1), "k3_s2": ("3-fold", "kfold", 3, 2),
    "k5_title": ("5-fold", "title", 5, 0), "k5_s1": ("5-fold", "kfold", 5, 1), "k5_s2": ("5-fold", "kfold", 5, 2),
    "k10_s0": ("10-fold", "kfold", 10, 0), "k10_s1": ("10-fold", "kfold", 10, 1), "k10_s2": ("10-fold", "kfold", 10, 2),
    "lofo": ("leave-one-film-out", "lofo", None, None),
    "holdout": ("80/20 holdout ×5", "holdout", 5, None),
}
GROUPS = ("3-fold", "5-fold", "10-fold", "leave-one-film-out", "80/20 holdout ×5")


# ---------------------------------------------------------------- folds (pure)
def _testable(pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str]) -> tuple[list[str], np.ndarray, list[str]]:
    pairs = pairs.sort_values("pair_id")
    ids = [p for p in pairs["pair_id"] if film_group[p] not in train_only]
    always = sorted(p for p in pairs["pair_id"] if film_group[p] in train_only)
    return ids, np.array([film_group[p] for p in ids]), always


def _fold(ids: list[str], groups: np.ndarray, test_mask: np.ndarray, always: list[str], val_fraction: float, rs: int) -> dict:
    rest = np.flatnonzero(~test_mask)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=rs).split(rest, groups=groups[rest]))
    pick = lambda ix: sorted(ids[i] for i in ix)
    return {"train": sorted(pick(rest[tr]) + always), "val": pick(rest[va]), "test": pick(np.flatnonzero(test_mask))}


def lofo_folds(pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str], val_fraction: float = 0.1) -> dict:
    """One fold per testable film group (sorted by name): test = that film's pairs; val = GroupShuffleSplit of the
    other testable films (random_state = fold index); train-only films always in train."""
    ids, groups, always = _testable(pairs, film_group, train_only)
    return {str(k): _fold(ids, groups, groups == g, always, val_fraction, k) for k, g in enumerate(sorted(set(groups)))}


def holdout_folds(pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str], draws: int = 5,
                  test_fraction: float = 0.2, val_fraction: float = 0.1) -> dict:
    """`draws` independent splits: fold d tests GroupShuffleSplit(test_size=test_fraction, random_state=d) of the
    testable film groups (a fraction of films, not of clips); val drawn from the rest as in lofo_folds."""
    ids, groups, always = _testable(pairs, film_group, train_only)
    out = {}
    for d in range(draws):
        _, te = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=d).split(ids, groups=groups))
        mask = np.zeros(len(ids), bool)
        mask[te] = True
        out[str(d)] = _fold(ids, groups, mask, always, val_fraction, d)
    return out


def check_scheme(folds: dict, pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str], partition: bool) -> None:
    """Every fold: train / val / test share no film group or track_id, cover every pair, test and val non-empty,
    train-only films in train. partition=True (k-fold, lofo): also the full check_folds (test folds partition the
    testable pairs, no film or track tested in two folds)."""
    if partition:
        check_folds(folds, pairs, film_group, train_only)
    track = dict(zip(pairs["pair_id"], pairs["track_id"]))
    all_ids = set(pairs["pair_id"])
    for k, f in folds.items():
        assert f["test"] and f["val"], f"fold {k}: empty test or val"
        assert set(f["train"]) | set(f["val"]) | set(f["test"]) == all_ids, f"fold {k} does not cover every pair"
        for grp in (film_group, track):
            s = {n: {grp[p] for p in f[n]} for n in ("train", "val", "test")}
            assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"]), f"fold {k}: splits share a group"
        assert not ({film_group[p] for p in f["test"] + f["val"]} & train_only), f"fold {k}: a train-only film is held out"


def scheme_folds(name: str, pairs: pd.DataFrame, film_group: dict[str, str], train_only: set[str], title: dict,
                 val_fraction: float = 0.1) -> dict:
    _, kind, n, seed = SCHEMES[name]
    if kind == "kfold":
        return make_film_folds(pairs, film_group, train_only, n, val_fraction, seed)
    if kind == "title":
        return title["folds"]
    if kind == "lofo":
        return lofo_folds(pairs, film_group, train_only, val_fraction)
    return holdout_folds(pairs, film_group, train_only, n, 0.2, val_fraction)


# ---------------------------------------------------------------- config / I/O
def scheme_cfg(name: str, extra: list[str] = ()) -> DictConfig:
    from bgm.config import load_config
    n = len(json.loads((Path(CLEAN["paths.manifests"]) / "cv" / f"{name}.json").read_text())["folds"])
    over = {**CLEAN, "paths.runs": str(RUNS / name), "paths.reports": "reports/cleaned", "splits.folds_file": f"cv/{name}.json",
            "splits.n_folds": n}
    return load_config("configs/default.yaml", [*(f"{k}={v}" for k, v in over.items()), *TRAIN_OVER, *extra])


def make() -> None:
    from bgm.config import load_config
    base = load_config("configs/default.yaml", [*(f"{k}={v}" for k, v in CLEAN.items()), "splits.folds_file=folds_title.json"])
    title = manifest.read_folds(base)
    pairs = manifest.read_pairs(base)
    film, train_only = title["film_group"], set(title["train_only"])
    (Path(CLEAN["paths.manifests"]) / "cv").mkdir(exist_ok=True)
    for name, (label, kind, n, seed) in SCHEMES.items():
        folds = scheme_folds(name, pairs, film, train_only, title, float(base.splits.val_fraction))
        check_scheme(folds, pairs, film, train_only, partition=kind != "holdout")
        doc = folds_document(folds, manifest.manifest_hash(base), film_group=film, train_only=sorted(train_only), split="film",
                             scheme={"name": name, "label": label, "kind": kind, "n": n, "split_seed": seed})
        path = Path(CLEAN["paths.manifests"]) / "cv" / f"{name}.json"
        path.write_text(json.dumps(doc, indent=1))
        sizes = np.array([[len(f[s]) for s in ("train", "val", "test")] for f in folds.values()])
        log.info("%s: %d folds, train %d-%d, val %d-%d, test %d-%d", name, len(folds), sizes[:, 0].min(),
                 sizes[:, 0].max(), sizes[:, 1].min(), sizes[:, 1].max(), sizes[:, 2].min(), sizes[:, 2].max())


def jobs() -> list[tuple[str, int, int]]:
    out = []
    for name in SCHEMES:
        n = len(json.loads((Path(CLEAN["paths.manifests"]) / "cv" / f"{name}.json").read_text())["folds"])
        out += [(name, k, s) for k in range(n) for s in SEEDS]
    return out


def run_path(name: str, k: int, s: int) -> Path:
    return RUNS / name / "semantic" / VARIANT / f"fold{k}" / f"seed{s}"


def train(worker: int, of: int) -> None:
    """Jobs worker, worker + of, ...; finished runs (metrics.json) are skipped, so it resumes. One FeatureStore per
    worker (it does not depend on the folds)."""
    import torch
    from bgm.eval.store import FeatureStore
    from bgm.train.semantic import train_one
    todo = [j for i, j in enumerate(jobs()) if i % of == worker and not (run_path(*j) / "metrics.json").exists()]
    log.info("worker %d/%d: %d runs to do", worker, of, len(todo))
    store, cfgs = None, {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, (name, k, s) in enumerate(todo):
        cfg = cfgs.setdefault(name, scheme_cfg(name))
        store = store or FeatureStore(cfg)
        train_one(cfg, store, k, s, VARIANT, device)
        if i % 25 == 0:
            log.info("worker %d: %d / %d done", worker, i + 1, len(todo))


# ---------------------------------------------------------------- report
def load_scheme(name: str, film: dict[str, str], music_track: dict[str, str]) -> pd.DataFrame:
    """One row per (fold, seed, test clip): rank against every music file, rank in the test-fold gallery, h, N."""
    doc = json.loads((Path(CLEAN["paths.manifests"]) / "cv" / f"{name}.json").read_text())
    n_tracks = pd.Series(music_track).value_counts()
    rows = []
    for k, f in doc["folds"].items():
        test = f["test"]
        h = n_tracks[[music_track[p] for p in test]].to_numpy()      # pair_id == music_id in the cleaned manifest
        for s in SEEDS:
            z = np.load(run_path(name, int(k), s) / "perquery.npz")
            assert len(z["track_all"]) == len(test), f"{name} fold {k} seed {s}: perquery does not match the test fold"
            assert list(z["track_all/clusters"]) == [music_track[p] for p in test]
            rows.append(pd.DataFrame({"fold": int(k), "seed": s, "pair_id": test, "film": [film[p] for p in test],
                                      "rank_all": z["track_all"], "rank_fold": z["track_test_fold"], "h": h,
                                      "N_all": len(music_track), "N_fold": len(test), "n_train": len(f["train"])}))
    df = pd.concat(rows, ignore_index=True)
    df["hit10"] = (df.rank_all <= 10).astype(float)
    df["hit1"] = (df.rank_all <= 1).astype(float)
    df["hit5"] = (df.rank_all <= 5).astype(float)
    df["rr"] = 1.0 / df.rank_all
    df["hit10_n"] = subsampled_hit(df.rank_all.to_numpy(), df.N_all.to_numpy(), df.h.to_numpy(), GALLERY_N, 10)
    df["hit10_fold"] = (df.rank_fold <= 10).astype(float)
    return df


def _mstd(v) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def _per_clip(df: pd.DataFrame, col: str) -> pd.Series:
    """Each test clip's value averaged over every run that tested it (seeds, split seeds, draws)."""
    return df.groupby("pair_id")[col].mean()


def _boot(df: pd.DataFrame, col: str, film: dict[str, str]) -> dict:
    v = _per_clip(df, col)
    return cluster_bootstrap(v.to_numpy(), np.array([film[p] for p in v.index]), B)


def group_rows(dfs: dict[str, pd.DataFrame], film: dict[str, str]) -> list[dict]:
    out = []
    for g in GROUPS:
        names = [n for n in SCHEMES if SCHEMES[n][0] == g]
        df = pd.concat([dfs[n].assign(scheme=n) for n in names], ignore_index=True)
        run = df.groupby(["scheme", "fold", "seed"]).agg(r1=("hit1", "mean"), r5=("hit5", "mean"), r10=("hit10", "mean"),
                                                         medr=("rank_all", "median"), mrr=("rr", "mean"), r10n=("hit10_n", "mean"),
                                                         r10f=("hit10_fold", "mean"), nq=("pair_id", "size"),
                                                         ntr=("n_train", "first"), nf=("N_fold", "first")).reset_index()
        # pooled over the scheme's test clips, each clip once per (split seed, train seed): R@10 per train seed / split seed
        by_seed = df.groupby("seed").hit10.mean()
        by_split = df.groupby("scheme").hit10.mean()
        out.append({"group": g, "names": names, "df": df, "run": run, "n_runs": len(run), "n_folds": run.groupby("scheme").fold.nunique().iloc[0],
                    "clips": df.pair_id.nunique(), "films": df.film.nunique(),
                    "ci10": _boot(df, "hit10", film), "ci10n": _boot(df, "hit10_n", film),
                    "pooled": {c: df[c].mean() for c in ("hit1", "hit5", "hit10", "rr", "hit10_n")},
                    "medr_pooled": float(df.rank_all.median()),
                    "seed_std": float(by_seed.std(ddof=1)), "split_std": float(by_split.std(ddof=1)) if len(by_split) > 1 else float("nan"),
                    "chance10": float(np.mean([chance_recall(N, h, 10) for N, h in zip(df.N_all, df.h)])),
                    "chance10n": float(np.mean([chance_recall(GALLERY_N, h, 10) for h in df.h]))})
    return out


def paired(ref: pd.DataFrame, other: pd.DataFrame, col: str, film: dict[str, str]) -> tuple[dict, int]:
    a, b = _per_clip(ref, col), _per_clip(other, col)
    common = a.index.intersection(b.index)
    return cluster_bootstrap((b[common] - a[common]).to_numpy(), np.array([film[p] for p in common]), B), len(common)


def report() -> Path:
    from bgm.config import load_config
    base = load_config("configs/default.yaml", [*(f"{k}={v}" for k, v in CLEAN.items()), "splits.folds_file=folds_title.json"])
    title = manifest.read_folds(base)
    pairs = manifest.read_pairs(base)
    film = title["film_group"]
    music_track = dict(zip(pairs.music_id, pairs.track_id))
    assert (pairs.pair_id == pairs.music_id).all()
    dfs = {n: load_scheme(n, film, music_track) for n in SCHEMES}
    G = group_rows(dfs, film)
    ref = next(g for g in G if g["group"] == "5-fold")
    k5t = dfs["k5_title"]
    L = ["# Cross-validation schemes (D133)", "",
         "Final single semantic head (SigLIP → CLAP+MERT, `siglip-both`, InfoNCE, the D116 settings), cleaned dataset "
         f"({len(pairs):,} pairs, {len(set(film.values()))} film groups; Konjum salangai is train-only, so {len(set(film[p] for p in pairs.pair_id) - set(title['train_only']))} "
         "films are testable). Every scheme groups by the same film groups; val (early stopping) is 10% of the non-test films "
         f"in each fold; training seeds {', '.join(map(str, SEEDS))} in every fold. Single heads, not the 10-head ensemble. "
         "src/bgm/analysis/cv_schemes.py; folds in cleaned_dataset/manifests/cv/; runs in runs/cleaned/cv/.", "",
         "**Metric.** Track retrieval, query = a whole test clip, hit = any music file of its track group. Gallery = **every "
         f"music file ({len(music_track):,})** for every scheme, so the numbers compare across schemes; the test-fold gallery "
         f"does not (table 3). R@10 @{GALLERY_N}: exact expected R@10 with the gallery cut at random to {GALLERY_N} files, every "
         "relevant file kept. mean ± std is over runs (folds × split seeds × training seeds); for leave-one-film-out the "
         "per-fold std is over 210 folds of 1-48 clips and is not meaningful, so read the pooled value and its CI. Pooled: each "
         "test clip's hit averaged over every run that tested it, then averaged over clips; 95% CI from a cluster bootstrap over "
         f"film groups ({B:,} resamples).", "",
         "## 1. Every scheme, gallery = all music files", "",
         "| scheme | folds | runs | test clips (films) | mean train clips | R@1 | R@5 | R@10 (mean ± std over runs) | R@10 pooled [95% CI] | "
         f"R@10 @{GALLERY_N} pooled [95% CI] | MedR ↓ | MRR |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ch = G[0]
    L.append(f"| random (exact chance) | | | | | | | {ch['chance10']:.3f} | {ch['chance10']:.3f} | {ch['chance10n']:.3f} | | |")
    for g in G:
        r = g["run"]
        lofo = g["group"] == "leave-one-film-out"
        r10 = f"{g['pooled']['hit10']:.3f} (per-fold std n/a)" if lofo else _mstd(r.r10)
        L.append(f"| {g['group']} | {g['n_folds']} | {g['n_runs']} | {g['clips']} ({g['films']}) | {r.ntr.mean():.0f} | "
                 f"{g['pooled']['hit1']:.3f} | {g['pooled']['hit5']:.3f} | {r10} | {_abs(g["ci10"])} | {_abs(g["ci10n"])} | "
                 f"{g['medr_pooled']:.0f} | {g['pooled']['rr']:.3f} |")
    L += ["", "R@1, R@5, MedR (median over every run × clip rank) and MRR are pooled. The 80/20 holdout tests 20% of films per "
          "draw, so its 5 draws cover only part of the clips (column 4).", "",
          "## 2. Seed levels", "",
          "Spread of pooled R@10 (all-file gallery) from each source of randomness. Training seed: std over seeds 0-2 of the "
          "scheme's pooled R@10 (folds and split seeds fixed). Split seed: std over the 3 partitions (k-fold) or 5 draws "
          "(holdout) of their pooled R@10 (training seeds averaged).", "",
          "| scheme | R@10 by training seed 0 / 1 / 2 | std (training seed) | R@10 by split seed / draw | std (split seed) |",
          "| --- | --- | --- | --- | --- |"]
    for g in G:
        df = g["df"]
        bs = df.groupby("seed").hit10.mean()
        if g["group"] == "80/20 holdout ×5":
            sp = df.groupby("fold").hit10.mean()
        else:
            sp = df.groupby("scheme").hit10.mean().reindex(g["names"])
        L.append(f"| {g['group']} | {' / '.join(f'{v:.3f}' for v in bs)} | {bs.std(ddof=1):.3f} | "
                 f"{' / '.join(f'{v:.3f}' for v in sp) if len(sp) > 1 else '—'} | {sp.std(ddof=1):.3f} |" if len(sp) > 1 else
                 f"| {g['group']} | {' / '.join(f'{v:.3f}' for v in bs)} | {bs.std(ddof=1):.3f} | — (one partition) | — |")
    L += ["", "Per k-fold partition (mean ± std over its folds × training seeds):", "",
          "| partition | test clips per fold | R@10 all files | R@10 test-fold gallery |", "| --- | --- | --- | --- |"]
    for n in SCHEMES:
        if SCHEMES[n][1] in ("kfold", "title"):
            r = G[GROUPS.index(SCHEMES[n][0])]["run"].query("scheme == @n")
            L.append(f"| {n} | {r.nq.min()}–{r.nq.max()} | {_mstd(r.r10)} | {_mstd(r.r10f)} |")
    L += ["", "## 3. Why the test-fold gallery cannot compare schemes", "",
          "Same runs, ranked only against the music of the test fold (the SPEC's per-fold track R@10). The gallery shrinks as k "
          "grows, so R@10 rises with k for reasons unrelated to the model; for leave-one-film-out the gallery is one film's "
          "music (1-48 files, every query of a ≤ 10-file film scores a hit).", "",
          "| scheme | test-fold gallery files (mean, min-max) | R@10 test-fold gallery | R@10 all files |", "| --- | --- | --- | --- |"]
    for g in G:
        r = g["run"]
        L.append(f"| {g['group']} | {r.nf.mean():.0f} ({r.nf.min()}–{r.nf.max()}) | {g['df'].hit10_fold.mean():.3f} | {g['pooled']['hit10']:.3f} |")
    L += ["", "## 4. Paired differences vs 5-fold", "",
          "Per test clip, R@10 (all-file gallery) averaged over the runs that tested it; Δ = scheme − 5-fold (all three 5-fold "
          "partitions) over the clips both test; film-group cluster bootstrap. More training clips per fold is the main "
          "difference between schemes.", "",
          "| scheme | mean train clips | clips | Δ R@10 [95% CI] | "
          f"Δ R@10 @{GALLERY_N} [95% CI] |", "| --- | --- | --- | --- | --- |"]
    for g in G:
        if g is ref:
            continue
        d, nc = paired(ref["df"], g["df"], "hit10", film)
        dn, _ = paired(ref["df"], g["df"], "hit10_n", film)
        L.append(f"| {g['group']} | {g['run'].ntr.mean():.0f} | {nc} | {_ci(d)} | {_ci(dn)} |")
    old = Path("runs/cleaned/semantic") / VARIANT
    if all((old / f"fold{k}" / f"seed{s}" / "metrics.json").exists() for k in range(5) for s in SEEDS):
        prev = [json.loads((old / f"fold{k}" / f"seed{s}" / "metrics.json").read_text())["test"]["track_all"]["R@10"]
                for k in range(5) for s in SEEDS]
        now = k5t.groupby(["fold", "seed"]).hit10.mean()
        L += ["", f"**Reproduction check.** The existing 5-fold runs (runs/cleaned/semantic/siglip-both, seeds 0-2, D116) give "
              f"all-file R@10 {_mstd(prev)}; retrained here on the same folds: {_mstd(now)} (max |Δ| per run "
              f"{np.max(np.abs(np.array(prev) - now.to_numpy())):.3f})."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    rows = [{"scheme": g["group"], "runs": g["n_runs"], "clips": g["clips"], "train_mean": g["run"].ntr.mean(),
             "r10_pooled": g["pooled"]["hit10"], "r10_lo": g["ci10"]["lo"], "r10_hi": g["ci10"]["hi"],
             "r10_mean_runs": g["run"].r10.mean(), "r10_std_runs": g["run"].r10.std(ddof=1), f"r10_at{GALLERY_N}": g["pooled"]["hit10_n"],
             "seed_std": g["seed_std"], "split_std": g["split_std"]} for g in G]
    pd.DataFrame(rows).to_csv(OUT.with_suffix(".csv"), index=False)
    return OUT


def main(argv: list[str]) -> None:
    from bgm.log import setup_logging
    setup_logging()
    cmd = argv[0] if argv else "report"
    if cmd == "make":
        make()
    elif cmd == "train":
        w = int(argv[argv.index("--worker") + 1]) if "--worker" in argv else 0
        n = int(argv[argv.index("--of") + 1]) if "--of" in argv else 1
        train(w, n)
    else:
        log.info("wrote %s", report())


if __name__ == "__main__":
    main(sys.argv[1:])
