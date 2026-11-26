"""Ensembles on the 1,248-pair film split (PROGRESS.md D65): how recall grows with the number of seeds, ensembles
across feature variants, and the paper baselines ensembled the same way (the fair comparison).

Run: python -m bgm.analysis.arch_ensemble   -> reports/arch_ensemble.md + ledger rows

An ensemble averages its members' cosine scores per (query, candidate). Single-model rows average per-query hits over
their members. Sections and ranks exactly as the heads are evaluated (track_test_fold, segment_all); each member's
seed-0 ranks are checked against its saved perquery.npz. Paired Δ R@10: 95% cluster bootstrap over film groups.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("reports/arch_ensemble.md")
HEAD_SEEDS = tuple(range(10))
# name -> [(kind, variant, seeds)]
GROUPS = {"both": [("semantic", "siglip-both", HEAD_SEEDS)], "clap": [("semantic", "siglip-clap", (0, 1, 2))],
          "clipclap": [("semantic", "clip-clap", (0, 1, 2))], "cfd": [("baselines", "cfd", (0, 1, 2))],
          "utcmvmr": [("baselines", "utcmvmr", (0, 1, 2))], "mvpt": [("baselines", "mvpt", (0, 1, 2))]}


def load_baseline_scorer(run: Path, store: FeatureStore, train_ids: list[str], device: str):
    from bgm.train.baselines import BaselineModel, BaselineScorer, SeqFeatures
    ck = torch.load(run / "ckpt.pt", map_location=device)
    rc = OmegaConf.load(run / "config.yaml")
    model = BaselineModel(ck["variant"], ck["dims"], rc.train.baselines, len(rc.features.attributes.moods)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    ctx = rc.train.baselines.mvpt_context_s if ck["variant"] == "mvpt" else 0.0
    return BaselineScorer(model, SeqFeatures(store, rc, train_ids), store, ctx, device)


def member_scores(store, sc, f) -> tuple[np.ndarray, list[np.ndarray]]:
    """(track scores [Q, G], [segment scores [W]] per test segment) of one model on one fold."""
    gal = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
    T = np.stack([sc.track(q, gal) for q in f["test"]])
    segs = test_segments(store, f["test"])
    S = [np.asarray(sc.window(q[0], q[2], q[3], segment_gallery(store, segs, q, "all")[0])) for q in segs]
    return T, S


def fold_truth(store, f):
    gal = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
    gtrack = np.array([store.music_track[m] for m in gal])
    thits = np.stack([gtrack == store.track[p] for p in f["test"]])
    segs = test_segments(store, f["test"])
    shit = []
    for q in segs:
        w, hit = segment_gallery(store, segs, q, "all")
        h = np.zeros(len(w), bool)
        h[hit] = True
        shit.append(h)
    return thits, shit, [q[0] for q in segs]


def ranks_of(members: list[tuple], thits, shit) -> dict:
    """Ranks of the ensemble of `members` [(T, S)]: {'track_test_fold': [Q], 'segment_all': [n_seg]}."""
    T = np.mean([m[0] for m in members], 0)
    tr = np.array([rank_of_first_hit(T[i], thits[i]) for i in range(len(T))])
    sr = np.array([rank_of_first_hit(np.mean([m[1][i] for m in members], 0), shit[i]) for i in range(len(shit))])
    return {"track_test_fold": tr, "segment_all": sr}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film = manifest.read_folds(cfg)["film_group"]
    store = FeatureStore(cfg)
    hit10 = lambda r: (r <= 10).astype(float)  # noqa: E731
    systems: dict[str, dict[str, list]] = {}
    CL = {"track_test_fold": [], "segment_all": []}

    def add(name, r):
        for sec in CL:
            systems.setdefault(name, {s: [] for s in CL})[sec].append(r[sec] if isinstance(r, dict) else r[sec])

    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        thits, shit, seg_pairs = fold_truth(store, f)
        M = {}
        for g, specs in GROUPS.items():
            for kind, variant, seeds in specs:
                for s in seeds:
                    run = FILM_ROOT / kind / variant / f"fold{k}" / f"seed{s}"
                    sc = load_scorer(run, store, "cuda") if kind == "semantic" else load_baseline_scorer(run, store, f["train"], "cuda")
                    M[(g, s)] = member_scores(store, sc, f)
                    if s == 0:
                        r = ranks_of([M[(g, s)]], thits, shit)
                        with np.load(run / "perquery.npz") as z:
                            same = min(float((r[sec] == z[sec]).mean()) for sec in CL)
                        log.info("fold %d %s seed 0 reproduces saved ranks on %.3f of queries", k, variant, same)
                        assert same > 0.97, (variant, same)
                    del sc
        single = lambda g, seeds: {sec: np.mean([hit10(ranks_of([M[(g, s)]], thits, shit)[sec]) for s in seeds], 0) for sec in CL}  # noqa: E731
        ens = lambda mem: {sec: hit10(v) for sec, v in ranks_of([M[x] for x in mem], thits, shit).items()}  # noqa: E731
        both = lambda n: [("both", s) for s in range(n)]  # noqa: E731
        three = lambda g: [(g, s) for s in (0, 1, 2)]  # noqa: E731
        add("heads, single model (mean of 10 seeds)", single("both", HEAD_SEEDS))
        for n in (2, 3, 5, 7, 10):
            add(f"heads, ensemble of {n} seeds", ens(both(n)))
        add("heads CLAP+MERT ×3 + CLAP-only ×3", ens(both(3) + three("clap")))
        add("heads CLAP+MERT ×3 + CLAP-only ×3 + CLIP ×3", ens(both(3) + three("clap") + three("clipclap")))
        add("heads CLAP+MERT ×10 + CLAP-only ×3", ens(both(10) + three("clap")))
        for g, nm in (("cfd", "CFD-style"), ("utcmvmr", "UT-CMVMR-style"), ("mvpt", "MVPt-style")):
            add(f"{nm}, single model (mean of 3 seeds)", single(g, (0, 1, 2)))
            add(f"{nm}, ensemble of 3 seeds", ens(three(g)))
        CL["track_test_fold"].append(np.array([film[p] for p in f["test"]]))
        CL["segment_all"].append(np.array([film[p] for p in seg_pairs]))
        log.info("fold %d done: track R@10 single %.3f, ens3 %.3f, ens10 %.3f", k, systems["heads, single model (mean of 10 seeds)"]["track_test_fold"][-1].mean(),
                 systems["heads, ensemble of 3 seeds"]["track_test_fold"][-1].mean(), systems["heads, ensemble of 10 seeds"]["track_test_fold"][-1].mean())
    cl = {sec: np.concatenate(v) for sec, v in CL.items()}
    V = {n: {sec: np.concatenate(v[sec]) for sec in CL} for n, v in systems.items()}
    base = "heads, single model (mean of 10 seeds)"
    ci = lambda a, b, sec: cluster_bootstrap(V[a][sec] - V[b][sec], cl[sec], B)  # noqa: E731
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Ensembles on the film split: size curve, feature variants, and ensembled baselines (D65)", "",
         "1,248-pair film split, test-fold gallery 196 (195–199). An ensemble averages its members' cosine scores. Hit@10 per query; "
         f"paired differences with a 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         "## Heads: number of seeds and feature variants", "",
         "| system | track R@10 | Δ vs single model [95% CI] | segment R@10 | Δ vs single model [95% CI] |", "| --- | --- | --- | --- | --- |"]
    heads = [n for n in V if n.startswith("heads")]
    for n in heads:
        t, s = V[n]["track_test_fold"].mean(), V[n]["segment_all"].mean()
        if n == base:
            L.append(f"| {n} | {t:.3f} | — | {s:.3f} | — |")
        else:
            L.append(f"| {n} | {t:.3f} | {fmt(ci(n, base, 'track_test_fold'))} | {s:.3f} | {fmt(ci(n, base, 'segment_all'))} |")
    L += ["", f"- 10 seeds against 3 seeds: track {fmt(ci('heads, ensemble of 10 seeds', 'heads, ensemble of 3 seeds', 'track_test_fold'))}; "
          f"segment {fmt(ci('heads, ensemble of 10 seeds', 'heads, ensemble of 3 seeds', 'segment_all'))}.", "",
          "## Paper baselines, ensembled the same way", "",
          "| system | track R@10 | segment R@10 | heads ensemble of 3 − this ensemble of 3, track [95% CI] | same, segment |", "| --- | --- | --- | --- | --- |"]
    h3 = "heads, ensemble of 3 seeds"
    for nm in ("CFD-style", "UT-CMVMR-style", "MVPt-style"):
        a, e = f"{nm}, single model (mean of 3 seeds)", f"{nm}, ensemble of 3 seeds"
        L.append(f"| {a} | {V[a]['track_test_fold'].mean():.3f} | {V[a]['segment_all'].mean():.3f} | | |")
        L.append(f"| {e} | {V[e]['track_test_fold'].mean():.3f} | {V[e]['segment_all'].mean():.3f} | {fmt(ci(h3, e, 'track_test_fold'))} | {fmt(ci(h3, e, 'segment_all'))} |")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    rows = []
    for n in heads[1:]:
        d, ds = ci(n, base, "track_test_fold"), ci(n, base, "segment_all")
        verdict = "KEEP" if d["lo"] > 0 else ("REJECT (worse)" if d["hi"] < 0 else "NULL")
        rows.append(f"| {date.today().isoformat()} | Ensemble: {n} | film | 196 (195–199) | — | {V[n]['track_test_fold'].mean():.3f} | — | "
                    f"{fmt(d)}; segment {fmt(ds)} | {verdict} |")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("\n".join(rows) + "\n")
    log.info("wrote %s and %d ledger rows", OUT, len(rows))


if __name__ == "__main__":
    main()
