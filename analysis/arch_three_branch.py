"""Three branches - semantic, dynamics, facial expression - with many-to-many targets (PROGRESS.md D77).
1,248-pair film split, 3 seeds, runs on CPU.

Run: python -m bgm.analysis.arch_three_branch   -> reports/arch_three_branch.md + ledger rows

Branches (each scores a video against music; scores are z-standardised per query and summed with weights):
  semantic   SigLIP frames vs CLAP+MERT, trained heads (siglip-both: one-to-one; siglip-both-m2m: many-to-many);
  dynamics   25 Hz motion / cut curves vs onset / beat / loudness curves, trained conv encoders (shiftneg); score =
             mean per-step cosine of the aligned spans;
  facial     per-second counts of faces and of their expressions vs CLAP+MERT, trained heads (face-both, face-both-m2m).
Many-to-many: music is clustered across films (data/manifests/music_clusters.json, 100 clusters); a video's positives
are its own music (weight 1) and the rest of that cluster (weight 0.5).
Metrics: R@10 primary (the paired music), R@10 any-hit (any music of the query's cluster) with its chance, segment
R@10 primary. Per-query values averaged over seeds; paired Δ vs the one-to-one semantic baseline; 95% cluster
bootstrap over film groups.
"""
from __future__ import annotations

import itertools
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch

from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.metrics import chance_recall, cluster_bootstrap, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids
from bgm.train.dynamics import DynModel

log = logging.getLogger(__name__)
OUT = Path("reports/arch_three_branch.md")
SEEDS, DEV, RATE = (0, 1, 2), "cpu", 25


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-3)


def fuse(branches: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    return sum(wb * z(branches[b]) for b, wb in w.items() if wb > 0)


class Dyn:
    """Dynamics branch scores from one trained run: mean per-step cosine of aligned spans."""

    def __init__(self, cfg, store, fold: int, seed: int):
        self.store, self.m, self.c = store, DynModel(cfg, DEV), {}
        self.m.load(torch.load(FILM_ROOT / "dynamics" / "shiftneg" / f"fold{fold}" / f"seed{seed}" / "ckpt.pt", map_location=DEV)["state"])
        self.m.eval()

    def enc(self, side: str, i: str) -> np.ndarray:
        if (side, i) not in self.c:
            self.c[(side, i)] = self.m.encode(side, self.store.dyn(side, i)).cpu().numpy()
        return self.c[(side, i)]

    def span(self, p: str, a: float, m: str, s: float, d: float) -> float:
        hv, hm = self.enc("video", p), self.enc("music", m)
        fa, fs = int(round(a * RATE)), int(round(s * RATE))
        L = max(1, min(int(round(d * RATE)), len(hv) - fa, len(hm) - fs))
        return float((hv[fa:fa + L] * hm[fs:fs + L]).sum(1).mean()) if L > 0 and fa < len(hv) and fs < len(hm) else 0.0

    def track(self, p: str, gallery: list[str]) -> np.ndarray:
        dur = self.store.duration("video", p)
        return np.array([self.span(p, 0.0, m, 0.0, min(dur, self.store.duration("music", m))) for m in gallery])

    def window(self, p: str, a: float, b: float, windows) -> np.ndarray:
        return np.array([self.span(p, a, m, s, d) for m, s, d in windows])


def weight_grid(step: float = 0.1):
    n = int(round(1 / step))
    return [{"sem": i / n, "dyn": j / n, "face": (n - i - j) / n} for i in range(n + 1) for j in range(n + 1 - i)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film = manifest.read_folds(cfg)["film_group"]
    clus = json.loads((manifest.manifests_dir(cfg) / "music_clusters.json").read_text())["theme"]
    store = FeatureStore(cfg)
    SYS = {  # name -> (semantic variant, face variant, weights or "val")
        "semantic, one-to-one (baseline)": ("siglip-both", None, {"sem": 1.0}),
        "semantic, many-to-many": ("siglip-both-m2m", None, {"sem": 1.0}),
        "dynamics alone": (None, None, {"dyn": 1.0}),
        "facial alone, one-to-one": (None, "face-both", {"face": 1.0}),
        "facial alone, many-to-many": (None, "face-both-m2m", {"face": 1.0}),
        "3 branches 0.5 / 0.3 / 0.2, one-to-one heads": ("siglip-both", "face-both", {"sem": 0.5, "dyn": 0.3, "face": 0.2}),
        "semantic 0.9 + facial 0.1, one-to-one heads": ("siglip-both", "face-both", {"sem": 0.9, "face": 0.1}),
        "semantic 0.8 + facial 0.2, one-to-one heads": ("siglip-both", "face-both", {"sem": 0.8, "face": 0.2}),
        "semantic 0.8 + dynamics 0.1 + facial 0.1, one-to-one heads": ("siglip-both", "face-both", {"sem": 0.8, "dyn": 0.1, "face": 0.1}),
        "3 branches, weights tuned on val, one-to-one heads": ("siglip-both", "face-both", "val"),
        "3 branches 0.5 / 0.3 / 0.2, many-to-many heads": ("siglip-both-m2m", "face-both-m2m", {"sem": 0.5, "dyn": 0.3, "face": 0.2}),
        "semantic 0.8 + facial 0.2, many-to-many heads": ("siglip-both-m2m", "face-both-m2m", {"sem": 0.8, "face": 0.2}),
        "3 branches, weights tuned on val, many-to-many heads": ("siglip-both-m2m", "face-both-m2m", "val"),
    }
    MET = ("R@10 primary", "R@10 any-hit", "seg R@10 primary")
    acc = {s: {m: [] for m in MET} for s in SYS}
    CL, CH, tuned = {"t": [], "s": []}, [], []
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        uniq = lambda ids: list(dict.fromkeys(store.music_of[p] for p in ids))  # noqa: E731
        gte, gva = uniq(f["test"]), uniq(f["val"])
        pri = lambda ids, g: [np.array([store.music_track[m] for m in g]) == store.track[p] for p in ids]  # noqa: E731
        anyh = [np.array([clus[m] for m in gte]) == clus[store.music_of[p]] for p in f["test"]]
        pri_te, pri_va = pri(f["test"], gte), pri(f["val"], gva)
        CH.append(np.array([chance_recall(len(h), int(h.sum()), 10) for h in anyh]))
        segs = test_segments(store, f["test"])
        gal = [segment_gallery(store, segs, q, "all") for q in segs]
        fold = {s: {m: [] for m in MET} for s in SYS}
        for seed in SEEDS:
            sc = {v: load_scorer(FILM_ROOT / "semantic" / v / f"fold{k}" / f"seed{seed}", store, DEV)
                  for v in ("siglip-both", "siglip-both-m2m", "face-both", "face-both-m2m")}
            dyn = Dyn(cfg, store, k, seed)
            T = {v: np.stack([s.track(q, gte) for q in f["test"]]) for v, s in sc.items()}
            T["dyn"] = np.stack([dyn.track(q, gte) for q in f["test"]])
            Tv = {v: np.stack([s.track(q, gva) for q in f["val"]]) for v, s in sc.items()}
            Tv["dyn"] = np.stack([dyn.track(q, gva) for q in f["val"]])
            S = {v: [np.asarray(s.window(q[0], q[2], q[3], w)) for q, (w, _) in zip(segs, gal)] for v, s in sc.items()}
            S["dyn"] = [dyn.window(q[0], q[2], q[3], w) for q, (w, _) in zip(segs, gal)]
            for name, (sv, fv, w) in SYS.items():
                key = {"sem": sv, "dyn": "dyn", "face": fv}
                if w == "val":
                    best, w = -1.0, None
                    for cand in weight_grid():
                        r10 = np.mean([rank_of_first_hit(fuse({b: Tv[key[b]][i] for b in cand}, cand), pri_va[i]) <= 10 for i in range(len(pri_va))])
                        if r10 > best:
                            best, w = r10, cand
                    tuned.append(w)
                use = {b: key[b] for b in w if w[b] > 0}
                ft = [fuse({b: T[v][i] for b, v in use.items()}, w) for i in range(len(pri_te))]
                fold[name]["R@10 primary"].append(np.array([float(rank_of_first_hit(s, h) <= 10) for s, h in zip(ft, pri_te)]))
                fold[name]["R@10 any-hit"].append(np.array([float(rank_of_first_hit(s, h) <= 10) for s, h in zip(ft, anyh)]))
                sr = []
                for i, (_, hit) in enumerate(gal):
                    s = fuse({b: S[v][i] for b, v in use.items()}, w)
                    sr.append(float(1 + int((s > s[hit]).sum()) <= 10))
                fold[name]["seg R@10 primary"].append(np.array(sr))
            log.info("fold %d seed %d: track R@10 " + ", ".join(f"{n.split(',')[0][:14]} {fold[n]['R@10 primary'][-1].mean():.3f}" for n in list(SYS)[:5]), k, seed)
        for s in SYS:
            for m in MET:
                acc[s][m].append(np.mean(fold[s][m], 0))
        CL["t"].append(np.array([film[p] for p in f["test"]])), CL["s"].append(np.array([film[q[0]] for q in segs]))
    V = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in acc.items()}
    cl = {m: np.concatenate(CL["s" if m.startswith("seg") else "t"]) for m in MET}
    base = "semantic, one-to-one (baseline)"
    fmt = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    tw = {b: float(np.mean([w[b] for w in tuned])) for b in ("sem", "dyn", "face")}   # over both tuned systems
    L = ["# Three branches (semantic, dynamics, facial expression) with many-to-many targets (D77)", "",
         "1,248-pair film split, test-fold gallery 196 (195–199), 3 seeds per system, single models (no ensemble). Each cell: value (paired Δ vs "
         f"the one-to-one semantic baseline [95% CI]; film-group bootstrap, {B:,} resamples). Chance: R@10 primary 0.058, R@10 any-hit "
         f"{np.concatenate(CH).mean():.3f}, segment R@10 0.020.", "", "| system | " + " | ".join(MET) + " |", "| --- | --- | --- | --- |"]
    rows = []
    for s in SYS:
        cells, ds = [], {}
        for m in MET:
            d = None if s == base else cluster_bootstrap(V[s][m] - V[base][m], cl[m], B)
            ds[m] = d
            cells.append(f"{V[s][m].mean():.3f}" + ("" if d is None else f" ({fmt(d)})"))
        L.append(f"| {s} | " + " | ".join(cells) + " |")
        if s != base:
            d = ds["R@10 primary"]
            rows.append(f"| {date.today().isoformat()} | Three-branch / many-to-many: {s} | film | 196 (195–199) | — | primary {V[s]['R@10 primary'].mean():.3f}; any-hit "
                        f"{V[s]['R@10 any-hit'].mean():.3f} | — | primary **{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]; segment {fmt(ds['seg R@10 primary'])} | "
                        f"{'KEEP' if d['lo'] > 0 else ('REJECT (worse)' if d['hi'] < 0 else 'NULL')} |")
    L += ["", f"Weights chosen on val (mean over {len(tuned)} runs): semantic {tw['sem']:.2f}, dynamics {tw['dyn']:.2f}, facial {tw['face']:.2f}.", ""]
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
