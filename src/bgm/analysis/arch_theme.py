"""One-to-many targets and scoring over theme groups (PROGRESS.md D66), 1,248-pair film split.

Run: python -m bgm.analysis.arch_theme   -> reports/arch_theme.md + ledger rows

Systems: the baseline heads (seeds 0-2) and the heads trained with theme targets (`train.semantic.theme_targets`
mask / multi), each as single models (hits averaged over seeds) and as its 3-seed ensemble.
Two ways of counting a hit, both on the usual galleries:
  strict  the heads' own protocol (track retrieval: a file of the query's track_id; segment: the query's own window);
  theme   a file / window whose music file is in the query's THEME group (data/manifests/themes.json).
Theme hits are easier by construction, so each column has its exact chance level and the comparison that matters
is between systems under the SAME rule. Paired Δ R@10: 95% cluster bootstrap over film groups.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_ensemble import member_scores
from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.data.themes import read_themes
from bgm.eval.metrics import chance_recall, cluster_bootstrap, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("reports/arch_theme.md")
VARIANTS = {"baseline heads": "siglip-both", "theme mask (mates are not negatives)": "siglip-both-thememask",
            "theme multi-positive (mates are positives)": "siglip-both-thememulti"}
SEEDS = (0, 1, 2)
KEYS = ("track/strict", "track/theme", "segment/strict", "segment/theme")


def fold_hits(store, f, theme) -> tuple[dict, list[str]]:
    """{key: per-query list of bool hit arrays over the query's candidates}, and the pair of each segment query."""
    gal = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
    gtrack, gtheme = np.array([store.music_track[m] for m in gal]), np.array([theme[m] for m in gal])
    H = {k: [] for k in KEYS}
    for p in f["test"]:
        H["track/strict"].append(gtrack == store.track[p])
        H["track/theme"].append(gtheme == theme[store.music_of[p]])
    segs = test_segments(store, f["test"])
    for q in segs:
        w, hit = segment_gallery(store, segs, q, "all")
        h = np.zeros(len(w), bool)
        h[hit] = True
        H["segment/strict"].append(h)
        H["segment/theme"].append(np.array([theme[m] == theme[store.music_of[q[0]]] for m, _, _ in w]))
    return H, [q[0] for q in segs]


def hit10(scores, hits) -> np.ndarray:
    return np.array([float(rank_of_first_hit(s, h) <= 10) for s, h in zip(scores, hits)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film, theme = manifest.read_folds(cfg)["film_group"], read_themes(cfg)
    store = FeatureStore(cfg)
    V = {(n, kind): {k: [] for k in KEYS} for n in VARIANTS for kind in ("single", "ens3")}
    CH = {k: [] for k in KEYS}
    CL = {"track": [], "segment": []}
    multi = {"track": [], "segment": []}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        H, seg_pairs = fold_hits(store, f, theme)
        for key in KEYS:
            CH[key].append(np.array([chance_recall(len(h), int(h.sum()), 10) for h in H[key]]))
        multi["track"].append(np.array([h.sum() > 1 for h in H["track/theme"]]))
        multi["segment"].append(np.array([h.sum() > 1 for h in H["segment/theme"]]))
        for name, variant in VARIANTS.items():
            M = [member_scores(store, load_scorer(FILM_ROOT / "semantic" / variant / f"fold{k}" / f"seed{s}", store, "cuda"), f) for s in SEEDS]
            for key in KEYS:
                i = 0 if key.startswith("track") else 1
                V[(name, "single")][key].append(np.mean([hit10(m[i], H[key]) for m in M], 0))
                ens = np.mean([m[0] for m in M], 0) if i == 0 else [np.mean([m[1][j] for m in M], 0) for j in range(len(M[0][1]))]
                V[(name, "ens3")][key].append(hit10(ens, H[key]))
        CL["track"].append(np.array([film[p] for p in f["test"]]))
        CL["segment"].append(np.array([film[p] for p in seg_pairs]))
        log.info("fold %d: track strict/theme R@10 baseline %.3f / %.3f, multi %.3f / %.3f", k, V[("baseline heads", "single")]["track/strict"][-1].mean(),
                 V[("baseline heads", "single")]["track/theme"][-1].mean(), V[("theme multi-positive (mates are positives)", "single")]["track/strict"][-1].mean(),
                 V[("theme multi-positive (mates are positives)", "single")]["track/theme"][-1].mean())
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}  # noqa: E731
    V = {n: cat(v) for n, v in V.items()}
    CH, CL, multi = cat(CH), cat(CL), cat(multi)
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    ci = lambda a, b, key: cluster_bootstrap(V[a][key] - V[b][key], CL[key.split("/")[0]], B)  # noqa: E731
    L = ["# One-to-many targets and scoring over theme groups (D66)", "",
         "1,248-pair film split, test-fold gallery 196 (195–199). Theme groups: same film and file-mean MERT and CLAP cosines both above "
         "the 99th percentile of different-film pairs (audio only; NOT yet verified by ear). "
         f"{100 * multi['track'].mean():.0f}% of test clips have another gallery file in their theme group. Hit@10 per query; paired "
         f"differences: 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         f"Chance R@10: track strict {CH['track/strict'].mean():.3f}, track theme {CH['track/theme'].mean():.3f}; segment strict "
         f"{CH['segment/strict'].mean():.3f}, segment theme {CH['segment/theme'].mean():.3f}.", ""]
    res = {}
    for kind, title in (("single", "Single models (mean of 3 seeds)"), ("ens3", "Ensembles of 3 seeds")):
        L += [f"## {title}", "", "| system | track R@10 strict | Δ vs baseline | track R@10 theme | Δ vs baseline | segment R@10 strict | Δ vs baseline | segment R@10 theme | Δ vs baseline |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n in VARIANTS:
            cells = []
            for key in KEYS:
                d = None if n == "baseline heads" else ci((n, kind), ("baseline heads", kind), key)
                res[(n, kind, key)] = (V[(n, kind)][key].mean(), d)
                cells += [f"{V[(n, kind)][key].mean():.3f}", "—" if d is None else fmt(d)]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    rows = []
    for n in list(VARIANTS)[1:]:
        (ts, dts), (tt, dtt) = res[(n, "single", "track/strict")], res[(n, "single", "track/theme")]
        dss, dst = res[(n, "single", "segment/strict")][1], res[(n, "single", "segment/theme")][1]
        verdict = "KEEP" if dts["lo"] > 0 else ("REJECT (worse)" if dts["hi"] < 0 else "NULL")
        rows.append(f"| {date.today().isoformat()} | One-to-many targets: {n} (train.semantic.theme_targets; themes unverified by ear) | film | 196 (195–199) | — | "
                    f"strict {ts:.3f}; theme {tt:.3f} | — | strict {fmt(dts)}; theme-level {fmt(dtt)}; segment strict {fmt(dss)}, theme {fmt(dst)} | {verdict} (strict rule) |")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("\n".join(rows) + "\n")
    log.info("wrote %s and %d ledger rows", OUT, len(rows))


if __name__ == "__main__":
    main()
