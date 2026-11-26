"""Architecture plan Phase 0 (user plan 2026-10-02; PROGRESS.md D52): diagnostics only, no training.

Run: python -m bgm.analysis.arch_phase0 quick   -> items 0.1 and 0.5 (existing perquery files, seconds)
     python -m bgm.analysis.arch_phase0         -> everything -> reports/arch_phase0.md, reports/listening_sheet.md

Items (plan numbering):
  0.1 heads vs the three trained baselines, paired per-query CIs (film split, 5,000 cluster-bootstrap resamples).
  0.2 new-film error anatomy on the extended split: film-level hit@10, rank of the true song among same-film songs,
      same-film songs per gallery (hits averaged over the 3 seeds; old-clip rows included as a reference).
  0.3 false-negative candidates: for seed-0 wrong-top-1 queries, MERT / CLAP file-mean cosine between the retrieved
      and the true song; flag same-film pairs above the 95th percentile of different-film similarity on BOTH
      features; listening sheet with 20 flagged + 10 random controls.
  0.4 training-fold recall of the current heads (film split), per-query galleries subsampled to the test-fold
      gallery size (own music + N-1 random train files, seeded per query) - the Phase 1d (LoRA) gate.
  0.5 fusion check: the existing M9/H3 paired result, quoted with its split caveat (fusion was never rerun on the
      film split; recomputing it needs dynamics runs that do not exist there).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.analysis.full_dataset import BASE_FOLDS, full_cfg, load_scorer
from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import SECTIONS, Split, _gallery, _runs
from bgm.eval.store import FeatureStore, fold_ids, l2
from bgm.train.semantic import SemanticInputs

log = logging.getLogger(__name__)
B = 5000
BASELINES = ("cfd", "utcmvmr", "mvpt")
FILM_ROOT = Path("runs/filmsplit")
FULL_RUNS = Path("runs/full/semantic/siglip-both")
OUT = Path("reports/arch_phase0.md")


def film_cfg():
    return load_config("configs/default.yaml", ["splits.folds_file=folds_film.json"])


# ---------------------------------------------------------------- 0.1 heads vs trained baselines
def _per_query(split: Split, runs: dict[int, list], sec: str, stat) -> tuple[np.ndarray, np.ndarray]:
    """stat of per-query ranks, averaged over the fold's runs, pooled over folds, with film clusters."""
    v, cl = [], []
    for k in sorted(runs):
        v.append(np.mean([stat(r[sec]) for r in runs[k]], 0))
        cl.append(split.clusters(k, sec))
    return np.concatenate(v), np.concatenate(cl)


def heads_vs_baselines(cfg) -> dict:
    split = Split(cfg, "folds_film.json", "film")
    heads = _runs(FILM_ROOT, "siglip-both")
    stats = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
    out = {}
    for name in BASELINES:
        base = _runs(FILM_ROOT, name, kind="baselines")
        common = sorted(set(heads) & set(base))
        out[name] = {}
        for sec in ("track_test_fold", "segment_all"):
            row = {"gallery": _gallery(split, sec, False)}
            for m, f in stats.items():
                a, cl = _per_query(split, {k: heads[k] for k in common}, sec, f)
                b, _ = _per_query(split, {k: base[k] for k in common}, sec, f)
                row[m] = cluster_bootstrap(a - b, cl, B)
            out[name][sec] = row
    return out


# ---------------------------------------------------------------- 0.2 / 0.3 extended-split error anatomy
def _film_of_music(store: FeatureStore, film: dict) -> dict:
    return {m: film[p] for p, m in store.music_of.items()}


def new_film_errors(cfg_full, store: FeatureStore, seeds=(0, 1, 2)) -> tuple[dict, list[dict]]:
    doc = manifest.read_folds(cfg_full)
    film = doc["film_group"]
    old = set(json.loads(BASE_FOLDS.read_text())["film_group"])
    fmus = _film_of_music(store, film)
    agg = {g: {"film_hit10": [], "rank_same": [], "n_same": []} for g in ("new", "old")}
    wrong = []                                                       # seed-0 wrong-top-1 rows for 0.3
    for k, f in sorted(doc["folds"].items(), key=lambda x: int(x[0])):
        test = f["test"]
        gallery = list(dict.fromkeys(store.music_of[p] for p in test))
        gfilm = np.array([fmus[m] for m in gallery])
        gtrack = np.array([store.music_track[m] for m in gallery])
        for s in seeds:
            sc = load_scorer(FULL_RUNS / f"fold{k}" / f"seed{s}", store, "cuda")
            for q in test:
                scores = sc.track(q, gallery)
                order = np.argsort(-scores)
                same = gfilm == film[q]
                grp = "old" if q in old else "new"
                top10 = order[:10]
                agg[grp]["film_hit10"].append(float(same[top10].any()))
                if s == seeds[0]:
                    sf = order[same[order]]                          # same-film gallery entries, best first
                    true_pos = np.flatnonzero(gtrack[sf] == store.track[q])
                    agg[grp]["rank_same"].append(int(true_pos[0]) + 1 if len(true_pos) else -1)
                    agg[grp]["n_same"].append(int(same.sum()))
                    top1 = gallery[order[0]]
                    if gtrack[order[0]] != store.track[q]:
                        wrong.append({"query": q, "true": store.music_of[q], "retrieved": top1, "fold": int(k),
                                      "same_film": bool(fmus[top1] == film[q]),
                                      "rank_true": int(np.flatnonzero(gtrack[order] == store.track[q])[0]) + 1})
        log.info("0.2 fold %s done (%d test queries)", k, len(test))
    return agg, wrong


def false_negatives(cfg_full, store: FeatureStore, wrong: list[dict], rng) -> dict:
    emb = {}
    def mean_emb(m, key):
        if (m, key) not in emb:
            emb[(m, key)] = l2(store.music(m)[key].astype(np.float32).mean(0))
        return emb[(m, key)]
    film = manifest.read_folds(cfg_full)["film_group"]
    fmus = _film_of_music(store, film)
    for w in wrong:
        w["mert_cos"] = float(mean_emb(w["true"], "sem_mert_1hz") @ mean_emb(w["retrieved"], "sem_mert_1hz"))
        w["clap_cos"] = float(mean_emb(w["true"], "sem_clap_1hz") @ mean_emb(w["retrieved"], "sem_clap_1hz"))
    music = sorted({w["true"] for w in wrong} | {w["retrieved"] for w in wrong})
    null_m, null_c = [], []
    for _ in range(20000):
        a, b = music[rng.integers(len(music))], music[rng.integers(len(music))]
        if a != b and fmus[a] != fmus[b]:
            null_m.append(float(mean_emb(a, "sem_mert_1hz") @ mean_emb(b, "sem_mert_1hz")))
            null_c.append(float(mean_emb(a, "sem_clap_1hz") @ mean_emb(b, "sem_clap_1hz")))
    p95_m, p95_c = float(np.percentile(null_m, 95)), float(np.percentile(null_c, 95))
    for w in wrong:
        w["flagged"] = bool(w["same_film"] and w["mert_cos"] > p95_m and w["clap_cos"] > p95_c)
    flagged = sorted([w for w in wrong if w["flagged"]], key=lambda w: -(w["mert_cos"] + w["clap_cos"]))
    controls = [w for w in wrong if not w["flagged"]]
    controls = [controls[i] for i in rng.choice(len(controls), min(10, len(controls)), replace=False)]
    _write_sheet(store, flagged[:20], controls, p95_m, p95_c, len(wrong))
    return {"wrong_top1": len(wrong), "same_film_wrong": int(sum(w["same_film"] for w in wrong)),
            "p95_mert": p95_m, "p95_clap": p95_c, "flagged": len(flagged),
            "flagged_share_of_wrong": len(flagged) / max(1, len(wrong))}


def _write_sheet(store, flagged, controls, p95_m, p95_c, n_wrong) -> None:
    raw = dict(zip(store.pairs.music_id, store.pairs.music_path))
    L = ["# Listening sheet: are wrong top-1 picks actually the same cue? (Phase 0.3)", "",
         f"Seed-0 wrong-top-1 queries on the extended split: {n_wrong}. Flagged = same film AND MERT cosine > {p95_m:.3f} "
         f"AND CLAP cosine > {p95_c:.3f} (95th percentiles of different-film pairs). 20 flagged + 10 controls below; "
         "listen to each pair and mark SAME CUE / RELATED / DIFFERENT.", ""]
    for title, rows in (("Flagged (suspected false negatives)", flagged), ("Random controls (not flagged)", controls)):
        L += [f"## {title}", "", "| # | query clip | true music wav | retrieved music wav | same film | MERT cos | CLAP cos | true rank | verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {i} | {w['query']} | {raw.get(w['true'], w['true'])} | {raw.get(w['retrieved'], w['retrieved'])} | "
              f"{'yes' if w['same_film'] else 'no'} | {w['mert_cos']:.3f} | {w['clap_cos']:.3f} | {w['rank_true']} |  |"
              for i, w in enumerate(rows, 1)]
        L.append("")
    Path("reports/listening_sheet.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- 0.4 training-fold recall
def train_recall(cfg, store: FeatureStore, seeds=(0, 1, 2)) -> dict:
    out = {"train_R@10": [], "test_R@10": [], "train_R@1": []}
    for k in range(5):
        f = fold_ids(cfg, k)
        n_gal = len(set(store.music_of[p] for p in f["test"]))
        train_mus = list(dict.fromkeys(store.music_of[p] for p in f["train"]))
        mt = np.array([store.music_track[m] for m in train_mus])
        pos = {m: i for i, m in enumerate(train_mus)}
        for s in seeds:
            sc = load_scorer(FILM_ROOT / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}", store, "cuda")
            inp = sc.inp
            M = np.stack([sc._emb(("tm", m), "music", lambda m=m: inp.music(m, 0.0, store.duration("music", m))) for m in train_mus])
            r1 = r10 = 0
            qs = f["train"]
            for qi, q in enumerate(qs):
                v = sc._emb(("tv", q), "video", lambda: inp.video(q, 0.0, store.duration("video", q)))
                scores = M @ v
                own = pos[store.music_of[q]]
                rng = np.random.default_rng([77, k, qi])
                others = rng.choice([i for i in range(len(train_mus)) if mt[i] != store.track[q]], n_gal - 1, replace=False)
                sub = np.concatenate([[own], others])
                rank = 1 + int((scores[sub] > scores[own]).sum())
                r1 += rank <= 1
                r10 += rank <= 10
            out["train_R@1"].append(r1 / len(qs))
            out["train_R@10"].append(r10 / len(qs))
            with np.load(FILM_ROOT / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}" / "perquery.npz") as z:
                out["test_R@10"].append(float((z["track_test_fold"] <= 10).mean()))
        log.info("0.4 fold %d: train R@10 %.3f vs test %.3f", k, np.mean(out["train_R@10"][-3:]), np.mean(out["test_R@10"][-3:]))
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for m, v in out.items()}


# ---------------------------------------------------------------- report
def _ci(d) -> str:
    return f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def write_report(vs: dict, anatomy: dict | None, fn: dict | None, tr: dict | None) -> Path:
    L = ["# Architecture plan, Phase 0: diagnostics (D52)", "",
         "Film split = 1,248 pairs (runs/filmsplit); extended split = 4,246 pairs (runs/full). Paired Δ: per-query, "
         f"hits averaged over seeds, 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         "## 0.1 Heads minus each trained baseline (positive = heads better)", ""]
    for sec, title in (("track_test_fold", SECTIONS["track_test_fold"]), ("segment_all", SECTIONS["segment_all"])):
        L += [f"### {title}", "", "| baseline | gallery | Δ R@1 | Δ R@10 | Δ MRR |", "| --- | --- | --- | --- | --- |"]
        L += [f"| heads − {n} | {vs[n][sec]['gallery']} | {_ci(vs[n][sec]['R@1'])} | {_ci(vs[n][sec]['R@10'])} | {_ci(vs[n][sec]['MRR'])} |"
              for n in BASELINES]
        L.append("")
    if anatomy:
        L += ["## 0.2 New-film error anatomy (extended split, test-fold galleries)", "",
              "| queries | film-level hit@10 (3 seeds) | true-song rank among same-film songs: =1 / median | same-film songs in gallery, median |",
              "| --- | --- | --- | --- |"]
        for g in ("new", "old"):
            a = anatomy[g]
            rs = np.array([r for r in a["rank_same"] if r > 0])
            L.append(f"| {g} ({len(a['rank_same'])}) | {np.mean(a['film_hit10']):.3f} | {float((rs == 1).mean()):.3f} / "
                     f"{float(np.median(rs)):.0f} | {float(np.median(a['n_same'])):.0f} |")
        L.append("")
    if fn:
        L += ["## 0.3 False-negative candidates (seed 0)", "",
              f"- Wrong top-1 queries: {fn['wrong_top1']}; retrieved-from-the-same-film among them: {fn['same_film_wrong']} "
              f"({fn['same_film_wrong'] / max(1, fn['wrong_top1']):.0%}).",
              f"- Flagged as suspected false negatives (same film, MERT > {fn['p95_mert']:.3f} and CLAP > {fn['p95_clap']:.3f}, "
              f"both 95th-pct of different-film pairs): **{fn['flagged']}** ({fn['flagged_share_of_wrong']:.1%} of wrong top-1s).",
              "- Sheet for the user: reports/listening_sheet.md (20 flagged + 10 controls, wav paths included).", ""]
    if tr:
        L += ["## 0.4 Training-fold recall (film split, per-query train galleries of test size)", "",
              f"- train R@1 {tr['train_R@1']['mean']:.3f} ± {tr['train_R@1']['std']:.3f}; "
              f"train R@10 **{tr['train_R@10']['mean']:.3f} ± {tr['train_R@10']['std']:.3f}** vs "
              f"test R@10 {tr['test_R@10']['mean']:.3f} ± {tr['test_R@10']['std']:.3f}.",
              "- Low train recall favours more capacity / LoRA (Phase 1d gate); a large train-test gap favours regularisation and data breadth.", ""]
    L += ["## 0.5 Fusion check (existing M9/H3 result; fusion has not been rerun on the film split)", "",
          "- v2, track_id split, 15 runs: fused − semantic-only window@1 **−0.001** [−0.003, +0.001]; track@1 −0.005 [−0.012, +0.000]; "
          "mean tuned fine weights sem 0.80 / dyn 0.11 / attr 0.09 (docs/PROGRESS.md, 'H3 on v2').",
          "- Fusion currently adds nothing over the semantic branch; Phase 2 decides whether the dynamics weight goes to 0.", ""]
    OUT.write_text("\n".join(L) + "\n")
    return OUT


def main(quick: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    vs = heads_vs_baselines(cfg)
    anatomy = fn = tr = None
    if not quick:
        cfgf = full_cfg()
        storef = FeatureStore(cfgf)
        anatomy, wrong = new_film_errors(cfgf, storef)
        fn = false_negatives(cfgf, storef, wrong, np.random.default_rng(0))
        tr = train_recall(cfg, FeatureStore(cfg))
    path = write_report(vs, anatomy, fn, tr)
    Path("reports/arch_phase0.json").write_text(json.dumps({"vs_baselines": vs, "anatomy": {k: {m: list(map(float, x)) for m, x in v.items()}
                                                            for k, v in (anatomy or {}).items()}, "false_negatives": fn, "train_recall": tr},
                                                           default=float))
    log.info("wrote %s", path)


if __name__ == "__main__":
    main(quick=sys.argv[1:] == ["quick"])
