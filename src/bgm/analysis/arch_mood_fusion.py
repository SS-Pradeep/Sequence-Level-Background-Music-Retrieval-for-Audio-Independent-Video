"""Semantic+mood fusion vs semantic-only on the film split (user step 2026-10-02; PROGRESS.md D55). CPU only.

Run: OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= nice python -m bgm.analysis.arch_mood_fusion
Writes reports/arch_mood_fusion.md, appends ledger rows to results/arch_ablation.md.

Per (fold, seed): the trained semantic head (runs/filmsplit/semantic/siglip-both, loaded on CPU) plus the fold's
attribute calibration (attr_logits_1hz, fit on train only, seed-independent). Fusion follows SPEC §5.8: each
branch's raw scores are z-standardized over the query's candidate set and summed with (1 - wa, wa); the dynamics
weight is fixed to 0 (D54: envelope test at chance). wa is tuned per (fold, seed) on the val fold - for the track
protocol on val track R@10 (val-music gallery), for the segment protocol on val segment R@10 (all-windows
variant); the ascending grid keeps the smaller wa on ties, so ties favour semantic-only. Test sections:
track_test_fold and segment_all, paired per-query Δ (fused − semantic-only) with hits averaged over seeds,
95% cluster bootstrap over film groups, 5,000 resamples (the ledger's decision rule, on the track CI).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.attributes import fit_calibration, s_attr
from bgm.eval.metrics import cluster_bootstrap, rank_of_first_hit, recall_summary
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids
from bgm.features.extract import feature_path
from bgm.retrieval.pipeline import fuse, zstats

log = logging.getLogger(__name__)
GRID = np.round(np.linspace(0.0, 1.0, 11), 2)              # candidate attr weights wa (sem weight = 1 - wa)
SEEDS = (0, 1, 2)
VARIANT = "siglip-both"
OUT = Path("reports/arch_mood_fusion.md")
SECTIONS = ("track_test_fold", "segment_all")


class LeanAttrStore:
    """Duck-typed stand-in for FeatureStore inside fit_calibration: loads and caches only attr_logits_1hz
    ([T1, 16] per item), so calibrating on ~990 train pairs never pulls the big semantic arrays into RAM."""

    def __init__(self, cfg, store: FeatureStore):
        self.cfg, self.music_of, self.track = cfg, store.music_of, store.track
        self._c: dict = {}

    def _get(self, side: str, item_id: str) -> dict:
        if (side, item_id) not in self._c:
            with np.load(feature_path(self.cfg, side, item_id)) as z:
                self._c[(side, item_id)] = {"attr_logits_1hz": z["attr_logits_1hz"]}
        return self._c[(side, item_id)]

    def video(self, p: str) -> dict:
        return self._get("video", p)

    def music(self, m: str) -> dict:
        return self._get("music", m)


class AttrScorer:
    """Mood branch with the fold's calibration; the span rule is applied exactly as on the semantic side."""

    def __init__(self, store: FeatureStore, cal):
        self.store, self.tv, self.tm = store, cal.transform("video"), cal.transform("music")

    def track(self, q: str, gallery: list[str]) -> np.ndarray:
        pv = self.store.whole("video", q, "attr_logits_1hz", self.tv)
        pm = np.stack([self.store.whole("music", m, "attr_logits_1hz", self.tm) for m in gallery])
        return s_attr(pv[None], pm)

    def window(self, q: str, a: float, b: float, windows) -> np.ndarray:
        pv = self.store.pooled("video", q, "attr_logits_1hz", a, b, self.tv)
        pm = np.stack([self.store.pooled("music", m, "attr_logits_1hz", s, s + d, self.tm) for m, s, d in windows])
        return s_attr(pv[None], pm)


def rank_at(raw: dict[str, np.ndarray], hits: np.ndarray, wa: float) -> int:
    """Fused rank (SPEC §5.8 standardization) of the best relevant candidate at attr weight wa."""
    return rank_of_first_hit(fuse(raw, zstats(raw), {"sem": 1.0 - wa, "attr": wa}), hits)


def tune_wa(queries: list[tuple[dict, np.ndarray]]) -> tuple[float, float]:
    """(wa, val R@10): ascending grid, strict improvement only, so ties keep the smaller wa."""
    best, best_wa = -1.0, 0.0
    for wa in GRID:
        r10 = float(np.mean([rank_at(raw, hits, wa) <= 10 for raw, hits in queries]))
        if r10 > best:
            best, best_wa = r10, float(wa)
    return best_wa, best


def _track_queries(store, at: AttrScorer, sc, ids: list[str], gallery: list[str], attr_cache: dict) -> list:
    gtrack = np.array([store.music_track[m] for m in gallery])
    out = []
    for q in ids:
        if q not in attr_cache:
            attr_cache[q] = at.track(q, gallery)
        out.append(({"sem": sc.track(q, gallery), "attr": attr_cache[q]}, gtrack == store.track[q]))
    return out


def _segment_queries(store, at: AttrScorer, sc, ids: list[str], attr_cache: dict) -> tuple[list, list[str]]:
    segs = test_segments(store, ids)
    out, owners = [], []
    for q in segs:
        windows, hit = segment_gallery(store, segs, q, "all")
        if q not in attr_cache:
            attr_cache[q] = at.window(q[0], q[2], q[3], windows)
        hits = np.zeros(len(windows), bool)
        hits[hit] = True
        out.append(({"sem": sc.window(q[0], q[2], q[3], windows), "attr": attr_cache[q]}, hits))
        owners.append(q[0])
    return out, owners


def run_all(cfg) -> dict:
    """{sec: {"fused"|"sem": {fold: [ranks per seed]}, "wa": [...], "clusters": {fold: [...]}, "mismatch": int}}."""
    film = manifest.read_folds(cfg)["film_group"]
    temps = list(cfg.features.attributes.temps)
    res = {sec: {"fused": {}, "sem": {}, "wa": [], "clusters": {}, "N": {}, "mismatch": 0, "n": 0} for sec in SECTIONS}
    for k in range(cfg.splits.n_folds):
        store = FeatureStore(cfg)                           # fresh per fold: caches stay bounded
        f = fold_ids(cfg, k)
        cal = fit_calibration(LeanAttrStore(cfg, store), f["train"], "attr_logits_1hz", temps, tag=f"fold{k}")
        at = AttrScorer(store, cal)
        gal_val = list(dict.fromkeys(store.music_of[p] for p in f["val"]))
        gal_test = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
        caches = {sec: {} for sec in SECTIONS + ("val_track", "val_seg")}   # attr scores, seed-independent
        for s in SEEDS:
            run = FILM_ROOT / "semantic" / VARIANT / f"fold{k}" / f"seed{s}"
            sc = load_scorer(run, store, "cpu")
            vq_t = _track_queries(store, at, sc, f["val"], gal_val, caches["val_track"])
            wa_t, val_t = tune_wa(vq_t)
            vq_s, _ = _segment_queries(store, at, sc, f["val"], caches["val_seg"])
            wa_s, val_s = tune_wa(vq_s)
            tq_t = _track_queries(store, at, sc, f["test"], gal_test, caches["track_test_fold"])
            tq_s, owners = _segment_queries(store, at, sc, f["test"], caches["segment_all"])
            with np.load(run / "perquery.npz") as z:
                saved = {"track_test_fold": z["track_test_fold"], "segment_all": z["segment_all"]}
            for sec, qs, wa, own in (("track_test_fold", tq_t, wa_t, f["test"]), ("segment_all", tq_s, wa_s, owners)):
                r = res[sec]
                r["fused"].setdefault(k, []).append(np.array([rank_at(raw, h, wa) for raw, h in qs], float))
                sem = np.array([rank_at(raw, h, 0.0) for raw, h in qs], float)
                r["sem"].setdefault(k, []).append(sem)
                r["wa"].append(wa)
                r["clusters"][k] = np.array([film[p] for p in own])
                r["N"][k] = np.array([len(h) for _, h in qs])
                r["mismatch"] += int((sem != saved[sec]).sum())
                r["n"] += len(sem)
            log.info("fold %d seed %d: wa track %.1f (val R@10 %.3f) seg %.1f (val R@10 %.3f) | test track R@10 fused %.3f sem %.3f",
                     k, s, wa_t, val_t, wa_s, val_s, float(np.mean(res["track_test_fold"]["fused"][k][-1] <= 10)),
                     float(np.mean(res["track_test_fold"]["sem"][k][-1] <= 10)))
    return res


# ---------------------------------------------------------------- aggregation and report
def paired(sec: dict, stat) -> dict:
    """Per-query stat averaged over each fold's seeds, Δ fused − sem pooled over folds, film-cluster bootstrap."""
    d, cl = [], []
    for k in sorted(sec["fused"]):
        a = np.mean([stat(r) for r in sec["fused"][k]], 0)
        b = np.mean([stat(r) for r in sec["sem"][k]], 0)
        d.append(a - b)
        cl.append(sec["clusters"][k])
    return cluster_bootstrap(np.concatenate(d), np.concatenate(cl), B)


def _abs_row(sec: dict, system: str) -> dict:
    per_run = [recall_summary(r) for k in sec[system] for r in sec[system][k]]
    return {m: f"{np.mean([x[m] for x in per_run]):.3f} ± {np.std([x[m] for x in per_run], ddof=1):.3f}"
            for m in ("R@1", "R@10", "MedR", "MRR")}


def write_report(res: dict) -> Path:
    stats = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
    cis = {sec: {m: paired(res[sec], f) for m, f in stats.items()} for sec in SECTIONS}
    ci = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    titles = {"track_test_fold": "Track retrieval, test-fold gallery", "segment_all": "Segment retrieval, all windows"}
    L = ["# Semantic+mood fusion vs semantic-only, film split (D55)", "",
         "Attribute branch: attr_logits_1hz with the fold's train-only calibration; SPEC §5.8 fusion with the "
         "dynamics weight fixed to 0 (D54). wa tuned per (fold, seed) on the val fold (ties keep the smaller wa). "
         f"15 runs; paired Δ fused − semantic-only, 95% cluster bootstrap over film groups, {B:,} resamples.", ""]
    for sec in SECTIONS:
        wa = np.array(res[sec]["wa"])
        a, b = _abs_row(res[sec], "fused"), _abs_row(res[sec], "sem")
        L += [f"## {titles[sec]}", "",
              f"Tuned wa: mean {wa.mean():.2f} (min {wa.min():.1f}, max {wa.max():.1f}; {int((wa == 0).sum())}/15 runs at 0). "
              f"Semantic-only ranks match the saved per-query files on {res[sec]['n'] - res[sec]['mismatch']}/{res[sec]['n']} "
              "queries (CPU vs GPU float noise).", "",
              "| system | R@1 | R@10 | MedR | MRR |", "| --- | --- | --- | --- | --- |",
              f"| semantic only | {b['R@1']} | {b['R@10']} | {b['MedR']} | {b['MRR']} |",
              f"| semantic+mood | {a['R@1']} | {a['R@10']} | {a['MedR']} | {a['MRR']} |", "",
              f"Δ fused − sem: R@1 {ci(cis[sec]['R@1'])}; R@10 {ci(cis[sec]['R@10'])}; MRR {ci(cis[sec]['MRR'])}", ""]
    keep = cis["track_test_fold"]["R@10"]
    verdict = "KEEP" if keep["lo"] > 0 else ("REJECT (worse)" if keep["hi"] < 0 else "NULL")
    L += [f"**Verdict (ledger rule, track R@10 CI): {verdict}.** Segment R@10 Δ: {ci(cis['segment_all']['R@10'])}.", ""]
    OUT.write_text("\n".join(L) + "\n")
    rows = []
    for sec, tag in (("track_test_fold", "track"), ("segment_all", "segment")):
        a = _abs_row(res[sec], "fused")
        n = np.concatenate([res[sec]["N"][k] for k in sorted(res[sec]["N"])])
        gal = f"{n.mean():.0f} ({n.min()}–{n.max()})" + ("" if sec == "track_test_fold" else " aligned windows")
        v = verdict if sec == "track_test_fold" else "segment-level, informational"
        rows.append(f"| 2026-10-02 | sem+mood fusion, dyn=0, wa val-tuned ({tag}) | film | {gal} | "
                    f"{a['R@1']} | {a['R@10']} | {a['MRR']} | {ci(cis[sec]['R@10'])} | {v} |")
    with open("results/arch_ablation.md", "a") as f:
        f.write("\n".join(rows) + "\n")
    return OUT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    res = run_all(film_cfg())
    path = write_report(res)
    log.info("wrote %s and the ledger rows", path)


if __name__ == "__main__":
    main()
