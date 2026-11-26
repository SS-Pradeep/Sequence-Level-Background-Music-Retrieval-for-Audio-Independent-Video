"""Training-free retrieval improvements on the 1,248-pair film split (PROGRESS.md D64): seed ensembling and
hubness correction, on the saved siglip-both heads.

Run: python -m bgm.analysis.arch_inference    -> reports/arch_inference.md + ledger rows

* ensemble: the three seeds' cosine scores averaged per (query, candidate).
* hubness correction with a query bank from the TRAIN fold only (no test query is used):
    csls  s'(q, j) = s(q, j) - mean of candidate j's top-K similarities to the bank queries (K = 10, the usual value,
          not tuned; the query-side CSLS term is constant per query and cannot change a ranking);
    is    inverted softmax (QB-Norm style): s'(q, j) = exp(s(q, j) / tau) / sum_b exp(s(b, j) / tau), with tau the
          model's own learned temperature (nothing tuned).
  Track retrieval: bank = the train videos (whole clip). Segment retrieval: bank = every train segment.
* Evaluated on track_test_fold and segment_all exactly as the heads are (same galleries, rank_of_first_hit); paired
  per-query hit@10 against the single heads (hits averaged over the three seeds), 95% cluster bootstrap over film
  groups, 5,000 resamples (the ledger rule).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap, rank_of_first_hit
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
K_CSLS = 10
SEEDS = (0, 1, 2)
SYSTEMS = ("heads", "ensemble", "csls", "is", "ensemble+csls", "ensemble+is")
OUT = Path("reports/arch_inference.md")


def csls(S: np.ndarray, hub: np.ndarray, k: int = K_CSLS) -> np.ndarray:
    """S: [Q, G] query-candidate scores; hub: [G, Nb] candidate-bank scores. Subtract each candidate's mean top-k bank score."""
    k = min(k, hub.shape[1])
    return S - np.sort(hub, 1)[:, -k:].mean(1)[None, :]


def inverted_softmax(S: np.ndarray, hub: np.ndarray, tau: float) -> np.ndarray:
    """exp(S / tau) / sum_b exp(hub / tau), in log space (ranking-equivalent): S / tau - logsumexp_b(hub / tau)."""
    h = hub / tau
    mx = h.max(1)
    return S / tau - (mx + np.log(np.exp(h - mx[:, None]).sum(1)))[None, :]


def _video_emb(sc, key, fn):
    return sc._emb(key, "video", fn)


def fold_scores(store, sc, f) -> dict:
    """Raw and bank scores of one model on one fold. track: (S [Q, G], hub [G, Nb], hits [Q, G]); segments: a list of
    (s [W], hub [W, Nb], hits [W]) per test segment."""
    st, inp = store, sc.inp
    gal = list(dict.fromkeys(st.music_of[p] for p in f["test"]))
    gtrack = np.array([st.music_track[m] for m in gal])
    whole = lambda p: _video_emb(sc, ("tv", p), lambda: inp.video(p, 0.0, st.duration("video", p)))  # noqa: E731
    M = np.stack([sc._emb(("tm", m), "music", lambda m=m: inp.music(m, 0.0, st.duration("music", m))) for m in gal])
    Q = np.stack([whole(p) for p in f["test"]])
    Bk = np.stack([whole(p) for p in f["train"]])
    track = (Q @ M.T, M @ Bk.T, np.stack([gtrack == st.track[p] for p in f["test"]]))
    Bs = np.stack([_video_emb(sc, ("wv", p, round(a, 3), round(b, 3)), lambda p=p, a=a, b=b: inp.video(p, a, b))
                   for p in f["train"] for a, b in st.segments[p]])
    segs = test_segments(st, f["test"])
    seg = []
    for q in segs:
        windows, hit = segment_gallery(st, segs, q, "all")
        s = sc.window(q[0], q[2], q[3], windows)
        W = np.stack([sc._cache[("mw", m, round(x, 3), round(d, 3))] for m, x, d in windows])
        hits = np.zeros(len(windows), bool)
        hits[hit] = True
        seg.append((s, W @ Bs.T, hits))
    return {"track": track, "segment": seg, "tau": float(sc.model.log_tau.exp().item()), "seg_pairs": [q[0] for q in segs]}


def _variants(S, hub, tau) -> dict:
    return {"raw": S, "csls": csls(S, hub), "is": inverted_softmax(S, hub, tau)}


def fold_ranks(per_seed: list[dict]) -> dict:
    """{section: {system: ranks}}; 'heads', 'csls', 'is' are [n_seeds, Q] (one row per seed), ensembles are [Q]."""
    out = {}
    # track
    V = [_variants(*d["track"][:2], d["tau"]) for d in per_seed]
    hits = per_seed[0]["track"][2]
    r = lambda S: np.array([rank_of_first_hit(S[i], hits[i]) for i in range(len(S))])  # noqa: E731
    out["track_test_fold"] = {"heads": np.stack([r(v["raw"]) for v in V]), "csls": np.stack([r(v["csls"]) for v in V]),
                              "is": np.stack([r(v["is"]) for v in V]), "ensemble": r(np.mean([v["raw"] for v in V], 0)),
                              "ensemble+csls": r(np.mean([v["csls"] for v in V], 0)), "ensemble+is": r(np.mean([v["is"] for v in V], 0))}
    # segments
    n = len(per_seed[0]["segment"])
    acc = {k: [] for k in SYSTEMS}
    for i in range(n):
        h = per_seed[0]["segment"][i][2]
        V = [_variants(d["segment"][i][0][None], d["segment"][i][1], d["tau"]) for d in per_seed]
        rr = lambda s: rank_of_first_hit(s[0], h)  # noqa: E731
        acc["heads"].append([rr(v["raw"]) for v in V]), acc["csls"].append([rr(v["csls"]) for v in V]), acc["is"].append([rr(v["is"]) for v in V])
        acc["ensemble"].append(rr(np.mean([v["raw"] for v in V], 0)))
        acc["ensemble+csls"].append(rr(np.mean([v["csls"] for v in V], 0)))
        acc["ensemble+is"].append(rr(np.mean([v["is"] for v in V], 0)))
    out["segment_all"] = {k: (np.array(v).T if np.ndim(v) == 2 else np.array(v)) for k, v in acc.items()}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film = manifest.read_folds(cfg)["film_group"]
    store = FeatureStore(cfg)
    R = {sec: {s: [] for s in SYSTEMS} for sec in ("track_test_fold", "segment_all")}
    CL = {"track_test_fold": [], "segment_all": []}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        per_seed = [fold_scores(store, load_scorer(FILM_ROOT / "semantic" / "siglip-both" / f"fold{k}" / f"seed{s}", store, "cuda"), f) for s in SEEDS]
        ranks = fold_ranks(per_seed)
        with np.load(FILM_ROOT / "semantic" / "siglip-both" / f"fold{k}" / "seed0" / "perquery.npz") as z:
            same = float((ranks["track_test_fold"]["heads"][0] == z["track_test_fold"]).mean()), float((ranks["segment_all"]["heads"][0] == z["segment_all"]).mean())
        for sec in R:
            for s in SYSTEMS:
                R[sec][s].append(ranks[sec][s])
        CL["track_test_fold"].append(np.array([film[p] for p in f["test"]]))
        CL["segment_all"].append(np.array([film[p] for p in per_seed[0]["seg_pairs"]]))
        log.info("fold %d: heads reproduce saved ranks on %.3f (track) / %.3f (segment) of queries | track R@10 heads %.3f ens %.3f csls %.3f ens+csls %.3f",
                 k, *same, float((ranks["track_test_fold"]["heads"] <= 10).mean()), float((ranks["track_test_fold"]["ensemble"] <= 10).mean()),
                 float((ranks["track_test_fold"]["csls"] <= 10).mean()), float((ranks["track_test_fold"]["ensemble+csls"] <= 10).mean()))
    names = {"heads": "single heads (baseline, mean of 3 seeds)", "ensemble": "ensemble of the 3 seeds", "csls": f"CSLS hubness correction (K = {K_CSLS}), single heads",
             "is": "inverted softmax (QB-Norm), single heads", "ensemble+csls": "ensemble + CSLS", "ensemble+is": "ensemble + inverted softmax"}
    titles = {"track_test_fold": "Track retrieval, test-fold gallery", "segment_all": "Segment retrieval, all windows"}
    L = ["# Training-free retrieval improvements: seed ensemble and hubness correction (D64)", "",
         "1,248-pair film split, saved siglip-both heads, 5 folds × 3 seeds. Hubness correction uses a query bank from the train fold "
         "only. Per-query hits are averaged over seeds for the single-model rows; ensembles give one ranking per query. "
         f"Paired Δ vs the single heads: 95% cluster bootstrap over film groups, {B:,} resamples.", ""]
    res = {}
    for sec, title in titles.items():
        cl = np.concatenate(CL[sec])
        stat = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
        pq = lambda s, f: np.concatenate([f(x).mean(0) if x.ndim == 2 else f(x) for x in R[sec][s]])  # noqa: E731
        L += [f"## {title}", "", "| system | R@1 | R@10 | MRR | Δ R@10 vs single heads [95% CI] |", "| --- | --- | --- | --- | --- |"]
        for s in SYSTEMS:
            v = {m: pq(s, f) for m, f in stat.items()}
            d = cluster_bootstrap(v["R@10"] - pq("heads", stat["R@10"]), cl, B) if s != "heads" else None
            res[(sec, s)] = (v, d)
            L.append(f"| {names[s]} | {v['R@1'].mean():.3f} | {v['R@10'].mean():.3f} | {v['MRR'].mean():.3f} | " +
                     ("—" if d is None else f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]") + " |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    rows = []
    for s in SYSTEMS[1:]:
        v, d = res[("track_test_fold", s)]
        ds = res[("segment_all", s)][1]
        verdict = "KEEP" if d["lo"] > 0 else ("REJECT (worse)" if d["hi"] < 0 else "NULL")
        rows.append(f"| {date.today().isoformat()} | Inference only: {names[s]} | film | 196 (195–199) | {v['R@1'].mean():.3f} | {v['R@10'].mean():.3f} | "
                    f"{v['MRR'].mean():.3f} | **{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]; segment **{ds['mean']:+.3f}** "
                    f"[{ds['lo']:+.3f}, {ds['hi']:+.3f}] | {verdict} |")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("\n".join(rows) + "\n")
    log.info("wrote %s and %d ledger rows", OUT, len(rows))


if __name__ == "__main__":
    main()
