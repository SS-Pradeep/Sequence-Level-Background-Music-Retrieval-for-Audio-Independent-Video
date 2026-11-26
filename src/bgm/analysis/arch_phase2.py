"""Architecture plan Phase 2 (user revision 2026-10-02; PROGRESS.md D54): training-free dynamics envelope test.

CPU only (cached dyn_25hz arrays + numpy), safe to run beside the Phase 1a GPU chain.

Run: python -m bgm.analysis.arch_phase2
Writes reports/arch_phase2.md, appends a ledger row to results/arch_ablation.md.

Plan items:
  2.1 ~1 Hz curves from the cached raw dyn_25hz channels (mean per whole second):
      video  = motion energy (log_motion) and cut density (cut_impulse);
      music  = short-term loudness (rms_db) and onset density (log_onset).
  2.2 training-free pair score, evaluated as track retrieval on the film split's test-fold galleries,
      R@10 against chance with a 95% cluster bootstrap over film groups (5,000 resamples):
      - corr:        mean over the two curve pairs of the Pearson r on the t=0-aligned overlap,
                     maximised over lags of +/-2 s (SPEC §3.4: music sample 0 = first video frame);
      - level:       -|z(video file-mean log_motion) - z(music file-mean rms_db)|, z over the split's
                     query videos / gallery music files;
      - corr+level:  per-query gallery z-score of each score, summed (the plan's scorer, headline row);
      - best-window: diagnostic only - max Pearson r over every 1 Hz start offset of the music file,
                     "is there envelope signal anywhere in the file".
  2.3 branch: not above chance -> dynamics fusion weight 0 (branch kept only in Viterbi and rendering);
      above chance -> retrain the 1-D conv encoders with other-song negatives.
Also: the val/train film-disjointness audit the user asked for (folds_film.json).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, film_cfg
from bgm.data import manifest
from bgm.data.folds import check_folds
from bgm.eval.metrics import chance_recall, chance_summary, cluster_bootstrap, recall_summary
from bgm.eval.store import FeatureStore

log = logging.getLogger(__name__)
OUT = Path("reports/arch_phase2.md")
MAX_LAG_S = 2
MIN_OVERLAP_S = 8
V_MOTION, V_CUT = 0, 2                      # dyn_25hz video channels: log_motion, cut_impulse
M_ONSET, M_RMS = 0, 2                       # dyn_25hz music channels: log_onset, rms_db
SCORERS = ("corr", "level", "corr+level", "best-window")


# ---------------------------------------------------------------- curves and correlations (pure)
def to_1hz(x_25hz: np.ndarray) -> np.ndarray:
    """[T25] raw 25 Hz curve -> [T1] mean over each whole second (trailing partial second dropped)."""
    t1 = len(x_25hz) // 25
    return x_25hz[: t1 * 25].reshape(t1, 25).mean(1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two equal-length 1-D arrays; 0.0 when either is (near-)constant."""
    sa, sb = a.std(), b.std()
    if len(a) < 2 or sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def lag_corr(v_1hz: np.ndarray, m_1hz: np.ndarray, max_lag: int = MAX_LAG_S, min_overlap: int = MIN_OVERLAP_S) -> float:
    """Max Pearson r between v[t] and m[t + d] over integer lags d in [-max_lag, max_lag] (1 Hz samples).

    Both curves start at t = 0 of their file; the true pair is aligned there (SPEC §3.4). -1.0 when no lag
    leaves at least `min_overlap` seconds of overlap."""
    best = -1.0
    for d in range(-max_lag, max_lag + 1):
        va, ma = (v_1hz[-d:], m_1hz) if d < 0 else (v_1hz, m_1hz[d:])
        n = min(len(va), len(ma))
        if n >= min_overlap:
            best = max(best, pearson(va[:n], ma[:n]))
    return best


def window_corr(v_1hz: np.ndarray, m_1hz: np.ndarray, min_overlap: int = MIN_OVERLAP_S) -> float:
    """Max Pearson r of v against every 1 Hz start offset of m (full sliding scan, diagnostic scorer)."""
    n = min(len(v_1hz), len(m_1hz))
    if n < min_overlap:
        return -1.0
    v = v_1hz[:n].astype(np.float64)
    m = m_1hz.astype(np.float64)
    sv = v.std()
    if sv < 1e-8:
        return 0.0
    dots = np.correlate(m, v, mode="valid")                        # [S] sum_t m[s+t] v[t]
    cs, cs2 = np.concatenate([[0.0], m.cumsum()]), np.concatenate([[0.0], (m**2).cumsum()])
    wmean = (cs[n:] - cs[:-n]) / n
    wstd = np.sqrt(np.maximum((cs2[n:] - cs2[:-n]) / n - wmean**2, 0.0))
    ok = wstd > 1e-8
    if not ok.any():
        return 0.0
    r = (dots[ok] / n - v.mean() * wmean[ok]) / (sv * wstd[ok])
    return float(r.max())


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


# ---------------------------------------------------------------- envelope retrieval
def _video_curves(store: FeatureStore, pair_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(motion_1hz, cut_1hz, file-mean raw log_motion) from the cached raw dyn_25hz."""
    d = store.video(pair_id)["dyn_25hz"]
    return to_1hz(d[:, V_MOTION]), to_1hz(d[:, V_CUT]), float(d[:, V_MOTION].mean())


def _music_curves(store: FeatureStore, music_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(rms_1hz, onset_1hz, file-mean raw rms_db) from the cached raw dyn_25hz."""
    d = store.music(music_id)["dyn_25hz"]
    return to_1hz(d[:, M_RMS]), to_1hz(d[:, M_ONSET]), float(d[:, M_RMS].mean())


def _rank(scores: np.ndarray, relevant: np.ndarray) -> float:
    """Rank of the best relevant gallery entry, ties shared (argmax-ties rule, PROGRESS.md)."""
    best = scores[relevant].max()
    greater, ties = int((scores > best).sum()), int((scores == best).sum())
    return greater + (ties + 1) / 2


def envelope_eval(cfg, store: FeatureStore) -> dict:
    """Track retrieval with the training-free envelope scorers on every test fold of the film split.

    Returns {scorer: {"ranks", "hit10_minus_chance", "clusters", "N", "h"}} pooled over folds."""
    doc = manifest.read_folds(cfg)
    film = doc["film_group"]
    folds = sorted(doc["folds"].items(), key=lambda x: int(x[0]))
    all_q = [q for _, f in folds for q in f["test"]]
    galleries = {k: list(dict.fromkeys(store.music_of[q] for q in f["test"])) for k, f in folds}
    v_lvl = {q: _video_curves(store, q)[2] for q in all_q}
    m_lvl = {m: _music_curves(store, m)[2] for k in galleries for m in galleries[k]}
    vz = dict(zip(v_lvl, zscore(np.array(list(v_lvl.values())))))
    mz = dict(zip(m_lvl, zscore(np.array(list(m_lvl.values())))))
    out = {s: {"ranks": [], "d10": [], "clusters": [], "N": [], "h": []} for s in SCORERS}
    for k, f in folds:
        gallery = galleries[k]
        gtrack = np.array([store.music_track[m] for m in gallery])
        mcur = {m: _music_curves(store, m) for m in gallery}
        glvl = np.array([mz[m] for m in gallery])
        for q in f["test"]:
            v_mot, v_cut, _ = _video_curves(store, q)
            corr = np.array([(lag_corr(v_mot, mcur[m][0]) + lag_corr(v_cut, mcur[m][1])) / 2 for m in gallery])
            win = np.array([(window_corr(v_mot, mcur[m][0]) + window_corr(v_cut, mcur[m][1])) / 2 for m in gallery])
            level = -np.abs(vz[q] - glvl)
            scores = {"corr": corr, "level": level, "corr+level": zscore(corr) + zscore(level), "best-window": win}
            relevant = gtrack == store.track[q]
            for s, sc in scores.items():
                r = _rank(sc, relevant)
                o = out[s]
                o["ranks"].append(r)
                o["d10"].append(float(r <= 10) - chance_recall(len(gallery), int(relevant.sum()), 10))
                o["clusters"].append(film[q])
                o["N"].append(len(gallery))
                o["h"].append(int(relevant.sum()))
        log.info("fold %s done (%d queries, gallery %d)", k, len(f["test"]), len(gallery))
    return out


# ---------------------------------------------------------------- fold audit (val film-disjointness)
def fold_audit(cfg) -> dict:
    """check_folds on folds_film.json plus explicit per-fold film-group intersections for the report."""
    doc = manifest.read_folds(cfg)
    pairs = manifest.read_pairs(cfg)
    film, train_only = doc["film_group"], set(doc["train_only"])
    check_folds(doc["folds"], pairs, film, train_only)
    rows = []
    for k, f in sorted(doc["folds"].items(), key=lambda x: int(x[0])):
        g = {s: {film[p] for p in f[s]} for s in ("train", "val", "test")}
        rows.append({"fold": int(k), **{f"{s}_clips": len(f[s]) for s in g}, **{f"{s}_films": len(g[s]) for s in g},
                     "train∩val": len(g["train"] & g["val"]), "train∩test": len(g["train"] & g["test"]),
                     "val∩test": len(g["val"] & g["test"]),
                     "konjum_val_test": sum(film[p] in train_only for p in f["val"] + f["test"])})
    return {"rows": rows, "train_only": sorted(train_only), "n_films": len(set(film.values()))}


# ---------------------------------------------------------------- report
def write_report(res: dict, audit: dict) -> tuple[Path, dict]:
    chance = chance_summary(res["corr"]["N"], res["corr"]["h"])
    cis = {s: cluster_bootstrap(res[s]["d10"], res[s]["clusters"], B) for s in SCORERS}
    verdict = lambda d: "ABOVE CHANCE" if d["lo"] > 0 else ("BELOW CHANCE" if d["hi"] < 0 else "AT CHANCE")  # noqa: E731
    n = np.array(res["corr"]["N"])
    gal = f"{n.mean():.0f} ({n.min()}–{n.max()})"
    L = ["# Architecture plan, Phase 2: training-free dynamics envelope test (D54)", "",
         "Film split, track retrieval on test-fold galleries, 5 folds pooled (deterministic, no seeds). 1 Hz curves "
         "from the cached raw dyn_25hz channels: video motion energy (log_motion) + cut density (cut_impulse) vs music "
         f"loudness (rms_db) + onset density (log_onset). Lag tolerance ±{MAX_LAG_S} s around the t=0 alignment. "
         f"Δ@10 = per-query hit@10 minus exact chance, 95% cluster bootstrap over film groups, {B:,} resamples.", "",
         f"Gallery {gal}; chance R@1 {chance['R@1']:.3f}, R@10 {chance['R@10']:.3f}, MRR {chance['MRR']:.3f}.", "",
         "| scorer | R@1 | R@10 | MedR | MRR | Δ R@10 vs chance [CI] | verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in SCORERS:
        m, d = recall_summary(res[s]["ranks"]), cis[s]
        L.append(f"| {s} | {m['R@1']:.3f} | {m['R@10']:.3f} | {m['MedR']:.0f} | {m['MRR']:.3f} | "
                 f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}] | {verdict(d)} |")
    L += ["", "corr+level is the plan's scorer (per-query gallery z-scores, summed); best-window is a diagnostic "
          "upper bound (best start offset anywhere in the music file, not just ±2 s).", "",
          "## Plan 2.3 branch", ""]
    head = cis["corr+level"]
    anysig = any(cis[s]["lo"] > 0 for s in SCORERS)
    if head["lo"] > 0:
        L.append("- The envelope scorer is above chance: retrain the 1-D conv dynamics encoders with other-song "
                 "negatives instead of time-shifted copies (plan 2.3, 'above chance' arm).")
    elif not anysig:
        L.append("- No scorer clears chance, including the best-window upper bound: set the dynamics fusion weight "
                 "to 0 and keep the branch only in Viterbi and rendering (plan 2.3, 'not above chance' arm).")
    else:
        up = ", ".join(s for s in SCORERS if cis[s]["lo"] > 0)
        L.append(f"- The plan's scorer is at chance but {up} clear(s) it; decision needs the user.")
    L += ["", "## Val/train film-disjointness audit (folds_film.json)", "",
          f"`check_folds` passed: train/val/test share no film group (and no track_id) in any fold; test folds "
          f"partition the non-train-only clips. {audit['n_films']} film groups; train-only: {audit['train_only']}.", "",
          "| fold | train clips/films | val clips/films | test clips/films | train∩val films | train∩test | val∩test | train-only clips in val+test |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in audit["rows"]:
        L.append(f"| {r['fold']} | {r['train_clips']}/{r['train_films']} | {r['val_clips']}/{r['val_films']} | "
                 f"{r['test_clips']}/{r['test_films']} | {r['train∩val']} | {r['train∩test']} | {r['val∩test']} | "
                 f"{r['konjum_val_test']} |")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    m, d = recall_summary(res["corr+level"]["ranks"]), head
    led = (f"| 2026-10-02 | Phase 2 envelope test (training-free dynamics, corr+level) | film | {gal} | "
           f"{m['R@1']:.3f} | {m['R@10']:.3f} | {m['MRR']:.3f} | vs chance: **{d['mean']:+.3f}** "
           f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {verdict(d)} (vs chance, not vs heads) |")
    with open("results/arch_ablation.md", "a") as f:
        f.write(led + "\n")
    return OUT, cis


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    audit = fold_audit(cfg)
    log.info("fold audit passed: %d film groups, train-only %s", audit["n_films"], audit["train_only"])
    res = envelope_eval(cfg, FeatureStore(cfg))
    path, cis = write_report(res, audit)
    for s in SCORERS:
        log.info("%-12s R@10 %.3f, vs chance %+.3f [%+.3f, %+.3f]", s, recall_summary(res[s]["ranks"])["R@10"],
                 cis[s]["mean"], cis[s]["lo"], cis[s]["hi"])
    log.info("wrote %s and the ledger row", path)


if __name__ == "__main__":
    main()
