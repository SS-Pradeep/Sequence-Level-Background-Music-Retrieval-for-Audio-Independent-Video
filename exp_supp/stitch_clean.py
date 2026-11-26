"""The stitching study (stitching/results/main_table.md + decision.md) re-run on the cleaned dataset (1,011 pairs), D150.

Same code as the original study; nothing in src/ or stitching/ is edited. The stitching modules are pointed at the
cleaned data by patching module attributes at start-up:

  lattices   stitching.data.CACHE -> runs/exp_supp/rerank/clean_t3/cache (D148b: C head retrained on cleaned train
             clips, theta on cleaned val, cleaned ens10 emission, 200 training sequences, 92 test sequences)
  configs    bgm.segmentation.retrieval_eval.CUE_OVERRIDES += the cleaned paths (as clean_t3.cue_cfg): every
             `load_config(..., CUE_OVERRIDES)` inside stitching/ reads cleaned_dataset/ and runs/cleaned/
  outputs    every module's OUT / CACHE -> runs/exp_supp/stitch_clean/{results,cache}

Stages (each re-uses the original module's code; the few loops that hard-code stitching/results are re-written here
line for line):
  cue_stats            stitching.cue_stats helpers on the cleaned interim music (L, rho of the training films' BGM)
  eval --fold k        rows 1-10 (stitching.evaluate.main minus the D143 reference check, which only exists for the
                       original data; replaced by: the cleaned final HMM row reproduces D148b's clean_t3/gold exactly)
  gold --fold k        rows 11-13 (stitching.gold_prior.main)
  phase0               rows 14-15 (stitching.phase0 classes)
  win10 --fold k       10 s window embeddings for the semi-Markov rows (stitching.win10.build, cleaned heads)
  tokens / cue-train   row 2c's change predictor on cleaned BaSSL tokens (stitching.cue_bassl recipe, cleaned C heads)
  phase2 --fold k      2a / 2b / 2c semi-Markov + equal-budget two-stage re-tunes (stitching.phase2.main)
  semi --fold k        segmental Viterbi (stitching.extra_rows loop)
  report               main_table.md (stitching.report.main), decision.md (stitching.decision rule, n = 92), plus a
                       film-cluster bootstrap table of every row vs the final HMM -> exp_supp/stitch_clean_table.{md,tex}

`.venv/bin/python -m exp_supp.stitch_clean <stage> [--fold k]`
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from exp_supp.common import ROOT, RUNS, fmt_ci, merge_clusters, paired_ci
from exp_supp.rerank.clean_t3 import CACHE as LATTICES, CLEAN, GOLD as D148B_GOLD

log = logging.getLogger("exp_supp.stitch_clean")
BASE = RUNS / "stitch_clean"
OUT = BASE / "results"
CACHE = BASE / "cache"
TABLE = ROOT / "stitch_clean_table"


def patch() -> None:
    """Point stitching/ at the cleaned data (module attributes only; files untouched)."""
    import bgm.segmentation.retrieval_eval as rev
    if not any(x.startswith("paths.manifests=cleaned") for x in rev.CUE_OVERRIDES):
        rev.CUE_OVERRIDES = [*rev.CUE_OVERRIDES, *CLEAN]
    import stitching.cue_bassl as cb
    import stitching.data as sd
    import stitching.decision as dec
    import stitching.evaluate as ev
    import stitching.gold_prior as gp
    import stitching.phase2 as p2
    import stitching.report as rp
    import stitching.win10 as w10
    sd.CACHE = LATTICES
    w10.CACHE = CACHE
    for m in (ev, gp, p2, rp, dec):
        m.OUT = OUT
    cb.TOKENS = CACHE / "bassl_tokens.npz"
    cb.attach_p = attach_p_clean
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def cue_cfg():
    from exp_supp.rerank.clean_t3 import cue_cfg as c
    cfg = c()
    assert cfg.paths.runs == "runs/cleaned"
    return cfg


# ---------------------------------------------------------------- cue statistics (stitching.cue_stats.main, cleaned)
def stage_cue_stats(jobs: int) -> Path:
    from joblib import Parallel, delayed
    import stitching.cue_stats as cs
    from bgm.data import manifest
    from bgm.data.prep import interim_music_path
    from bgm.eval.store import fold_ids
    cfg = cue_cfg()
    pairs = manifest.read_pairs(cfg)
    fp = CACHE / "cue_feats.npz"
    if fp.exists():
        feats = np.load(fp, allow_pickle=True)["d"].item()
    else:
        mids = sorted(set(pairs["music_id"]))
        res = Parallel(n_jobs=jobs, verbose=5)(delayed(cs.file_features)(str(interim_music_path(cfg, m))) for m in mids)
        feats = dict(zip(mids, res))
        np.savez(fp, d=np.array(feats, dtype=object))
    pairs = pairs.set_index("pair_id")
    fg = manifest.read_folds(cfg)["film_group"]
    rng = np.random.default_rng(0)
    out = {"method": cs.__doc__, "data": "cleaned_dataset (1,011 pairs)", "folds": {}}
    for k in range(5):
        files = {}
        for p in fold_ids(cfg, k)["train"]:
            files.setdefault(pairs.loc[p, "music_id"], fg[p])
        total, n_cues, film_cues = 0.0, 0, {}
        for m in sorted(files):
            X, C, dur = feats[m]
            if not len(X):
                continue
            cu = cs.cues_of(X, C)
            total += dur
            n_cues += len(cu)
            film_cues.setdefault(files[m], []).extend([c for c in cu if len(c) >= 4])
        L = total / n_cues
        allc = [(f, c) for f, cu in film_cues.items() for c in cu]
        null = []
        for _ in range(20000):
            i, j = rng.integers(len(allc), size=2)
            if allc[i][0] != allc[j][0]:
                null.append(cs.xcorr(allc[i][1], allc[j][1]))
        thr = float(np.quantile(null, 0.99))
        n_rep, n_elig = 0, 0
        for f, cu in film_cues.items():
            for i in range(len(cu)):
                n_elig += 1
                if i and max(cs.xcorr(cu[h], cu[i]) for h in range(max(0, i - 60), i)) > thr:
                    n_rep += 1
        out["folds"][str(k)] = {"L_s": L, "rho": n_rep / max(n_elig, 1), "n_cues": n_cues, "n_files": len(files),
                                "music_minutes": total / 60, "n_cues_ge4s": n_elig, "n_reprises": n_rep,
                                "xcorr_threshold_p99_null": thr, "n_films": len(film_cues)}
        log.warning("cue stats fold %d: %s", k, out["folds"][str(k)])
    p = OUT / "cue_stats.json"
    p.write_text(json.dumps(out, indent=1))
    return p


# ---------------------------------------------------------------- rows 1-10 (stitching.evaluate.main, cleaned)
def stage_eval(k: int, device: str) -> Path:
    import torch
    import stitching.evaluate as ev
    from stitching.crf import CRF, ctx_features
    from stitching.data import load_fold
    from stitching.dtw_single import PretetDTW
    from stitching.greedy import Greedy
    from stitching.hmm_derived import HandSet, HMMDerived
    from stitching.metrics import evaluate_path, fold_entropy
    from stitching.smooth import Smooth
    from stitching.two_stage import TwoStage
    t0 = time.time()
    cfg = cue_cfg()
    F = load_fold(k)
    assert F.get("data") == "cleaned_dataset"
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    cs = json.loads((OUT / "cue_stats.json").read_text())["folds"][str(k)]
    L, rho = float(cs["L_s"]), float(cs["rho"])
    tau = ev.ensemble_tau(cfg, k)
    mu = float(F["d75"]["penalties"]["mu"])
    out = {"fold": k, "data": "cleaned_dataset", "tau": tau, "L_s": L, "rho": rho, "theta_segmenter": F["theta"],
           "gallery": {s: F[f"gallery_{s}"] for s in ("train", "val", "test")},
           "n_episodes": {"train": len(tr), "val": len(va), "test": len(te)},
           "gold_stats": {s: ev.gold_stats(e) for s, e in (("train", tr), ("val", va), ("test", te))},
           "shortlist": {"track_R@10": float(np.mean(np.concatenate([e.shortlist_rank for e in te]) <= 10)),
                         "track_MRR": float(np.mean(1.0 / np.concatenate([e.shortlist_rank for e in te])))},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "methods": {}, "timelines": {}, "films": {e.key: sorted(set(e.films)) for e in te},
           "test_meta": {str(i): {"segs": e.segs.tolist(), "true_sw": list(e.true_sw),
                                  "truth_files": [e.files[g["mid"]] for g in e.gold_win]} for i, e in enumerate(te)}}
    gal = F["gallery_test"]

    def run(key, st, seed=None):
        t1 = time.time()
        params = st.fit(tr, va)
        fit_s = time.time() - t1
        paths = [st.timed(e) for e in te]
        rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
        rec = {"name": dict(ev.ROWS)[key], "params": params, "fit_s": fit_s, "rows": rows,
               "entropy": fold_entropy(te, paths, gal), "seed": seed}
        log.warning("fold %d %-14s seed %s fit %.0f s | test acc %.3f F1 %.3f", k, key, seed, fit_s,
                    np.mean([r["track_acc"] for r in rows]), np.mean([r["switch_f1"] for r in rows]))
        if seed in (None, 0) and key in ev.TIMELINE:
            out["timelines"][key] = [[e.files[m] for m in p.mid] for e, p in zip(te, paths)]
        return rec, st

    for key, st in (("greedy", Greedy()), ("smooth", Smooth()), ("two_stage", TwoStage()), ("dtw", PretetDTW()),
                    ("handset", HandSet(cfg, mu)), ("hmm_l1", HMMDerived(tau, L, rho, 1.0)),
                    ("hmm_tuned", HMMDerived(tau, L, rho, tune_lam=True))):
        out["methods"][key], _ = run(key, st)
    dev = device if torch.cuda.is_available() else "cpu"
    out["crf_val"] = {}
    for key, ctx, gam in (("crf_noctx", False, (0.0,)), ("crf_ctx", True, (0.0,)), ("crf_ctx_prior", True, (0.01, 0.1, 1.0))):
        recs = []
        for s in ev.CRF_SEEDS:
            rec, st = run(key, CRF(tau, L, rho, ctx, gam, seed=s, device=dev, name=key), s)
            recs.append(rec)
            if s == 0:
                vp = [st.decode(e) for e in va]
                vr = [evaluate_path(e, p) for e, p in zip(va, vp)]
                marg = [st.switch_marginals(e) for e in va]
                sig = [np.array([ctx_features(e, t)[0] for t in range(1, e.T)]) for e in va]
                lab = [np.array([any(abs(b - x) < 1e-6 for x in e.true_sw) for b in e.btimes], float) for e in va]
                out["crf_val"][key] = {"pred_switches_per_min": float(np.mean([r["switches_per_min"] for r in vr])),
                                       "true_switches_per_min": float(np.mean([r["true_switches_per_min"] for r in vr])),
                                       "pred_switch_rate_per_boundary": float(np.mean(np.concatenate([np.diff(p.mid) != 0 for p in vp if len(p.mid) > 1]))),
                                       "mean_posterior_switch_prob": float(np.mean(np.concatenate(marg))),
                                       "true_change_rate_per_boundary": float(np.mean(np.concatenate(lab))),
                                       "derived_p_change_mean": float(np.mean(np.concatenate([1 - np.exp(-e.scene_dur[1:] / L) for e in va]))),
                                       "marginals": np.concatenate(marg).tolist(), "siglip_dist": np.concatenate(sig).tolist(),
                                       "label": np.concatenate(lab).tolist()}
        out["methods"][key] = {"name": recs[0]["name"], "seeds": recs}
    p = OUT / f"fold{k}.json"
    p.write_text(json.dumps(out, default=float))
    log.warning("fold %d eval -> %s [%.0f s]", k, p, time.time() - t0)
    return p


# ---------------------------------------------------------------- rows 11-13 (stitching.gold_prior.main, cleaned)
def stage_gold(k: int, device: str) -> Path:
    import stitching.gold_prior as gp
    p = gp.main(k, device)
    mine = json.loads(p.read_text())
    ref = json.loads((D148B_GOLD / f"fold{k}_gold.json").read_text())
    chk = {}
    for key in ("hmm_gold_l1", "hmm_gold_tuned"):
        a = np.array([[r["track_acc"], r["switch_f1"], r["TBIoU"]] for r in mine["methods"][key]["rows"]])
        b = np.array([[r["track_acc"], r["switch_f1"], r["TBIoU"]] for r in ref["methods"][key]["rows"]])
        chk[key] = float(np.nanmax(np.abs(a - b)))
    mine["reproduces_D148b"] = chk
    assert max(chk.values()) < 1e-9, f"fold {k}: final HMM row does not reproduce D148b: {chk}"
    p.write_text(json.dumps(mine, default=float))
    log.warning("fold %d gold: reproduces D148b %s", k, chk)
    return p


# ---------------------------------------------------------------- rows 14-15 (stitching.phase0, cleaned)
def stage_phase0() -> None:
    from stitching.data import load_fold
    from stitching.metrics import evaluate_path, fold_entropy
    from stitching.phase0 import PHASE0_ROWS, EveryCut, NeverSwitch
    for k in range(5):
        F = load_fold(k)
        te = F["episodes"]["test"]
        out = {"fold": k, "methods": {}}
        for key, st in (("every_cut", EveryCut()), ("never_switch", NeverSwitch())):
            paths = [st.timed(e) for e in te]
            rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
            out["methods"][key] = {"name": dict(PHASE0_ROWS)[key], "params": {}, "rows": rows,
                                   "entropy": fold_entropy(te, paths, F["gallery_test"])}
            log.warning("fold %d %s acc %.3f F1 %.3f", k, key, np.mean([r["track_acc"] for r in rows]),
                        np.mean([r["switch_f1"] for r in rows]))
        (OUT / f"fold{k}_phase0.json").write_text(json.dumps(out, default=float))


# ---------------------------------------------------------------- segmental Viterbi (stitching.extra_rows, cleaned)
def stage_semi(k: int) -> None:
    from stitching.data import load_fold
    from stitching.extra_rows import SEMI_ROWS
    from stitching.metrics import evaluate_path, fold_entropy
    from stitching.semi_viterbi import SemiViterbi
    F = load_fold(k)
    tr, va, te = (F["episodes"][s] for s in ("train", "val", "test"))
    st = SemiViterbi()
    params = st.fit(tr, va)
    paths = [st.timed(e) for e in te]
    rows = [dict(evaluate_path(e, p), key=e.key) for e, p in zip(te, paths)]
    log.warning("fold %d semi: %s | acc %.3f F1 %.3f", k, params, np.mean([r["track_acc"] for r in rows]),
                np.mean([r["switch_f1"] for r in rows]))
    out = {"fold": k, "methods": {"semi_viterbi": {"name": SEMI_ROWS[0][1], "params": params, "rows": rows,
                                                   "entropy": fold_entropy(te, paths, F["gallery_test"])}},
           "timelines": {"semi_viterbi": [[e.files[m] for m in p.mid] for e, p in zip(te, paths)]}}
    (OUT / f"fold{k}_semi.json").write_text(json.dumps(out, default=float))


# ---------------------------------------------------------------- row 2c change predictor (stitching.cue_bassl, cleaned)
def stage_tokens(device: str) -> Path:
    """As stitching.cue_bassl.build_tokens: every cleaned clip segmented out-of-fold by the cleaned C head of the fold
    where it is test (train-only clips: fold 0's head), at that fold's cleaned val theta."""
    import stitching.cue_bassl as cb
    from bgm.cue.tokens import clip_segments, segment_features
    from bgm.eval.store import FeatureStore
    from bgm.segmentation.bassl_features import BaSSL
    from bgm.segmentation.report import Fold
    from bgm.segmentation.retrieval_eval import segments_from_cuts
    from bgm.segmentation.train_head import SegData
    from exp_supp.rerank.clean_t3 import seg_cfg, seg_file
    cfg_seg = seg_cfg()
    data = SegData(cfg_seg, need_siglip=True)
    store = FeatureStore(cfg_seg)
    th = json.loads(seg_file().read_text())["thetas"]
    test_of = {p: int(k) for k, f in data.folds["folds"].items() for p in f["test"]}
    allp = list(store.pairs.pair_id)
    bassl = BaSSL(device)
    out = {}
    for k in range(5):
        clips = [p for p in allp if test_of.get(p, 0) == k]
        data.frozen = lambda kk, split, c=clips: [[p] for p in c]          # noqa: E731
        fo = Fold(data, bassl, k, "x", {"combo": "combo"})
        for i, p in enumerate(clips):
            c = fo.chains("combo")[i]
            segs = segments_from_cuts(fo.cuts("combo", i, float(th[str(k)]["combo"]), False), float(c.duration))
            v = store.video(p)
            dyn = np.asarray(v["dyn_25hz"])
            out[p] = segment_features(v["sem_siglip_1hz"], dyn[:, 0], clip_segments(segs, len(dyn)))
        log.warning("tokens fold %d: %d clips", k, len(clips))
    np.savez(cb.TOKENS, **out)
    return cb.TOKENS


def stage_cue_train() -> None:
    import stitching.cue_bassl as cb
    cfg = cb.normal_cfg()
    assert cfg.paths.runs == "runs/cleaned" and cfg.cue.exp == "cue_bassl", cfg.paths.runs
    cb.train("normal", list(range(5)))


def attach_p_clean(k: int, eps, split: str) -> None:
    """stitching.cue_bassl.attach_p with the cleaned segment file (val / test parts of the cleaned sequences)."""
    import stitching.cue_bassl as cb
    from bgm.eval.store import FeatureStore
    from exp_supp.rerank.clean_t3 import seg_file
    cfg = cb.normal_cfg()
    assert cfg.paths.runs == "runs/cleaned"
    store = FeatureStore(cfg)
    seg = json.loads(seg_file().read_text())["folds"][str(k)][split]
    raws = [cb.episode_tokens(store, seg[int(e.key.split("/")[-1])]["parts"], e.segs) for e in eps]
    for e, r, p in zip(eps, raws, cb.p_change(cfg, k, raws)):
        assert len(r) == e.T and len(p) == e.T - 1
        e.p_change = np.asarray(p, float)


# ---------------------------------------------------------------- report
SUFFIXES = (("gold", "stitching.gold_prior", "GOLD_ROWS"), ("phase0", "stitching.phase0", "PHASE0_ROWS"),
            ("phase2", "stitching.phase2", "PHASE2_ROWS"), ("semi", "stitching.extra_rows", "SEMI_ROWS"))
FINAL = "hmm_gold_tuned"
DMETRICS = ("track_acc", "switch_f1", "TBIoU", "switches_per_min", "jumps_per_min", "align_success")


def load_all():
    import importlib
    import stitching.evaluate as ev
    folds = [json.loads((OUT / f"fold{k}.json").read_text()) for k in range(5)]
    rows = list(ev.ROWS)
    for suf, mod, name in SUFFIXES:
        if all((OUT / f"fold{k}_{suf}.json").exists() for k in range(5)):
            rows += getattr(importlib.import_module(mod), name)
            for k, f in enumerate(folds):
                g = json.loads((OUT / f"fold{k}_{suf}.json").read_text())
                f["methods"].update(g["methods"])
                if suf == "gold":
                    f["gold_prior"], f["reproduces_D148b"] = g["gold_prior"], g["reproduces_D148b"]
    return folds, rows


def stage_report() -> None:
    import stitching.report as rp
    from scipy.stats import wilcoxon
    folds, ROWS = load_all()
    # stitching.report.main writes main_table.md / wilcoxon.md / plots; the D143 line needs a placeholder for cleaned data
    for k in range(5):
        p = OUT / f"fold{k}.json"
        d = json.loads(p.read_text())
        if "reference_check_handset_vs_D143" not in d:
            d["reference_check_handset_vs_D143"] = {"params_equal": "n/a (cleaned)", "max_abs_diff_objective": float("nan")}
            p.write_text(json.dumps(d, default=float))
    true_rate = float(np.mean([np.mean([r["true_switches_per_min"] for r in f["methods"]["greedy"]["rows"]]) for f in folds]))
    orig_derive = rp._derive

    def derive(r):                         # |switches/min − true rate| with the cleaned true rate
        r = orig_derive(r)
        r["rate_err"] = abs(r["switches_per_min"] - true_rate)
        return r
    rp._derive = derive
    rp.COLS = [(c, (f"|switches/min − {true_rate:.2f}|" if c == "rate_err" else n), d) for c, n, d in rp.COLS]
    rp.main()
    md = (OUT / "main_table.md").read_text()
    md = md.replace("# Stitching ablation (title split, 5 folds;",
                    "# Stitching ablation, cleaned dataset (1,011 pairs; title split, 5 folds;")
    md = md.replace("- Hand-set row reproduces D143's combo/own R0_tuned exactly in every fold: " +
                    ", ".join(f"f{k} n/a (cleaned)/nan" for k in range(5)),
                    "- Lattices = D148b's cleaned Table 3 chain (runs/exp_supp/rerank/clean_t3/cache). Row 12 (final system) "
                    "reproduces D148b's cleaned final HMM exactly in every fold (max |Δ| per sequence: " +
                    ", ".join(f"f{f['fold']} {max(f['reproduces_D148b'].values()):.0e}" for f in folds) + ").")
    (OUT / "main_table.md").write_text(md)

    # pooled per-sequence rows
    pooled, names = {}, dict(ROWS)
    for f in folds:
        for key, _ in ROWS:
            r, _ = rp.per_episode(f["methods"][key])
            pooled.setdefault(key, []).extend(r)
    keys = [r["key"] for r in pooled[FINAL]]
    films = {k: v for f in folds for k, v in f["films"].items()}
    clusters = merge_clusters([films[k] for k in keys])
    n = len(keys)

    # decision (stitching.decision rule, unchanged) on the cleaned rows
    import stitching.decision as dec

    def wx(a, b, m):
        x = np.array([r[m] for r in a]) - np.array([r[m] for r in b])
        return float(x.mean()), (float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0)
    D = ["# Phase 3 decision and paired tests, cleaned dataset (1,011 pairs)", "",
         f"Same pre-registered rule as stitching/results/decision.md (stitching.decision, unchanged), on the {n} cleaned "
         "test sequences (5 folds pooled; CRF rows = mean over 3 seeds per sequence). Cross-fitted CRF rows were never "
         "completed on the original data either and are not run here.", ""]
    verdicts = []
    for c in dec.CANDIDATES:
        if c not in pooled:
            continue
        refs = ["two_stage"] + [r for r in dec.NEW[c] if r in pooled]
        ok_all, parts = True, []
        for r in refs:
            da, pa = wx(pooled[c], pooled[r], "track_acc")
            df, pf = wx(pooled[c], pooled[r], "switch_f1")
            ok = (da > 0 and pa < 0.05) and not (df < 0 and pf < 0.05)
            ok_all &= ok
            parts.append(f"vs {names[r]}: Δacc {da:+.3f} (p {pa:.2g}), ΔF1 {df:+.3f} (p {pf:.2g}) → {'pass' if ok else 'fail'}")
        verdicts.append((c, ok_all, parts))
    win = [c for c, ok, _ in verdicts if ok]
    D.insert(2, f"**Rule outcome: {', '.join(names[w] for w in win)} passes.**" if win else
             "**Rule outcome: no candidate (semi-Markov 2a / 2b / 2c, segmental Viterbi) passes against two-stage.**")
    D += ["", "## Rule check per candidate", ""]
    for c, ok, parts in verdicts:
        D.append(f"- **{names[c]}**: {'PASS' if ok else 'fail'}")
        D += [f"  - {x}" for x in parts]
    D += ["", "## Wilcoxon, every new row vs two-stage (row 3), greedy (row 1) and the final HMM (row 12)", "",
          "| row | vs | Δ track acc | p | Δ switch F1 | p |", "|---|---|---|---|---|---|"]
    for c in list(dec.NEW) + ["two_stage"]:
        if c not in pooled:
            continue
        for r in ("two_stage", "greedy", FINAL) + tuple(x for x in dec.NEW.get(c, []) if x in pooled):
            if r == c:
                continue
            da, pa = wx(pooled[c], pooled[r], "track_acc")
            df, pf = wx(pooled[c], pooled[r], "switch_f1")
            D.append(f"| {names[c]} | {names[r]} | {da:+.3f} | {pa:.2g} | {df:+.3f} | {pf:.2g} |")
    (OUT / "decision.md").write_text("\n".join(D) + "\n")

    # every row vs the final HMM, film-cluster bootstrap (exp_supp convention) -> exp_supp/stitch_clean_table.{md,tex}
    def fstat(key, m):
        per = []
        for f in folds:
            r, ent = rp.per_episode(f["methods"][key])
            per.append(ent if m == "entropy" else float(np.nanmean([x[m] for x in r])))
        return float(np.mean(per)), float(np.std(per))
    T = ["# Stitching study on the cleaned dataset (1,011 pairs), every row vs the final system", "",
         f"{n} test sequences, 5 title folds, D148b lattices; every hyperparameter tuned on the fold's val sequences. Mean ± std "
         "over folds for track acc. and switch F1, fold means elsewhere. Δ = row − final HMM per sequence, 95% cluster bootstrap "
         f"over film groups ({len(set(clusters))} clusters), 10,000 resamples; * = CI excludes 0.", "",
         "| # | Decoder | Track acc. | Switch F1 ±1 s | TBIoU | Sw./min | Jumps/min | Align. succ. | Δ track acc. [95% CI] | Δ switch F1 [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    deltas = {}
    for i, (key, nm) in enumerate(ROWS, 1):
        P = pooled[key]
        assert [r["key"] for r in P] == keys
        deltas[key] = {m: paired_ci(np.array([a[m] - b[m] for a, b in zip(P, pooled[FINAL])], float), clusters)
                       for m in ("track_acc", "switch_f1")}
        a, b = fstat(key, "track_acc"), fstat(key, "switch_f1")
        rest = " | ".join(f"{fstat(key, m)[0]:.{2 if 'min' in m else 3}f}" for m in DMETRICS[2:])
        dd = "—" if key == FINAL else None
        T.append(f"| {i} | {'**' + nm + ' (final)**' if key == FINAL else nm} | {a[0]:.3f} ± {a[1]:.3f} | {b[0]:.3f} ± {b[1]:.3f} | {rest} | "
                 f"{dd or fmt_ci(deltas[key]['track_acc'])} | {dd or fmt_ci(deltas[key]['switch_f1'])} |")
    T += ["", f"True changes {true_rate:.2f} / min. Shortlist track R@10 "
          f"{np.mean([f['shortlist']['track_R@10'] for f in folds]):.3f} ± {np.std([f['shortlist']['track_R@10'] for f in folds]):.3f}."]
    TABLE.with_suffix(".md").write_text("\n".join(T) + "\n")
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Stitching decoders on the cleaned dataset (1{,}011 pairs, %d test sequences, 5 title folds). " % n
           + r"$\Delta$ vs.\ the final system (HMM, gold-path prior, $\alpha$ tuned); bold = 95\% film-level cluster-bootstrap CI excludes 0.}",
           r"\label{tab:supp_stitch_cleaned}", r"\begin{tabular}{@{}rlcccccccc@{}}", r"\toprule",
           r"\# & Decoder & Acc. & F1 & TBIoU & Sw./min & J./min & Align. & $\Delta$Acc. & $\Delta$F1 \\", r"\midrule"]
    sig = lambda r: r["lo"] > 0 or r["hi"] < 0  # noqa: E731
    for i, (key, nm) in enumerate(ROWS, 1):
        cells = [f"{fstat(key, m)[0]:.{2 if 'min' in m else 3}f}" for m in DMETRICS]
        dl = ["--", "--"] if key == FINAL else [
            (r"$\mathbf{%+.3f}$" if sig(deltas[key][m]) else r"$%+.3f$") % deltas[key][m]["mean"] for m in ("track_acc", "switch_f1")]
        name = nm.replace("π", r"$\pi$").replace("λ", r"$\lambda$").replace("é", r"\'e")
        tex.append(f"{i} & {r'\textbf{' + name + '}' if key == FINAL else name} & " + " & ".join(cells + dl) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    TABLE.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print((OUT / "main_table.md").read_text().split("\n## ")[0])
    print("\n".join(T))
    print("\n".join(D[:20]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cue_stats", "eval", "gold", "phase0", "win10", "tokens", "cue_train", "phase2",
                                      "semi", "report"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--rows", default="two_stage_768,two_stage_3072,semi_sum,semi_pool")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    patch()
    ks = [a.fold] if a.fold is not None else list(range(5))
    if a.stage == "cue_stats":
        stage_cue_stats(a.jobs)
    elif a.stage == "eval":
        [stage_eval(k, a.device) for k in ks]
    elif a.stage == "gold":
        [stage_gold(k, a.device) for k in ks]
    elif a.stage == "phase0":
        stage_phase0()
    elif a.stage == "win10":
        import stitching.win10 as w10
        [w10.build(k, a.device) for k in ks]
    elif a.stage == "tokens":
        stage_tokens(a.device)
    elif a.stage == "cue_train":
        stage_cue_train()
    elif a.stage == "phase2":
        import stitching.phase2 as p2
        [p2.main(k, a.rows.split(",")) for k in ks]
    elif a.stage == "semi":
        [stage_semi(k) for k in ks]
    else:
        stage_report()
