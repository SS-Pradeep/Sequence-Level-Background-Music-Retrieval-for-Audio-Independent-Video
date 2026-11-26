"""Grounding with chance levels (`bgm eval --exp grounding`): re-score the grounding predictions the trained models
already made, under grounding_rules (exact ±0.1 s, strict = within 0.5 s, sync), with each rule's chance level.

No training. The M7 dynamics runs are re-scored from their saved per-query errors (no model loaded); the M9 fused
system needs inference only (the run's frozen checkpoints and fine weights); the hand-crafted baseline is
deterministic and recomputed. Sets and systems:
  real test folds (5 folds x 3 seeds): learned dynamics with / without shifted negatives, fused (M9), hand-crafted;
  synthetic beat-cut set (runs/synthetic_grounding, seed 0): learned dynamics, hand-crafted.
Stage 1 writes one per-query file per run (grounding_rescored.npz; grounding_fused.npz for M9); stage 2 aggregates
them: per query, hit = the rule's array at the predicted offset, chance = the array's mean; hits averaged over seeds
first; every mean with a 95% cluster-bootstrap CI over track groups; paired differences on (hit - chance).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from bgm.eval.metrics import GROUNDING_RULES, RULE_LABELS, cluster_bootstrap, top_offsets
from bgm.eval.protocol import GroundingQuery, grounding_queries, score_predictions
from bgm.eval.store import FeatureStore, fold_ids
from bgm.features.dynamics import M_CHANNELS
from bgm.train.dynamics import grounding_arrays
from bgm.train.shifts import shift_in_sync

log = logging.getLogger(__name__)
RATE = 25
SYSTEMS = {"shiftneg": "learned dynamics, shifted negatives (M7)", "noshift": "learned dynamics, no shifted negatives (M7)",
           "fused": "fused semantic + dynamics + mood (M9)", "hand": "hand-crafted dynamics (cut-on-beat + motion/energy)"}


# ---------------------------------------------------------------- stage 1: per-run predictions -> per-query hits
def check_alignment(store, queries: list[GroundingQuery], saved: dict, pred: np.ndarray, cfg, where: str) -> None:
    """The rebuilt queries must be the ones the run scored: its saved strict and (old) sync hits must be reproduced
    exactly from the predicted offsets, with the old definitions (protocol.grounding_legacy)."""
    true = np.array([q.true for q in queries])
    err = (pred - true) / RATE
    strict_old = np.abs(err) <= cfg.eval.grounding_tol_s
    assert np.array_equal(strict_old, np.asarray(saved["strict"]).astype(bool)), f"{where}: strict not reproduced"
    d = cfg.train.dynamics
    sync_old = np.array([abs(e) <= cfg.eval.sync_tol_s or shift_in_sync(
        store.dyn("music", q.music_id)[:, M_CHANNELS.index("log_onset")], store.music(q.music_id)["beat_times_s"], q.true, q.L,
        int(p - q.true), RATE, d.guard_s, d.max_ac, d.ac_ctx_steps) for q, e, p in zip(queries, err, pred)])
    assert np.array_equal(sync_old, np.asarray(saved["sync"]).astype(bool)), f"{where}: sync not reproduced"


def rescore_saved_dynamics(cfg, store, variant: str, queries: dict) -> None:
    """runs/dynamics/<variant>/fold*/seed*: predicted offsets from the saved errors (no model loaded)."""
    for k, qs in queries.items():
        for s in cfg.train.seeds:
            d = Path(cfg.paths.runs) / "dynamics" / variant / f"fold{k}" / f"seed{s}"
            z = dict(np.load(d / "perquery.npz"))
            assert len(z["err_s"]) == len(qs), (d, len(z["err_s"]), len(qs))
            pred = np.array([q.true for q in qs]) + np.round(z["err_s"] * RATE).astype(int)
            check_alignment(store, qs, z, pred, cfg, str(d))
            np.savez(d / "grounding_rescored.npz", **grounding_arrays(score_predictions(store, qs, pred, cfg)))


def hand_predictions(cfg, store, queries: list[GroundingQuery]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Hand-crafted dynamics (deterministic; hand_offsets needs no calibration): first argmax and the tied top offsets."""
    from bgm.eval.zeroshot import ZeroShot
    zs = ZeroShot.__new__(ZeroShot)
    zs.store, zs.cfg = store, cfg
    scores = [zs.hand_offsets(q.pair_id, q.a, q.b, q.music_id) for q in queries]
    return np.array([int(np.argmax(s)) for s in scores]), [top_offsets(s) for s in scores]


def rescore_hand(cfg, store, queries: dict) -> None:
    """runs/zeroshot/fold*: recomputed, and checked against the hits the fold's run saved."""
    for k, qs in queries.items():
        d = Path(cfg.paths.runs) / "zeroshot" / f"fold{k}"
        z = dict(np.load(d / "perquery.npz"))
        pred, tied = hand_predictions(cfg, store, qs)
        check_alignment(store, qs, {"strict": z["grounding/hand_dynamics/strict"], "sync": z["grounding/hand_dynamics/sync"]},
                        pred, cfg, str(d))
        np.savez(d / "grounding_rescored.npz", **grounding_arrays(score_predictions(store, qs, pred, cfg, tied)))


def fused_offset(index, sq, q: GroundingQuery, w_fine: dict) -> tuple[int, np.ndarray]:
    """Best fused offsets of one query over every 25 Hz offset of its own file: raw branch scores of the windows,
    each branch standardized over the offsets (SPEC §5.8), weighted sum; (first argmax, offsets tied at the top).
    With w_fine dyn = 0 the score only changes at 1 Hz feature boundaries, so ties are common."""
    from bgm.retrieval.pipeline import fuse, zstats
    n_off = len(index.store.music(q.music_id)["dyn_25hz"]) - q.L + 1
    raw = index.window_scores(sq, np.full(n_off, index.pos[q.music_id]), np.arange(n_off) / RATE, q.b - q.a)
    s = fuse(raw, zstats(raw), w_fine)
    return int(np.argmax(s)), top_offsets(s)


def _stale(out: Path, deps: list[Path]) -> bool:
    """Missing, older than a dependency, or scored before ties were handled (no n_tied)."""
    return (not out.exists() or any(p.stat().st_mtime > out.stat().st_mtime for p in deps)
            or "n_tied" not in np.load(out).files)


@torch.no_grad()
def rescore_fused(cfg, store, queries: dict) -> None:
    """runs/pipeline/fold*/seed*: the run's frozen models and fine weights (inference only). Skipped when
    grounding_fused.npz is newer than fusion.json and both checkpoints."""
    from bgm.retrieval.index import FoldModels, MusicIndex, pair_query
    runs = Path(cfg.paths.runs)
    for k, qs in queries.items():
        f = fold_ids(cfg, k)
        for s in cfg.train.seeds:
            run = runs / "pipeline" / f"fold{k}" / f"seed{s}"
            out = run / "grounding_fused.npz"
            deps = [run / "fusion.json", runs / "semantic" / cfg.pipeline.sem_variant / f"fold{k}" / f"seed{s}" / "ckpt.pt",
                    runs / "dynamics" / cfg.pipeline.dyn_variant / f"fold{k}" / f"seed{s}" / "ckpt.pt"]
            if not _stale(out, deps):
                continue
            w = json.loads((run / "fusion.json").read_text())["w_fine"]
            models = FoldModels.load(cfg, store, k, s, cfg.pipeline.sem_variant, cfg.pipeline.dyn_variant, train_ids=f["train"])
            index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
            vqs, pred, tied = {}, [], []
            for q in qs:
                if q.pair_id not in vqs:
                    vqs[q.pair_id] = pair_query(store, q.pair_id)
                    vqs[q.pair_id].prepare(models, store.min_span)
                p, t = fused_offset(index, vqs[q.pair_id].segment(q.a, q.b), q, w)
                pred.append(p), tied.append(t)
            g = score_predictions(store, qs, np.array(pred), cfg, tied)
            first = score_predictions(store, qs, np.array(pred), cfg)          # ties broken by the first offset (for comparison)
            np.savez(out, **grounding_arrays(g), **{f"{r}_first_index": first.hits[r] for r in GROUNDING_RULES})
            log.info("fused grounding fold %d seed %d: %d queries, %.0f%% with tied top offsets", k, s, len(qs),
                     100 * np.mean(g.n_tied > 1))
            del models, index
            torch.cuda.empty_cache()


def rescore_synthetic(root: Path) -> None:
    """runs/synthetic_grounding: the saved checkpoint's val predictions, and the hand-crafted baseline on the same queries."""
    cfg = OmegaConf.load(root / "config.yaml")
    m = json.loads((root / "metrics.json").read_text())
    store = FeatureStore(cfg)
    qs = grounding_queries(store, m["val_ids"])
    saved = json.loads((root / "val_predictions.json").read_text())
    assert [(q.pair_id, q.seg_idx, q.true) for q in qs] == [(r["pair_id"], r["seg_idx"], r["true_frame"]) for r in saved]
    pred = np.array([r["pred_frame"] for r in saved])
    check_alignment(store, qs, dict(np.load(root / "perquery.npz")), pred, cfg, str(root))
    np.savez(root / "grounding_rescored.npz", **grounding_arrays(score_predictions(store, qs, pred, cfg)))
    hp, ht = hand_predictions(cfg, store, qs)
    np.savez(root / "grounding_hand.npz", **grounding_arrays(score_predictions(store, qs, hp, cfg, ht)))


# ---------------------------------------------------------------- stage 2: aggregation
def _load(path: Path, hits: str = "") -> SimpleNamespace:
    """hits: key suffix of the hit arrays ("" = tie-aware; "_first_index" = ties broken by the first offset, fused only)."""
    z = dict(np.load(path))
    return SimpleNamespace(hits={r: z[r + hits] for r in GROUNDING_RULES}, chance={r: z[f"{r}_chance"] for r in GROUNDING_RULES},
                           true=z["true"], clusters=z["clusters"], err_s=z["err_s"], n_tied=z["n_tied"])


def summarize(results_by_fold: dict[int, list], clusters_by_fold: dict[int, np.ndarray], n_boot: int) -> dict:
    """results_by_fold: fold -> [per-seed results over the same queries]. Per query: hits averaged over seeds; then
    the folds pooled. Mean accuracy, chance and (hit - chance), each with a cluster-bootstrap CI."""
    out, per = {}, {}
    folds = sorted(results_by_fold)
    cl = np.concatenate([clusters_by_fold[k] for k in folds])
    for r in GROUNDING_RULES:
        hit = np.concatenate([np.mean([np.asarray(g.hits[r], float) for g in results_by_fold[k]], 0) for k in folds])
        ch = np.concatenate([np.mean([np.asarray(g.chance[r], float) for g in results_by_fold[k]], 0) for k in folds])
        per[r] = hit - ch
        out[r] = {"accuracy": cluster_bootstrap(hit, cl, n_boot), "chance": cluster_bootstrap(ch, cl, n_boot),
                  "excess": cluster_bootstrap(hit - ch, cl, n_boot)}
    out["_per_query_excess"], out["_clusters"] = per, cl
    out["n_queries"], out["n_clusters"] = int(len(cl)), int(len(np.unique(cl)))
    out["n_runs"] = int(sum(len(v) for v in results_by_fold.values()))
    return out


def paired(a: dict, b: dict, n_boot: int) -> dict:
    """a - b on per-query (hit - chance), same queries in the same order."""
    assert np.array_equal(a["_clusters"], b["_clusters"])
    return {r: cluster_bootstrap(a["_per_query_excess"][r] - b["_per_query_excess"][r], a["_clusters"], n_boot) for r in GROUNDING_RULES}


def _check_same_queries(runs: dict[str, dict[int, list]]) -> None:
    ref = next(iter(runs.values()))
    for name, by_fold in runs.items():
        for k, gs in by_fold.items():
            for g in gs:
                assert np.array_equal(g.true, ref[k][0].true) and np.array_equal(g.clusters, ref[k][0].clusters), (name, k)


def before_values(cfg) -> dict:
    """The previously reported numbers (old definitions): mean over runs (learned) or folds (hand) of strict, sync."""
    runs = Path(cfg.paths.runs)
    out = {}
    for v in ("shiftneg", "noshift"):
        M = [json.loads(p.read_text())["test"]["grounding"] for p in sorted(runs.glob(f"dynamics/{v}/fold*/seed*/metrics.json"))]
        out[v] = {r: float(np.mean([m[r] for m in M])) for r in ("strict", "sync")}
    Z = [json.loads(p.read_text())["grounding"]["hand_dynamics"] for p in sorted(runs.glob("zeroshot/fold*/metrics.json"))]
    out["hand"] = {r: float(np.mean([z[r] for z in Z])) for r in ("strict", "sync")}
    out["fused"] = {}
    return out


def aggregate(cfg: DictConfig) -> dict:
    n_boot, runs = cfg.eval.bootstrap_resamples, Path(cfg.paths.runs)
    folds, seeds = range(cfg.splits.n_folds), list(cfg.train.seeds)
    by = {v: {k: [_load(runs / "dynamics" / v / f"fold{k}" / f"seed{s}" / "grounding_rescored.npz") for s in seeds] for k in folds}
          for v in ("shiftneg", "noshift")}
    by["fused"] = {k: [_load(runs / "pipeline" / f"fold{k}" / f"seed{s}" / "grounding_fused.npz") for s in seeds] for k in folds}
    by["hand"] = {k: [_load(runs / "zeroshot" / f"fold{k}" / "grounding_rescored.npz")] for k in folds}
    _check_same_queries(by)
    cl = {k: by["hand"][k][0].clusters for k in folds}
    real = {name: summarize(by[name], cl, n_boot) for name in SYSTEMS}
    for name in SYSTEMS:
        real[name]["median_abs_err_s"] = float(np.median(np.abs(np.concatenate([g.err_s for k in folds for g in by[name][k]]))))
        real[name]["tied_share"] = float(np.mean(np.concatenate([g.n_tied > 1 for k in folds for g in by[name][k]])))
    first = {k: [_load(runs / "pipeline" / f"fold{k}" / f"seed{s}" / "grounding_fused.npz", "_first_index") for s in seeds] for k in folds}
    real["fused_first_index"] = summarize(first, cl, n_boot)
    real["n_all_within_0.5s"] = int(sum((by["hand"][k][0].chance["strict"] == 1).sum() for k in folds))
    real["paired"] = {"shiftneg − hand": paired(real["shiftneg"], real["hand"], n_boot),
                      "noshift − hand": paired(real["noshift"], real["hand"], n_boot),
                      "shiftneg − noshift": paired(real["shiftneg"], real["noshift"], n_boot),
                      "fused − hand": paired(real["fused"], real["hand"], n_boot)}
    real["before"] = before_values(cfg)
    root = runs / "synthetic_grounding"
    sl, sh = _load(root / "grounding_rescored.npz"), _load(root / "grounding_hand.npz")
    scl = {0: sl.clusters}
    syn = {"shiftneg": summarize({0: [sl]}, scl, n_boot), "hand": summarize({0: [sh]}, scl, n_boot)}
    syn["paired"] = {"shiftneg − hand": paired(syn["shiftneg"], syn["hand"], n_boot)}
    syn["hand"]["tied_share"] = float(np.mean(sh.n_tied > 1))
    m = json.loads((root / "metrics.json").read_text())
    syn["before"] = {"shiftneg": {"sync": m["previous_val_sync"]}, "hand": {}}
    syn["rerun"] = {"val_sync_old_definition": m["val_sync_at_best_epoch"], "previous_val_sync": m["previous_val_sync"],
                    "best_epoch": m["best_epoch"], "n_train_clips": m["train_clips"], "n_val_clips": len(m["val_ids"])}
    return {"real": real, "synthetic": syn}


def strip(d):
    """Drop the per-query arrays (keys starting with '_') before writing JSON."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def run_grounding_rescore(cfg: DictConfig, skip_scoring: bool = False) -> dict:
    if not skip_scoring:
        store = FeatureStore(cfg)
        queries = {k: grounding_queries(store, fold_ids(cfg, k)["test"]) for k in range(cfg.splits.n_folds)}
        for v in ("shiftneg", "noshift"):
            rescore_saved_dynamics(cfg, store, v, queries)
        rescore_hand(cfg, store, queries)
        log.info("re-scored the M7 dynamics runs and the hand-crafted baseline")
        rescore_fused(cfg, store, queries)
        rescore_synthetic(Path(cfg.paths.runs) / "synthetic_grounding")
    res = aggregate(cfg)
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grounding_chance.json").write_text(json.dumps(strip(res), indent=1))
    (out / "grounding_chance.md").write_text("\n".join(["# Grounding with chance levels", ""] + markdown(res)) + "\n")
    from bgm.eval.run import refresh_zeroshot_report
    refresh_zeroshot_report(cfg)                               # results/zeroshot.md: hand-crafted grounding, same rules
    for name in SYSTEMS:
        s = res["real"][name]
        log.info("%s: %s", name, "; ".join(f"{r} {s[r]['accuracy']['mean']:.3f} (chance {s[r]['chance']['mean']:.3f})" for r in GROUNDING_RULES))
    return res


# ---------------------------------------------------------------- tables
def _ci(b: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(b['mean'])} [{f.format(b['lo'])}, {f.format(b['hi'])}]"


def set_table(rows: list[tuple[str, dict, dict]]) -> list[str]:
    """One table for one query set: rows of (system label, summary, previously reported values)."""
    L = ["| system | rule | before | accuracy | chance | accuracy − chance (95% CI) |", "| --- | --- | --- | --- | --- | --- |"]
    for label, s, before in rows:
        for r in GROUNDING_RULES:
            b = f"{before[r]:.3f}" if r in before else "—"
            L.append(f"| {label} | {RULE_LABELS[r]} | {b} | {_ci(s[r]['accuracy'])} | {_ci(s[r]['chance'])} | **{_ci(s[r]['excess'], True)}** |")
    return L


RULES_NOTE = ("Each query: a test segment at its exact duration against every 25 Hz offset of its own music file. Rules: "
              "exact = |error| ≤ 0.1 s; within 0.5 s = |error| ≤ 0.5 s (key `strict`); sync = exact, or the predicted shift "
              "still lines up (`lines_up`: a guard of `sample_negative_shifts` fires). Per query: hit = the rule at the predicted "
              "offset (when several offsets tie at the top score, the rule's mean over them: the expected hit under a random "
              "tie-break); chance = the share of offsets the rule accepts (a uniformly random offset's hit rate). Hits are "
              "averaged over seeds per query, then pooled over folds; means with 95% cluster-bootstrap CIs over track groups "
              "(10,000 resamples). \"before\" = the number reported before this change (mean over runs; old definitions: strict "
              "the same, sync = |error| ≤ 0.1 s or the sampler guards, judged on the beat grid alone near file edges); exact is new.")


def markdown(res: dict, heading: str = "###") -> list[str]:
    real, syn = res["real"], res["synthetic"]
    r0 = real["shiftneg"]
    L = [RULES_NOTE, "",
         f"{heading} Real test folds ({r0['n_queries']} queries from {r0['n_clusters']} track groups; learned and fused: "
         "5 folds × 3 seeds, hand-crafted: 5 folds, deterministic)", ""]
    L += set_table([(SYSTEMS[n], real[n], real["before"].get(n, {})) for n in SYSTEMS])
    L += ["", "Paired differences in (hit − chance), same queries (chance cancels, so this is also the difference in accuracy):", ""]
    for name, p in real["paired"].items():
        L.append(f"- {name}: " + "; ".join(f"{RULE_LABELS[r].split(' (')[0]} {_ci(p[r], True)}" for r in GROUNDING_RULES))
    fi, fu = real["fused_first_index"], real["fused"]
    L += ["", "Median absolute error (s): " + ", ".join(f"{n} {real[n]['median_abs_err_s']:.2f}" for n in SYSTEMS) + ".",
          f"- Ties: {100 * fu['tied_share']:.0f}% of fused queries have several offsets at the top score (in the runs whose "
          "tuned w_fine gives the dynamics branch weight 0, the score changes only at 1 Hz feature boundaries); learned "
          f"dynamics and hand-crafted scores are continuous ({100 * real['hand']['tied_share']:.1f}% tied, hand-crafted). "
          "Breaking fused ties by the first offset instead (offset 0 is the true start of every video's first segment) "
          "would report " + "; ".join(f"{RULE_LABELS[r].split(' (')[0]} {fi[r]['accuracy']['mean']:.3f} "
                                     f"(excess {fi[r]['excess']['mean']:+.3f}, vs {fu[r]['excess']['mean']:+.3f})" for r in GROUNDING_RULES) + ".",
          f"- {real['n_all_within_0.5s']} of {r0['n_queries']} queries leave ≤ 1 s of slack between segment and file, so every "
          "offset is within 0.5 s (chance 1, excess 0); they raise accuracy and chance alike.", ""]
    ru = syn["rerun"]
    L += [f"{heading} Synthetic beat-cut set ({syn['shiftneg']['n_queries']} queries from {ru['n_val_clips']} val clips; "
          f"one learned run, seed 0)", ""]
    L += set_table([(SYSTEMS["shiftneg"].replace(" (M7)", ""), syn["shiftneg"], syn["before"]["shiftneg"]),
                    (SYSTEMS["hand"], syn["hand"], {})])
    p = syn["paired"]["shiftneg − hand"]
    L += ["", "- learned − hand-crafted: " + "; ".join(f"{RULE_LABELS[r].split(' (')[0]} {_ci(p[r], True)}" for r in GROUNDING_RULES),
          f"- The synthetic model is the only one retrained (user-approved; the original synthetic checkpoint had not been "
          f"saved): same generator, seed and settings as the M7 acceptance test ({ru['n_train_clips']} training clips, "
          f"max_epochs 80), checkpoint now kept in runs/synthetic_grounding/. Val sync {ru['val_sync_old_definition']:.3f} "
          f"(old definition, best epoch {ru['best_epoch']}) vs the previously reported {ru['previous_val_sync']:.3f} (the "
          "acceptance test's printout). These val clips also choose the epoch, so this set is a sanity check, not a "
          "held-out result; the CIs rest on 8 clips.", ""]
    return L
