"""Leave-one-out ablation of the final system on the cleaned title split (PROGRESS.md D127).

python -m bgm.analysis.ablation   -> results/cleaned/ablation.md

The final system (D116 + D121): 10-head ensemble of SigLIP → CLAP+MERT heads, semantic-only coarse-to-fine retrieval,
Viterbi stitching, cue-switch predictor (R4). Each row changes one component and is paired against the final system
at the stage that component acts on:
  * loss function (D124 runs): the 4 alternative losses against InfoNCE, ensembles and single heads, with the
    diagnostics of bgm.analysis.loss_why (why they lose);
  * retrieval (track / segment R@k): encoder swaps as 10-head ensembles (members 0-9, run seed 0) against
    siglip-both-ens10 run seed 0; single heads (seeds 0-9, hits averaged over seeds) against the same ensemble;
  * pipeline accuracy (per test segment, before Viterbi): runs/cleaned/pipeline_ens10 run seeds 0-2 against
    no coarse stage (the same runs' fused_no_coarse), single heads (runs/cleaned/pipeline), + dynamics / + mood;
  * stitching (test concatenations): Viterbi against greedy (the same runs) and against the single heads' Viterbi;
  * cue-switch (in-film boundary F1 on blocked test sequences, runs/cleaned/cue/eval, D122): R4 against the stored
    D75 penalties (R0) and penalties re-tuned on val sequences.
The pipeline is hard-wired to SigLIP → CLAP+MERT features (bgm.retrieval.index), so the encoder rows stop at retrieval.
Δ = ablated − final, per query averaged over seeds first; 95% cluster bootstrap over film groups (concatenations /
sequences for stitching and cue-switch), 10,000 resamples. Development numbers, not the final-table protocol.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bgm.analysis.cleaned_compare import CLEAN, make_cfg
from bgm.cue.report import Table, boot_f1, f1_of, load_evals
from bgm.eval.final import _ci, _ms
from bgm.eval.final_film import rank_runs
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import Split, _cell, _hits, _row
from bgm.eval.report_pipeline_filmsplit import load_runs, per_run, seed_mean

log = logging.getLogger(__name__)
ROOT = Path(CLEAN["paths.runs"])
OUT = Path(CLEAN["paths.results"]) / "ablation.md"
SEEDS = tuple(range(10))
FINAL_ENS = "siglip-both"
WHY = Path("reports/cleaned/loss/why.json")                     # python -m bgm.analysis.loss_why
LOSSES = (("siglip-both-sigmoid", "sigmoid", "Sigmoid (SigLIP pairwise)"), ("siglip-both-hnnce", "hn_nce", "Hard-negative InfoNCE (β 0.5)"),
          ("siglip-both-dcl", "dcl", "Decoupled contrastive (DCL)"), ("siglip-both-vsemax", "vse_max", "VSE++ max-violation hinge (m 0.2)"))
ENCODERS = (("siglip-clap", "− MERT (SigLIP → CLAP)"), ("siglip-mert", "− CLAP (SigLIP → MERT)"),
            ("clip-both", "CLIP for SigLIP (CLIP → CLAP+MERT)"))
PIPELINES = (("pipeline_ens10", "fused_no_coarse", "− coarse stage (every window of every file)"),
             ("pipeline", "fused", "− ensemble (single heads, seeds 0–2)"),
             ("pipeline_ens10_dyn20", "fused", "+ learned dynamics (fine 0.2)"),
             ("pipeline_ens10_attr20", "fused", "+ mood (coarse and fine 0.2)"))
STITCH = (("pipeline_ens10", "greedy", "greedy for Viterbi"), ("pipeline", "viterbi", "− ensemble (single heads, Viterbi)"))
CUE = (("R0_stored", "− cue-switch (stored D75 penalties)"), ("R0_tuned", "− cue-switch (penalties re-tuned on val sequences)"))


def verdict(b: dict) -> str:
    """Reading of an ablated − final CI."""
    if b["hi"] < 0:
        return "worse"
    return "better" if b["lo"] > 0 else "no significant difference"


def paired_seed_mean(a: dict, key_a: str, b: dict, key_b: str, pair_key: str, cluster_of, B: int) -> tuple[float, float, dict]:
    """(mean of a, mean of b, cluster bootstrap of a − b) over per-query values averaged over each fold's seeds.
    a, b: load_runs outputs; both must hold the same queries in the same order (asserted). cluster_of: pair -> cluster
    (a dict, or None to use the pair_key values themselves as clusters)."""
    assert sorted(a) == sorted(b), f"folds differ: {sorted(a)} vs {sorted(b)}"
    va, pa = seed_mean(a, key_a, pair_key)
    vb, pb = seed_mean(b, key_b, pair_key)
    assert np.array_equal(pa, pb), "the two systems hold different queries"
    cl = pa if cluster_of is None else np.array([cluster_of[p] for p in pa])
    return float(np.nanmean(va)), float(np.nanmean(vb)), cluster_bootstrap(va - vb, cl, B)


def _ret_delta(split: Split, a: dict, b: dict, sec: str, k: int, B: int) -> tuple[float, float, dict]:
    ha, cl = _hits(split, a, sec, k, False)
    hb, clb = _hits(split, b, sec, k, False)
    assert np.array_equal(cl, clb)
    return float(ha.mean()), float(hb.mean()), cluster_bootstrap(ha - hb, cl, B)


def _complete(runs: dict, n_seeds: int) -> bool:
    return sorted(runs) == [0, 1, 2, 3, 4] and all(len(v) == n_seeds for v in runs.values())


def section_loss(split: Split, B: int) -> tuple[list[str], list[tuple]]:
    final = rank_runs(ROOT, "semantic_ens", f"{FINAL_ENS}-ens10", (0,))
    single_final = rank_runs(ROOT, "semantic", FINAL_ENS, SEEDS)
    L = ["## 1. Loss function of the semantic heads", "",
         "Every setting except `train.semantic.loss` is the shipped recipe (D124; losses at literature defaults, not tuned). "
         "Δ against InfoNCE with the same seeds (single heads, seeds 0–9) or members (10-head ensemble, members 0–9).", "",
         "| loss | heads | track R@1 | track R@10 | track MRR | segment R@10 | Δ track R@10 [95% CI] | Δ segment R@10 [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = []
    for kind, name, n, ref, label in (("semantic_ens", f"{FINAL_ENS}-ens10", 1, None, "**InfoNCE (shipped)**"),
                                      ("semantic", FINAL_ENS, len(SEEDS), None, "**InfoNCE (shipped)**")):
        runs = final if kind == "semantic_ens" else single_final
        t, s = _row(split, runs, "track_test_fold", False), _row(split, runs, "segment_all", False)
        L.append(f"| {label} | {'10-head ensemble' if n == 1 else 'single, seeds 0–9'} | {_cell(t['R@1'])} | {_cell(t['R@10'])} | "
                 f"{_cell(t['MRR'])} | {_cell(s['R@10'])} | — | — |")
        for v, _, lname in LOSSES:
            r = rank_runs(ROOT, kind, f"{v}-ens10" if n == 1 else v, (0,) if n == 1 else SEEDS)
            base = final if n == 1 else single_final
            if not _complete(r, n):
                L.append(f"| {lname} | pending | | | | | | |")
                continue
            t, s = _row(split, r, "track_test_fold", False), _row(split, r, "segment_all", False)
            at, bt, ct = _ret_delta(split, r, base, "track_test_fold", 10, B)
            _, _, cs = _ret_delta(split, r, base, "segment_all", 10, B)
            L.append(f"| {lname} | {'10-head ensemble' if n == 1 else 'single, seeds 0–9'} | {_cell(t['R@1'])} | {_cell(t['R@10'])} | "
                     f"{_cell(t['MRR'])} | {_cell(s['R@10'])} | {_ci(ct)} | {_ci(cs)} |")
            if n == 1:
                summary.append((f"{lname} for InfoNCE", "retrieval", "track R@10", bt, at, ct))
    return L + [""] + why_lines(), summary


def why_lines() -> list[str]:
    """The D127 diagnostics (bgm.analysis.loss_why) as tables; the reading of them lives in PROGRESS.md."""
    if not WHY.exists():
        return ["### Why", "", f"Pending: run `python -m bgm.analysis.loss_why` ({WHY}).", ""]
    w = json.loads(WHY.read_text())
    p, h, hs, tr = w["push"], w["heads"], w["heads_std"], w["training"]
    names = [("siglip-both", "info_nce", "InfoNCE (shipped)")] + list(LOSSES)
    L = ["### Why: where each loss pushes", "",
         "Simulated training batches (one clip per video, film cap 30, batch 128) scored by the shipped InfoNCE heads (seeds 0–2, "
         "5 folds). For each video, the share of the loss's gradient on in-batch negative music that lands on music **from the "
         "same film**, and on **near-duplicates** of the video's own music (same film, frozen MERT cosine > "
         f"{w['near_dup_p95']['mert']:.3f} and CLAP > {w['near_dup_p95']['clap']:.3f}: the 95th percentiles of different-film "
         "pairs, the D52 rule). Uniform = the share a loss blind to difficulty would give (the share of negatives that are "
         "same-film). Same heads for every row, so only the loss's weighting differs. Sigmoid uses its initial bias (−10; "
         f"the trained bias is not saved). VSE++ rows count only videos whose hinge is active ({w['vse_active_rows']:.1%} of them).", "",
         "| loss | push on same-film music | push on near-duplicates | × uniform (same-film) |", "| --- | --- | --- | --- |",
         f"| uniform (reference) | {p['uniform']['same_film']:.3f} | {p['uniform']['near_dup']:.4f} | 1.0 |"]
    for _, key, lname in names:
        L.append(f"| {lname} | {p[key]['same_film']:.3f} | {p[key]['near_dup']:.4f} | {p[key]['same_film'] / p['uniform']['same_film']:.1f} |")
    L += ["", "### Why: train vs test, and the geometry each loss learns", "",
          "Each loss's own heads, seeds 0–2 × 5 folds (mean ± std over the 15). Train R@10: training clips against a size-matched "
          "random gallery of training music (as D52 0.4). Cosines and margin on the test fold; margin = true pair − hardest "
          "other-track music. Uniformity: log E exp(−2‖x − y‖²) of the test music embeddings (lower = more spread). Training: "
          "seeds 0–9.", "",
          "| loss | train R@10 | test R@10 | gap | true-pair cos | same-film neg cos | other-film neg cos | margin vs hardest | "
          "music uniformity | epochs run | best epoch | val segment R@10 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v, _, lname in names:
        a, sd, t = h[v], hs[v], tr[v]
        L.append(f"| {lname} | {a['train_R@10']:.3f} ± {sd['train_R@10']:.3f} | {a['test_R@10']:.3f} ± {sd['test_R@10']:.3f} | "
                 f"{a['train_R@10'] - a['test_R@10']:.3f} | {a['pos_cos']:.3f} | {a['neg_cos_same_film']:.3f} | "
                 f"{a['neg_cos_other_film']:.3f} | {a['margin_vs_hardest']:+.3f} | {a['uniformity']:.2f} | {t['epochs_run']:.1f} | "
                 f"{t['best_epoch']:.1f} | {t['val_segment_R@10']:.3f} |")
    return L + [""]


def section_retrieval(split: Split, B: int) -> tuple[list[str], list[tuple]]:
    final = rank_runs(ROOT, "semantic_ens", f"{FINAL_ENS}-ens10", (0,))
    single_final = rank_runs(ROOT, "semantic", FINAL_ENS, SEEDS)
    # (row name, runs, seeds per fold, reference runs, summary name or None)
    systems = [("**final: 10-head ensemble, SigLIP → CLAP+MERT**", final, 1, None, None)]
    systems += [(f"{name}, 10-head ensemble", rank_runs(ROOT, "semantic_ens", f"{v}-ens10", (0,)), 1, final, name)
                for v, name in ENCODERS]
    systems.append(("− ensemble (single heads SigLIP → CLAP+MERT, seeds 0–9)", single_final, len(SEEDS), final,
                    "− ensemble (single heads, seeds 0–9)"))
    systems += [(f"{name}, single heads", rank_runs(ROOT, "semantic", v, SEEDS), len(SEEDS), single_final, None)
                for v, name in ENCODERS]
    L = ["## 2. Encoders and ensemble (retrieval)", "",
         "Test clips of all 5 folds (track: test-fold gallery) and test segments (segment: all windows). Ensembles: members 0–9, "
         "one run per fold; single heads: seeds 0–9 per fold. Δ: encoder ensembles and single heads (SigLIP → CLAP+MERT) against "
         "the final ensemble; encoder single heads against SigLIP → CLAP+MERT single heads (same seeds).", "",
         "| system | runs | track R@1 | track R@10 | track MRR | segment R@10 | Δ track R@10 [95% CI] | Δ segment R@10 [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = []
    for name, runs, n, ref, short in systems:
        if not _complete(runs, n):
            L.append(f"| {name} | pending | | | | | | |")
            continue
        t, s = _row(split, runs, "track_test_fold", False), _row(split, runs, "segment_all", False)
        dt = ds = "—"
        if ref is not None and _complete(ref, len(next(iter(ref.values())))):
            at, bt, ct = _ret_delta(split, runs, ref, "track_test_fold", 10, B)
            _, _, cs = _ret_delta(split, runs, ref, "segment_all", 10, B)
            dt, ds = _ci(ct), _ci(cs)
            if short:
                summary.append((short, "retrieval", "track R@10", bt, at, ct))
        L.append(f"| {name} | {sum(len(v) for v in runs.values())} | {_cell(t['R@1'])} | {_cell(t['R@10'])} | {_cell(t['MRR'])} | "
                 f"{_cell(s['R@10'])} | {dt} | {ds} |")
    return L + [""], summary


def section_pipeline(film: dict, B: int) -> tuple[list[str], list[tuple]]:
    final = load_runs(ROOT / "pipeline_ens10", "perquery.npz")
    _, pairs = seed_mean(final, "fused/window@1", "pair")
    L = ["## 3. Pipeline accuracy", "",
         f"Per test segment ({len(pairs):,}), after the coarse and fine stages, before Viterbi. Track@1: the top window's file is "
         "in the right track group; window@1 / @J: the query's own file within 0.5 s of the aligned offset (J = 20). Final = "
         "pipeline_ens10, run seeds 0–2 (members 10e … 10e + 9).", "",
         "| system | runs | track@1 | window@1 | window@J | Δ track@1 [95% CI] | Δ window@1 [95% CI] | Δ window@J [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |",
         f"| **final: 10-head ensemble, semantic only** | {sum(len(v) for v in final.values())} | " +
         " | ".join(_ms(per_run(final, f"fused/{m}")) for m in ("track@1", "window@1", "window@J")) + " | — | — | — |"]
    summary = []
    for exp, key, name in PIPELINES:
        P = final if exp == "pipeline_ens10" else load_runs(ROOT / exp, "perquery.npz")
        if not _complete(P, 3):
            L.append(f"| {name} | pending | | | | | | |")
            continue
        cis = {m: paired_seed_mean(P, f"{key}/{m}", final, f"fused/{m}", "pair", film, B) for m in ("track@1", "window@1", "window@J")}
        L.append(f"| {name} | {sum(len(v) for v in P.values())} | " +
                 " | ".join(_ms(per_run(P, f"{key}/{m}")) for m in ("track@1", "window@1", "window@J")) + " | " +
                 " | ".join(_ci(cis[m][2]) for m in ("track@1", "window@1", "window@J")) + " |")
        a, b, c = cis["window@1"]
        summary.append((name, "pipeline", "window@1", b, a, c))
    return L + [""], summary


def section_stitch(B: int) -> tuple[list[str], list[tuple]]:
    final = load_runs(ROOT / "pipeline_ens10", "stitch_perquery.npz")
    metrics = (("switch_f1", "switch F1 ±1 s"), ("part_track_acc", "part track accuracy"), ("switches_per_min", "switches / min"))
    _, ccl = seed_mean(final, "concat/viterbi/switch_f1", "concat/clusters")
    L = ["## 4. Stitching", "",
         f"Test concatenations ({len(ccl)}: 3–4 test clips of different track groups joined end to end, concat_sets_title.json); "
         "penalties tuned per run on val concatenations. CIs resample concatenations.", "",
         "| system | runs | " + " | ".join(t for _, t in metrics) + " | " + " | ".join(f"Δ {t} [95% CI]" for _, t in metrics) + " |",
         "| --- | --- | " + " | ".join("---" for _ in range(2 * len(metrics))) + " |",
         f"| **final: Viterbi, 10-head ensemble** | {sum(len(v) for v in final.values())} | " +
         " | ".join(_ms(per_run(final, f"concat/viterbi/{k}")) for k, _ in metrics) + " | " + " | ".join("—" for _ in metrics) + " |"]
    summary = []
    for exp, kind, name in STITCH:
        S = final if exp == "pipeline_ens10" else load_runs(ROOT / exp, "stitch_perquery.npz")
        cis = {k: paired_seed_mean(S, f"concat/{kind}/{k}", final, f"concat/viterbi/{k}", "concat/clusters", None, B) for k, _ in metrics}
        L.append(f"| {name} | {sum(len(v) for v in S.values())} | " + " | ".join(_ms(per_run(S, f"concat/{kind}/{k}")) for k, _ in metrics) +
                 " | " + " | ".join(_ci(cis[k][2]) for k, _ in metrics) + " |")
        a, b, c = cis["switch_f1"]
        summary.append((name, "stitching", "switch F1 (concatenations)", b, a, c))
    return L + [""], summary


def section_cue(B: int) -> tuple[list[str], list[tuple]]:
    cfg = make_cfg(CLEAN)
    T = Table(load_evals(cfg))
    cl = T.clusters("blocked")
    final = T.counts("R4", "blocked", "in_film/1")
    L = ["## 5. Cue-switch predictor", "",
         f"In-film music-change boundary F1 (±1 s, pooled over sequences) on the {len(cl)} blocked test sequences of D122 "
         "(1–3 min of joined test clips; runs/cleaned/cue/eval). Emission = the final system's; R4 seeds 0–2 averaged. CIs "
         "resample film-group clusters of sequences.", "",
         "| system | boundary F1 [95% CI] | Δ [95% CI] |", "| --- | --- | --- |",
         f"| **final: R4 learned change probabilities** | {_ci_plain(boot_f1(final, cl))} | — |"]
    summary = []
    for s, name in CUE:
        c = T.counts(s, "blocked", "in_film/1")
        d = boot_f1(c, cl, final)
        L.append(f"| {name} | {_ci_plain(boot_f1(c, cl))} | {_ci(d)} |")
        summary.append((name, "cue-switch", "in-film boundary F1", f1_of(final), f1_of(c), d))
    return L + [""], summary


def _ci_plain(b: dict) -> str:
    return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def write_report() -> Path:
    cfg = make_cfg(CLEAN)
    B = int(cfg.eval.bootstrap_resamples)
    split = Split(cfg, "folds_title.json", "film")
    s0, r0 = section_loss(split, B)
    s1, r1 = section_retrieval(split, B)
    s2, r2 = section_pipeline(split.film, B)
    s3, r3 = section_stitch(B)
    s4, r4 = section_cue(B)
    L = ["# Ablation of the final system (D127)", "",
         "Generated by `python -m bgm.analysis.ablation` (src/bgm/analysis/ablation.py). Cleaned dataset (D115), title split, "
         "Konjum salangai train-only. Each row changes one component of the final system and is compared, paired, at the "
         f"stage that component acts on. Δ = ablated − final; 95% cluster bootstrap, {B:,} resamples. **Development numbers** "
         "(one ensemble run per fold for the encoder rows), not the 5 folds × 3 seeds final-table protocol.", "",
         "## Summary", "",
         "| change to the final system | stage | measure | final | changed | Δ [95% CI] | reading |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, stage, measure, f, a, c in r0 + r1 + r2 + r3 + r4:
        L.append(f"| {name} | {stage} | {measure} | {f:.3f} | {a:.3f} | {_ci(c)} | {verdict(c)} |")
    L += ["", "The pipeline reads SigLIP → CLAP+MERT features only, so the loss and encoder rows are measured at retrieval.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L + s0 + s1 + s2 + s3 + s4))
    log.info("wrote %s", OUT)
    return OUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    write_report()
