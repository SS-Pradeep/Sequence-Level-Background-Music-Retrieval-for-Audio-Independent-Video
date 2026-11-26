"""Dynamics and mood branches with real fusion weight on the film split (PROGRESS.md D69).

Run: python -m bgm.analysis.arch_branches   -> reports/arch_branches.md + ledger rows
Pipelines (runs/filmsplit/<exp>, 5 folds x 3 seeds, dynamics encoders trained on the film split) against the
semantic-only pipeline of D60 (runs/filmsplit/pipeline), paired per test segment; hits averaged over seeds; 95%
cluster bootstrap over film groups.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import film_cfg
from bgm.data import manifest
from bgm.eval.final import _ms
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_pipeline_filmsplit import load_runs, seed_mean

log = logging.getLogger(__name__)
ROOT, OUT, B = Path("runs/filmsplit"), Path("reports/arch_branches.md"), 10000
EXPS = {"pipeline_dyn20": "semantic 0.8 + dynamics 0.2", "pipeline_attr20": "semantic 0.8 + mood 0.2",
        "pipeline_both20": "semantic 0.6 + dynamics 0.2 + mood 0.2", "pipeline_tuned": "weights tuned on val per run"}
KEYS = {"fused/track@1": "track@1", "fused/window@1": "window@1", "fused/window@J": "window@J", "offset/fused/1": "right track and within ±1 s"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    film = manifest.read_folds(film_cfg())["film_group"]
    base = load_runs(ROOT / "pipeline", "perquery.npz")
    fmt = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    dyn = [json.loads(p.read_text()) for p in sorted((ROOT / "dynamics" / "shiftneg").glob("fold*/seed*/metrics.json"))]
    L = ["# Dynamics and mood branches with real weight, film split (D69)", "",
         "1,248-pair film split, test-fold gallery 196 (195–199), 15 runs per pipeline. Dynamics encoders trained on the film split "
         f"(shift negatives). Paired Δ vs the semantic-only pipeline per test segment; 95% cluster bootstrap over film groups, {B:,} resamples.", ""]
    if dyn:
        t = dyn[0]["test"]
        L += ["Dynamics encoders alone, test grounding (find the offset within the segment's own music file): " +
              "; ".join(f"{k} {_ms([m['test'][k] for m in dyn])}" for k in t if isinstance(t[k], (int, float)))[:900] + ".", ""]
    L += ["| pipeline | " + " | ".join(KEYS.values()) + " |", "| --- | " + " | ".join("---" for _ in KEYS) + " |",
          "| semantic only (D60) | " + " | ".join(f"{seed_mean(base, k)[0].mean():.3f}" for k in KEYS) + " |"]
    rows, single = [], None
    for exp, name in EXPS.items():
        P = load_runs(ROOT / exp, "perquery.npz")
        if sum(len(v) for v in P.values()) < 15:
            continue
        cells, d1 = [], None
        for k in KEYS:
            a, pa = seed_mean(P, k, "pair")
            b, pb = seed_mean(base, k, "pair")
            assert (pa == pb).all()
            d = cluster_bootstrap(a - b, np.array([film[p] for p in pa]), B)
            d1 = d1 or d
            cells.append(f"{a.mean():.3f} ({fmt(d)})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        verdict = "KEEP" if d1["lo"] > 0 else ("REJECT (worse)" if d1["hi"] < 0 else "NULL")
        rows.append(f"| {date.today().isoformat()} | Branch weights: {name} (pipeline, per test segment) | film | 196 (195–199) | track@1 {seed_mean(P, 'fused/track@1')[0].mean():.3f} | "
                    f"window@J {seed_mean(P, 'fused/window@J')[0].mean():.3f} | window@1 {seed_mean(P, 'fused/window@1')[0].mean():.3f} | track@1 vs semantic only: **{d1['mean']:+.3f}** "
                    f"[{d1['lo']:+.3f}, {d1['hi']:+.3f}] | {verdict} |")
        single = single or P
        if exp == "pipeline_tuned":
            M = [json.loads(p.read_text()) for p in sorted((ROOT / exp).glob("fold*/seed*/metrics.json"))]
            wf = [m["fusion"]["w_fine"] for m in M]
            L.append(f"|   ↳ weights chosen on val (mean of {len(M)} runs) | " + ", ".join(f"{b} {np.mean([w[b] for w in wf]):.2f}" for b in ("sem", "dyn", "attr")) + " | | | |")
    if single:
        L += ["", "Each branch alone (same shortlist): " + "; ".join(
            f"{n} track@1 {seed_mean(single, s + '/track@1')[0].mean():.3f}, window@1 {seed_mean(single, s + '/window@1')[0].mean():.3f}"
            for s, n in (("sem", "semantic"), ("dyn", "dynamics"), ("attr", "mood"))) + ".", ""]
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
