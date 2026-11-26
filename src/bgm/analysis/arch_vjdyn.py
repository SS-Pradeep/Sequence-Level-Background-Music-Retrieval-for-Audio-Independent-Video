"""Dynamics branch with V-JEPA 2 on the video side against the hand-made curves (PROGRESS.md D70), film split.

Run: python -m bgm.analysis.arch_vjdyn   -> reports/arch_vjdyn.md + ledger rows
Grounding (the branch's own task): for each test segment, score every 25 Hz offset of its own music file and take
the best; exact = within 0.1 s, strict = within 0.5 s, sync = exact or a shift at which the music still lines up
with itself (D33), each against its per-query chance. Per-query values averaged over seeds; 95% cluster bootstrap.
Clusters are track groups (the saved per-query clusters); film groups contain them.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import film_cfg
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_pipeline_filmsplit import load_runs, seed_mean

log = logging.getLogger(__name__)
ROOT, OUT, B = Path("runs/filmsplit"), Path("reports/arch_vjdyn.md"), 10000
VAR = {"shiftneg": "hand-made curves (motion, cuts, luminance)", "vjepa-shiftneg": "V-JEPA 2 tubelet series"}
PIPE = {"pipeline_dyn20": "semantic 0.8 + dynamics 0.2, hand-made curves", "pipeline_vjdyn20": "semantic 0.8 + dynamics 0.2, V-JEPA 2"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fmt = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Dynamics branch: V-JEPA 2 video input against the hand-made curves (D70)", "",
         "1,248-pair film split, 5 folds × 3 seeds. Music side unchanged (onsets, beats, loudness, spectral change, novelty at 25 Hz). "
         "Video side: per-frame-pair V-JEPA 2 features (12.5 Hz, z-scored per file) instead of the 5 curves; same encoders, loss and "
         "shifted negatives.", "", "## Grounding: find the offset inside the segment's own music file", "",
         "| video input | rule | hit | chance | minus chance [95% CI] |", "| --- | --- | --- | --- | --- |"]
    G, rows = {}, []
    for v, name in VAR.items():
        R = load_runs(ROOT / "dynamics" / v, "perquery.npz")
        if sum(len(x) for x in R.values()) < 15:
            continue
        G[v] = R
        cl = np.concatenate([R[k][0]["clusters"] for k in sorted(R)])
        for rule in ("exact", "strict", "sync"):
            h, c = seed_mean(R, rule)[0], seed_mean(R, f"{rule}_chance")[0]
            L.append(f"| {name} | {rule} | {h.mean():.3f} | {c.mean():.3f} | {fmt(cluster_bootstrap(h - c, cl, B))} |")
    if len(G) == 2:
        cl = np.concatenate([G["shiftneg"][k][0]["clusters"] for k in sorted(G["shiftneg"])])
        L += ["", "V-JEPA 2 minus curves, paired: " + "; ".join(
            f"{r} {fmt(cluster_bootstrap(seed_mean(G['vjepa-shiftneg'], r)[0] - seed_mean(G['shiftneg'], r)[0], cl, B))}" for r in ("exact", "strict", "sync")) + "."]
        d = cluster_bootstrap(seed_mean(G["vjepa-shiftneg"], "sync")[0] - seed_mean(G["shiftneg"], "sync")[0], cl, B)
        rows.append(f"| {date.today().isoformat()} | Dynamics branch video input: V-JEPA 2 tubelet series vs hand-made curves (grounding sync, not R@k) | film | own file | "
                    f"exact {seed_mean(G['vjepa-shiftneg'], 'exact')[0].mean():.3f} | sync {seed_mean(G['vjepa-shiftneg'], 'sync')[0].mean():.3f} | strict "
                    f"{seed_mean(G['vjepa-shiftneg'], 'strict')[0].mean():.3f} | sync, V-JEPA − curves: **{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                    f"{'KEEP' if d['lo'] > 0 else ('REJECT (worse)' if d['hi'] < 0 else 'NULL')} |")
    film = manifest.read_folds(film_cfg())["film_group"]
    base = load_runs(ROOT / "pipeline", "perquery.npz")
    keys = {"fused/track@1": "track@1", "fused/window@1": "window@1", "fused/window@J": "window@J", "offset/fused/1": "right track and within ±1 s"}
    L += ["", "## Pipeline, per test segment (Δ vs semantic only, film-group CIs)", "", "| pipeline | " + " | ".join(keys.values()) + " |",
          "| --- | " + " | ".join("---" for _ in keys) + " |", "| semantic only (D60) | " + " | ".join(f"{seed_mean(base, k)[0].mean():.3f}" for k in keys) + " |"]
    for exp, name in PIPE.items():
        P = load_runs(ROOT / exp, "perquery.npz")
        if sum(len(x) for x in P.values()) < 15:
            continue
        cells = []
        for k in keys:
            a, pa = seed_mean(P, k, "pair")
            cells.append(f"{a.mean():.3f} ({fmt(cluster_bootstrap(a - seed_mean(base, k)[0], np.array([film[p] for p in pa]), B))})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
        if exp == "pipeline_vjdyn20":
            L += ["", "Dynamics branch alone in the pipeline: track@1 " + f"{seed_mean(P, 'dyn/track@1')[0].mean():.3f}, window@1 {seed_mean(P, 'dyn/window@1')[0].mean():.3f}."]
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
