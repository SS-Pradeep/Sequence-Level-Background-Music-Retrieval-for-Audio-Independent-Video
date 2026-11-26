"""Generic paired ledger evaluation of a film-split semantic variant against the siglip-both baseline heads.

python -m bgm.analysis.arch_paired <variant> --change "<ledger change column>" [--report NAME] [--decide track|segment]

Used by 1c and later ablations (1a has its own module with the layer-weight table). Protocol: paired per-query Δ
vs runs/filmsplit/semantic/siglip-both, hits averaged over each fold's seeds, 95% cluster bootstrap over film
groups, 5,000 resamples; verdict by the R@10 CI of the deciding section (ledger rule). Writes
reports/<report>.md and appends one ledger row to results/arch_ablation.md.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, FILM_ROOT, _per_query, film_cfg
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import Split, _cell, _gallery, _row, _runs

log = logging.getLogger(__name__)
STATS = {"R@1": lambda r: (r <= 1).astype(float), "R@10": lambda r: (r <= 10).astype(float), "MRR": lambda r: 1.0 / r}
TITLES = {"track_test_fold": "Track retrieval, test-fold gallery", "segment_all": "Segment retrieval, all windows"}


def paired_eval(variant: str, baseline: str = "siglip-both") -> dict:
    """{section: {gallery, abs, base, R@1, R@10, MRR (paired CIs)}} over the folds both variants completed."""
    split = Split(film_cfg(), "folds_film.json", "film")
    base = _runs(FILM_ROOT, baseline)
    new = _runs(FILM_ROOT, variant)
    common = sorted(set(base) & set(new))
    assert len(common) == 5, f"folds present for both: {common}"
    out = {}
    for sec in TITLES:
        row = {"gallery": _gallery(split, sec, False), "abs": _row(split, new, sec, False), "base": _row(split, base, sec, False)}
        for m, f in STATS.items():
            a, cl = _per_query(split, new, sec, f)
            b, _ = _per_query(split, base, sec, f)
            row[m] = cluster_bootstrap(a - b, cl, B)
        out[sec] = row
    return out


def write_outputs(out: dict, variant: str, change: str, report: str, decide: str) -> str:
    dec = out[decide]["R@10"]
    verdict = "KEEP" if dec["lo"] > 0 else ("REJECT (worse)" if dec["hi"] < 0 else "NULL")
    ci = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = [f"# {change}", "",
         f"runs/filmsplit/semantic/{variant} vs the siglip-both baseline heads: paired per-query Δ, hits averaged "
         f"over seeds, 95% cluster bootstrap over film groups, {B:,} resamples. Deciding section: {TITLES[decide]}.", ""]
    for sec, title in TITLES.items():
        r = out[sec]
        L += [f"## {title} (gallery {r['gallery']})", "",
              "| system | R@1 | R@10 | MRR |", "| --- | --- | --- | --- |",
              "| baseline heads | " + " | ".join(_cell(r["base"][m]) for m in ("R@1", "R@10", "MRR")) + " |",
              f"| {variant} | " + " | ".join(_cell(r["abs"][m]) for m in ("R@1", "R@10", "MRR")) + " |",
              "", f"Δ vs baseline: R@1 {ci(r['R@1'])}; R@10 {ci(r['R@10'])}; MRR {ci(r['MRR'])}", ""]
    L += [f"**Verdict ({TITLES[decide]} R@10 CI): {verdict}.**", ""]
    Path(f"reports/{report}.md").write_text("\n".join(L) + "\n")
    a, sec = out[decide]["abs"], decide
    note = "" if decide == "track_test_fold" else "; decided on segment R@10"
    led = (f"| {date.today().isoformat()} | {change} | film | {out[sec]['gallery']} | {_cell(a['R@1'])} | {_cell(a['R@10'])} | "
           f"{_cell(a['MRR'])} | {ci(out[sec]['R@10'])}{note} | {verdict} |")
    with open("results/arch_ablation.md", "a") as f:
        f.write(led + "\n")
    return verdict


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--change", required=True, help="ledger 'change' column text")
    ap.add_argument("--report", default=None, help="reports/<name>.md (default arch_<variant>)")
    ap.add_argument("--decide", choices=["track", "segment"], default="track")
    args = ap.parse_args()
    out = paired_eval(args.variant)
    decide = "track_test_fold" if args.decide == "track" else "segment_all"
    v = write_outputs(out, args.variant, args.change, args.report or f"arch_{args.variant}", decide)
    for sec in TITLES:
        d = out[sec]["R@10"]
        log.info("%s Δ R@10 %+.3f [%+.3f, %+.3f]", sec, d["mean"], d["lo"], d["hi"])
    log.info("verdict %s", v)


if __name__ == "__main__":
    main()
