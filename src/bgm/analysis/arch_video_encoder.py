"""Heads on temporal video features (PROGRESS.md D67): V-JEPA 2 alone and SigLIP + V-JEPA 2, against the SigLIP heads,
on the clip-level split (folds.json; track-group clusters) and the film split (folds_film.json; film-group clusters).

Run: python -m bgm.analysis.arch_video_encoder [suffix ...]   -> reports/arch_video_encoder.md + ledger rows
A suffix (e.g. -d2) adds the variants trained with that suffix (deeper heads).
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

from bgm.analysis.arch_phase0 import B, film_cfg
from bgm.config import load_config
from bgm.eval.final import _hit_at, _seed_mean
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.report_filmsplit import Split

log = logging.getLogger(__name__)
OUT = Path("reports/arch_video_encoder.md")
SECS = {"track_test_fold": "track", "segment_all": "segment"}
NAMES = {"vjepa-both": "V-JEPA 2 only", "siglipvjepa-both": "SigLIP + V-JEPA 2"}


def load(root: Path, variant: str, film: dict | None, split: Split | None):
    """{sec: (hit@10 per query averaged over seeds, hit@1, clusters)}; None if the variant has no complete runs."""
    d = root / "semantic" / variant
    if len(list(d.glob("fold*/seed*/perquery.npz"))) < 15:
        return None
    out = {}
    for sec in SECS:
        a10, ex = _seed_mean(str(d), "perquery.npz", [sec], _hit_at(10))
        a1, _ = _seed_mean(str(d), "perquery.npz", [sec], _hit_at(1))
        cl = ex[f"{sec}/clusters"] if split is None else np.concatenate([split.clusters(k, sec) for k in sorted(split.folds)])
        out[sec] = (a10[sec], a1[sec], cl)
    return out


def main(suffixes: list[str]) -> None:
    logging.basicConfig(level=logging.INFO)
    fmt = lambda d: f"**{d['mean']:+.3f}** [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# Temporal video encoder: V-JEPA 2 features against SigLIP frames (D67)", "",
         "Music side CLAP+MERT throughout; same heads and training as the baseline. V-JEPA 2 ViT-L sees 16 frames over about 1.9 s around "
         "each second. 5 folds × 3 seeds; hit@10 per query averaged over seeds; paired Δ vs the SigLIP heads of the same split, 95% cluster "
         f"bootstrap ({B:,} resamples) over track groups (clip-level split) or film groups (film split).", ""]
    rows = []
    splits = (("Clip-level split (folds.json: grouped by song; the same film can be in train and test)", Path("runs"), None, "clip-level", "≈ 250"),
              ("Film split (folds_film.json: no film in both train and test)", Path("runs/filmsplit"), Split(film_cfg(), "folds_film.json", "film"), "film", "196 (195–199)"))
    for title, root, split, tag, gal in splits:
        base = load(root, "siglip-both", None, split)
        L += [f"## {title}", "", "| video features | head | track R@1 | track R@10 | Δ R@10 vs SigLIP | segment R@10 | Δ R@10 vs SigLIP |", "| --- | --- | --- | --- | --- | --- | --- |",
              f"| SigLIP frames (baseline) | 1 hidden layer | {base['track_test_fold'][1].mean():.3f} | {base['track_test_fold'][0].mean():.3f} | — | {base['segment_all'][0].mean():.3f} | — |"]
        for suf in [""] + suffixes:
            for v, name in NAMES.items():
                r = load(root, v + suf, None, split)
                if r is None:
                    continue
                dt = cluster_bootstrap(r["track_test_fold"][0] - base["track_test_fold"][0], base["track_test_fold"][2], B)
                ds = cluster_bootstrap(r["segment_all"][0] - base["segment_all"][0], base["segment_all"][2], B)
                head = "1 hidden layer" if not suf else suf.replace("-d", "") + " hidden layers"
                L.append(f"| {name} | {head} | {r['track_test_fold'][1].mean():.3f} | {r['track_test_fold'][0].mean():.3f} | {fmt(dt)} | {r['segment_all'][0].mean():.3f} | {fmt(ds)} |")
                verdict = "KEEP" if dt["lo"] > 0 else ("REJECT (worse)" if dt["hi"] < 0 else "NULL")
                rows.append(f"| {date.today().isoformat()} | Video encoder: {name}, {head} (train.semantic.video_feature) | {tag} | {gal} | {r['track_test_fold'][1].mean():.3f} | "
                            f"{r['track_test_fold'][0].mean():.3f} | — | {fmt(dt)}; segment {fmt(ds)} | {verdict} |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    have = Path("results/arch_ablation.md").read_text()
    new = [r for r in rows if "|".join(r.split("|")[2:4]) not in have]
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(x + "\n" for x in new))
    log.info("wrote %s, %d new ledger rows", OUT, len(new))


if __name__ == "__main__":
    main(sys.argv[1:])
