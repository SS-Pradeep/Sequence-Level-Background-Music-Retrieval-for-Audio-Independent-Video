"""Seed ensembles evaluated like single runs, for the film-split final report (PROGRESS.md D99).

An ensemble averages its members' cosine scores per (query, candidate) (`MeanScorer`, the D65 rule) and is scored with
the semantic trainer's own `evaluate`, so its perquery.npz has the sections, rank conventions and clusters of a
trained run. Run seed e of an ensemble of size n uses member seeds n·e .. n·e + n − 1, so the run seeds are disjoint
ensembles (CLAUDE.md rule 11: 5 folds × 3 seeds).

Run: python -m bgm.eval.ensemble [name ...]   -> runs/filmsplit/semantic_ens/<name>/fold<k>/seed<e>/
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from bgm.eval.store import FeatureStore, fold_ids
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)
EXP = "semantic_ens"
SECTIONS = ("track_test_fold", "track_all", "segment_all", "segment_other")


@dataclass(frozen=True)
class EnsembleSpec:
    kind: str                  # "semantic" | "baselines": the folder under paths.runs
    variant: str
    size: int                  # members per ensemble
    run_seeds: tuple[int, ...]

    def members(self, e: int) -> list[int]:
        return list(range(self.size * e, self.size * (e + 1)))


SPECS = {"siglip-both-ens10": EnsembleSpec("semantic", "siglip-both", 10, (0, 1, 2)),
         "siglip-both-ens3": EnsembleSpec("semantic", "siglip-both", 3, (0,)),
         "cfd-ens3": EnsembleSpec("baselines", "cfd", 3, (0,)),
         "utcmvmr-ens3": EnsembleSpec("baselines", "utcmvmr", 3, (0,)),
         "mvpt-ens3": EnsembleSpec("baselines", "mvpt", 3, (0,)),
         **{f"siglip-both-{v}-ens10": EnsembleSpec("semantic", f"siglip-both-{v}", 10, (0,))     # D124 loss variants
            for v in ("sigmoid", "hnnce", "dcl", "vsemax")},
         **{f"{v}-ens10": EnsembleSpec("semantic", v, 10, (0,))                                     # D127 encoder ablations
            for v in ("siglip-clap", "siglip-mert", "clip-both")}}


def spec_of(name: str) -> EnsembleSpec:
    """A named spec, or for any other '<variant>-ens10' the 10-member semantic ensemble of that variant, run seed 0 (D135)."""
    if name in SPECS:
        return SPECS[name]
    assert name.endswith("-ens10"), name
    return EnsembleSpec("semantic", name.removesuffix("-ens10"), 10, (0,))


def load_member(cfg: DictConfig, store: FeatureStore, spec: EnsembleSpec, fold: int, seed: int, train_ids: list[str], device: str):
    run = Path(cfg.paths.runs) / spec.kind / spec.variant / f"fold{fold}" / f"seed{seed}"
    if spec.kind == "semantic":
        from bgm.analysis.full_dataset import load_scorer
        return load_scorer(run, store, device)
    from bgm.analysis.arch_ensemble import load_baseline_scorer
    return load_baseline_scorer(run, store, train_ids, device)


def evaluate_ensemble(cfg: DictConfig, store: FeatureStore, name: str, fold: int, e: int, device: str = "cuda") -> dict:
    """Score one ensemble on one fold's test clips and write it as a run; returns its test metrics."""
    from bgm.person.evaluate import MeanScorer
    from bgm.train.semantic import evaluate
    spec = spec_of(name)
    f = fold_ids(cfg, fold)
    members = [load_member(cfg, store, spec, fold, s, f["train"], device) for s in spec.members(e)]
    with torch.no_grad():
        metrics, per = evaluate(cfg, store, MeanScorer(members), f["test"])
    d = run_dir(cfg, f"{EXP}/{name}", fold, e)
    np.savez(d / "perquery.npz", **per)
    rc = OmegaConf.create(OmegaConf.to_container(cfg))
    rc.ensemble = {"name": name, "kind": spec.kind, "variant": spec.variant, "members": spec.members(e)}
    write_run(d, rc, {"test": metrics, "members": spec.members(e)}, e)
    log.info("%s fold %d seed %d (members %s): track R@10 %.3f, segment R@10 %.3f", name, fold, e, spec.members(e),
             metrics["track_test_fold"]["R@10"], metrics["segment_all"]["R@10"])
    return metrics


def run_ensembles(cfg: DictConfig, names: list[str], device: str = "cuda") -> None:
    store = FeatureStore(cfg)
    for name in names:
        for k in range(cfg.splits.n_folds):
            for e in spec_of(name).run_seeds:
                spec, root = spec_of(name), Path(cfg.paths.runs)
                if (root / EXP / name / f"fold{k}" / f"seed{e}" / "metrics.json").exists():
                    continue
                missing = [s for s in spec.members(e) if not (root / spec.kind / spec.variant / f"fold{k}" / f"seed{s}" / "metrics.json").exists()]
                if missing:
                    log.warning("%s fold %d seed %d: members %s not trained yet, skipped", name, k, e, missing)
                    continue
                evaluate_ensemble(cfg, store, name, k, e, device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from bgm.config import load_config
    over = [a for a in sys.argv[1:] if "=" in a]               # dotted overrides, e.g. splits.folds_file=folds_title.json (D102)
    names = [a for a in sys.argv[1:] if "=" not in a]
    run_ensembles(load_config("configs/default.yaml", ["splits.folds_file=folds_film.json", "paths.runs=runs/filmsplit", *over]),
                  names or list(SPECS))
