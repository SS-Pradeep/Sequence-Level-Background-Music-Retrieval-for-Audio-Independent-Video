"""Run directories (CLAUDE.md rule 10): config.yaml, metrics.json, git_commit.txt and the seed for every run."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from bgm.tools import REPO_ROOT


def git_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "no-git"     # PROGRESS.md D3


def run_dir(cfg: DictConfig, exp: str, fold: int | str, seed: int | None = None) -> Path:
    d = Path(cfg.paths.runs) / exp / f"fold{fold}"
    if seed is not None:
        d = d / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run(d: Path, cfg: DictConfig, metrics: dict, seed: int | None = None) -> None:
    OmegaConf.save(cfg, d / "config.yaml")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (d / "git_commit.txt").write_text(git_commit() + "\n")
    (d / "seed.txt").write_text(f"{seed}\n")
