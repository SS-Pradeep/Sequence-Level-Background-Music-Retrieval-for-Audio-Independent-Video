"""Config loading: configs/default.yaml plus dotted overrides (e.g. `train.semantic.lr_finetune=3e-4`)."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | tuple[str, ...] = ()) -> DictConfig:
    """Load a YAML config and apply `key.sub=value` overrides.

    The config is in struct mode, so an override naming a key that does not exist raises
    instead of silently adding a typo'd setting.
    """
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)
    bad = [o for o in overrides if "=" not in o]
    if bad:
        raise ValueError(f"overrides must look like key.sub=value, got {bad}")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
