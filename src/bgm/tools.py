"""Locating external tools (ffmpeg, ffprobe, fpcalc)."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_tool(name: str) -> str | None:
    """PATH first, then the repo-local `.checkpoints/bin/` (where a static fpcalc lives on this machine)."""
    found = shutil.which(name)
    if found:
        return found
    local = REPO_ROOT / ".checkpoints" / "bin" / name
    return str(local) if local.exists() else None
