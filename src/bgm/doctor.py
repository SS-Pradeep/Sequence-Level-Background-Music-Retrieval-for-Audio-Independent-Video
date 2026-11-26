"""`bgm doctor`: report the environment the pipeline depends on."""
from __future__ import annotations

import importlib
import platform
import subprocess

from bgm.tools import find_tool

PACKAGES = ["torch", "torchvision", "torchaudio", "transformers", "numpy", "scipy", "pandas", "pyarrow", "sklearn",
            "librosa", "soundfile", "soxr", "cv2", "scenedetect", "pyloudnorm", "pedalboard", "panns_inference",
            "typer", "omegaconf", "pytest", "sentencepiece"]


def _version_line(tool: str, flag: str) -> str:
    path = find_tool(tool)
    if not path:
        return "MISSING"
    out = subprocess.run([path, flag], capture_output=True, text=True)
    line = (out.stdout or out.stderr).strip().splitlines()
    return f"{line[0] if line else '?'}  ({path})"


def report() -> list[str]:
    lines = [f"python      {platform.python_version()} ({platform.platform()})"]
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = (f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB"
               if cuda else "none")
        lines.append(f"torch       {torch.__version__}, CUDA {torch.version.cuda if cuda else 'unavailable'}, GPU {gpu}")
    except ImportError:
        lines.append("torch       MISSING")
    lines += [f"ffmpeg      {_version_line('ffmpeg', '-version')}",
              f"ffprobe     {_version_line('ffprobe', '-version')}",
              f"fpcalc      {_version_line('fpcalc', '-version')}"]
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            lines.append(f"  {name:<16}{getattr(mod, '__version__', 'installed')}")
        except Exception:  # noqa: BLE001 -- a broken import is reported, not raised
            lines.append(f"  {name:<16}MISSING")
    return lines
