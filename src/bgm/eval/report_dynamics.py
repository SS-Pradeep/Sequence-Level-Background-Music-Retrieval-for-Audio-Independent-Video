"""results/dynamics.md (SPEC M7, H2): learned vs hand-crafted dynamics, with vs without shifted negatives.

Grounding on the test folds with chance levels, from results/grounding_chance.json (`bgm eval --exp grounding`, which
re-scores the saved predictions; no training). Per-query rates are averaged over seeds; comparisons are paired, with
a cluster bootstrap over track groups.
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from bgm.eval.rescore_grounding import markdown


def load_grounding_chance(cfg) -> dict:
    p = Path(cfg.paths.results) / "grounding_chance.json"
    assert p.exists(), f"{p} is missing: run `bgm eval --exp grounding` first"
    return json.loads(p.read_text())


def _verdict(b: dict) -> str:
    return f"(Δ {b['mean']:+.3f}, CI [{b['lo']:+.3f}, {b['hi']:+.3f}])"


def h2_lines(res: dict) -> list[str]:
    real = res["real"]
    ex = real["shiftneg"]["sync"]["excess"]
    lh, sn = real["paired"]["shiftneg − hand"]["sync"], real["paired"]["shiftneg − noshift"]["sync"]
    return [f"- Learned dynamics (shifted negatives) above chance on sync: **{'yes' if ex['lo'] > 0 else 'not shown'}** {_verdict(ex)}.",
            f"- Learned dynamics beats hand-crafted on sync: **{'yes' if lh['lo'] > 0 else 'not shown'}** {_verdict(lh)}.",
            f"- Shifted negatives are necessary: **{'yes' if sn['lo'] > 0 else 'not shown'}** {_verdict(sn)}."]


def write_dynamics_report(cfg: DictConfig) -> Path:
    res = load_grounding_chance(cfg)
    run0 = json.loads((Path(cfg.paths.runs) / "dynamics" / "shiftneg" / "fold0" / "seed0" / "metrics.json").read_text())
    L = ["# Dynamics (M7, H2)", "", "Grounding with chance levels (`bgm eval --exp grounding`): the saved predictions of the "
         "trained real-data models are re-scored (no real-data model retrained; M9 fused needs inference only). v1 numbers "
         "(before the motion-feature fix, D27) remain in results/final_v1.md.", ""]
    L += markdown(res, "##")
    L += ["## H2", ""] + h2_lines(res)
    L += ["", f"Training: {run0['val']['too_short_for_8s']} of {run0['val']['train_clips']} fold-0 training clips are too short for "
          "an 8 s window (> 10%), so 6 s / 4 s length buckets are on (SPEC §3.5); no clip is skipped."]
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamics.md").write_text("\n".join(L) + "\n")
    return out / "dynamics.md"
