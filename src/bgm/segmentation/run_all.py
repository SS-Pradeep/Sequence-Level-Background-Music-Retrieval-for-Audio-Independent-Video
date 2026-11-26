"""Step 1 segmentation study driver (D140): `python -m bgm.segmentation.run_all <stage> [overrides]`.

Stages, in order:
  shots     TransNetV2 on every interim video -> data/interim/segmentation/shots_transnetv2.json (+ per-frame probs);
            PySceneDetect shots.csv in the same format -> shots_pyscenedetect.json; sanity report
  bassl     frozen BaSSL shot features per clip -> data/features/segmentation/bassl_shots.npz (config-hashed)
  train     boundary heads, 5 folds x 3 seeds -> runs/segmentation/<variant>/fold<k>/seed<s>/
  eval      the 5 variants on the frozen title-split val / test sequences -> results/segmentation.{csv,tex,md},
            per-boundary probabilities and segments -> data/interim/segmentation/
  figure    results/segmentation.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.config import load_config

log = logging.getLogger(__name__)

FOLDS_FILE = "folds_title.json"
SEQ_FILE = "cue_sequences_title.json"


def out_dir(cfg) -> Path:
    d = Path(cfg.paths.interim) / "segmentation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_shots(cfg) -> None:
    from bgm.data import manifest
    from bgm.data.prep import interim_video_path
    from bgm.segmentation import shots_transnet as st
    pairs = manifest.read_pairs(cfg)
    ids = list(pairs["pair_id"])
    d = out_dir(cfg)
    res = st.run({p: interim_video_path(cfg, p) for p in ids}, threshold=0.5)
    (d / "shots_transnetv2.json").write_text(json.dumps({p: res[p]["shots"] for p in ids}))
    np.savez_compressed(d / "transnetv2_prob.npz", **{p: res[p]["prob"] for p in ids})
    shots = manifest.read_shots(cfg)
    dur = dict(zip(pairs["pair_id"], pairs["video_duration_s"]))
    psd = {p: st.shots_from_cuts(manifest.cut_times(shots, p), res[p]["shots"][-1][3]) for p in ids}
    (d / "shots_pyscenedetect.json").write_text(json.dumps(psd))
    report_shots(cfg, {"transnetv2": {p: res[p]["shots"] for p in ids}, "pyscenedetect": psd}, dur)


def report_shots(cfg, det: dict, dur: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["# Shot detection sanity report (D140)", "", "| detector | shots | shots / min | median length s | mean length s | p90 s |",
             "| --- | --- | --- | --- | --- | --- |"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, shots in det.items():
        L = np.array([s[3] - s[2] for v in shots.values() for s in v])
        tot = sum(v[-1][3] for v in shots.values()) / 60
        lines.append(f"| {name} | {len(L)} | {len(L) / tot:.1f} | {np.median(L):.2f} | {L.mean():.2f} | {np.quantile(L, 0.9):.2f} |")
        ax.hist(np.clip(L, 0, 20), bins=80, range=(0, 20), histtype="step", lw=1.5, label=f"{name} ({len(L) / tot:.1f}/min)")
    ax.set_xlabel("shot length (s, clipped at 20)")
    ax.set_ylabel("shots")
    ax.legend(frameon=False)
    fig.tight_layout()
    rep = Path(cfg.paths.reports) / "segmentation"
    rep.mkdir(parents=True, exist_ok=True)
    fig.savefig(rep / "shot_lengths.png", dpi=130)
    (rep / "shots.md").write_text("\n".join(lines) + "\n")
    log.info("\n%s", "\n".join(lines))


def bassl_cache_path(cfg) -> Path:
    from bgm.segmentation.bassl_features import config_hash
    h = config_hash(out_dir(cfg) / "shots_transnetv2.json")
    return Path(cfg.paths.features) / "segmentation" / f"bassl_shots_{h}.npz"


def stage_bassl(cfg) -> None:
    from bgm.data.prep import interim_video_path
    from bgm.segmentation import bassl_features as bf
    path = bassl_cache_path(cfg)
    if path.exists():
        log.info("bassl shot features cached: %s", path)
        return
    shots = json.loads((out_dir(cfg) / "shots_transnetv2.json").read_text())
    feats = bf.run({p: interim_video_path(cfg, p) for p in shots}, shots)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **feats)
    log.info("bassl shot features: %d clips, %d shots -> %s", len(feats), sum(len(v) for v in feats.values()), path)


def stage_train(cfg) -> None:
    from bgm.segmentation.train_head import run_train
    run_train(cfg)


def stage_eval(cfg) -> None:
    from bgm.segmentation.report import run_eval
    run_eval(cfg)


def stage_figure(cfg) -> None:
    from bgm.segmentation.report import figure
    figure(cfg)


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stage, overrides = argv[0], argv[1:]
    cfg = load_config("configs/default.yaml", [f"splits.folds_file={FOLDS_FILE}", *overrides])
    {"shots": stage_shots, "bassl": stage_bassl, "train": stage_train, "eval": stage_eval, "figure": stage_figure}[stage](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
