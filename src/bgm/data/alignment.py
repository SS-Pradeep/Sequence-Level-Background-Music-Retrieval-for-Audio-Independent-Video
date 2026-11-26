"""Alignment audit (SPEC M4): is each pair's music time-aligned with its video?

For every pair with eligible internal cuts, the excess cut-on-beat rate of its own music (rate minus the mean
over 20 circular shifts). If pairs are aligned, cuts sit on beats more than chance: the excess is positive
(Wilcoxon signed-rank, one-sided). Writes reports/alignment.md with histograms of segment length and count.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import wilcoxon

from bgm.data import manifest
from bgm.eval.metrics import excess_cut_on_beat
from bgm.features.extract import feature_path, load_features

log = logging.getLogger(__name__)


def run_alignment_audit(cfg: DictConfig) -> dict:
    pairs, shots, segs = manifest.read_pairs(cfg), manifest.read_shots(cfg), manifest.read_segments(cfg)
    rows = []
    for pid, mid in zip(pairs.pair_id, pairs.music_id):
        m = load_features(feature_path(cfg, "music", mid))
        assert "beat_times_s" in m, f"{mid}: run `bgm extract --what dyn_music` first"
        rng = np.random.default_rng([cfg.seed, len(rows)])
        rate, shifted, excess = excess_cut_on_beat(manifest.cut_times(shots, pid), m["beat_times_s"], m["meta"]["duration_s"],
                                                   rng, cfg.eval.cut_beat_tol_s, cfg.eval.circular_shift_draws)
        rows.append({"pair_id": pid, "rate": rate, "shifted": shifted, "excess": excess, "tempo_bpm": float(m["tempo_bpm"])})
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["excess"])
    nz = ok.excess[ok.excess != 0]
    stat = wilcoxon(nz, alternative="greater") if len(nz) > 10 else None
    res = {"pairs_with_cuts": len(ok), "pairs_without_eligible_cuts": int(df.excess.isna().sum()),
           "mean_rate": float(ok.rate.mean()), "mean_shifted": float(ok.shifted.mean()), "mean_excess": float(ok.excess.mean()),
           "share_positive": float((ok.excess > 0).mean()), "share_negative": float((ok.excess < 0).mean()),
           "wilcoxon_p_greater": float(stat.pvalue) if stat else float("nan")}
    rep = Path(cfg.paths.reports)
    rep.mkdir(parents=True, exist_ok=True)
    df.to_csv(rep / "alignment_per_pair.csv", index=False)
    _histograms(segs, rep / "segments_hist.png")
    L = segs.end_s - segs.start_s
    per = segs.groupby("pair_id").size()
    lines = ["# Alignment audit (M4)", "",
             f"- Pairs with eligible internal cuts: {res['pairs_with_cuts']} (no eligible cut: {res['pairs_without_eligible_cuts']})",
             f"- Cut-on-beat rate (±{cfg.eval.cut_beat_tol_s * 1000:.0f} ms): own music **{res['mean_rate']:.3f}**, "
             f"circularly shifted **{res['mean_shifted']:.3f}**, mean excess **{res['mean_excess']:+.3f}**",
             f"- Pairs with positive excess: {res['share_positive']:.1%}; negative: {res['share_negative']:.1%}",
             f"- Wilcoxon signed-rank (excess > 0, zeros dropped): p = {res['wilcoxon_p_greater']:.3g}", "",
             "## Segments", "",
             f"- {len(segs)} segments over {segs.pair_id.nunique()} videos; per video median {per.median():.0f}, max {per.max()}",
             f"- Length s: min {L.min():.2f}, 10% {L.quantile(.1):.2f}, median {L.median():.2f}, 90% {L.quantile(.9):.2f}, max {L.max():.2f}",
             f"- Segments shorter than {cfg.segments.min_shot_s} s: {int((L < cfg.segments.min_shot_s).sum())} (clips under 2 s only)",
             f"- Segments per video: {per.value_counts().sort_index().to_dict()}",
             "", "![segments](segments_hist.png)"]
    (rep / "alignment.md").write_text("\n".join(lines) + "\n")
    log.info("alignment: excess %+.3f, positive %.1f%%, Wilcoxon p = %.3g", res["mean_excess"], 100 * res["share_positive"], res["wilcoxon_p_greater"])
    return res


def _histograms(segs: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(segs.end_s - segs.start_s, bins=40, color="#4c72b0")
    ax[0].set(xlabel="segment length (s)", ylabel="segments")
    per = segs.groupby("pair_id").size()
    ax[1].hist(per, bins=np.arange(per.max() + 2) - 0.5, color="#55a868")
    ax[1].set(xlabel="segments per video", ylabel="videos")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
