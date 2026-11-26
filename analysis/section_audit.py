"""Section-level alignment audit: do scene boundaries coincide with musical section changes?

Scene boundaries: internal segment boundaries (segments.csv), > `edge_s` from either end.
Music events, per file:
  * novelty peaks: Foote novelty on sem_mert_1hz (checkerboard half-width 8 s, the SPEC value, and 4 s, suited
    to these short clips); local maxima above the file's median with prominence >= 0.5 x std, at k + 0.5 s;
  * loudness changes: |mean rms_db over the next 1 s - mean over the previous 1 s| (25 Hz), local maxima >= 3 dB,
    at least 1 s apart.
Events within `edge_s` of either end are dropped (the zero-padded novelty kernel is unreliable there), before and
after shifting. A boundary is a hit if an event lies within +/- tol_s. Chance: the same rate with the event times
circularly shifted by U(0, duration), 20 draws. Excess = rate - shifted rate, per pair; Wilcoxon (excess > 0) and a
cluster bootstrap over track groups; overall and by song / background label.
Run: python -m bgm.analysis.section_audit -> reports/section_alignment.md
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import wilcoxon

from bgm.config import load_config
from bgm.data import manifest
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.store import FeatureStore
from bgm.features.dynamics import M_CHANNELS, foote_novelty

log = logging.getLogger(__name__)


def novelty_peaks(mert_1hz: np.ndarray, half_width: int) -> np.ndarray:
    nov = foote_novelty(mert_1hz, half_width)
    if len(nov) < 3 or nov.std() < 1e-9:
        return np.zeros(0)
    idx, _ = find_peaks(nov, height=np.median(nov), prominence=0.5 * nov.std())
    return idx + 0.5


def loudness_changes(rms_db_25hz: np.ndarray, min_db: float = 3.0, rate: int = 25) -> np.ndarray:
    x = np.asarray(rms_db_25hz, float)
    if len(x) < 2 * rate + 1:
        return np.zeros(0)
    cs = np.r_[0.0, np.cumsum(x)]
    t = np.arange(rate, len(x) - rate)
    change = np.abs((cs[t + rate] - cs[t]) / rate - (cs[t] - cs[t - rate]) / rate)
    idx, _ = find_peaks(change, height=min_db, distance=rate)
    return (t[idx]) / rate


def hit_rate(bounds: np.ndarray, events: np.ndarray, tol: float) -> float:
    if not len(bounds):
        return np.nan
    if not len(events):
        return 0.0
    return float((np.abs(bounds[:, None] - events[None, :]).min(1) <= tol).mean())


def excess(bounds, events, dur, rng, tol=1.0, edge=1.0, draws=20) -> tuple[float, float, float]:
    keep = lambda e: e[(e > edge) & (e < dur - edge)]  # noqa: E731
    b = bounds[(bounds > edge) & (bounds < dur - edge)]
    if not len(b):
        return np.nan, np.nan, np.nan
    r = hit_rate(b, keep(events), tol)
    sh = float(np.mean([hit_rate(b, keep((events + rng.uniform(0, dur)) % dur), tol) for _ in range(draws)]))
    return r, sh, r - sh


def run(cfg) -> pd.DataFrame:
    store = FeatureStore(cfg)
    labels = pd.read_csv(manifest.manifests_dir(cfg) / "song_labels.csv", dtype={"music_id": str}).set_index("music_id")
    rows = []
    for i, p in enumerate(store.pairs.pair_id):
        m = store.music_of[p]
        dur = min(store.duration("video", p), store.duration("music", m))
        segs = store.segments[p]
        bounds = np.array([a for a, _ in segs[1:]], float)
        md = store.music(m)
        ev = {"novelty8": novelty_peaks(md["sem_mert_1hz"].astype(np.float32), 8),
              "novelty4": novelty_peaks(md["sem_mert_1hz"].astype(np.float32), 4),
              "loudness": loudness_changes(md["dyn_25hz"][:, M_CHANNELS.index("rms_db")])}
        ev["any4"] = np.sort(np.r_[ev["novelty4"], ev["loudness"]])
        row = {"pair_id": p, "track_id": store.track[p], "label": labels.loc[m, "label"], "n_bounds": int(((bounds > 1) & (bounds < dur - 1)).sum())}
        for name, e in ev.items():
            rng = np.random.default_rng([cfg.seed, i, len(name)])
            row[f"{name}_rate"], row[f"{name}_shifted"], row[f"{name}_excess"] = excess(bounds, e, dur, rng)
            row[f"{name}_events"] = int(len(e))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cfg) -> str:
    L = ["# Section-level alignment audit", "",
         "Do scene (segment) boundaries fall within ±1 s of musical section changes? Rate = share of eligible boundaries "
         "with a music event within ±1 s; shifted = the same with the music's events circularly shifted (20 draws); "
         "excess = rate − shifted, per pair. p: Wilcoxon signed-rank, excess > 0. CI: 95% cluster bootstrap over track groups.", ""]
    events = {"novelty8": "novelty peaks (8 s kernel, SPEC)", "novelty4": "novelty peaks (4 s kernel)",
              "loudness": "loudness changes ≥ 3 dB", "any4": "novelty (4 s) or loudness"}
    for subset, g in [("all pairs", df), *[(f"{lab}", df[df.label == lab]) for lab in ("song", "background", "uncertain")]]:
        L += [f"## {subset}", "", "| music event | pairs | rate | shifted | excess [95% CI] | pairs with excess > 0 | Wilcoxon p |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, title in events.items():
            d = g.dropna(subset=[f"{name}_excess"])
            if len(d) < 5:
                continue
            x = d[f"{name}_excess"].to_numpy()
            b = cluster_bootstrap(x, d.track_id.to_numpy(), cfg.eval.bootstrap_resamples)
            nz = x[x != 0]
            p = wilcoxon(nz, alternative="greater").pvalue if len(nz) > 10 else np.nan
            L.append(f"| {title} | {len(d)} | {d[f'{name}_rate'].mean():.3f} | {d[f'{name}_shifted'].mean():.3f} | "
                     f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] | {(x > 0).mean():.1%} | {p:.3g} |")
        L.append("")
    L += ["Labels (data/manifests/song_labels.csv): " + ", ".join(f"{k} {v}" for k, v in df.label.value_counts().items()) +
          f". Pairs with ≥ 1 eligible boundary: {int(df.n_bounds.gt(0).sum())} of {len(df)}."]
    return "\n".join(L) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    df = run(cfg)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/section_alignment_per_pair.csv", index=False)
    Path("reports/section_alignment.md").write_text(summarize(df, cfg))
    print(Path("reports/section_alignment.md").read_text())


if __name__ == "__main__":
    main()
