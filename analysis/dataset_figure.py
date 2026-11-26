"""Dataset figure: example pairs in film order, keyframes above the time-aligned music, music changes marked.

python -m bgm.analysis.dataset_figure   -> outputs/dataset_figure/{example_pairs.png,.pdf,rows.csv}

Each row is one stretch of a SunNXT upload (the only source with clip positions, meta_movie_start_s / _end_s). Every
coloured block is one pair: a silent video clip and the music that plays under it, sample 0 at its first frame (SPEC
§3.4). Consecutive cues use different tracks, so the music changes between them; the stretches between cues are not in
the dataset. Shot cuts inside a cue (shots.csv) are marked to show that the music keeps going across most cuts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.analysis.retrieval_figure import mel_db, style, thumb

log = logging.getLogger(__name__)
OUT = Path("outputs/dataset_figure")
ROWS = [("206", 1, 4), ("295", 0, 3), ("36", 0, 2), ("526", 2, 4), ("139", 1, 3)]   # (sunnxt film, first clip, last clip)
COLS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
SLOT_FRAC = 1 / 15                       # keyframe width as a fraction of the row's time span


def stretch(pairs: pd.DataFrame, film: str, i: int, j: int) -> list[dict]:
    """Clips i..j of one SunNXT upload in film order: pair_id, track_id, film-time start / end (s), shot cuts (film s)."""
    g = pairs[pairs.meta_film == film].copy()
    g["a"], g["b"] = g.meta_movie_start_s.astype(float), g.meta_movie_end_s.astype(float)
    g = g.sort_values("a").iloc[i:j + 1]
    return [dict(pair_id=r.pair_id, music_id=r.music_id, track_id=r.track_id, a=r.a, b=r.b) for r in g.itertuples()]


def crop_bars(img: np.ndarray, thr: float = 0.06) -> np.ndarray:
    """Drop letterbox rows (near-black top / bottom bands) from an [H, W, 3] frame."""
    x = img.astype(float) / (255.0 if img.dtype == np.uint8 else 1.0)
    bright = np.where(x.mean((1, 2)) > thr)[0]
    return img[bright[0]:bright[-1] + 1] if len(bright) else img


def single(film: str, i: int, j: int, name: str) -> Path:
    """One stretch of one film, large: keyframes (letterbox cropped, true aspect) above the time-aligned music,
    shot cuts as ticks, music changes in red. -> outputs/dataset_figure/<name>.png/.pdf"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("outputs/retrieval_figure/thumbs")
    pairs = pd.read_csv("data/manifests/pairs.csv", dtype=str, keep_default_na=False)
    pairs = pairs[pairs.meta_source == "sunnxt"]
    shots = pd.read_csv("data/manifests/shots.csv", dtype=str, keep_default_na=False)
    cuts_of = {p: [float(x) for x in c.split()] for p, c in zip(shots.pair_id, shots.cut_times_s)}
    t = pd.read_csv("outputs/film_titles.csv", dtype=str)
    title = dict(zip(t.source + ":" + t.source_id, t.title))[f"sunnxt:{film}"]
    cues = stretch(pairs, film, i, j)
    t0, t1 = cues[0]["a"], cues[-1]["b"]
    W, H = 16.0, 5.6
    fig = plt.figure(figsize=(W, H))
    left, right = 0.11, 0.99
    rows = {"band": (0.80, 0.07), "video": (0.47, 0.32), "cuts": (0.425, 0.04), "music": (0.13, 0.29)}
    ax = {k: fig.add_axes([left, y, right - left, h]) for k, (y, h) in rows.items()}
    for a in ax.values():
        a.set_xlim(t0 - 0.3, t1 + 0.3)
        a.axis("off")
    ax["band"].set_ylim(0, 1), ax["video"].set_ylim(0, 1), ax["cuts"].set_ylim(0, 1), ax["music"].set_ylim(0, 64)
    sec_per_in = (t1 - t0 + 0.6) / (W * (right - left))
    tile = 0.32 * H * 16 / 9 * sec_per_in                       # seconds covered by one 16:9 keyframe at full height
    gap = 0.12
    for c, q in enumerate(cues):
        col, letter, d = COLS[c], "ABCDE"[c], q["b"] - q["a"]
        ax["band"].add_patch(FancyBboxPatch((q["a"], 0.05), d, 0.9, boxstyle="round,pad=0,rounding_size=0.25", fc=col, ec="none"))
        ax["band"].text(q["a"] + d / 2, 0.5, f"Pair {c + 1}  ·  music {letter}  ·  {d:.1f} s", color="w", fontsize=13,
                        fontweight="bold", ha="center", va="center")
        k = max(1, int(round(d / tile)))
        w = d / k
        for m in range(k):
            img = crop_bars(thumb(q["pair_id"], cache, min(d - 0.2, (m + 0.5) * w)))
            ax["video"].imshow(img, extent=(q["a"] + m * w + gap / 2, q["a"] + (m + 1) * w - gap / 2, 0, 1), aspect="auto")
        cuts = cuts_of.get(q["pair_id"], [])
        ax["cuts"].plot(q["a"] + np.array(cuts), [0.5] * len(cuts), "^", color="#333333", ms=7)
        ax["cuts"].plot([q["a"], q["b"]], [0.5, 0.5], color=col, lw=1, alpha=0.6, zorder=0)
        M, dur = mel_db(q["music_id"], cache)
        ax["music"].imshow(M, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=(q["a"], q["a"] + min(dur, d), 0, 64))
        ax["music"].add_patch(Rectangle((q["a"], 0), d, 64, fill=False, ec=col, lw=3))
    for a, b in zip(cues[:-1], cues[1:]):
        x = (a["b"] + b["a"]) / 2
        for k in ("video", "cuts", "music"):
            top = 64 if k == "music" else 1
            ax[k].add_patch(Rectangle((a["b"], 0), b["a"] - a["b"], top, fc="#f2f2f2", ec="none", zorder=-1))
        fig.add_artist(plt.Line2D([left + (x - t0 + 0.3) / (t1 - t0 + 0.6) * (right - left)] * 2, [0.115, 0.79],
                                  color="#d62839", lw=2.5, ls=(0, (5, 3))))
        i0 = "ABCDE"[cues.index(a)]
        ax["band"].text(x, 1.25, f"music changes\n{i0} → {'ABCDE'[cues.index(b)]}", color="#d62839", fontsize=11.5,
                        fontweight="bold", ha="center", va="bottom", linespacing=1.1)
    lab = dict(fontsize=12.5, ha="right", va="center", color="#333333", transform=fig.transFigure)
    fig.text(left - 0.01, 0.63, "silent video\n(keyframes)", **lab)
    fig.text(left - 0.01, 0.445, "shot cuts", **lab)
    fig.text(left - 0.01, 0.275, "music\n(mel spectrogram)", **lab)
    tk = np.arange(np.ceil(t0 / 5) * 5, t1 + 1e-9, 5)
    axis = fig.add_axes([left, 0.115, right - left, 0.001])
    axis.set_xlim(t0 - 0.3, t1 + 0.3)
    axis.set_yticks([])
    axis.set_xticks(tk, [f"{x:.0f} s" for x in tk], fontsize=10.5)
    for sp in ("top", "right", "left"):
        axis.spines[sp].set_visible(False)
    fig.text(0.5 * (left + right), 0.035, f"time in the source video ({title})", ha="center", fontsize=12)
    fig.text(left, 0.985, f"{title}: {len(cues)} consecutive pairs", fontsize=15, fontweight="bold", va="top")
    fig.text(left, 0.925, "▲ shot cut inside a pair (the music continues)    grey: between cues, not in the dataset",
             fontsize=10.5, va="top", ha="left", color="#555555", fontfamily="DejaVu Sans")
    out = OUT / name
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    log.info("wrote %s", out.with_suffix(".png"))
    return out


def main() -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("outputs/retrieval_figure/thumbs")
    pairs = pd.read_csv("data/manifests/pairs.csv", dtype=str, keep_default_na=False)
    pairs = pairs[pairs.meta_source == "sunnxt"]
    shots = pd.read_csv("data/manifests/shots.csv", dtype=str, keep_default_na=False)
    cuts_of = {p: [float(x) for x in c.split()] for p, c in zip(shots.pair_id, shots.cut_times_s)}
    t = pd.read_csv("outputs/film_titles.csv", dtype=str)
    title = dict(zip(t.source + ":" + t.source_id, t.title))

    n = len(ROWS)
    fig = plt.figure(figsize=(16, 2.55 * n + 0.6))
    g = fig.add_gridspec(n, 2, width_ratios=[0.11, 1], hspace=0.55, wspace=0.01)
    table = []
    for r, (film, i, j) in enumerate(ROWS):
        cues = stretch(pairs, film, i, j)
        t0, t1 = cues[0]["a"], cues[-1]["b"]
        span = t1 - t0
        slot = span * SLOT_FRAC
        sub = g[r, 1].subgridspec(3, 1, height_ratios=[0.2, 1, 0.8], hspace=0.0)
        bar, kf, sp = (fig.add_subplot(sub[k]) for k in range(3))
        for ax in (bar, kf, sp):
            ax.set_xlim(t0 - 0.5, t1 + 0.5)
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
        for ax in (bar, kf):
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        bar.set_ylim(0, 1)
        kf.set_ylim(0, 1)
        sp.set_ylim(0, 64)
        letters = {}
        for c, q in enumerate(cues):
            col = COLS[c % len(COLS)]
            letter = letters.setdefault(q["track_id"], "ABCDE"[len(letters)])
            d = q["b"] - q["a"]
            bar.add_patch(Rectangle((q["a"], 0.08), d, 0.84, color=col, lw=0))
            text = f"pair {c + 1} · music {letter} · {d:.1f} s" if d / span > 0.16 else f"music {letter} · {d:.0f} s"
            bar.text(q["a"] + d / 2, 0.5, text, color="w", fontsize=9.5,
                     fontweight="bold", ha="center", va="center", clip_on=True)
            k = max(1, int(round(d / slot)))
            w = d / k
            for m in range(k):
                img = thumb(q["pair_id"], cache, min(d - 0.2, (m + 0.5) * w))
                kf.imshow(img, extent=(q["a"] + m * w, q["a"] + (m + 1) * w, 0, 1), aspect="auto", zorder=1)
            kf.add_patch(Rectangle((q["a"], 0), d, 1, fill=False, ec=col, lw=2.5, zorder=3))
            M, dur = mel_db(q["music_id"], cache)
            sp.imshow(M, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                      extent=(q["a"], q["a"] + min(dur, d), 0, 64), zorder=1)
            sp.add_patch(Rectangle((q["a"], 0), d, 64, fill=False, ec=col, lw=2.5, zorder=3))
            for x in cuts_of.get(q["pair_id"], []):
                for ax, top in ((kf, 1), (sp, 64)):
                    ax.plot([q["a"] + x] * 2, [0, top], color="white", lw=1.3, ls=(0, (2, 2)), zorder=2)
            table.append(dict(film=title[f"sunnxt:{film}"], pair=c + 1, music=letter, pair_id=q["pair_id"],
                              track_id=q["track_id"], film_start_s=q["a"], film_end_s=q["b"],
                              shot_cuts=len(cuts_of.get(q["pair_id"], []))))
        for a, b in zip(cues[:-1], cues[1:]):
            for ax, top in ((kf, 1), (sp, 64)):
                ax.add_patch(Rectangle((a["b"], 0), b["a"] - a["b"], top, fill=False, hatch="///", ec="#bbbbbb", lw=0))
            x = (a["b"] + b["a"]) / 2
            for ax in (bar, kf, sp):
                ax.axvline(x, color="#d62839", lw=2, ls=(0, (4, 2)), zorder=5)
            bar.text(x, 1.12, "music changes", color="#d62839", fontsize=9, ha="center", va="bottom", fontweight="bold")
        ticks = np.arange(np.ceil(t0 / 10) * 10, t1 + 1e-9, 10 if span < 70 else 20)
        sp.set_xticks(ticks, [f"{x:.0f}" for x in ticks], fontsize=9)
        sp.set_xlabel("time in the source video (s)", fontsize=9.5, labelpad=1)
        lab = fig.add_subplot(g[r, 0])
        lab.axis("off")
        lab.text(1.0, 0.62, title[f"sunnxt:{film}"], fontsize=13.5, fontweight="bold", ha="right", va="center")
        lab.text(1.0, 0.42, f"{len(cues)} pairs · {span:.0f} s", fontsize=10.5, ha="right", va="center", color="#444")
        lab.text(1.0, 0.18, "mel spectrogram ↓", fontsize=9, ha="right", va="center", color="#666", style="italic")
    handles = [Patch(fc=COLS[0], label="one pair: silent clip + the music under it (music starts at the clip's first frame)"),
               Line2D([], [], color="#d62839", lw=2, ls=(0, (4, 2)), label="music changes (next cue uses a different track)"),
               Line2D([], [], color="#888888", lw=1.5, ls=(0, (2, 2)), label="shot cut inside a pair (the music continues across it)"),
               Patch(fc="white", ec="#999999", hatch="///", label="between cues (not in the dataset)")]
    fig.subplots_adjust(bottom=0.07)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.savefig(OUT / "example_pairs.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "example_pairs.pdf", bbox_inches="tight")
    pd.DataFrame(table).to_csv(OUT / "rows.csv", index=False)
    log.info("wrote %s", OUT / "example_pairs.png")
    return OUT / "example_pairs.png"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import sys
    if len(sys.argv) > 1:
        single(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
