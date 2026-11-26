"""Slide images for "Steps 3-4: predict and decode" (viva deck): change probability per boundary, and the Viterbi route.

python -m bgm.analysis.slide_steps34   -> outputs/slide_steps34/{change_probability,viterbi_path}.png

Same held-out test video and lattice as viterbi_real_figure (outputs/viterbi_explainer/lattice.json, D125): 4 clips of
7aum Arivu, 72 s, 11 scenes. p_i are the cue-switch transformer's change probabilities; the route is decoded over every
candidate with the tuned costs and asserted to equal the final system's saved route. Sized for the 760 x 445 px
placeholders on the slide (1.71 : 1), on the deck's background colour.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np

from bgm.analysis.viterbi_real_figure import decode, load, move, shown_rows

log = logging.getLogger(__name__)
OUT = Path("outputs/slide_steps34")
BG, INK, MUTED, FAINT = "#e9e5db", "#141c2e", "#5f6672", "#c9c3b6"
ACCENT, LOW = "#b04a12", "#a7a196"
MOVE_COLOR = {"continue": INK, "jump": "#c98a1b", "switch": ACCENT}
MOVE_LS = {"continue": "-", "jump": (0, (4.0, 1.8)), "switch": (0, (1.2, 1.4))}
SIZE = (7.6, 4.45)


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans", "font.size": 10, "axes.edgecolor": MUTED, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": BG,
                         "axes.facecolor": BG, "savefig.facecolor": BG})
    return plt


def thumbs(fig, gs_row, d: dict, n: int) -> None:
    """One keyframe per scene with its time span, along one gridspec row."""
    from PIL import Image
    segs = d["segments"]
    for i in range(n):
        tx = fig.add_subplot(gs_row[i])
        tx.imshow(Image.open(io.BytesIO(base64.b64decode(d["thumbs_jpg_b64"][i]))), aspect="auto")
        tx.set_xticks([]), tx.set_yticks([])
        for sp in tx.spines.values():
            sp.set_visible(False)
        tx.set_title(f"s{i + 1}", fontsize=9, pad=2, color=INK, fontweight="bold")
        tx.text(0.5, -0.08, f"{segs[i]['a']:.0f}–{segs[i]['b']:.0f} s", transform=tx.transAxes, ha="center", va="top",
                fontsize=7, color=MUTED)


def change_probability(d: dict) -> None:
    """Bars of p_i between the scene keyframes; true clip changes marked."""
    plt = setup()
    P, segs = d["cue"]["p"], d["segments"]
    n = len(segs)
    true_b = [any(abs(s["b"] - t) < 0.05 for t in d["case"]["true_change_times_s"]) for s in segs[:-1]]
    fig = plt.figure(figsize=SIZE)
    gs = fig.add_gridspec(2, n, height_ratios=[0.55, 2.6], hspace=0.32, wspace=0.08, left=0.075, right=0.99, top=0.93,
                          bottom=0.17)
    thumbs(fig, [gs[0, i] for i in range(n)], d, n)
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(n - 1) + 0.5
    ax.bar(x, P, width=0.42, color=[ACCENT if p >= 0.5 else LOW for p in P], zorder=3)
    for xi, p in zip(x, P):
        ax.text(xi, p + 0.025, f"{p:.2f}", ha="center", va="bottom", fontsize=9, zorder=4,
                color=ACCENT if p >= 0.5 else MUTED, fontweight="bold" if p >= 0.5 else "normal")
    for i, t in enumerate(true_b):
        if t:
            ax.plot(x[i], -0.06, marker="^", ms=8, color=INK, clip_on=False, zorder=5)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(n - 0.52, 0.515, "p = 0.5", ha="right", va="bottom", fontsize=8, color=MUTED)
    for xi in x:
        ax.axvline(xi, color=FAINT, lw=0.6, zorder=1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [f"{s['b']:.1f} s" for s in segs[:-1]], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=20)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8, length=3)
    ax.set_ylabel("P(music changes)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="", marker="^", ms=8, color=INK, label="real music change in the original film")],
               loc="lower left", bbox_to_anchor=(0.065, -0.01), frameon=False, fontsize=8.5, handletextpad=0.3)
    fig.text(0.99, 0.022, "one bar per scene boundary (cut time)", ha="right", fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "change_probability.png", dpi=250)
    plt.close(fig)


def viterbi_path(d: dict) -> None:
    """Candidate windows per scene (best window of the top-3 songs, plus the route's own window), route highlighted."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    plt = setup()
    V, B, path = decode(d)
    assert path == d["saved_picks"]["final"], (path, d["saved_picks"]["final"])
    C, P, cue = d["cands"], d["cue"]["p"], d["cue"]
    n = len(C)
    lab = {f["code"]: f["label"] for f in d["files"]}
    moves = [move(C[i][path[i]], C[i + 1][path[i + 1]], P[i], cue, d["tol_s"])[0] for i in range(n - 1)]
    rows = shown_rows(d, path)
    log.info("route %s; moves %s", [f"{lab[C[i][j]['f']]}@{C[i][j]['off']:g}" for i, j in enumerate(path)], moves)

    fig = plt.figure(figsize=SIZE)
    gs = fig.add_gridspec(2, n, height_ratios=[0.55, 2.6], hspace=0.3, wspace=0.08, left=0.075, right=0.99, top=0.93,
                          bottom=0.1)
    thumbs(fig, [gs[0, i] for i in range(n)], d, n)
    ax = fig.add_subplot(gs[1, :])
    w, h = 0.78, 0.5
    Y = lambda r: -r * 0.72 - (0.22 if r == 3 else 0.0)  # noqa: E731
    pos = {(i, j): Y(r) for i in range(n) for r, j in enumerate(rows[i]) if j is not None}
    for (a_i, _), ya in pos.items():                    # faint: every move between shown windows
        for (b_i, _), yb in pos.items():
            if b_i == a_i + 1:
                ax.plot([a_i + w / 2, b_i - w / 2], [ya, yb], color=FAINT, lw=0.5, zorder=1)
    for i, k in enumerate(moves):                       # the route
        ax.annotate("", xy=(i + 1 - w / 2, pos[(i + 1, path[i + 1])]), xytext=(i + w / 2, pos[(i, path[i])]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=MOVE_COLOR[k], lw=2.0,
                                    linestyle=MOVE_LS[k], shrinkA=0, shrinkB=0.5))
    for (i, j), y in pos.items():
        c, on = C[i][j], path[i] == j
        ax.add_patch(FancyBboxPatch((i - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=INK if on else "#f6f3ec", ec=INK if on else FAINT, lw=1.0, zorder=4))
        ax.text(i, y + 0.08, lab[c["f"]], ha="center", va="center", fontsize=10, zorder=6,
                color="white" if on else INK, fontweight="bold" if on else "normal")
        ax.text(i, y - 0.12, f"@{c['off']:.1f} s", ha="center", va="center", fontsize=6.5, zorder=6,
                color="#d9dce2" if on else MUTED)
    ax.plot([-0.5, n - 0.5], [Y(3) + h / 2 + 0.1] * 2, color=MUTED, lw=0.6, ls=":")
    ax.text(-0.58, Y(1), "top-3 songs", rotation=90, ha="right", va="center", fontsize=9, color=MUTED)
    ax.text(-0.58, Y(3), "outside\ntop 3", rotation=90, ha="right", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(Y(3) - h / 2 - 0.05, h / 2 + 0.05)
    ax.axis("off")
    handles = [Line2D([], [], color=MOVE_COLOR[k], lw=2.0, ls=MOVE_LS[k], label=k) for k in ("continue", "jump", "switch")]
    handles.append(FancyBboxPatch((0, 0), 1, 1, fc=INK, ec=INK, label="chosen window"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=4, frameon=False, fontsize=8.5,
               handlelength=2.4, columnspacing=1.6)
    fig.text(0.99, 0.022, "box = song, start of window in it", ha="right", fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "viterbi_path.png", dpi=250)
    plt.close(fig)


def main() -> None:
    d = load()
    change_probability(d)
    viterbi_path(d)
    log.info("wrote %s/{change_probability,viterbi_path}.png", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
