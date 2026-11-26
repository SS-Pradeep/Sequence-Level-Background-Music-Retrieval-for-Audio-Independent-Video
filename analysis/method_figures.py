"""Method-section figures: the Viterbi trellis (continue / jump / switch) and the seam crossfade gains.

python -m bgm.analysis.method_figures trellis   -> outputs/method_figures/viterbi_trellis.{pdf,png}
python -m bgm.analysis.method_figures seam      -> outputs/method_figures/seam_crossfade.{pdf,png}
python -m bgm.analysis.method_figures cue       -> outputs/method_figures/viterbi_cue.{pdf,png}   (full width)
python -m bgm.analysis.method_figures           -> all three

Trellis: a toy lattice (4 segments, 3 top-fit windows each plus one contiguous continuation of the previous segment's
window, as add_continuations builds it). The highlighted path is not drawn by hand: it is the argmin of the shipped
`viterbi` with `transition_cost` and the default penalties (configs/default.yaml stitch: lam_jump 1, lam_switch 2,
mu 0.5), asserted against the intended continue -> jump -> switch story. Greedy (best fit per segment) would switch
three times on the same lattice. Seam: gains from the shipped `crossfade_gains` (render.py); summed power
out^2 + in^2 is the mid-fade level on uncorrelated music (D46: measured -2.9 dB for linear on 1.mp4, theory -3.0).
Both figures are single-column (3.45 in) for a two-column paper.

Cue figure (the paper's decoder: per-boundary costs from the change probability p_i): the same toy lattice with
p = (0.1, 0.3, 0.9), alpha 1, lam_jump 0.25, mu 0.5 (round numbers so the reader can check the arithmetic). Costs
T(c, c'; p_i): continue -alpha log(1 - p_i), jump -alpha log p_i + lam_jump, switch -alpha log p_i + mu Delta(k, k').
The figure's own recursion (score minus cost) is asserted to give the route of the shipped `decode_lattice_cue`.
(a) trellis with V_i(c) and back-pointers, (b) the three costs against p, (c) one recursion step written out.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from bgm.retrieval.stitch import Candidate, greedy, transition_cost, viterbi

log = logging.getLogger(__name__)
OUT = Path("outputs/method_figures")
COL_W = 3.45                                           # in, one column of a two-column paper
FILE_COLOR = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}   # Okabe-Ito: blue, vermillion, green
INK, FAINT = "#222222", "#b8b8b8"
TC = dict(lam_jump=1.0, lam_switch=2.0, mu=0.5, tol_s=0.04)     # configs/default.yaml stitch

# Toy lattice. Segment durations (s); per segment: top-3 windows by fit, then the continuation of the previous
# segment's path window. (file, offset s, key pitch class, tempo BPM, unary cost = -z fine score).
DURS = (6.0, 5.0, 7.0, 6.0)
KEY_TEMPO = {"A": (0, 100.0), "B": (7, 100.0), "C": (3, 140.0)}   # B a fifth from A at the same tempo; C far
LATTICE = (
    [("A", 12, -2.0), ("B", 40, -1.6), ("C", 3, -1.2)],
    [("C", 20, -1.4), ("B", 75, -1.2), ("A", 50, -1.0), ("A", 18, -1.1)],
    [("A", 64, -1.8), ("B", 10, -1.5), ("C", 44, -1.0), ("A", 23, -0.3)],
    [("B", 31, -3.2), ("C", 55, -1.6), ("A", 90, -0.8), ("A", 71, -0.5)],
)
N_TOP = 3
EXPECTED = [0, 3, 0, 0]                                # A12 -continue-> A18 -jump-> A64 -switch-> B31


def style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6})


def lattice() -> list[list[Candidate]]:
    return [[Candidate(f, float(o), d, *KEY_TEMPO[f], c) for f, o, c in seg] for seg, d in zip(LATTICE, DURS)]


def edge_kind(a: Candidate, b: Candidate) -> str:
    if a.music_id != b.music_id:
        return "switch"
    return "continue" if abs(b.offset - (a.offset + a.dur)) <= TC["tol_s"] else "jump"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    log.info("wrote %s/%s.{pdf,png}", OUT, name)


def trellis() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    cands = lattice()
    path = viterbi(cands, **TC)
    assert path == EXPECTED, f"Viterbi path {path} is not the intended continue/jump/switch path {EXPECTED}"
    kinds = [edge_kind(cands[i][path[i]], cands[i + 1][path[i + 1]]) for i in range(len(cands) - 1)]
    assert kinds == ["continue", "jump", "switch"], kinds
    g = greedy(cands)
    log.info("viterbi %s %s; greedy %s with %d switches", path, kinds, g,
             sum(cands[i][g[i]].music_id != cands[i + 1][g[i + 1]].music_id for i in range(len(cands) - 1)))

    style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    w, h, gap = 0.56, 0.50, 0.42                       # node box (x units = segments, y units = rows)

    def y_of(r: int) -> float:                         # continuation row sits below a small gap
        return -(r + (gap if r >= N_TOP else 0.0))

    for i in range(len(cands) - 1):                    # every transition, faint
        for ra in range(len(cands[i])):
            for rb in range(len(cands[i + 1])):
                ax.plot([i + w / 2, i + 1 - w / 2], [y_of(ra), y_of(rb)], color=FAINT, lw=0.4, alpha=0.55,
                        zorder=1, solid_capstyle="butt")

    ls = {"continue": "-", "jump": (0, (4.0, 1.6)), "switch": (0, (1.0, 1.3))}
    for i, k in enumerate(kinds):                      # the decoded path
        x0, y0, x1, y1 = i + w / 2, y_of(path[i]), i + 1 - w / 2, y_of(path[i + 1])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=INK, lw=1.5,
                                    linestyle=ls[k], shrinkA=0, shrinkB=0.5))
        t = 0.5 if k != "jump" else 0.42
        ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0) + 0.13, k, ha="center", va="bottom", fontsize=6.8,
                style="italic", color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    for i, seg in enumerate(cands):                    # nodes
        for r, c in enumerate(seg):
            on = path[i] == r
            cont = r >= N_TOP
            col = FILE_COLOR[c.music_id]
            ax.add_patch(FancyBboxPatch((i - w / 2, y_of(r) - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                        fc="white" if cont else col, ec=INK if on else col,
                                        lw=1.6 if on else 0.8, ls="--" if cont and not on else "-", zorder=4))
            ax.text(i, y_of(r), f"{c.music_id}$\\,${c.offset:.0f}s", ha="center", va="center", zorder=6,
                    fontsize=7.2, fontweight="bold" if on else "normal", color=col if cont else "white")
        if i == 0:
            ax.text(i, y_of(N_TOP), "–", ha="center", va="center", color="#999", fontsize=7)
        ax.text(i, 0.62, f"$s_{i + 1}$", ha="center", va="bottom", fontsize=8.5)
        ax.text(i, 0.43, f"{DURS[i]:.0f} s", ha="center", va="bottom", fontsize=6.5, color="#555")

    ax.plot([-0.5, len(cands) - 0.5], [y_of(N_TOP) + h / 2 + gap / 2 + 0.02] * 2, color="#999", lw=0.5, ls=":")
    ax.text(-0.42, -(N_TOP - 1) / 2, "top-$J$ windows\n(by fit)", rotation=90, ha="right", va="center",
            fontsize=6.8, color="#444")
    ax.text(-0.42, y_of(N_TOP), "contin-\nuation", rotation=90, ha="right", va="center", fontsize=6.8,
            color="#444")

    handles = [Line2D([], [], color=INK, lw=1.5, ls=ls["continue"], label=r"continue: $0$"),
               Line2D([], [], color=INK, lw=1.5, ls=ls["switch"],
                      label=r"switch: $\lambda_{\mathrm{switch}}+\mu\,(\Delta_{\mathrm{key}}+\Delta_{\mathrm{tempo}})$"),
               Line2D([], [], color=INK, lw=1.5, ls=ls["jump"], label=r"jump: $\lambda_{\mathrm{jump}}$")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
              fontsize=6.8, handlelength=2.4, columnspacing=1.0, handletextpad=0.5, borderaxespad=0)
    ax.set_xlim(-0.62, len(cands) - 0.45)
    ax.set_ylim(y_of(N_TOP) - h / 2 - 0.08, 0.9)
    ax.axis("off")
    save(fig, "viterbi_trellis")
    plt.close(fig)


def seam() -> None:
    import matplotlib.pyplot as plt
    from bgm.render.render import crossfade_gains

    n = 2001
    t = (np.arange(n) + 0.5) / n
    curves = {name: crossfade_gains(n, name) for name in ("equal_power", "linear")}
    db = {k: 10 * np.log10(o ** 2 + i ** 2) for k, (o, i) in curves.items()}
    mid = n // 2
    assert abs(db["equal_power"]).max() < 1e-9 and abs(db["linear"][mid] + 3.01) < 0.01

    style()
    fig, (ga, pa) = plt.subplots(1, 2, figsize=(COL_W, 1.45), gridspec_kw=dict(wspace=0.42))
    ep, lin = "#0072B2", "#D55E00"
    (o, i), (lo, li) = curves["equal_power"], curves["linear"]
    ga.plot(t, o, color=ep, lw=1.3)
    ga.plot(t, i, color=ep, lw=1.3, label="equal power")
    ga.plot(t, lo, color=lin, lw=1.3, ls=(0, (4, 1.6)))
    ga.plot(t, li, color=lin, lw=1.3, ls=(0, (4, 1.6)), label="linear")
    ga.text(0.07, 0.6, "out", fontsize=6.5, color="#555", va="center")
    ga.text(0.93, 0.6, "in", fontsize=6.5, color="#555", va="center", ha="right")
    ga.set_ylim(0, 1.05)
    ga.set_yticks([0, 0.5, 1])
    ga.set_ylabel("gain", labelpad=1)
    ga.set_title("(a) fade gains", fontsize=7.5, pad=3)

    pa.axhline(0, color=ep, lw=1.3)
    pa.plot(t, db["linear"], color=lin, lw=1.3, ls=(0, (4, 1.6)))
    pa.annotate("−3 dB", xy=(0.5, db["linear"][mid]), xytext=(0.5, -1.7), ha="center", va="bottom", fontsize=6.8,
                color=lin, arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=lin, lw=0.7))
    pa.set_ylim(-3.6, 0.6)
    pa.set_yticks([-3, -2, -1, 0])
    pa.set_ylabel(r"$g_\mathrm{out}^2+g_\mathrm{in}^2$ (dB)", labelpad=1)
    pa.set_title("(b) summed power", fontsize=7.5, pad=3)

    for ax in (ga, pa):
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "½", "1"])
        ax.set_xlabel("fade time $t$", labelpad=1)
        ax.tick_params(length=2, pad=1.5, width=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ga.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2,
               frameon=False, fontsize=6.8, handlelength=2.4)
    save(fig, "seam_crossfade")
    plt.close(fig)


CUE_P = (0.1, 0.3, 0.9)
CUE = dict(alpha=1.0, lam_jump=0.25, mu=0.5)
MOVE_COLOR = {"continue": INK, "jump": "#b4741a", "switch": "#d1365f"}
MOVE_LS = {"continue": "-", "jump": (0, (4.0, 1.6)), "switch": (0, (1.0, 1.3))}


def cue_cost(a: Candidate, b: Candidate, p: float) -> tuple[str, float]:
    """(move, T(a, b; p)) of the paper's decoder."""
    from bgm.features.keys import fifths_distance
    from bgm.retrieval.stitch_eval import tempo_matrix
    k = edge_kind(a, b)
    if k == "continue":
        return k, -CUE["alpha"] * np.log(1 - p)
    if k == "jump":
        return k, -CUE["alpha"] * np.log(p) + CUE["lam_jump"]
    delta = fifths_distance(a.key, b.key) + float(tempo_matrix(np.array([a.tempo]), np.array([b.tempo]))[0, 0])
    return k, -CUE["alpha"] * np.log(p) + CUE["mu"] * delta


def cue_decode(cands) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """V_i(c) = score_i(c) + max_c' [V_{i-1}(c') - T(c', c; p_{i-1})], back-pointers, route."""
    V, B = [np.array([-c.cost for c in cands[0]])], [None]
    for i in range(1, len(cands)):
        M = np.array([[V[-1][a] - cue_cost(pa, cb, CUE_P[i - 1])[1] for cb in cands[i]] for a, pa in enumerate(cands[i - 1])])
        B.append(M.argmax(0))
        V.append(M.max(0) - np.array([c.cost for c in cands[i]]))
    path = [int(V[-1].argmax())]
    for i in range(len(cands) - 1, 0, -1):
        path.append(int(B[i][path[-1]]))
    return V, B, path[::-1]


def shipped_cue_route(cands) -> list[int]:
    """Route of bgm.cue.stitch.decode_lattice_cue (the system's decoder) on the same lattice."""
    from bgm.cue.stitch import decode_lattice_cue, prob_to_logit
    from bgm.retrieval.stitch_eval import Layer
    code = {f: n for n, f in enumerate(sorted({c.music_id for cs in cands for c in cs}))}
    layers = []
    for i, cs in enumerate(cands):
        L = Layer(np.array([code[c.music_id] for c in cs]), np.array([c.offset for c in cs]), np.array([c.dur for c in cs]),
                  np.array([c.key for c in cs]), np.array([c.tempo for c in cs]), np.array([c.cost for c in cs]), len(cs))
        if i:
            P = layers[-1]
            L.cont_of = np.array([next((j for j in range(len(cs)) if L.mid[j] == P.mid[q] and abs(L.off[j] - P.off[q] - P.dur[q]) <= 0.04), -1)
                                  for q in range(len(P.mid))])
        layers.append(L)
    return decode_lattice_cue(layers, prob_to_logit(np.array(CUE_P)), CUE["alpha"], CUE["lam_jump"], CUE["mu"], None, 0.04)


def cue_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    cands = lattice()
    V, B, path = cue_decode(cands)
    assert path == shipped_cue_route(cands) == EXPECTED, (path, EXPECTED)
    moves = [cue_cost(cands[i][path[i]], cands[i + 1][path[i + 1]], CUE_P[i]) for i in range(len(cands) - 1)]
    assert [m[0] for m in moves] == ["continue", "jump", "switch"], moves
    name = lambda c: f"({c.music_id}, {c.offset:.0f})"   # noqa: E731  c = (k, tau)

    style()
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1], height_ratios=[1, 1.05], wspace=0.16, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    w, h, gap = 0.6, 0.5, 0.42
    y_of = lambda r: -(r + (gap if r >= N_TOP else 0.0))   # noqa: E731
    n = len(cands)
    for i in range(n - 1):                             # every transition, faint
        for ra in range(len(cands[i])):
            for rb in range(len(cands[i + 1])):
                ax.plot([i + w / 2, i + 1 - w / 2], [y_of(ra), y_of(rb)], color=FAINT, lw=0.35, alpha=0.5, zorder=1)
    for i in range(1, n):                              # best arrival into every window (back-pointers)
        for rb in range(len(cands[i])):
            ra = int(B[i][rb])
            if path[i] == rb and path[i - 1] == ra:
                continue
            ax.annotate("", xy=(i - w / 2, y_of(rb)), xytext=(i - 1 + w / 2, y_of(ra)), zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color="#777", lw=0.7, shrinkA=0, shrinkB=0))
    for i, (k, t) in enumerate(moves):                 # the decoded route
        x0, y0, x1, y1 = i + w / 2, y_of(path[i]), i + 1 - w / 2, y_of(path[i + 1])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=MOVE_COLOR[k], lw=1.7,
                                    linestyle=MOVE_LS[k], shrinkA=0, shrinkB=0.5))
        ax.text(i + 0.5, 0.43, f"{k}\n$T$ = {t:.2f}", ha="center", va="bottom", fontsize=6.0, style="italic",
                color=MOVE_COLOR[k], linespacing=1.0)
    for i, seg in enumerate(cands):                    # windows: (k, tau), score, V
        for r, c in enumerate(seg):
            on, cont, col = path[i] == r, r >= N_TOP, FILE_COLOR[c.music_id]
            ax.add_patch(FancyBboxPatch((i - w / 2, y_of(r) - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="white" if cont else col, ec=INK if on else col, lw=1.6 if on else 0.8,
                                        ls="--" if cont and not on else "-", zorder=4))
            tc = col if cont else "white"
            ax.text(i, y_of(r) + 0.08, name(c), ha="center", va="center", fontsize=6.9, color=tc, zorder=6,
                    fontweight="bold" if on else "normal")
            ax.text(i, y_of(r) - 0.13, f"score {-c.cost:.1f}", ha="center", va="center", fontsize=5.4, color=tc, zorder=6)
            ax.text(i + w / 2 - 0.02, y_of(r) + h / 2, f"{V[i][r]:.2f}", ha="center", va="center", fontsize=5.4, color="white",
                    zorder=7, bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=INK, ec="none"))
        ax.text(i, 0.72, f"$s_{i + 1}$", ha="center", va="bottom", fontsize=8.5)
        ax.text(i, 0.52, f"$d_{i + 1}$ = {DURS[i]:.0f} s", ha="center", va="bottom", fontsize=5.8, color="#555")
    for i, p in enumerate(CUE_P):                      # change probability at each boundary
        ax.text(i + 0.5, 0.86, f"$p_{i + 1}$ = {p:.1f}", ha="center", va="bottom", fontsize=6.6, color="#b4741a",
                bbox=dict(boxstyle="round,pad=0.18", fc="#fbf1e2", ec="none"))
    ax.text(0, y_of(N_TOP), "$\\tilde{\\mathcal{C}}_1 = \\mathcal{C}_1$", ha="center", va="center", fontsize=6.4, color="#888")
    ax.plot([-0.45, n - 0.55], [y_of(N_TOP) + h / 2 + gap / 2 + 0.03] * 2, color="#999", lw=0.5, ls=":")
    ax.text(-0.43, -(N_TOP - 1) / 2, "top candidates $\\mathcal{C}_i$", rotation=90, ha="right", va="center", fontsize=6.4, color="#444")
    ax.text(-0.36, y_of(N_TOP), "$c^+$", ha="right", va="center", fontsize=7.5, color="#444")
    handles = [Line2D([], [], color=MOVE_COLOR[k], lw=1.5, ls=MOVE_LS[k], label=k) for k in MOVE_COLOR]
    handles += [Line2D([], [], color="#777", lw=0.7, marker=">", ms=3, label="best arrival (back-pointer)"),
                Line2D([], [], ls="", marker="s", ms=5, mfc=INK, mec=INK, label="$V_i(c)$: best route score")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=5, frameon=False, fontsize=5.8,
              handlelength=2.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(y_of(N_TOP) - h / 2 - 0.08, 1.12)
    ax.axis("off")
    ax.set_title("(a) fill $V_i(c)$ scene by scene, keep each best arrival, trace back from the best end",
                 fontsize=7, loc="left", pad=2)

    # (b) the three costs against p
    bx = fig.add_subplot(gs[0, 1])
    pp = np.linspace(0.01, 0.99, 400)
    a = CUE["alpha"]
    bx.plot(pp, -a * np.log(1 - pp), color=MOVE_COLOR["continue"], lw=1.2, ls=MOVE_LS["continue"], label=r"continue $-\alpha\log(1-p)$")
    bx.plot(pp, -a * np.log(pp) + CUE["lam_jump"], color=MOVE_COLOR["jump"], lw=1.2, ls=MOVE_LS["jump"], label=r"jump $-\alpha\log p+\lambda_\mathrm{jump}$")
    bx.plot(pp, -a * np.log(pp), color=MOVE_COLOR["switch"], lw=1.2, ls=MOVE_LS["switch"], label=r"switch $-\alpha\log p+\mu\Delta$ ($\Delta{=}0$)")
    for i, (p, (k, t)) in enumerate(zip(CUE_P, moves)):
        bx.axvline(p, color="#ccc", lw=0.6, zorder=0)
        bx.plot(p, t, "o", ms=3.4, color=MOVE_COLOR[k], zorder=5)
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 3.2)
    bx.set_xticks([0, *CUE_P, 1], ["0", *[f"$p_{i + 1}$" for i in range(len(CUE_P))], "1"])
    for t in bx.get_xticklabels()[1:-1]:
        t.set_color("#b4741a")
    bx.text(0.15, 2.85, r"jump $-\alpha\log p+\lambda_\mathrm{jump}$", fontsize=5.8, color=MOVE_COLOR["jump"], va="center")
    bx.text(0.15, 2.45, r"switch $-\alpha\log p+\mu\Delta$ (here $\Delta{=}0$)", fontsize=5.8, color=MOVE_COLOR["switch"], va="center")
    bx.text(0.79, 1.95, r"continue $-\alpha\log(1-p)$", fontsize=5.8, color=MOVE_COLOR["continue"], va="center", ha="right")
    bx.set_xlabel("change probability $p_i$", labelpad=0.5, fontsize=6.4)
    bx.set_ylabel("cost $T$", labelpad=1, fontsize=6.4)
    bx.tick_params(length=2, pad=1, labelsize=5.8)
    bx.spines[["top", "right"]].set_visible(False)
    bx.set_title("(b) move costs against $p_i$; dots: the route's moves", fontsize=7, loc="left", pad=3)

    # (c) one recursion step, written out
    cx = fig.add_subplot(gs[1, 1])
    cx.axis("off")
    i, j = n - 1, path[-1]
    c = cands[i][j]
    rows = [(name(pc), V[i - 1][q], *cue_cost(pc, c, CUE_P[i - 1])) for q, pc in enumerate(cands[i - 1])]
    best = int(B[i][j])
    cx.set_title(f"(c) one step: the best arrival at ({c.music_id}, {c.offset:.0f}) in $s_4$", fontsize=7, loc="left", pad=2)
    y = 0.86
    cx.text(0.0, y, "$c'$", fontsize=6.4, color="#555", transform=cx.transAxes)
    for x, t in ((0.3, "$V_3(c')$"), (0.52, "move"), (0.76, "$T$"), (0.98, "$V_3(c') - T$")):
        cx.text(x, y, t, fontsize=6.4, color="#555", ha="right" if x != 0.52 else "center", transform=cx.transAxes)
    for q, (nm, v, k, t) in enumerate(rows):
        y -= 0.15
        wgt = "bold" if q == best else "normal"
        if q == best:
            cx.add_patch(plt.Rectangle((-0.02, y - 0.045), 1.02, 0.14, transform=cx.transAxes, fc="#eef1f6", ec="none"))
        cx.text(0.0, y, nm, fontsize=6.4, color=FILE_COLOR[nm[1]], fontweight=wgt, transform=cx.transAxes)
        cx.text(0.3, y, f"{v:.2f}", fontsize=6.4, ha="right", fontweight=wgt, transform=cx.transAxes)
        cx.text(0.52, y, k, fontsize=6.4, ha="center", style="italic", color=MOVE_COLOR[k], transform=cx.transAxes)
        cx.text(0.76, y, f"{t:.2f}", fontsize=6.4, ha="right", fontweight=wgt, transform=cx.transAxes)
        cx.text(0.98, y, f"{v - t:.2f}", fontsize=6.4, ha="right", fontweight=wgt, transform=cx.transAxes)
        if q == best:
            cx.text(1.0, y, " ← max", fontsize=6.4, fontweight=wgt, transform=cx.transAxes)
    y -= 0.17
    cx.text(0.0, y, f"$V_4$ = score + max = {-c.cost:.1f} + {V[i - 1][best] - rows[best][3]:.2f} = {V[i][j]:.2f}, the best final window",
            fontsize=6.4, transform=cx.transAxes)
    save(fig, "viterbi_cue")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    which = sys.argv[1:] or ["trellis", "seam", "cue"]
    for name in which:
        {"trellis": trellis, "seam": seam, "cue": cue_figure}[name]()


if __name__ == "__main__":
    main()
