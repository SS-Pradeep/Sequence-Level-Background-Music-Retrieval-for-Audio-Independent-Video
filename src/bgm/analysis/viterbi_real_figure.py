"""Viterbi stitching on the real held-out test video (D125), in the paper's notation; full-width paper figure.

python -m bgm.analysis.viterbi_real_figure   -> outputs/method_figures/viterbi_real.{pdf,png}

Input: outputs/viterbi_explainer/lattice.json (the worked example's full lattice, change probabilities and tuned costs;
made by python -m bgm.analysis.viterbi_explainer). The recursion runs over EVERY candidate (20-37 per scene) with the
paper's costs T(c, c'; p_i): continue -alpha log(1 - p_i), jump -alpha log p_i + lam_jump, switch -alpha log p_i +
mu Delta(k, k'), at the fold's tuned alpha 1.17, lam_jump -0.25, mu 0. Its route is asserted to equal the final system's
saved route. The figure draws, per scene, the best window of each of the 3 highest-scoring songs, plus the route's
window when it is not among them (else the continuation c+ of the previous route window); V values are from the full
lattice.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
SRC = Path("outputs/viterbi_explainer/lattice.json")
OUT = Path("outputs/method_figures")
INK, FAINT, MUTED = "#222222", "#bdbdbd", "#666666"
MOVE_COLOR = {"continue": INK, "jump": "#b4741a", "switch": "#d1365f"}
MOVE_LS = {"continue": "-", "jump": (0, (4.0, 1.6)), "switch": (0, (1.0, 1.3))}
STEP_BOUNDARY = 3                                       # panel (c): the recursion into scene 4 (true change at 20.28 s)


def load() -> dict:
    d = json.loads(SRC.read_text())
    d["cands"] = [[dict(zip(("f", "off", "dur", "key", "tempo", "z"), c)) for c in L["c"]] for L in d["layers"]]
    return d


def move(a: dict, b: dict, p: float, cue: dict, tol: float) -> tuple[str, float]:
    """(kind, T(a, b; p)) with the paper's per-boundary costs."""
    from bgm.features.keys import fifths_distance
    from bgm.retrieval.stitch_eval import tempo_matrix
    al = cue["alpha"]
    if a["f"] == b["f"]:
        if abs(b["off"] - (a["off"] + a["dur"])) <= tol:
            return "continue", -al * np.log1p(-p)
        return "jump", -al * np.log(p) + cue["lam_jump"]
    delta = fifths_distance(a["key"], b["key"]) + float(tempo_matrix(np.array([a["tempo"]]), np.array([b["tempo"]]))[0, 0])
    return "switch", -al * np.log(p) + cue["mu"] * delta


def decode(d: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """V_i(c) = score_i(c) + max_c' [V_{i-1}(c') - T(c', c; p_{i-1})] over every candidate; back-pointers; route."""
    C, P = d["cands"], d["cue"]["p"]
    V, B = [np.array([c["z"] for c in C[0]])], [None]
    for i in range(1, len(C)):
        M = np.array([[V[-1][k] - move(a, b, P[i - 1], d["cue"], d["tol_s"])[1] for b in C[i]] for k, a in enumerate(C[i - 1])])
        B.append(M.argmax(0))
        V.append(M.max(0) + np.array([c["z"] for c in C[i]]))
    path = [int(V[-1].argmax())]
    for i in range(len(C) - 1, 0, -1):
        path.append(int(B[i][path[-1]]))
    return V, B, path[::-1]


def shown_rows(d: dict, path: list[int]) -> list[list[int]]:
    """Per scene: best window of the top-3 songs, then the route window (if not shown) or the c+ of the previous one."""
    rows = []
    for i, C in enumerate(d["cands"]):
        best = {}
        for j in sorted(range(len(C)), key=lambda j: -C[j]["z"]):
            best.setdefault(C[j]["f"], j)
        r = list(best.values())[:3]
        extra = None
        if path[i] not in r:
            extra = path[i]
        elif i:
            a = d["cands"][i - 1][path[i - 1]]
            extra = next((j for j, c in enumerate(C) if c["f"] == a["f"] and abs(c["off"] - a["off"] - a["dur"]) <= d["tol_s"]), None)
            extra = None if extra in r else extra
        rows.append(r + [extra])
    return rows


def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from PIL import Image
    from bgm.analysis.method_figures import style
    from bgm.analysis.worked_example import OTHER, TRUE_COLORS

    d = load()
    V, B, path = decode(d)
    assert path == d["saved_picks"]["final"], (path, d["saved_picks"]["final"])
    C, P, segs, cue = d["cands"], d["cue"]["p"], d["segments"], d["cue"]
    n = len(C)
    lab = {f["code"]: f["label"] for f in d["files"]}
    col = lambda f: TRUE_COLORS[int(lab[f][1]) - 1] if lab[f].startswith("S") else OTHER   # noqa: E731
    txt = lambda f: "white" if lab[f].startswith("S") else "#222"                           # noqa: E731
    moves = [move(C[i][path[i]], C[i + 1][path[i + 1]], P[i], cue, d["tol_s"]) for i in range(n - 1)]
    true_b = [any(abs(s["b"] - t) < 0.05 for t in d["case"]["true_change_times_s"]) for s in segs[:-1]]
    rows = shown_rows(d, path)
    log.info("route %s; moves %s", [f"{lab[C[i][j]['f']]}@{C[i][j]['off']:g}" for i, j in enumerate(path)], [m[0] for m in moves])

    style()
    fig = plt.figure(figsize=(7.0, 4.45))
    gs = fig.add_gridspec(5, n, height_ratios=[0.42, 2.45, 0.34, 0.2, 1.32], hspace=0.08, wspace=0.06,
                          left=0.075, right=0.995, top=0.925, bottom=0.07)
    for i in range(n):                                 # keyframes, true song, time
        tx = fig.add_subplot(gs[0, i])
        tx.imshow(Image.open(io.BytesIO(base64.b64decode(d["thumbs_jpg_b64"][i]))), aspect="auto")
        tx.set_xticks([]), tx.set_yticks([])
        for sp in tx.spines.values():
            sp.set_visible(False)
        tx.set_title(f"$s_{{{i + 1}}}$  {segs[i]['a']:.1f}–{segs[i]['b']:.1f} s", fontsize=5.4, pad=1.5)
    fig.text(0.075, 0.995, "(a) Viterbi on a held-out test video: 4 clips of one film, 4 songs, 72 s, 11 scenes", fontsize=7.2, va="top")

    ax = fig.add_subplot(gs[1, :])
    w, h = 0.8, 0.56
    Y = lambda r: -r * 0.78 - (0.3 if r == 3 else 0.0)  # noqa: E731
    pos = {}
    for i in range(n):
        for r, j in enumerate(rows[i]):
            if j is not None:
                pos[(i, j)] = Y(r)
    for i in range(1, n):                              # faint: moves between shown windows
        for (a_i, a), ya in pos.items():
            if a_i != i - 1:
                continue
            for (b_i, b), yb in pos.items():
                if b_i == i:
                    ax.plot([i - 1 + w / 2, i - w / 2], [ya, yb], color=FAINT, lw=0.3, alpha=0.5, zorder=1)
    for i in range(1, n):                              # best arrival (back-pointer) when its source is shown
        for (b_i, b), yb in pos.items():
            if b_i != i or path[i] == b:
                continue
            a = int(B[i][b])
            if (i - 1, a) in pos:
                ax.annotate("", xy=(i - w / 2, yb), xytext=(i - 1 + w / 2, pos[(i - 1, a)]), zorder=2,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.22,head_width=0.1", color="#888", lw=0.6, shrinkA=0, shrinkB=0))
    for i, (k, t) in enumerate(moves):                 # the route
        ax.annotate("", xy=(i + 1 - w / 2, pos[(i + 1, path[i + 1])]), xytext=(i + w / 2, pos[(i, path[i])]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=MOVE_COLOR[k], lw=1.5,
                                    linestyle=MOVE_LS[k], shrinkA=0, shrinkB=0.5))
        ax.text(i + 0.5, 0.86, f"$p_{{{i + 1}}}$={P[i]:.2f}", ha="center", va="bottom", fontsize=5.3,
                color="#b4741a", fontweight="bold" if P[i] >= 0.5 else "normal")
        ax.text(i + 0.5, 0.46, f"{k}\n{t:.2f}", ha="center", va="bottom", fontsize=5.1, style="italic", color=MOVE_COLOR[k], linespacing=0.95)
        if true_b[i]:
            ax.plot([i + 0.5] * 2, [Y(3) - h / 2 - 0.05, 0.42], color=INK, lw=0.6, ls=(0, (2.5, 1.5)), zorder=0)
    for (i, j), y in pos.items():                      # windows (k, tau), score, V
        c, on, cont = C[i][j], path[i] == j, rows[i][3] == j and j not in rows[i][:3]
        ax.add_patch(FancyBboxPatch((i - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=col(c["f"]), ec=INK if on else col(c["f"]), lw=1.5 if on else 0.6, zorder=4))
        ax.text(i, y + 0.09, f"({lab[c['f']]}, {c['off']:.1f})", ha="center", va="center", fontsize=5.4, color=txt(c["f"]),
                fontweight="bold" if on else "normal", zorder=6)
        ax.text(i, y - 0.13, f"score {c['z']:.2f}", ha="center", va="center", fontsize=4.6, color=txt(c["f"]), zorder=6)
        ax.text(i + w / 2 - 0.05, y + h / 2 + 0.02, f"{V[i][j]:.1f}", ha="center", va="center", fontsize=4.5, color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.2", fc=INK, ec="none"))
    ax.plot([-0.5, n - 0.5], [Y(3) + h / 2 + 0.11] * 2, color="#aaa", lw=0.5, ls=":")
    ax.text(-0.56, Y(1), "best window of the\ntop-3 songs in $\\tilde{\\mathcal{C}}_i$", rotation=90, ha="right", va="center", fontsize=5.4, color="#444")
    ax.text(-0.56, Y(3), "route / $c^+$", rotation=90, ha="right", va="center", fontsize=5.4, color="#444")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(Y(3) - h / 2 - 0.06, 1.12)
    ax.axis("off")

    bx = fig.add_subplot(gs[2, :])                     # true song vs chosen song, per scene
    for i in range(n):
        tl = f"S{segs[i]['true_clip']}"
        cf = C[i][path[i]]["f"]
        ok = lab[cf] == tl
        bx.add_patch(Rectangle((i - 0.48, 0.55), 0.96, 0.4, fc=TRUE_COLORS[segs[i]["true_clip"] - 1], ec="none"))
        bx.text(i, 0.75, tl, ha="center", va="center", fontsize=5.2, color="white", fontweight="bold")
        bx.add_patch(Rectangle((i - 0.48, 0.05), 0.96, 0.4, fc=col(cf), ec="none"))
        bx.text(i, 0.25, f"{lab[cf]} {'✓' if ok else '✗'}", ha="center", va="center", fontsize=5.2, color=txt(cf), fontweight="bold",
                fontfamily="DejaVu Sans")
    bx.text(-0.56, 0.75, "true song", ha="right", va="center", fontsize=5.4, color="#444")
    bx.text(-0.56, 0.25, "chosen", ha="right", va="center", fontsize=5.4, color="#444")
    bx.set_xlim(-0.5, n - 0.5)
    bx.set_ylim(0, 1)
    bx.axis("off")

    sub = gs[4, :].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    cx = fig.add_subplot(sub[0, 0])                    # (b) costs against p, with the 10 real boundaries
    pp = np.linspace(0.01, 0.99, 400)
    al, lj = cue["alpha"], cue["lam_jump"]
    cx.plot(pp, -al * np.log1p(-pp), color=MOVE_COLOR["continue"], lw=1.1, ls=MOVE_LS["continue"])
    cx.plot(pp, -al * np.log(pp) + lj, color=MOVE_COLOR["jump"], lw=1.1, ls=MOVE_LS["jump"])
    cx.plot(pp, -al * np.log(pp), color=MOVE_COLOR["switch"], lw=1.1, ls=MOVE_LS["switch"])
    for i, (p, (k, t)) in enumerate(zip(P, moves)):
        cx.plot(p, t, "o", ms=3, color=MOVE_COLOR[k], zorder=5, mec="white", mew=0.4)
    cx.text(0.86, 2.75, r"continue $-\alpha\log(1-p)$", fontsize=5.4, ha="right", color=MOVE_COLOR["continue"])
    cx.text(0.2, 2.75, r"switch $-\alpha\log p$", fontsize=5.4, color=MOVE_COLOR["switch"])
    cx.text(0.2, 2.3, rf"jump $-\alpha\log p{lj:+.2f}$", fontsize=5.4, color=MOVE_COLOR["jump"])
    cx.set_xlim(0, 1)
    cx.set_ylim(-0.4, 3.1)
    cx.set_xlabel("change probability $p_i$ (dots: the 10 boundaries, move taken)", fontsize=5.6, labelpad=0.5)
    cx.set_ylabel("cost $T$", fontsize=5.8, labelpad=1)
    cx.tick_params(length=2, pad=1, labelsize=5.4)
    cx.spines[["top", "right"]].set_visible(False)
    cx.set_title(rf"(b) costs at the tuned $\alpha$={al:.2f}, $\lambda_\mathrm{{jump}}$={lj:.2f}, $\mu$={cue['mu']:g}", fontsize=6.6, loc="left", pad=2)

    tx = fig.add_subplot(sub[0, 1])                    # (c) one real recursion step
    tx.axis("off")
    i, j = STEP_BOUNDARY, path[STEP_BOUNDARY]
    c = C[i][j]
    arr = [(k, V[i - 1][k], *move(a, c, P[i - 1], cue, d["tol_s"])) for k, a in enumerate(C[i - 1])]
    arr.sort(key=lambda r: -(r[1] - r[3]))
    best = int(B[i][j])
    assert arr[0][0] == best
    tx.set_title(f"(c) one step: best arrival at ({lab[c['f']]}, {c['off']:.1f}) in $s_{{{i + 1}}}$, "
                 f"top 4 of {len(arr)} arrivals ($p_{{{i}}}$={P[i - 1]:.2f}, true change)", fontsize=6.6, loc="left", pad=2)
    yy = 0.83
    heads = ((0.0, "$c'$", "left"), (0.36, "$V_{%d}(c')$" % i, "right"), (0.55, "move", "center"), (0.72, "$T$", "right"),
             (0.94, "$V_{%d}(c')-T$" % i, "right"))
    for x, t, ha in heads:
        tx.text(x, yy, t, fontsize=5.8, color=MUTED, ha=ha, transform=tx.transAxes)
    for q, (k, v, kind, t) in enumerate(arr[:4]):
        yy -= 0.15
        a = C[i - 1][k]
        wgt = "bold" if k == best else "normal"
        if k == best:
            tx.add_patch(Rectangle((-0.02, yy - 0.045), 0.98, 0.14, transform=tx.transAxes, fc="#eef1f6", ec="none"))
        tx.text(0.0, yy, f"({lab[a['f']]}, {a['off']:.1f})", fontsize=5.8, fontweight=wgt, transform=tx.transAxes,
                color=col(a["f"]) if lab[a["f"]].startswith("S") else "#444")
        tx.text(0.36, yy, f"{v:.2f}", fontsize=5.8, ha="right", fontweight=wgt, transform=tx.transAxes)
        tx.text(0.55, yy, kind, fontsize=5.8, ha="center", style="italic", color=MOVE_COLOR[kind], transform=tx.transAxes)
        tx.text(0.72, yy, f"{t:.2f}", fontsize=5.8, ha="right", fontweight=wgt, transform=tx.transAxes)
        tx.text(0.94, yy, f"{v - t:.2f}" + (" ← max" if k == best else ""), fontsize=5.8, ha="right" if k != best else "right",
                fontweight=wgt, transform=tx.transAxes)
    yy -= 0.17
    tx.text(0.0, yy, f"$V_{{{i + 1}}}$ = score + max = {c['z']:.2f} + {arr[0][1] - arr[0][3]:.2f} = {V[i][j]:.2f}",
            fontsize=5.8, transform=tx.transAxes)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "viterbi_real.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "viterbi_real.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log.info("wrote %s/viterbi_real.{pdf,png}", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    figure()
