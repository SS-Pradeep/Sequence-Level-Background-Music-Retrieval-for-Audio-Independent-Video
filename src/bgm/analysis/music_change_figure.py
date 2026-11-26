"""Paper figure for "Where does the music change?" (sec:realchanges).

python -m bgm.analysis.music_change_figure
  -> outputs/paper_viterbi/figures/music_change.{pdf,png}

Reads only saved reports (no model runs):
  (a) results/titlesplit/cue_switch.md  headline table: in-film boundary F1 +-1 s with 95% CIs (D121).
  (b) paired deltas R4 - R3 and R4nodur - R4 from results/titlesplit/cue_switch.md (1-3 min, D121),
      results/cleaned/cue_switch.md (D122) and results/titlesplit/cue_switch_long.md (5-15 min, D123).
  (c) results/segmentation_improve.json: boundary-head AP at clip joins vs AP on real music changes
      inside test clips (D142 music-awareness probe).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "paper_viterbi" / "figures"
TEXT_W = 6.875  # CVPR two-column \textwidth, inches

NUM = r"([+\-−]?\d+\.\d+)"
CI_RE = re.compile(NUM + r"\s*\[" + NUM + r",\s*" + NUM + r"\]")

BLUE, ORANGE, GREY, RED, GREEN = "#2b6cb0", "#dd6b20", "#8a8a8a", "#c2412c", "#1d8a5a"


def _f(s: str) -> float:
    return float(s.replace("−", "-"))


def _row(md: str, label: str) -> list[str]:
    """Cells of the first markdown table row whose first cell equals ``label``."""
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise KeyError(label)


def _ci(cell: str) -> tuple[float, float, float]:
    m = CI_RE.search(cell)
    if not m:
        raise ValueError(cell)
    return _f(m.group(1)), _f(m.group(2)), _f(m.group(3))


def load() -> dict:
    title = (ROOT / "results/titlesplit/cue_switch.md").read_text()
    cleaned = (ROOT / "results/cleaned/cue_switch.md").read_text()
    long_ = (ROOT / "results/titlesplit/cue_switch_long.md").read_text()

    systems = [
        ("greedy (no transitions, reference)", "Greedy (no switch cost)"),
        ("R0 fixed penalties, stored D75", "Fixed penalties"),
        ("R0 fixed penalties, re-tuned", "Fixed, re-tuned"),
        ("R4 without log duration (diagnostic)", "Transformer, no duration"),
        ("R3 learned, pair-only", "Pair-only predictor"),
        ("R4 learned + context", "Transformer predictor"),
        ("oracle p (true changes, reference)", "Oracle $p_i$ (true changes)"),
    ]
    a = [(name, *_ci(_row(title, key)[1])) for key, name in systems]

    sets = [("1–3 min", title), ("cleaned", cleaned), ("5–15 min", long_)]
    b = {
        "ctx": [(s, *_ci(_row(md, "R4 − R3")[1])) for s, md in sets],
        "dur": [(s, *_ci(_row(md, "R4nodur − R4")[1])) for s, md in sets],
    }

    seg = json.loads((ROOT / "results/segmentation_improve.json").read_text())
    heads = [
        ("emb_p", "BaSSL + head"),
        ("focal", "+ focal loss"),
        ("rank", "+ ranking loss"),
        ("contrast", "+ contrast feats"),
        ("gru", "+ BiGRU"),
        ("combo", "contrast + BiGRU"),
    ]
    c = [
        (name, seg["variants"][k]["AP"]["mean"], *seg["variants"][k]["AP"]["ci"],
         seg["probe"][k]["AP"]["mean"], *seg["probe"][k]["AP"]["ci"])
        for k, name in heads
    ]
    probe = {"n": seg["probe"]["n"], "n_pos": seg["probe"]["n_pos"], "base": seg["probe"]["base_rate"]}
    return {"a": a, "b": b, "c": c, "probe": probe}


def plot(d: dict) -> None:
    plt.rcParams.update({"font.size": 6.6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2, "ytick.major.size": 2,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.05),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.1], "wspace": 0.95})

    # (a) ladder with oracle headroom
    ax = axes[0]
    rows = d["a"]
    ys = range(len(rows))
    for y, (name, m, lo, hi) in zip(ys, rows):
        col = GREEN if name.startswith("Oracle") else BLUE if "predictor" in name and "Transformer" in name \
            else ORANGE if name.startswith("Pair") else GREY
        ax.barh(y, m, height=0.62, color=col, alpha=0.9 if col != GREY else 0.55, lw=0)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor="#222", elinewidth=0.6, capsize=1.4)
        ax.text(hi + 0.01, y, f"{m:.3f}", va="center", ha="left", fontsize=5.8)
    r4 = next(r for r in rows if r[0] == "Transformer predictor")[1]
    orc = next(r for r in rows if r[0].startswith("Oracle"))[1]
    yo, y4 = len(rows) - 1, len(rows) - 2
    ax.annotate("", xy=(orc, yo - 0.5), xytext=(r4, yo - 0.5),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREEN, shrinkA=0, shrinkB=0))
    ax.text(orc + 0.01, yo - 0.5, f"+{orc - r4:.2f}", ha="left", va="center", fontsize=5.8, color=GREEN)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(0.4, 0.92)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("in-film boundary F1 (±1 s)", labelpad=1)
    ax.set_title("(a) Room left for a better predictor", fontsize=7.2, loc="left", x=-0.95, pad=4)

    # (b) forest plot of paired differences
    ax = axes[1]
    labels, y = [], 0
    blocks = [("ctx", "Transformer − pair-only", BLUE, "o"), ("dur", "drop log duration", RED, "s")]
    yt = []
    for key, head, col, mk in blocks:
        for s, m, lo, hi in d["b"][key]:
            sig = lo > 0 or hi < 0
            ax.errorbar(m, -y, xerr=[[m - lo], [hi - m]], fmt=mk, ms=2.8, color=col, mfc=col if sig else "white",
                        mew=0.7, elinewidth=0.8, capsize=1.4)
            labels.append(s)
            yt.append(-y)
            y += 1
        ax.text(-0.073, -(y - 3) + 0.62, head, fontsize=6.0, color=col, ha="left", va="bottom", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
        y += 0.9
    ax.axvline(0, color="#222", lw=0.6, ls=(0, (2.5, 1.5)), zorder=0)
    ax.set_yticks(yt, labels, fontsize=6.0)
    ax.set_xlim(-0.075, 0.035)
    ax.set_ylim(min(yt) - 0.6, 1.05)
    ax.set_xlabel("Δ in-film boundary F1 [95% CI]", labelpad=1)
    ax.set_title("(b) Context vs. segment length", fontsize=7.2, loc="left", x=-0.42, pad=4)

    # (c) clip joins vs real music changes inside clips
    ax = axes[2]
    rows = d["c"]
    base = d["probe"]["base"]
    for i, (name, aj, ajl, ajh, ap, apl, aph) in enumerate(rows):
        y = -i
        ax.plot([ap, aj], [y, y], color="#bbb", lw=0.8, zorder=1)
        ax.errorbar(aj, y, xerr=[[aj - ajl], [ajh - aj]], fmt="o", ms=3, color=BLUE, elinewidth=0.6, capsize=1.2, zorder=3)
        ax.errorbar(ap, y, xerr=[[ap - apl], [aph - ap]], fmt="D", ms=2.8, color=RED, elinewidth=0.6, capsize=1.2, zorder=3)
    ax.axvline(base, color=RED, lw=0.7, ls=(0, (1, 1.2)), zorder=0)
    ax.text(base + 0.012, 0.75, f"chance {base:.2f}", fontsize=5.6, color=RED, ha="left", va="center")
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows], fontsize=6.0)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-len(rows) + 0.4, 1.25)
    ax.set_xlabel("average precision", labelpad=1)
    ax.plot([], [], "D", ms=2.8, color=RED, label=f"real changes in clips ({d['probe']['n_pos']}/{d['probe']['n']:,} cuts)")
    ax.plot([], [], "o", ms=3, color=BLUE, label="clip joins (benchmark)")
    ax.legend(frameon=False, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.3, -0.40), ncol=1,
              handletextpad=0.3, borderaxespad=0)
    ax.set_title("(c) Visual boundary heads on real changes", fontsize=7.2, loc="left", x=-0.72, pad=4)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "music_change.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "music_change.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    log.info("wrote %s", OUT / "music_change.pdf")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    d = load()
    for k in ("a", "b", "c"):
        log.info("%s: %s", k, d[k])
    plot(d)


if __name__ == "__main__":
    main()
