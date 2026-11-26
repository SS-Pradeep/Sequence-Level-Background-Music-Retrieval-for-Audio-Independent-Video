"""Two section figures for one rendered video (godfather.mp4, or a dataset test clip), from the render's own intermediates.

  sec1_training_coarse_to_fine.png  Training & window-level coarse-to-fine retrieval
  sec2_transition_viterbi.png       Transition costs & exact Viterbi decoding
  sections.json                     every number quoted in the figures

Re-plans with `poster_pipeline.capture_plan` and asserts the path equals the rendered one, exactly like
poster_pipeline. Same CLI and overrides as poster_pipeline:
  python -m bgm.analysis.godfather_sections --video godfather.mp4 --name godfather_with_bgm \
      --out outputs/godfather/sections --run runs/titlesplit/pipeline_ens10/fold0/seed0 <render overrides>
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from bgm.analysis.poster_pipeline import (BG, GRID, INK, INK2, MUTED, OTHER, SERIES, capture_plan, colour_map, film_of,
                                          frame_at, plt, save)

log = logging.getLogger(__name__)
SEM_ROOT = Path("runs/titlesplit/semantic/siglip-both")


# ----------------------------------------------------------------------------- path costs
# `pens`: one penalty dict per boundary (len(L) - 1), as bgm.cue.stitch.boundary_penalties gives. Units: fused z.
# As in bgm.cue.stitch, the constant continue cost -alpha*log(1 - p_i) is dropped at every boundary (no decision changes).
def path_cost(L, picks, pens, tol_s=0.04, L_fit=None) -> dict:
    """Split the total cost of a path into -fit, replay charge, switch, jump. L = the decoded (replay-charged) lattice;
    L_fit = the same lattice before replay charges (None: no split)."""
    from bgm.retrieval.stitch_eval import transition_matrix
    unary = float(sum(L[i].cost[j] for i, j in enumerate(picks)))
    fit = unary if L_fit is None else float(sum(L_fit[i].cost[j] for i, j in enumerate(picks)))
    replay = unary - fit
    sw = jp = 0.0
    n_sw = n_jp = 0
    for i in range(1, len(L)):
        a, b = picks[i - 1], picks[i]
        t = float(transition_matrix(L[i - 1].subset(np.array([a])), L[i].subset(np.array([b])), pens[i - 1], tol_s)[0, 0])
        if L[i - 1].mid[a] != L[i].mid[b]:
            sw, n_sw = sw + t, n_sw + 1
        elif abs(L[i].off[b] - (L[i - 1].off[a] + L[i - 1].dur[a])) > tol_s:
            jp, n_jp = jp + t, n_jp + 1
    return {"neg_fit": fit, "replay": replay, "switch": sw, "jump": jp, "total": unary + sw + jp, "n_switch": n_sw,
            "n_jump": n_jp}


def brute_force(L, pens, a, b, tol_s=0.04):
    """Exhaustive minimum over every candidate tuple of segments a..b-1 (no DP): (cost, path, n_paths)."""
    from bgm.retrieval.stitch_eval import transition_matrix
    tot = L[a].cost.copy()
    for i in range(a + 1, b):
        T = transition_matrix(L[i - 1], L[i], pens[i - 1], tol_s)          # [n_{i-1}, n_i]
        tot = tot[..., None] + T.reshape([1] * (tot.ndim - 1) + list(T.shape)) + L[i].cost
    k = int(np.argmin(tot))
    return float(tot.flat[k]), [int(x) for x in np.unravel_index(k, tot.shape)], int(tot.size)


def dp_window(L, pens, a, b, tol_s=0.04):
    """Plain Viterbi (no beam) over segments a..b-1 with all candidates: (cost, path)."""
    from bgm.retrieval.stitch_eval import transition_matrix
    D, backs = L[a].cost.copy(), []
    for i in range(a + 1, b):
        tot = D[:, None] + transition_matrix(L[i - 1], L[i], pens[i - 1], tol_s)
        backs.append(tot.argmin(0))
        D = tot.min(0) + L[i].cost
    j = int(np.argmin(D))
    path = [j]
    for bk in reversed(backs):
        j = int(bk[j])
        path.append(j)
    return float(D.min()), path[::-1]


# ----------------------------------------------------------------------------- section 1
def training_logs(runs: Path, seeds=range(10)) -> list[pd.DataFrame]:
    return [pd.read_csv(runs / f"seed{s}" / "train_log.csv") for s in seeds if (runs / f"seed{s}" / "train_log.csv").exists()]


def fig_section1(out, cap, plan, store, cmap, video, label, true_music=None, note=None) -> dict:
    from bgm.retrieval.pipeline import fuse
    res, w, index = cap["results"], cap["w_fine"], cap["index"]
    segs = plan["segments"]
    fig = plt.figure(figsize=(17, 10.5))
    SEM_RUNS = SEM_ROOT / f"fold{plan['fold']}"
    fig.suptitle(f"Training & window-level coarse-to-fine retrieval  ·  {label} ({res[-1].b:.0f} s, {len(res)} segments)", x=0.04,
                 ha="left", fontsize=17, weight="bold")
    fig.text(0.04, 0.935, "Frozen encoders (SigLIP for the picture; CLAP + MERT for the music) feed small trained heads. "
             "The heads rank whole music files first (coarse), then every window inside the shortlisted files (fine).",
             fontsize=11, color=INK2)

    # A: training curves of the 10 ensemble heads (fold 0)
    logs = training_logs(SEM_RUNS)
    best = [json.loads((SEM_RUNS / f"seed{s}" / "metrics.json").read_text())["val"] for s in range(len(logs))]
    axl = fig.add_axes([0.04, 0.56, 0.25, 0.30])
    axr = fig.add_axes([0.335, 0.56, 0.25, 0.30])
    for d, b in zip(logs, best):
        axl.plot(d.epoch, d.loss, color=SERIES[0], lw=1, alpha=0.55)
        axr.plot(d.epoch, d["val_seg_R@10"], color=SERIES[1], lw=1, alpha=0.55)
        e = int(b["best_epoch"])
        axr.plot(e, float(d.loc[d.epoch == e, "val_seg_R@10"].iloc[0]), "o", ms=5, color=INK, zorder=3)
    axl.set_xlabel("epoch"), axl.set_ylabel("InfoNCE loss (train)")
    axr.set_xlabel("epoch"), axr.set_ylabel("validation segment R@10")
    for a_ in (axl, axr):
        a_.grid(axis="y", color=GRID, lw=0.6)
    vr = np.array([b["segment_R@10"] for b in best])
    axl.set_title(f"A  Training the picture ↔ music heads ({len(logs)} seeds = the ensemble)", loc="left", fontsize=11,
                  weight="bold")
    axr.set_title(f"   early stopping on validation (dot = kept epoch; R@10 {vr.mean():.3f} ± {vr.std():.3f})", loc="left",
                  fontsize=10, color=INK2)
    fig.text(0.04, 0.475,
             "• contrastive InfoNCE with a learnable temperature; one segment per video in a batch\n"
             "• clips sharing a track_id are masked out of the negatives; encoders stay frozen, only the heads train\n"
             f"• title split (fold {plan['fold']}): " + (note or "Godfather is not in the dataset, so it is a fully unseen query"),
             fontsize=9.5, color=INK2, va="top", linespacing=1.6)

    # B: funnel from library to the Viterbi candidates
    nwin = np.array([len(r.fi) for r in res])
    G = len(index.ids)
    stages = [("library files", G), (f"shortlist (top {cap['K']} per span, union)", len(cap["short"])),
              ("windows scored / segment (median)", int(np.median(nwin))),
              (f"candidates kept / segment (top {cap['J']}, ≤4 per file)", cap["J"])]
    axf = fig.add_axes([0.66, 0.56, 0.31, 0.30])
    y = np.arange(len(stages))[::-1]
    axf.barh(y, [s[1] for s in stages], color=[OTHER, SERIES[0], SERIES[3], SERIES[1]], height=0.6)
    axf.set_xscale("log")
    for yi, (lab, v) in zip(y, stages):
        axf.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=10, weight="bold")
        axf.text(1.05, yi + 0.38, lab, va="bottom", fontsize=9, color=INK2)
    axf.set_yticks([]), axf.set_xlim(1, G * 6), axf.set_ylim(-0.6, len(stages) - 0.1)
    axf.set_xlabel("count (log scale)")
    axf.set_title("B  Coarse to fine on this video", loc="left", fontsize=11, weight="bold")

    # C: every window of the longest segment, top 5 files
    k = int(np.argmax([r.b - r.a for r in res]))
    r = res[k]
    S = fuse(r.raw, r.stats, w)
    axc = fig.add_axes([0.04, 0.07, 0.36, 0.30])
    best_f = {int(f): float(S[r.fi == f].max()) for f in np.unique(r.fi)}
    topf = sorted(best_f, key=lambda f: -best_f[f])[:5]
    for i, f in enumerate(topf):
        m = r.fi == f
        mid = index.ids[f]
        tag = ("  ← kept by Viterbi" if mid == segs[k]["music_id"] else "") + \
              ("  (TRUE music)" if true_music and store.music_track[mid] == store.music_track[true_music] else "")
        axc.plot(r.st[m], S[m], lw=2.6 if tag else 1.6, color=SERIES[i], label=textwrap.shorten(film_of(store, mid), 26) + tag)
    j = int(np.argmax(S))
    axc.plot(r.st[j], S[j], "o", ms=9, color=INK, mfc="none", mew=2)
    axc.annotate("best window by fit", (r.st[j], S[j]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    axc.set_xlabel("window start inside the music file (s)"), axc.set_ylabel("fused window score (z)")
    axc.grid(axis="y", color=GRID, lw=0.6)
    axc.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="top 5 files", title_fontsize=9)
    axc.set_title(f"C  Fine: segment {k + 1} ({r.a:.1f}–{r.b:.1f} s), a {r.b - r.a:.1f} s window slid every "
                  f"{plan['fusion']['window_hop_s']} s", loc="left", fontsize=11, weight="bold")
    ins = fig.add_axes([0.415, 0.09, 0.11, 0.085])
    ins.imshow(frame_at(Path(video), (r.a + r.b) / 2), aspect="auto")
    ins.set_xticks([]), ins.set_yticks([])

    # D: what the window step adds: best window vs the same file started at 0 s
    gains, start0, bestw = [], [], []
    for r_ in res:
        S_ = fuse(r_.raw, r_.stats, w)
        jj = int(np.argmax(S_))
        m = (r_.fi == r_.fi[jj])
        s0 = float(S_[m][np.argmin(r_.st[m])])
        start0.append(s0), bestw.append(float(S_[jj])), gains.append(float(S_[jj]) - s0)
    axd = fig.add_axes([0.66, 0.07, 0.31, 0.30])
    x = np.arange(1, len(res) + 1)
    axd.vlines(x, start0, bestw, color=MUTED, lw=1.2)
    axd.plot(x, start0, "o", ms=5, color=OTHER, mec=INK2, mew=0.6, label="same file, played from 0 s (file-level only)")
    axd.plot(x, bestw, "o", ms=6, color=SERIES[0], label="best window inside the file (window-level)")
    axd.set_xticks(x[::2], [f"{i}" for i in x[::2]], fontsize=8)
    axd.set_xlabel("segment"), axd.set_ylabel("fused score (z)")
    axd.grid(axis="y", color=GRID, lw=0.6)
    axd.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    axd.set_ylim(min(start0) - 1.0, max(bestw) + 0.4)
    g = np.array(gains)
    axd.set_title(f"D  Window search raises fit by {np.median(g):.2f} z (median; {int((g > 0.05).sum())}/{len(g)} segments)",
                  loc="left", fontsize=11, weight="bold")
    save(fig, out, "sec1_training_coarse_to_fine")
    return {"train_seeds": len(logs), "val_seg_R@10_mean": float(vr.mean()), "val_seg_R@10_std": float(vr.std()),
            "best_epochs": [int(b["best_epoch"]) for b in best], "library": G, "shortlist": len(cap["short"]),
            "windows_per_segment": [int(nwin.min()), int(np.median(nwin)), int(nwin.max())], "J": cap["J"],
            "example_segment": k + 1, "window_gain_z": {"median": float(np.median(g)), "per_segment": gains}}


# ----------------------------------------------------------------------------- section 2
def fig_section2(out, cap, plan, store, cmap, label) -> dict:
    from bgm.cue.stitch import boundary_penalties, decode_lattice_cue, prob_to_logit
    from bgm.retrieval.stitch_eval import decode_lattice, transition_matrix
    L0, Lc, picks, index, res = cap["lattice"], cap["lattice_charged"], list(cap["picks"]), cap["index"], cap["results"]
    n = len(Lc)
    cue = plan["cue"]
    prm = cue["params"]
    p = np.asarray(cue["p"], float)
    z = prob_to_logit(p)
    pens = boundary_penalties(z, prm["alpha"], prm["lam_jump"], prm["mu"])
    lam = np.array([q["lam_switch"] for q in pens])
    tuned = plan["tuned_penalties"]
    # the three paths: greedy (no transition cost), fixed-penalty Viterbi (val-tuned), cue Viterbi (rendered)
    greedy = [int(np.argmin(L.cost)) for L in L0]
    fixed = decode_lattice(L0, tuned, 100)
    assert decode_lattice_cue(Lc, z, prm["alpha"], prm["lam_jump"], prm["mu"], 100) == picks, \
        "cue decode of the charged lattice differs from the rendered path"
    t0 = time.perf_counter()
    exact = decode_lattice_cue(Lc, z, prm["alpha"], prm["lam_jump"], prm["mu"], None)
    t_exact = time.perf_counter() - t0
    paths = {"greedy": greedy, "fixed": fixed, "cue": picks}
    costs = {k: path_cost(Lc, v, pens, L_fit=L0) for k, v in paths.items()}       # all scored with the objective the render used
    ce = path_cost(Lc, exact, pens, L_fit=L0)
    bf = []
    W = min(4, n)
    for a in range(0, n - W + 1):
        c_bf, p_bf, npaths = brute_force(Lc, pens, a, a + W)
        c_dp, p_dp = dp_window(Lc, pens, a, a + W)
        bf.append({"segments": [a + 1, a + W], "paths": npaths, "brute": c_bf, "viterbi": c_dp, "same_path": p_bf == p_dp})
    assert all(abs(b["brute"] - b["viterbi"]) < 1e-9 for b in bf)
    n_cand = [len(L.cost) for L in Lc]
    log10_paths = float(np.sum(np.log10(n_cand)))
    sw_at = [Lc[i - 1].mid[picks[i - 1]] != Lc[i].mid[picks[i]] for i in range(1, n)]

    fig = plt.figure(figsize=(17, 13))
    fig.suptitle(f"Transition costs & exact Viterbi decoding  ·  {label} ({n} segments, {n - 1} boundaries)", x=0.04,
                 ha="left", fontsize=17, weight="bold")
    fig.text(0.04, 0.952,
             "cost(path) = Σ −fit(segment, window) + Σ transition(boundary).   At boundary i the trained cue-switch predictor "
             "gives a change probability pᵢ, and λᵢ = −α·logit(pᵢ):\n"
             f"continue the same music = 0  ·  switch file = λᵢ + μ·(key + tempo distance)  ·  jump inside a file = λᵢ + "
             f"{prm['lam_jump']:.1f}    (α = {prm['alpha']:.2f}, μ = {prm['mu']:.1f}, tuned on validation)",
             fontsize=10.5, color=INK2, va="top", linespacing=1.6)

    # A: p_i and lambda_i per boundary
    axa = fig.add_axes([0.04, 0.69, 0.56, 0.20])
    xb = np.arange(1, n)
    axa.bar(xb, p, width=0.6, color=[SERIES[1] if s else OTHER for s in sw_at], zorder=3)
    axa.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    axa.set_ylim(0, 1.05), axa.set_xlim(0.3, n - 0.3)
    axa.set_ylabel("change probability pᵢ")
    axa.set_xticks(xb, [f"{i}|{i + 1}" for i in xb], fontsize=7, rotation=90)
    axa.set_xlabel("boundary (segment i | i + 1)   ·   line = λᵢ: high p → λ < 0, switching is cheap; low p → λ > 0, keep the music going")
    ax2 = axa.twinx()
    ax2.plot(xb, lam, "o-", color=INK, ms=4, lw=1.2, zorder=4)
    ax2.axhline(0, color=INK, lw=0.5)
    ax2.set_ylabel("switch cost λᵢ", color=INK)
    ax2.spines["right"].set_visible(True)
    axa.set_title("A  Learned transition cost per boundary (orange bar = the decoded path switches music here)",
                  loc="left", fontsize=10.5, weight="bold")

    # B: one real transition matrix, at a boundary where the path continues
    stay = [i for i in range(1, n) if not sw_at[i - 1]]
    i = max(stay, key=lambda k: lam[k - 1]) if stay else max(1, n // 2)
    P = np.unique(np.r_[np.arange(min(20, len(Lc[i - 1].cost))), picks[i - 1]]).astype(int)
    C = np.unique(np.r_[np.arange(min(20, len(Lc[i].cost))), picks[i], [int(c) for c in Lc[i].cont_of[P] if c >= 0]]).astype(int)
    T = transition_matrix(Lc[i - 1].subset(P), Lc[i].subset(C), pens[i - 1])
    po, co = np.argsort(Lc[i - 1].mid[P], kind="stable"), np.argsort(Lc[i].mid[C], kind="stable")
    T, P, C = T[po][:, co], P[po], C[co]
    axt = fig.add_axes([0.70, 0.69, 0.22, 0.20])
    im = axt.imshow(T, cmap="Oranges", vmin=0, vmax=float(T.max()), aspect="auto")
    for a_, b_ in zip(*np.where(T == 0)):
        axt.plot(b_, a_, "s", ms=4, color=SERIES[2])
    pa, pb = int(np.where(P == picks[i - 1])[0][0]), int(np.where(C == picks[i])[0][0])
    axt.add_patch(Rectangle((pb - 0.5, pa - 0.5), 1, 1, fill=False, ec=INK, lw=2.2))
    axt.set_xlabel(f"seg {i + 1} candidates (grouped by file)", fontsize=9)
    axt.set_ylabel(f"seg {i} candidates", fontsize=9)
    axt.set_xticks([]), axt.set_yticks([])
    cb = fig.colorbar(im, ax=axt, fraction=0.05, pad=0.02)
    cb.set_label("transition cost", color=INK2, fontsize=9)
    axt.set_title(f"B  Transition matrix, boundary {i}|{i + 1}\n    (p = {p[i - 1]:.2f}, λ = {lam[i - 1]:.2f}); "
                  "green = continue, box = chosen", loc="left", fontsize=10, weight="bold")

    # C: cost breakdown of the three paths under the render's objective
    axc = fig.add_axes([0.10, 0.45, 0.40, 0.17])
    lab = {"greedy": "greedy\n(best window\nper segment)", "fixed": f"fixed-penalty\nViterbi (λ={tuned['lam_switch']:.0f})",
           "cue": "cue Viterbi\n(rendered)"}
    for k, (name, c) in enumerate(costs.items()):
        axc.barh(k, c["neg_fit"], color=SERIES[0], height=0.55, label="−fit" if k == 0 else None)
        left = 0.0
        for part, col in (("replay", SERIES[4]), ("switch", SERIES[1]), ("jump", SERIES[3])):
            v = c[part]
            axc.barh(k, v, left=left, color=col, height=0.55,
                     label=(part if part != "replay" else "replay charge") if k == 0 else None)
            left += v
        axc.plot(c["total"], k, "D", ms=9, color=INK, zorder=4)
        axc.text(max(left, 0) + 1.5, k, f"total {c['total']:.2f}  ·  {c['n_switch']} switches, {c['n_jump']} jumps",
                 va="center", fontsize=9)
    axc.axvline(0, color=INK2, lw=0.8)
    axc.set_yticks(range(3), [lab[k] for k in costs], fontsize=9)
    axc.invert_yaxis()
    axc.set_xlim(min(c["neg_fit"] for c in costs.values()) * 1.08, 42)
    axc.set_xlabel("path cost under the rendered objective (lower is better; ◆ = total)")
    axc.set_ylim(3.1, -0.5)                                       # room under the last bar for the legend
    axc.legend(frameon=False, fontsize=9, loc="lower right", ncol=4)
    axc.set_title("C  What each decoder pays: fit vs transitions", loc="left", fontsize=10.5, weight="bold")

    # D: exactness
    axd = fig.add_axes([0.62, 0.45, 0.35, 0.17])
    axd.axis("off")
    axd.set_title("D  Is the decoding exact?", loc="left", fontsize=10.5, weight="bold")
    same_ex = exact == picks
    rows = [("search space", f"{n_cand[0]} × {n_cand[1]} × {n_cand[2]} × … candidates = 10^{log10_paths:.0f} paths"),
            ("Viterbi work", f"{sum(a * b for a, b in zip(n_cand, n_cand[1:])):,} transition costs, "
                             f"{t_exact * 1000:.0f} ms on CPU (no beam)"),
            ("exact vs render (beam 100)", (f"identical path in all {n} segments" if same_ex else "paths DIFFER") +
             f", cost {ce['total']:.3f}"),
            ("brute force check", f"every path in {len(bf)} windows of {W} segments (≤ {max(b['paths'] for b in bf):,} each): "
                                  f"min = Viterbi in {sum(abs(b['brute'] - b['viterbi']) < 1e-9 for b in bf)}/{len(bf)}")]
    for r_, (a_, b_) in enumerate(rows):
        axd.text(0.0, 0.88 - r_ * 0.25, a_, fontsize=9, color=INK2, transform=axd.transAxes, va="top")
        axd.text(0.0, 0.88 - r_ * 0.25 - 0.10, b_, fontsize=10, weight="bold", transform=axd.transAxes, va="top")

    # E: lattice
    axl = fig.add_axes([0.04, 0.235, 0.93, 0.15])
    ranks = []
    for k, L in enumerate(Lc):
        order = np.argsort(L0[k].cost)
        rk = np.empty(len(order), int)
        rk[order] = np.arange(len(order))
        ranks.append(rk)
        for jj in order[:15]:
            axl.plot(k, rk[jj] + 1, "o", ms=4, color=cmap.get(index.ids[L.mid[jj]], OTHER), mec=BG, mew=0.5, zorder=2)
    for name, col, ls, lw in (("greedy", MUTED, "--", 1.3), ("fixed", SERIES[6], ":", 1.6), ("cue", INK, "-", 2.2)):
        axl.plot(range(n), [ranks[k][paths[name][k]] + 1 for k in range(n)], color=col, ls=ls, lw=lw,
                 label=lab[name].replace("\n", " "), zorder=3)
    for k in range(n):
        axl.plot(k, ranks[k][picks[k]] + 1, "o", ms=8, color=cmap.get(index.ids[Lc[k].mid[picks[k]]], OTHER), mec=INK,
                 mew=1.4, zorder=4)
    ymax = max(ranks[k][paths[nm][k]] + 1 for nm in paths for k in range(n))
    axl.set_ylim(max(ymax, 15) + 1.5, 0), axl.set_xlim(-0.5, n - 0.5)
    axl.set_xticks(range(n), [f"{k + 1}" for k in range(n)], fontsize=8)
    axl.set_ylabel("rank by fit")
    axl.grid(axis="y", color=GRID, lw=0.5)
    axl.legend(frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.5, loc="lower right", ncol=3)
    axl.set_title("E  Lattice: column = a segment's candidates (colour = music file); lines = the decoded paths",
                  loc="left", fontsize=10.5, weight="bold")

    def bars(ax, sel, title):
        prev, sw = None, 0
        for k, j in enumerate(sel):
            mid = index.ids[Lc[k].mid[j]]
            ax.add_patch(Rectangle((res[k].a, 0), res[k].b - res[k].a, 1, fc=cmap.get(mid, OTHER), ec=BG, lw=1.2))
            if prev is not None and mid != prev:
                sw += 1
                ax.plot([res[k].a] * 2, [-0.15, 1.15], color=INK, lw=1.1)
            prev = mid
        ax.set_xlim(0, res[-1].b), ax.set_ylim(-0.2, 1.2), ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{title}: {sw} file switches ({60 * sw / res[-1].b:.1f}/min)", loc="left", fontsize=10, weight="bold")
    bars(fig.add_axes([0.04, 0.155, 0.93, 0.035]), greedy, "F  Greedy soundtrack")
    bars(fig.add_axes([0.04, 0.09, 0.93, 0.035]), fixed, f"G  Fixed-penalty Viterbi (λ = {tuned['lam_switch']:.0f})")
    axv = fig.add_axes([0.04, 0.025, 0.93, 0.035])
    bars(axv, picks, "H  Cue Viterbi (rendered)")
    axv.tick_params(labelbottom=True)
    axv.set_xlabel("video time (s) · black tick = music changes file")
    save(fig, out, "sec2_transition_viterbi")
    return {"cue": {"variant": cue["variant"], "params": prm, "p": p.tolist(), "lam_switch": lam.tolist()},
            "tuned_fixed_penalties": tuned, "boundary_example": [i, i + 1], "costs": costs, "exact_no_beam": ce,
            "exact_equals_render": same_ex, "exact_ms": t_exact * 1000, "candidates_per_segment": n_cand,
            "log10_paths": log10_paths, "brute_force": bf, "paths": paths}


def main(argv=None):
    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    from bgm.render.query import featurize
    from bgm.render.scene import brief_path, ensure_brief
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--label", default=None, help="title label (default: the video file name)")
    ap.add_argument("--true-music", default=None, help="music_id of the clip's own music (dataset clips)")
    ap.add_argument("--note", default=None, help="third training bullet (default: the Godfather wording)")
    a, over = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config, over)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(cfg.paths.renders) / a.name
    feat = featurize(Path(a.video), work, cfg)
    rendered = json.loads((Path(cfg.paths.renders) / f"{a.name}.plan.json").read_text())
    store = FeatureStore(cfg)
    brief = ensure_brief(cfg, brief_path(cfg, Path(rendered["output"])), work / "silent25.mp4", feat["segments"], a.video)
    plan, cap = capture_plan(cfg, store, feat, Path(a.run), brief)
    assert [s["music_id"] for s in plan["segments"]] == [s["music_id"] for s in rendered["segments"]], \
        "the captured plan differs from the rendered one: check the overrides match the render"
    cmap = colour_map([s["music_id"] for s in rendered["segments"]], [s["end_s"] - s["start_s"] for s in rendered["segments"]])
    label = a.label or Path(a.video).name
    stats = {"section1": fig_section1(out, cap, rendered, store, cmap, a.video, label, a.true_music, a.note),
             "section2": fig_section2(out, cap, rendered, store, cmap, label)}
    (out / "sections.json").write_text(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
