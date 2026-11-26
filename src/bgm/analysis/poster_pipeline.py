"""Poster figures: one image per pipeline stage for one rendered video (e.g. godfather.mp4).

Re-runs `bgm.render.run.plan_for` with the same config as the render and captures every stage's intermediates by
wrapping the functions it calls (shortlist, fine window scoring, lattice, replay-aware decoding), so each figure shows
exactly what the render did. Needs the render's cache: renders/<name>/query.npz, renders/<name>.brief.yaml,
renders/<name>.plan.json and renders/<name>/bgm.wav.

Usage:
  python -m bgm.analysis.poster_pipeline --video godfather.mp4 --name godfather_bgm --out outputs/godfather \
      --run runs/titlesplit/pipeline_ens10/fold0/seed0 [dotted overrides, same as the render]
Writes 00_overview.png, 01_segmentation.png ... 06_rendering.png (+ .pdf) and stages.json into --out.
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import FancyBboxPatch, Rectangle

log = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BG, OTHER = "#141a2e", "#474e5e", "#8a8678", "#d6d0c2", "#ffffff", "#c9c5bb"
plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------------------- capture
def capture_plan(cfg, store, feat, run: Path, brief):
    """plan_for with its intermediates recorded: returns (plan, cap)."""
    import bgm.render.branches as br
    import bgm.render.replay as rp
    import bgm.render.run as rr
    cap: dict = {}
    o_short, o_fine, o_lat, o_rep = br.branch_shortlist, rr.fine, rr.build_lattice, rp.decode_without_replays

    def short(index, vq, w, K):
        cap.update(index=index, vq=vq, w_coarse=dict(w), K=K)
        out = o_short(index, vq, w, K)
        cap["short_core"] = list(out)
        return out

    def fine(index, vq, files, hop):
        cap["short"] = list(files)
        out = o_fine(index, vq, files, hop)
        cap["results"] = out
        return out

    def lat(index, vq, results, w, J, mx=None):
        out = o_lat(index, vq, results, w, J, mx)
        cap.update(lattice=out, w_fine=dict(w), J=J)
        return out

    def rep(decode, layers, pen, mp, *a, **k):
        cap["picks_plain"] = decode(layers)                     # Viterbi without the replay charge
        picks, L2, info = o_rep(decode, layers, pen, mp, *a, **k)
        cap.update(lattice_charged=L2, picks=picks)
        return picks, L2, info

    br.branch_shortlist, rr.fine, rr.build_lattice, rp.decode_without_replays = short, fine, lat, rep
    try:
        plan = rr.plan_for(cfg, store, feat, run, brief=brief)
    finally:
        br.branch_shortlist, rr.fine, rr.build_lattice, rp.decode_without_replays = o_short, o_fine, o_lat, o_rep
    return plan, cap


# ----------------------------------------------------------------------------- helpers
def frame_at(video: Path, t_s: float, width: int = 480) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, fr = cap.read()
    cap.release()
    assert ok, f"no frame at {t_s:.2f} s"
    rows = np.where(fr.max(axis=(1, 2)) > 24)[0]
    if len(rows) > 10:
        fr = fr[rows[0]:rows[-1] + 1]
    h, w = fr.shape[:2]
    fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out / f"{name}.png")


def film_of(store, mid: str) -> str:
    p = store.pairs
    row = p[p.music_id == mid]
    f = str(row.meta_film.iloc[0]) if len(row) and isinstance(row.meta_film.iloc[0], str) else mid
    f = f.replace("_", " ").replace(" ESubs", "")
    if f.isdigit():
        f = f"SunNXT film #{f}"
    if mid.endswith(tuple(f"_bgm_{k:04d}" for k in range(1, 100))):
        f += f" (cue {int(mid[-4:]) + 1})"
    return f


def colour_map(ids: list[str], durs: list[float] | None = None) -> dict[str, str]:
    """The 8 files that play longest get the categorical slots, in order of first appearance; the rest share the neutral."""
    tot: dict[str, float] = {}
    for m, d in zip(ids, durs or [1.0] * len(ids)):
        tot[m] = tot.get(m, 0.0) + d
    top = set(sorted(tot, key=lambda m: -tot[m])[:len(SERIES)])
    out = {}
    for m in dict.fromkeys(ids):
        out[m] = SERIES[sum(v != OTHER for v in out.values())] if m in top else OTHER
    return out


def thumbs_row(fig, rect, video, times, labels=None, border=None):
    """Equal-width thumbnails in figure rect (x0, y0, w, h)."""
    x0, y0, w, h = rect
    n = len(times)
    gap = 0.004
    tw = (w - gap * (n - 1)) / n
    axes = []
    for i, t in enumerate(times):
        ax = fig.add_axes([x0 + i * (tw + gap), y0, tw, h])
        ax.imshow(frame_at(video, t), aspect="auto")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(border[i] if border else GRID)
            s.set_linewidth(2.5 if border else 0.8)
        if labels:
            ax.set_title(labels[i], fontsize=8, color=INK2, pad=2)
        axes.append(ax)
    return axes


# ----------------------------------------------------------------------------- figures
def fig_segmentation(out, video, feat, cfg):
    from bgm.data.segments import span_embedding
    dur = float(feat["meta"]["duration_s"])
    cuts = np.asarray(feat["cut_times_s"], float)
    segs = [tuple(map(float, s)) for s in feat["segments"]]
    sig = feat["sem_siglip_1hz"].astype(np.float32)
    bounds = [0.0] + list(cuts) + [dur]
    shots = list(zip(bounds[:-1], bounds[1:]))
    cos = [float(span_embedding(sig, *a) @ span_embedding(sig, *b)) for a, b in zip(shots, shots[1:])]
    seg_starts = np.array([a for a, _ in segs[1:]])
    kept = [bool(np.any(np.abs(seg_starts - c) < 0.05)) for c in cuts]
    splits = [a for a in seg_starts if not np.any(np.abs(cuts - a) < 0.05)]
    n_short = sum(b - a < cfg.segments.min_shot_s for a, b in shots)

    fig = plt.figure(figsize=(16, 8.6))
    fig.suptitle("Step 1 · Scene segmentation: shots → scene segments", x=0.06, ha="left", fontsize=17, weight="bold")
    fig.text(0.06, 0.915, f"{len(cuts)} cuts → {len(shots)} shots → merge visually similar neighbours "
             f"(SigLIP cosine ≥ {cfg.segments.merge_cosine}, ≤ {cfg.segments.max_len_s:.0f} s), fold shots < "
             f"{cfg.segments.min_shot_s:.0f} s, split shots > {cfg.segments.max_len_s:.0f} s → {len(segs)} segments",
             fontsize=11, color=INK2)
    # A: one frame per shot sample (up to 14, evenly spread)
    pick = np.linspace(0, len(shots) - 1, min(14, len(shots))).round().astype(int)
    fig.text(0.06, 0.885, "A  Shots found by the cut detector (PySceneDetect AdaptiveDetector on the silent 25 fps copy)",
             fontsize=11, weight="bold")
    thumbs_row(fig, (0.06, 0.69, 0.90, 0.15), video, [(shots[i][0] + shots[i][1]) / 2 for i in pick],
               [f"shot {i + 1} · {shots[i][0]:.0f}s" for i in pick])

    ax = fig.add_axes([0.06, 0.55, 0.90, 0.10])
    for i, (a, b) in enumerate(shots):
        ax.add_patch(Rectangle((a, 0), b - a, 1, color="#e6e2d8" if i % 2 else "#cfcabd", lw=0))
    ax.set_xlim(0, dur), ax.set_ylim(0, 1), ax.set_yticks([0.5], ["shots"])
    ax.tick_params(labelbottom=False)
    ax.set_title("B  Shot timeline (alternating shades = consecutive shots)", loc="left", fontsize=11, weight="bold")

    ax = fig.add_axes([0.06, 0.30, 0.90, 0.19])
    c_keep, c_merge, c_fold = SERIES[1], SERIES[0], SERIES[6]
    kind = ["keep" if k else ("merge" if v >= cfg.segments.merge_cosine else "fold") for v, k in zip(cos, kept)]
    col = {"keep": c_keep, "merge": c_merge, "fold": c_fold}
    for c, v, k in zip(cuts, cos, kind):
        ax.plot([c, c], [0, v], color=col[k], lw=2, solid_capstyle="round")
        ax.plot(c, v, "o", ms=6, color=col[k], mec=BG, mew=1)
    ax.axhline(cfg.segments.merge_cosine, color=INK2, lw=1, ls="--")
    ax.text(dur, cfg.segments.merge_cosine + 0.01, f"merge threshold {cfg.segments.merge_cosine}", ha="right",
            va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(0, dur), ax.set_ylim(min(0.3, min(cos) - 0.05) if cos else 0, 1.02)
    ax.set_ylabel("cosine(shot, next shot)")
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.plot([], [], "o", color=c_keep, label=f"kept as a segment boundary ({kind.count('keep')})")
    ax.plot([], [], "o", color=c_merge, label=f"merged: same scene, similar look ({kind.count('merge')})")
    ax.plot([], [], "o", color=c_fold, label=f"merged: shot under {cfg.segments.min_shot_s:.0f} s folded into a neighbour ({kind.count('fold')})")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=True, facecolor=BG, edgecolor="none", framealpha=0.95, ncol=3, fontsize=9)
    ax.set_title("C  Visual similarity across each cut (SigLIP so400m, 1 frame/s)", loc="left", fontsize=11, weight="bold")

    ax = fig.add_axes([0.06, 0.13, 0.90, 0.11])
    for i, (a, b) in enumerate(segs):
        ax.add_patch(Rectangle((a, 0), b - a, 1, fc=SERIES[i % len(SERIES)], ec=BG, lw=2, alpha=0.9))
        if b - a > dur / 60:
            ax.text((a + b) / 2, 0.5, str(i + 1), ha="center", va="center", fontsize=9, color="white", weight="bold")
    for s in splits:
        ax.plot([s, s], [-0.15, 1.15], color=INK, lw=1.2, ls=":")
    ax.set_xlim(0, dur), ax.set_ylim(-0.2, 1.2), ax.set_yticks([0.5], ["segments"])
    ax.set_xlabel("video time (s)")
    note = f"; dotted = long shot split at {cfg.segments.max_len_s:.0f} s" if splits else ""
    ax.set_title(f"D  Final {len(segs)} segments, the units music is chosen for (colours repeat; numbers = order){note}",
                 loc="left", fontsize=11, weight="bold")
    save(fig, out, "01_segmentation")
    return {"n_cuts": len(cuts), "n_shots": len(shots), "n_short_shots": int(n_short), "n_segments": len(segs),
            "cuts_kept": int(sum(kept)), "cuts_merged": int(len(kept) - sum(kept)), "split_points_s": [round(s, 2) for s in splits],
            "segment_lengths_s": [round(b - a, 2) for a, b in segs]}


def fig_brief(out, video, brief, n_show: int = 8):
    """n_show evenly spaced segment cards + one emotion strip over every segment (fits a poster panel)."""
    emo = sorted({b["emotion"] for b in brief})
    ecol = {e: SERIES[i % len(SERIES)] for i, e in enumerate(emo)}
    pick = np.linspace(0, len(brief) - 1, min(n_show, len(brief))).round().astype(int)
    cols, rows = 4, int(np.ceil(len(pick) / 4))
    H = 2.2 + 3.0 * rows
    fig = plt.figure(figsize=(16, H))
    fig.text(0.04, 1 - 0.45 / H, "Step 2 · Scene understanding: what is happening, and what should the music feel like?",
             fontsize=17, weight="bold", va="top")
    fig.text(0.04, 1 - 1.0 / H, f"A local vision-language model (Qwen2.5-VL-3B) watches each of the {len(brief)} segments and writes a "
             "scene brief: scene, emotion, music text.\nThe music text is matched to the library with CLAP and becomes one "
             f"retrieval branch. {len(pick)} of the {len(brief)} briefs are shown.", fontsize=11, color=INK2, va="top")
    dur = brief[-1]["end_s"]
    ax = fig.add_axes([0.04, 1 - 1.95 / H, 0.92, 0.32 / H])
    for i, b in enumerate(brief):
        ax.add_patch(Rectangle((b["start_s"], 0), b["end_s"] - b["start_s"], 1, fc=ecol[b["emotion"]], ec=BG, lw=1.2))
    for i in pick:
        b = brief[i]
        ax.text((b["start_s"] + b["end_s"]) / 2, 0.5, str(i + 1), ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.set_xlim(0, dur), ax.set_ylim(0, 1), ax.set_yticks([]), ax.tick_params(labelsize=8)
    counts = ", ".join(f"{e} {sum(b['emotion'] == e for b in brief)}" for e in emo)
    ax.set_title(f"Emotion of every segment ({counts}); numbered = shown below", loc="left", fontsize=10, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    for k, i in enumerate(pick):
        b = brief[i]
        r, c = divmod(k, cols)
        w = 0.92 / cols - 0.015
        x0, top = 0.04 + c * (0.92 / cols), 1 - (2.45 + r * 3.0) / H
        axi = fig.add_axes([x0, top - 1.35 / H, w * 0.55, 1.35 / H])
        axi.imshow(frame_at(video, (b["start_s"] + b["end_s"]) / 2), aspect="auto")
        axi.set_xticks([]), axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(True), s.set_color(ecol[b["emotion"]]), s.set_linewidth(3)
        tx = x0 + w * 0.58
        fig.text(tx, top - 0.15 / H, f"Segment {i + 1}", fontsize=11, weight="bold", va="top")
        fig.text(tx, top - 0.5 / H, f"{b['start_s']:.1f}–{b['end_s']:.1f} s", fontsize=9.5, color=INK2, va="top")
        fig.text(tx, top - 0.95 / H, f" {b['emotion']} ", fontsize=10.5, color="white", weight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.25", fc=ecol[b["emotion"]], ec="none"))
        txt = textwrap.fill(textwrap.shorten(b["scene"], 120, placeholder="…"), 50) + "\n" + \
            textwrap.fill(textwrap.shorten("♪ " + b["music"], 110, placeholder="…"), 50)
        fig.text(x0, top - 1.45 / H, txt, fontsize=8.6, color=INK, va="top", linespacing=1.3)
    save(fig, out, "02_scene_brief")
    return {"emotions": {e: sum(b["emotion"] == e for b in brief) for e in emo}}


def fig_coarse(out, cap, store):
    from bgm.retrieval.pipeline import fuse, zstats
    index, vq, w, K = cap["index"], cap["vq"], cap["w_coarse"], cap["K"]
    spans = [(0.0, vq.duration)] + list(vq.segments)
    S = []
    for a, b in spans:
        gv, pv = vq.pooled(a, b)
        raw = {**index.file_scores(gv, pv), **index.file_branches(a, b)}
        S.append(fuse(raw, zstats(raw), w))
    S = np.array(S)                                   # [spans, G]
    Z = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    short = sorted(cap["short"], key=lambda g: -Z[1:, g].max())
    G = Z.shape[1]
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Step 3 · Coarse retrieval: shortlist whole music files from the library", x=0.05, ha="left",
                 fontsize=17, weight="bold")
    fig.text(0.05, 0.905, f"Each span (whole video, then each segment) is scored against all {G:,} library files; the top "
             f"{K} of every span are kept → {len(cap['short'])} shortlisted files ({100 * len(cap['short']) / G:.1f}% of the library)",
             fontsize=11, color=INK2)
    fig.text(0.05, 0.875, "Score = ensemble of 10 trained video↔music heads (SigLIP ↔ CLAP+MERT) + pace + scene-brief text + length fit, "
             "each z-scored and summed", fontsize=11, color=INK2)
    ax = fig.add_axes([0.05, 0.12, 0.30, 0.66])
    s0 = Z[0]
    top = np.argsort(-s0)[:K]
    ax.hist(s0, bins=60, color=OTHER, edgecolor=BG, lw=0.5)
    for g in top:
        ax.axvline(s0[g], color=SERIES[1], lw=1.2, alpha=0.9)
    ax.set_xlabel("fused score (z over the library)")
    ax.set_ylabel("number of music files")
    ax.set_title(f"A  Whole video vs every library file\n(orange = top {K}, kept)", loc="left", fontsize=11, weight="bold")
    ax2 = fig.add_axes([0.42, 0.12, 0.50, 0.66])
    M = Z[:, short]
    im = ax2.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(3.5, float(np.percentile(M, 99))))
    order = np.argsort(-Z, axis=1)[:, :K]
    for i in range(len(spans)):
        for j, g in enumerate(short):
            if g in order[i]:
                ax2.plot(j, i, "s", ms=4, mfc="none", mec=SERIES[1], mew=1.1)
    ax2.set_yticks(range(len(spans)), ["whole video"] + [f"seg {i}" for i in range(1, len(spans))], fontsize=8)
    ax2.set_xticks([])
    ax2.set_xlabel(f"{len(short)} shortlisted music files (sorted by best segment score)")
    ax2.set_title(f"B  Score of each span for each shortlisted file (orange box = in that span's top {K})",
                  loc="left", fontsize=11, weight="bold")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    cb.set_label("z score", color=INK2)
    save(fig, out, "03_coarse_shortlist")
    return {"library_files": int(G), "K": int(K), "shortlist": len(cap["short"])}


def fig_fine(out, cap, plan, store, cmap):
    from bgm.retrieval.pipeline import fuse
    res, w, index = cap["results"], cap["w_fine"], cap["index"]
    segs = plan["segments"]
    # example: the longest segment
    k = int(np.argmax([r.b - r.a for r in res]))
    r = res[k]
    S = fuse(r.raw, r.stats, w)
    fig = plt.figure(figsize=(16, 8.2))
    fig.suptitle("Step 4 · Fine retrieval: find the best-fitting window inside each shortlisted file", x=0.05,
                 ha="left", fontsize=17, weight="bold")
    nwin = [len(x.fi) for x in res]
    fig.text(0.05, 0.905, f"For a segment of length d, every window of length d is slid every {plan['fusion']['window_hop_s']} s "
             f"over the shortlisted files and scored ({min(nwin):,}–{max(nwin):,} windows per segment).", fontsize=11, color=INK2)
    ax = fig.add_axes([0.05, 0.14, 0.33, 0.62])
    best_per_file = {}
    for f in np.unique(r.fi):
        m = r.fi == f
        best_per_file[int(f)] = float(S[m].max())
    topf = sorted(best_per_file, key=lambda f: -best_per_file[f])[:5]
    chosen = segs[k]["music_id"]
    for i, f in enumerate(topf):
        m = r.fi == f
        mid = index.ids[f]
        tag = "  ← used after stitching (Step 5)" if mid == chosen else ""
        ax.plot(r.st[m], S[m], lw=2.6 if tag else 1.8, color=SERIES[i], label=textwrap.shorten(film_of(store, mid), 34) + tag)
    j = int(np.argmax(S))
    ax.plot(r.st[j], S[j], "o", ms=9, color=INK, mfc="none", mew=2)
    ax.annotate("best window by fit alone", (r.st[j], S[j]), xytext=(10, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("window start inside the music file (s)")
    ax.set_ylabel("fused window score (z)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=9, title="top 5 files for this segment (source film)", title_fontsize=9.5,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"A  Segment {k + 1} ({r.a:.1f}–{r.b:.1f} s, {r.b - r.a:.1f} s long): score of every window, per file",
                 loc="left", fontsize=11, weight="bold")
    ins = fig.add_axes([0.05, 0.80, 0.11, 0.075])
    ins.imshow(frame_at(Path(plan["video"]), (r.a + r.b) / 2), aspect="auto")
    ins.set_xticks([]), ins.set_yticks([])
    # B: branch contributions of each segment's chosen window
    ax = fig.add_axes([0.66, 0.14, 0.32, 0.62])
    names = {"sem": "picture ↔ music (trained heads)", "pace": "pace (motion ↔ energy)", "brief": "scene-brief text (CLAP)",
             "length": "length fit (no loops)"}
    br = [b for b in ("sem", "pace", "brief", "length") if w.get(b, 0) > 0]
    y = np.arange(len(segs))
    left_p, left_n = np.zeros(len(segs)), np.zeros(len(segs))
    for i, b in enumerate(br):
        v = np.array([s["z"].get(b, 0.0) * w[b] for s in segs])
        lp = np.where(v >= 0, left_p, left_n)
        ax.barh(y, v, left=lp, height=0.7, color=SERIES[[0, 3, 4, 2][i]], edgecolor=BG, lw=1, label=names[b])
        left_p, left_n = left_p + np.maximum(v, 0), left_n + np.minimum(v, 0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [f"seg {i + 1}" for i in y], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("contribution to the chosen window's score (z × weight)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2)
    ax.set_title("B  Why each chosen window won: score per branch", loc="left", fontsize=11, weight="bold", pad=48)
    save(fig, out, "04_fine_windows")
    return {"example_segment": k + 1, "windows_per_segment": [min(nwin), max(nwin)]}


def fig_viterbi(out, cap, plan, store, cmap):
    from bgm.retrieval.stitch_eval import greedy_lattice
    L, Lc, picks = cap["lattice"], cap["lattice_charged"], cap["picks"]
    greedy = greedy_lattice(L)
    index, res = cap["index"], cap["results"]
    n = len(L)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Step 5 · Stitching with Viterbi: fit each segment, but keep the music continuous", x=0.05,
                 ha="left", fontsize=17, weight="bold")
    pen = plan["penalties"]
    fig.text(0.05, 0.91, f"Each segment keeps its top {cap['J']} windows + the continuation of every previous candidate. "
             f"Path cost = −fit + switch penalty λ={pen['lam_switch']:.1f} (+ key/tempo distance × μ={pen['mu']:.1f}) "
             f"+ jump-within-file penalty {pen['lam_jump']:.1f}; continuing the same music costs 0.", fontsize=11, color=INK2)
    ax = fig.add_axes([0.05, 0.40, 0.90, 0.45])
    ranks = []
    ymax = 0
    for i, Li in enumerate(L):
        order = np.argsort(Li.cost)
        rank = np.empty(len(order), int)
        rank[order] = np.arange(len(order))
        ranks.append(rank)
        show = order[:25]
        for jj in show:
            mid = index.ids[Li.mid[jj]]
            ax.plot(i, rank[jj] + 1, "o", ms=6, color=cmap.get(mid, OTHER), mec=BG, mew=0.8,
                    alpha=1 if mid in cmap else 0.7, zorder=2)
        ymax = max(ymax, 25, rank[picks[i]] + 1, rank[greedy[i]] + 1)
    gy = [ranks[i][greedy[i]] + 1 for i in range(n)]
    vy = [ranks[i][picks[i]] + 1 for i in range(n)]
    ax.plot(range(n), gy, color=MUTED, lw=1.5, ls="--", zorder=1, label="greedy: best window per segment, independently")
    ax.plot(range(n), vy, color=INK, lw=2.5, zorder=3, label="Viterbi: best whole path")
    for i in range(n):
        mid = index.ids[L[i].mid[picks[i]]]
        ax.plot(i, vy[i], "o", ms=11, color=cmap.get(mid, OTHER), mec=INK, mew=1.8, zorder=4)
    ax.set_ylim(ymax + 1, 0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n), [f"seg {i + 1}" for i in range(n)], fontsize=8.5)
    ax.set_ylabel("candidate rank by fit\n(1 = best-fitting window)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True, facecolor=BG, edgecolor=GRID, framealpha=1)
    ax.set_title("A  Candidate lattice (dot colour = music file; grey = other files)", loc="left", fontsize=11, weight="bold")

    def bars(ax, sel, title):
        prev = None
        sw = 0
        for i, j in enumerate(sel):
            mid = index.ids[L[i].mid[j]]
            ax.add_patch(Rectangle((res[i].a, 0), res[i].b - res[i].a, 1, fc=cmap.get(mid, OTHER), ec=BG, lw=1.5))
            if prev is not None and mid != prev:
                sw += 1
                ax.plot([res[i].a] * 2, [-0.15, 1.15], color=INK, lw=1.2)
            prev = mid
        ax.set_xlim(0, res[-1].b), ax.set_ylim(-0.2, 1.2), ax.set_yticks([])
        ax.set_title(f"{title}: {sw} music switches ({60 * sw / res[-1].b:.1f}/min)", loc="left", fontsize=10.5, weight="bold")
        return sw
    sw_g = bars(fig.add_axes([0.05, 0.27, 0.90, 0.06]), greedy, "B  Greedy soundtrack")
    sw_v = bars(ax3 := fig.add_axes([0.05, 0.15, 0.90, 0.06]), picks, "C  Viterbi soundtrack (used)")
    ax3.set_xlabel("video time (s)  ·  black ticks = the music changes")
    used = list(dict.fromkeys(index.ids[L[i].mid[picks[i]]] for i in range(n)))
    hs = [Rectangle((0, 0), 1, 1, fc=cmap[m]) for m in used if cmap.get(m, OTHER) != OTHER]
    n_other = sum(cmap.get(m, OTHER) == OTHER for m in used)
    fig.legend(hs, [textwrap.shorten(film_of(store, m), 32) for m in used if cmap.get(m, OTHER) != OTHER],
               loc="upper left", bbox_to_anchor=(0.05, 0.075), ncol=4, frameon=False, fontsize=9,
               title=f"music in the final soundtrack, by source film ({len(used)} files; grey = {n_other} shorter-playing files)",
               title_fontsize=9.5, alignment="left")
    save(fig, out, "05_viterbi")
    return {"switches_greedy": sw_g, "switches_viterbi": sw_v, "files_used": len(used)}


def fig_render(out, plan, store, cmap, cfg, cuts):
    from bgm.render.render import Run, crossfade_gains, crossfade_lengths
    wav, sr = sf.read(plan["bgm_wav"])
    x = wav.mean(1) if wav.ndim > 1 else wav
    dur = len(x) / sr
    pieces = plan["pieces"]
    seams = plan["seams_s"]
    runs = [Run(**p) for p in pieces]
    xf = crossfade_lengths(runs, cfg.render)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Step 6 · Rendering: beat-snap, crossfade, loudness-match, mux", x=0.05, ha="left", fontsize=17,
                 weight="bold")
    fig.text(0.05, 0.91, f"{len(plan['segments'])} segments → {len(pieces)} music pieces (consecutive continuations merge). "
             f"Output {plan['output_lufs']:.1f} LUFS (target {cfg.render.target_lufs}), peak {plan['output_peak_dbfs']:.1f} dBFS; "
             "music sample 0 = first video frame.", fontsize=11, color=INK2)
    # A: beat snap
    ax = fig.add_axes([0.05, 0.62, 0.42, 0.22])
    bs = plan.get("beat_sync", [])
    sh = [b["shift_s"] for b in bs]
    ax.bar(range(1, len(sh) + 1), sh, color=[cmap.get(b["music_id"], OTHER) for b in bs], width=0.7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("music piece")
    ax.set_ylabel("offset shift (s)")
    nb, na, nc = (sum(b["on_beat_before"] for b in bs), sum(b["on_beat_after"] for b in bs), sum(b["inner_cuts"] for b in bs))
    ax.set_title(f"A  Beat snap: each piece's start in its music file moves ≤ {cfg.render.beat_sync_max_shift_s:.0f} s so the\n"
                 f"   music enters on a beat at the video cut (median |shift| {np.median(np.abs(sh)):.2f} s)", loc="left", fontsize=11, weight="bold")
    ax.grid(axis="y", color=GRID, lw=0.5)
    # B: crossfade shape on one seam
    ax = fig.add_axes([0.56, 0.62, 0.39, 0.22])
    if seams:
        i = int(np.argmax(xf)) if len(xf) else 0
        d = xf[i] if len(xf) else 1.0
        n = int(d * 1000)
        g_out, g_in = crossfade_gains(n, cfg.render.crossfade_shape)
        tt = np.linspace(-d / 2, d / 2, n)
        ax.plot(tt, g_out, lw=2.5, color=SERIES[0], label="outgoing music")
        ax.plot(tt, g_in, lw=2.5, color=SERIES[1], label="incoming music")
        ax.plot(tt, np.sqrt(g_out ** 2 + g_in ** 2), lw=1.2, ls="--", color=INK2, label="total power (constant)")
        ax.set_xlabel(f"time around the seam at {seams[i]:.1f} s (s)")
        ax.set_ylabel("gain")
        ax.legend(frameon=False, fontsize=9, loc="center right")
        ax.set_ylim(0, 1.12)
    ax.set_title(f"B  Equal-power crossfade at each seam ({cfg.render.crossfade_min_s:.0f}–{cfg.render.crossfade_max_s:.0f} s, "
                 f"{cfg.render.crossfade_beats} beats) + loudness-matching gain ramps", loc="left", fontsize=11, weight="bold")
    # C: waveform envelope coloured by piece
    ax = fig.add_axes([0.05, 0.14, 0.90, 0.36])
    hop = int(0.05 * sr)
    env = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop])
    t = np.arange(len(env)) * hop / sr
    for k, p in enumerate(pieces):
        a, b = p["start"], p["end"]
        m = (t >= a) & (t <= b)
        ax.fill_between(t[m], -env[m], env[m], color=cmap.get(p["music_id"], OTHER), lw=0)
    for s, j in zip(seams, plan.get("seam_loudness_jump_LU", [])):
        ax.axvline(s, color=INK, lw=0.8, ls=":")
    for c in cuts:
        ax.plot([c, c], [-1.05 * env.max(), -0.95 * env.max()], color=INK2, lw=0.8)
    for a, b in plan.get("silence", []) or []:
        ax.axvspan(a, b, color=GRID, alpha=0.6, lw=0)
        ax.text((a + b) / 2, 0.9 * env.max(), "no music (brief)", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(0, dur)
    ax.set_yticks([])
    ax.set_xlabel("video time (s)  ·  dotted = seams between pieces, ticks below = video cuts")
    ax.set_title("C  Final soundtrack (RMS envelope, coloured by music file), normalised to −16 LUFS with a −1 dBFS limiter, "
                 "then muxed onto the video", loc="left", fontsize=11, weight="bold")
    save(fig, out, "06_rendering")
    return {"pieces": len(pieces), "seams": len(seams), "crossfade_s": [round(v, 2) for v in xf],
            "cuts_on_beat_before_after_of": [nb, na, nc]}


def fig_overview(out, video, plan, stats):
    steps = [("1  Segment", f"{stats['seg']['n_shots']} shots →\n{stats['seg']['n_segments']} scene segments"),
             ("2  Understand", "VLM scene brief:\nscene · emotion · music text"),
             ("3  Shortlist", f"{stats['coarse']['library_files']:,} files →\n{stats['coarse']['shortlist']} candidates"),
             ("4  Fine match", f"slide windows,\n{stats['fine']['windows_per_segment'][1]:,} max per segment"),
             ("5  Viterbi", f"greedy {stats['vit']['switches_greedy']} switches →\n{stats['vit']['switches_viterbi']} switches"),
             ("6  Render", f"beat-snap, crossfade,\n{plan['output_lufs']:.0f} LUFS, mux")]
    fig = plt.figure(figsize=(16, 3.4))
    fig.suptitle("From a silent video to a full soundtrack", x=0.03, ha="left", fontsize=17, weight="bold")
    axv = fig.add_axes([0.01, 0.15, 0.12, 0.55])
    axv.imshow(frame_at(video, plan["duration_s"] * 0.4)), axv.axis("off")
    axv.set_title(f"silent video\n{plan['duration_s']:.0f} s", fontsize=10)
    ax = fig.add_axes([0.14, 0.05, 0.85, 0.85])
    ax.set_xlim(0, 6), ax.set_ylim(0, 1), ax.axis("off")
    for i, (h, s) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.25), 0.82, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=SERIES[i], ec="none", alpha=0.95))
        ax.text(i + 0.47, 0.62, h, ha="center", va="center", fontsize=13, color="white", weight="bold")
        ax.text(i + 0.47, 0.42, s, ha="center", va="center", fontsize=10, color="white")
        if i < 5:
            ax.annotate("", (i + 1.06, 0.5), (i + 0.9, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    save(fig, out, "00_overview")


def main(argv=None):
    from omegaconf import OmegaConf

    from bgm.config import load_config
    from bgm.eval.store import FeatureStore
    from bgm.render.query import featurize
    from bgm.render.scene import brief_path, ensure_brief
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True, help="the render's output stem (renders/<name>/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    a, over = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config, over)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(a.video)
    work = Path(cfg.paths.renders) / a.name
    feat = featurize(video, work, cfg)                      # cached by the render
    rendered = json.loads((Path(cfg.paths.renders) / f"{a.name}.plan.json").read_text())
    store = FeatureStore(cfg)
    brief = ensure_brief(cfg, brief_path(cfg, Path(rendered["output"])), work / "silent25.mp4", feat["segments"], str(video))
    plan, cap = capture_plan(cfg, store, feat, Path(a.run), brief)
    same = [s["music_id"] for s in plan["segments"]] == [s["music_id"] for s in rendered["segments"]]
    log.info("re-planned path matches the render: %s", same)
    assert same, "the captured plan differs from the rendered one: check the overrides match the render"
    rendered["video"] = str(video)
    cmap = colour_map([s["music_id"] for s in rendered["segments"]], [s["end_s"] - s["start_s"] for s in rendered["segments"]])
    stats = {"seg": fig_segmentation(out, video, feat, cfg),
             "brief": fig_brief(out, video, brief),
             "coarse": fig_coarse(out, cap, store),
             "fine": fig_fine(out, cap, rendered, store, cmap),
             "vit": fig_viterbi(out, cap, rendered, store, cmap),
             "render": fig_render(out, rendered, store, cmap, cfg, np.asarray(feat["cut_times_s"], float))}
    fig_overview(out, video, rendered, stats)
    stats["music"] = [{"start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2), "music_id": s["music_id"],
                       "film": film_of(store, s["music_id"]), "offset_s": round(s["offset_s"], 2)} for s in rendered["segments"]]
    stats["config"] = OmegaConf.to_container(cfg.render, resolve=True)
    (out / "stages.json").write_text(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
