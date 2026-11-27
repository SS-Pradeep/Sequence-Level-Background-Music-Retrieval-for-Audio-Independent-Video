"""Run the final system on a video and show every step while it runs.

    .venv/bin/python tools/show_steps.py input.mp4 [output.mp4] [--no-open] [--cached]

This is the same run as add_music.sh: the same config, the same fold 0 run and the 10-head ensemble, and the output
video is the same file. The script only adds hooks around the pipeline's own functions (nothing is re-implemented).
Each hook draws one picture as soon as its step finishes and saves it to

    renders/<output name>/steps/NN_<step>.png

and steps.html in that folder opens in the browser and refreshes itself, so each picture shows up live during the run.

--cached   reuse renders/<name>/query.npz (by default it is deleted, so the video steps run again on camera)
--no-open  do not open the browser page
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONFIG = "configs/best_semantic.yaml"
RUN = "runs/cleaned/pipeline_ens10/fold0/seed0"
OVERRIDES = ["pipeline.dyn_variant=null", "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"]
COLORS = plt.get_cmap("tab10").colors


def short(mid: str) -> str:
    """sunnxt_94_94_bgm_0002 -> sunnxt 94 #0002 (the same short names as Demo_Outputs/*/summary.txt)."""
    m = re.match(r"(?:combined_full_bgm_dataset_)?(\w+?)_+(\d+)_+\d+_bgm_(\d+)(_p\d)?$", mid)
    return f"{m[1]} {m[2]} #{m[3]}{m[4] or ''}" if m else mid


# ---------------------------------------------------------------- live report
class Report:
    def __init__(self, folder: Path, video: Path, auto_open: bool):
        self.dir = folder
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        self.video, self.steps, self.t0, self.t_last = video, [], time.time(), time.time()
        self.page = folder / "steps.html"
        self.write(done=False)
        if auto_open:
            webbrowser.open(self.page.as_uri())

    def step(self, title: str, what: str, facts: list[str], fig=None):
        n = len(self.steps) + 1
        now = time.time()
        name = None
        if fig is not None:
            name = f"{n:02d}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}.png"
            fig.savefig(self.dir / name, dpi=110, bbox_inches="tight")
            plt.close(fig)
        self.steps.append((title, what, facts, name, now - self.t_last))
        print(f"\n\033[1;36m━━ STEP {n}: {title} \033[0m (+{now - self.t_last:.1f} s)", flush=True)
        print(f"   {what}", flush=True)
        for f in facts:
            print(f"   • {f}", flush=True)
        if name:
            print(f"   picture: {self.dir / name}", flush=True)
        self.t_last = now
        self.write(done=False)

    def write(self, done: bool):
        cards = []
        for i, (title, what, facts, img, dt) in enumerate(self.steps, 1):
            li = "".join(f"<li>{html.escape(f)}</li>" for f in facts)
            im = f'<img src="{img}" alt="{html.escape(title)}">' if img else ""
            cards.append(f'<section id="s{i}"><h2><span>{i}</span>{html.escape(title)}<small>+{dt:.1f} s</small></h2>'
                         f"<p>{html.escape(what)}</p><ul>{li}</ul>{im}</section>")
        status = (f"Done in {time.time() - self.t0:.0f} s" if done else
                  f'<span class="dot"></span>Running… step {len(self.steps) + 1} in progress')
        refresh = "" if done else '<meta http-equiv="refresh" content="2">'
        self.page.write_text(f"""<!doctype html><html><head><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pipeline Steps</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#5d6676;--acc:#2563eb;--line:#e3e6eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11141a;--card:#1a1f28;--ink:#e6e9ef;--mute:#9aa3b2;--acc:#60a5fa;--line:#2a313d}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:1}}
header h1{{margin:0;font-size:18px}} header p{{margin:2px 0 0;color:var(--mute)}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}}
h2{{margin:0 0 6px;font-size:17px;display:flex;align-items:center;gap:10px}}
h2 span{{background:var(--acc);color:#fff;border-radius:50%;width:26px;height:26px;display:inline-grid;place-items:center;font-size:13px}}
h2 small{{margin-left:auto;color:var(--mute);font-weight:400}}
p{{margin:0 0 6px;color:var(--mute)}} ul{{margin:0 0 10px;padding-left:20px}}
img{{max-width:100%;border-radius:6px;background:#fff}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;background:#16a34a;margin-right:6px;animation:b 1s infinite}}
@keyframes b{{50%{{opacity:.2}}}}
</style></head><body>
<header><h1>Background music pipeline · {html.escape(self.video.name)}</h1><p>{status}</p></header>
<main>{''.join(cards)}</main>
<script>
const k="steps:"+location.pathname, n={len(self.steps)};
try{{const o=JSON.parse(sessionStorage.getItem(k)||"{{}}");
 if(o.n===n) window.scrollTo(0,o.y||0); else {{const s=document.getElementById("s"+n); if(s) s.scrollIntoView();}}
 addEventListener("beforeunload",()=>sessionStorage.setItem(k,JSON.stringify({{n,y:scrollY}})));}}catch(e){{}}
</script></body></html>""")


# ---------------------------------------------------------------- pictures
def filmstrip(ax, frames, dur):
    """Evenly spaced frames side by side along the time axis, as many as fit without squashing them."""
    fw, fh = ax.figure.get_size_inches()
    box = ax.get_position()
    ar = frames.shape[2] / frames.shape[1]
    n = int(np.clip(round(box.width * fw / (box.height * fh * ar)), 1, len(frames)))
    for j in range(n):
        a, b = j * dur / n, (j + 1) * dur / n
        k = min(len(frames) - 1, int((a + b) / 2))
        ax.imshow(frames[k], extent=[a, b, 0, 1], aspect="auto")
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1)
    ax.set_yticks([])


def fig_frames(frames):
    n = len(frames)
    idx = np.linspace(0, n - 1, min(n, 18)).round().astype(int)
    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.1 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, k in zip(np.ravel(axes), idx):
        ax.imshow(frames[k])
        ax.set_title(f"{k + 0.5:.1f} s", fontsize=9)
    fig.suptitle(f"One frame per second from the silent copy ({n} frames; {len(idx)} shown)", fontsize=12)
    return fig


def fig_shots(frames, cuts, dyn, dur):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    filmstrip(a1, frames, dur)
    t = np.arange(len(dyn)) / 25.0
    a2.plot(t, dyn[:, 0], color="#2563eb", lw=0.8, label="motion (optical flow)")
    a2.fill_between(t, dyn[:, 0], alpha=0.15, color="#2563eb")
    for c in cuts:
        a1.axvline(c, color="red", lw=1.5)
        a2.axvline(c, color="red", lw=1, ls="--")
    a2.set_xlabel("video time (s)")
    a2.set_ylabel("motion")
    a2.legend(loc="upper right", fontsize=8)
    a1.set_title(f"Shot cuts found by PySceneDetect: {len(cuts)} cuts → {len(cuts) + 1} shots (red lines)")
    return fig


def fig_similarity(sig, cuts, segs, dur):
    x = sig.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    S = x @ x.T
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(S, cmap="magma", vmin=max(0.3, float(np.percentile(S, 2))), vmax=1, extent=[0, len(S), len(S), 0])
    for c in cuts:
        ax.axvline(c, color="white", lw=0.4, alpha=0.5)
        ax.axhline(c, color="white", lw=0.4, alpha=0.5)
    for i, (a, b) in enumerate(segs):
        ax.add_patch(plt.Rectangle((a, a), b - a, b - a, fill=False, ec=COLORS[i % 10], lw=2.2))
    fig.colorbar(im, ax=ax, label="SigLIP cosine similarity")
    ax.set_xlabel("second")
    ax.set_ylabel("second")
    ax.set_title("SigLIP: how alike every pair of seconds looks\n(boxes = the segments built from this)")
    return fig


def fig_segments(frames, cuts, segs, dur):
    fig, ax = plt.subplots(figsize=(14, 2.6))
    filmstrip(ax, frames, dur)
    for i, (a, b) in enumerate(segs):
        ax.axvspan(a, b, ymin=0, ymax=0.18, color=COLORS[i % 10])
        ax.text((a + b) / 2, 0.08, f"seg {i}\n{b - a:.1f}s", ha="center", va="center", fontsize=8, color="white",
                weight="bold", transform=ax.get_xaxis_transform())
        ax.axvline(a, color="white", lw=3)
    for c in cuts:
        ax.axvline(c, color="red", lw=0.8, ls=":")
    ax.set_xlabel("video time (s)")
    ax.set_title(f"{len(cuts) + 1} shots merged into {len(segs)} segments (merge while SigLIP cosine ≥ 0.85, 2–20 s each)")
    return fig


def fig_coarse(ids, scores, short_idx, K):
    order = np.argsort(-scores)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    in_short = np.zeros(len(ids), bool)
    in_short[short_idx] = True
    a1.hist(scores[~in_short], bins=40, color="#9ca3af", label="not kept")
    a1.hist(scores[in_short], bins=40, color="#2563eb", alpha=0.85, label=f"shortlist ({in_short.sum()} files)")
    a1.set_xlabel("whole-video match score (z)")
    a1.set_ylabel("music files")
    a1.legend()
    a1.set_title(f"All {len(ids)} library files scored against the whole video")
    top = order[:15][::-1]
    a2.barh([short(ids[i]) for i in top], scores[top], color=["#2563eb" if in_short[i] else "#9ca3af" for i in top])
    a2.set_xlabel("score (z)")
    a2.set_title("Top 15 files for the whole video")
    a2.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Coarse stage: keep the top K={K} files for the whole video and for each segment", fontsize=12)
    fig.tight_layout()
    return fig


def fig_fine(ids, results, w):
    from bgm.retrieval.pipeline import fuse
    files = sorted({int(f) for r in results for f in np.unique(r.fi)})
    pos = {f: j for j, f in enumerate(files)}
    M = np.full((len(results), len(files)), np.nan)
    best = []
    for i, r in enumerate(results):
        S = fuse(r.raw, r.stats, w)
        for f in np.unique(r.fi):
            M[i, pos[int(f)]] = S[r.fi == f].max()
        k = int(np.argmax(S))
        best.append((pos[int(r.fi[k])], r.st[k]))
    order = np.argsort(-np.nanmean(M, 0))[:40]
    M = M[:, order]
    inv = {j: n for n, j in enumerate(order)}
    fig, ax = plt.subplots(figsize=(14, 1.6 + 0.45 * len(results)))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    for i, (j, st) in enumerate(best):
        if j in inv:
            ax.plot(inv[j], i, marker="*", color="red", ms=13)
            ax.text(inv[j] + 0.5, i, f"{st:.1f}s", color="white", fontsize=7, va="center")
    ax.set_yticks(range(len(results)), [f"seg {i}  {r.a:.0f}–{r.b:.0f}s" for i, r in enumerate(results)], fontsize=8)
    ax.set_xticks(range(len(order)), [short(ids[files[j]]) for j in order], rotation=70, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="best window score (z)")
    n_win = sum(len(r.fi) for r in results)
    ax.set_title(f"Fine stage: 0.5 s sliding windows of each shortlisted file scored against each segment "
                 f"({n_win:,} windows; ★ = best window, with its start time)")
    return fig


def fig_cue(p, segs):
    b = [s[1] for s in segs[:-1]]
    fig, ax = plt.subplots(figsize=(14, 3.4))
    col = ["#dc2626" if x >= 0.5 else "#16a34a" for x in p]
    ax.bar(range(len(p)), p, color=col)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    for i, x in enumerate(p):
        ax.text(i, x + 0.02, f"{x:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(p)), [f"seg {i}→{i + 1}\n{t:.1f} s" for i, t in enumerate(b)], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("P(music changes)")
    ax.set_title("Transformer change predictor: red = likely change (switching is cheap), green = keep the same music")
    return fig


def fig_viterbi(ids, lat, picks, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    J = max(L.n_top for L in lat)
    path_xy, labels = [], []
    music_col = {}
    for i, (L, j) in enumerate(zip(lat, picks)):
        order = np.argsort(L.cost)
        rank = np.empty(len(order), int)
        rank[order] = np.arange(len(order))
        show = order[: max(J, 1)]
        ax.scatter(np.full(len(show), i), rank[show], c=-L.cost[show], cmap="Greys", vmin=-L.cost.max(), vmax=-L.cost.min(),
                   s=30, edgecolors="none", alpha=0.7)
        m = ids[L.mid[j]]
        music_col.setdefault(m, COLORS[len(music_col) % 10])
        path_xy.append((i, rank[j]))
        labels.append((m, L.off[j], j >= L.n_top))
    xs, ys = zip(*path_xy)
    ax.plot(xs, ys, color="red", lw=2, zorder=3)
    for (x, y), (m, off, cont) in zip(path_xy, labels):
        ax.scatter([x], [y], s=160, color=music_col[m], edgecolors="red", linewidths=2, zorder=4)
        ax.annotate(f"{short(m)}\n@{off:.1f}s" + (" (continues)" if cont else ""), (x, y), xytext=(6, 8),
                    textcoords="offset points", fontsize=8, zorder=5)
    ax.invert_yaxis()
    ax.set_xticks(range(len(lat)), [f"seg {i}\n{r.a:.0f}–{r.b:.0f}s" for i, r in enumerate(results)], fontsize=8)
    ax.set_ylabel("candidate rank by own score (0 = best)")
    ax.set_title(f"Viterbi lattice: grey dots = top {J} candidate windows per segment; red path = the chosen soundtrack\n"
                 "(the best window is not always picked: keeping the same song can win when a change is unlikely)")
    return fig


def fig_final(frames, dur, mix, sr, pieces, seams, segs):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    filmstrip(a1, frames, dur)
    for a, b in segs:
        a1.axvline(a, color="white", lw=2)
    step = max(1, len(mix) // 6000)
    env = np.abs(mix[: len(mix) // step * step]).reshape(-1, step).max(1)
    t = np.arange(len(env)) * step / sr
    col = {}
    for p in pieces:
        c = col.setdefault(p["music_id"], COLORS[len(col) % 10])
        sel = (t >= p["start"]) & (t <= p["end"])
        a2.fill_between(t[sel], -env[sel], env[sel], color=c, lw=0)
        a2.text((p["start"] + p["end"]) / 2, 1.05, short(p["music_id"]), ha="center", fontsize=8, color=c,
                transform=a2.get_xaxis_transform())
    for s in seams:
        a2.axvline(s, color="black", ls=":", lw=1)
    a2.set_xlabel("video time (s)")
    a2.set_ylabel("soundtrack")
    fig.suptitle("Rendered soundtrack: beat-synced pieces, crossfades at the dotted seams, loudness normalised", fontsize=12)
    fig.subplots_adjust(hspace=0.35)
    return fig


# ---------------------------------------------------------------- hooks
def install_hooks(rep: Report):
    import bgm.cue.hook as hook
    import bgm.render.query as query
    import bgm.render.run as run_mod
    from bgm.retrieval.pipeline import fuse, zstats

    S = {"plans": 0}

    def wrap(mod, name, after):
        orig = getattr(mod, name)

        def f(*a, **k):
            out = orig(*a, **k)
            try:
                after(out, *a, **k)
            except Exception as e:                       # a picture must never break the run
                print(f"   (picture for {name} failed: {e!r})", flush=True)
            return out
        setattr(mod, name, f)

    # step 1-5: featurize (render/query.py)
    def a_silent(_, src, dst, cfg):
        rep.step("Silent copy", "The original sound is removed; the system only ever sees the pictures.",
                 [f"{Path(src).name} → {Path(dst).name} (25 fps, short side {cfg.media.video_short_side} px, no audio)"])

    def a_cuts(cuts, *a, **k):
        S["cuts"] = np.asarray(cuts, float)

    def a_frames(frames, path, T1):
        S["frames"] = frames
        rep.step("Sample frames", "One frame from the middle of every second; these are what the encoders see.",
                 [f"{len(frames)} frames of {frames.shape[2]}×{frames.shape[1]} px"], fig_frames(frames))

    def a_dyn(dyn, silent, cuts, *a, **k):
        S["dyn"] = dyn
        dur = len(dyn) / 25.0
        rep.step("Shot cuts + motion", "PySceneDetect finds every cut; optical flow measures how much the picture moves.",
                 [f"{len(cuts)} cuts", f"cut times (s): {', '.join(f'{c:.1f}' for c in cuts[:20])}" + (" …" if len(cuts) > 20 else ""),
                  "frames encoded with SigLIP → 1,152 numbers per second"], fig_shots(S["frames"], cuts, dyn, dur))

    def a_segs(segs, cuts, dur, sig, *a, **k):
        sp = [(float(s), float(e)) for s, e, _ in segs]
        rep.step("SigLIP similarity", "Each second's SigLIP vector compared with every other second: bright = looks alike.",
                 [f"{len(sig)} × {len(sig)} matrix; shots inside a bright block look like the same scene"],
                 fig_similarity(sig, cuts, sp, dur))
        rep.step("Segments (scenes)", "Similar neighbouring shots are merged into segments; each gets its own music.",
                 [f"{len(segs)} segments: " + ", ".join(f"{a:.1f}–{b:.1f}s" for a, b in sp)],
                 fig_segments(S["frames"], cuts, sp, dur))

    wrap(query, "silent_copy", a_silent)
    wrap(query, "detect_cuts", a_cuts)
    wrap(query, "frames_1hz", a_frames)
    wrap(query, "video_curves", a_dyn)
    wrap(query, "make_segments", a_segs)

    # featurize from cache: draw the video steps from the cached arrays
    def a_feat(feat, src, workdir, cfg):
        if "frames" in S:
            return
        from bgm.features.extract import frames_1hz
        silent = Path(workdir) / "silent25.mp4"
        sig = np.asarray(feat["sem_siglip_1hz"], np.float32)
        S["frames"] = frames_1hz(silent, len(sig))
        cuts, dur = np.asarray(feat["cut_times_s"], float), float(feat["meta"]["duration_s"])
        sp = [tuple(map(float, s)) for s in feat["segments"]]
        rep.step("Video features (from cache)", "query.npz already existed, so steps 1–5 were loaded instead of recomputed.",
                 [f"{len(cuts)} cuts, {len(sp)} segments"], fig_shots(S["frames"], cuts, np.asarray(feat["dyn_25hz"]), dur))
        rep.step("SigLIP similarity", "Each second compared with every other second.", [], fig_similarity(sig, cuts, sp, dur))
        rep.step("Segments (scenes)", "Similar neighbouring shots merged into segments.",
                 [", ".join(f"{a:.1f}–{b:.1f}s" for a, b in sp)], fig_segments(S["frames"], cuts, sp, dur))

    wrap(run_mod, "featurize", a_feat)

    # steps 6-9: plan_for (render/run.py) — only the first plan; later plans are the other seeds' agreement check
    def first():
        return S["plans"] == 0

    def a_coarse(short_idx, index, vq, w_coarse, K):
        if not first():
            return
        S["index"] = index
        raw = index.file_scores(*vq.pooled(0.0, vq.duration))
        sc = fuse(raw, zstats(raw), w_coarse)
        rep.step("Coarse retrieval", "10 trained heads embed the video and every music file in one space; files are ranked.",
                 [f"library: {len(index.ids)} music files", f"shortlist: {len(short_idx)} files (top {K} for the video + each segment)",
                  "best for whole video: " + ", ".join(short(index.ids[i]) for i in np.argsort(-sc)[:3])],
                 fig_coarse(index.ids, sc, short_idx, K))

    def a_fine(results, index, vq, files, hop):
        if not first():
            return
        S["results"] = results

    def a_lattice(lat, index, vq, results, w, J, *a):
        if not first():
            return
        S["lat"], S["w"] = lat, w
        rep.step("Fine retrieval", "Windows slide every 0.5 s across each shortlisted song and are scored per segment.",
                 [f"{sum(len(r.fi) for r in results):,} windows scored", f"top {J} windows per segment kept as candidates "
                  "(+ the 'keep playing' continuation of each previous one)"], fig_fine(index.ids, results, w))

    def a_cue(out, cfg, feat, n_layers, fold, stitch):
        if not first():
            return
        _, info = out
        segs = [tuple(map(float, s)) for s in feat["segments"]]
        p = info["p"]
        rep.step("Music-change predictor", "A 2-layer Transformer reads the segment sequence and predicts, at each boundary, "
                 "whether the music should change.",
                 [f"{len(info['seeds'])} models averaged (fold {fold})", "P(change): " + ", ".join(f"{x:.2f}" for x in p)],
                 fig_cue(p, segs) if p else None)

    def a_chosen(chosen, index, lat, picks):
        if not first():
            return
        S["plans"] += 1
        res = S["results"]
        sw = sum(a.music_id != b.music_id for a, b in zip(chosen, chosen[1:]))
        rep.step("Viterbi stitching", "Picks one window per segment: good match to its scene + smooth changes between scenes.",
                 [f"{len(chosen)} segments, {len(set(c.music_id for c in chosen))} songs, {sw} switches"] +
                 [f"seg {i} ({r.a:.1f}–{r.b:.1f}s): {short(c.music_id)} from {c.offset:.1f}s" for i, (r, c) in enumerate(zip(res, chosen))],
                 fig_viterbi(index.ids, lat, picks, res))
        if len(chosen) > 1:
            print("   (now planning the fold's other seeds for the agreement check; no pictures for those)", flush=True)

    wrap(run_mod, "coarse_shortlist", a_coarse)
    wrap(run_mod, "fine", a_fine)
    wrap(run_mod, "build_lattice", a_lattice)
    wrap(hook, "cue_decoder", a_cue)
    wrap(run_mod, "chosen_candidates", a_chosen)

    # step 10: the soundtrack (render/render.py, imported into run.py)
    def a_sound(out, runs, total, music_path, cfg_render):
        mix, seams = out
        S["mix"], S["seams"], S["pieces"], S["dur"] = mix, seams, [r.__dict__ for r in runs], total
        S["sr"] = int(cfg_render.sr)

    wrap(run_mod, "render_soundtrack", a_sound)

    def a_mux(_, video, wav, out, *a, **k):
        segs = [(r.a, r.b) for r in S["results"]]
        rep.step("Render + mux", "Pieces moved onto beats, joined with crossfades, loudness evened out, put on the original video.",
                 [f"{len(S['pieces'])} pieces, {len(S['seams'])} crossfades", f"output: {out}"],
                 fig_final(S["frames"], S["dur"], S["mix"], S["sr"], S["pieces"], S["seams"], segs))

    wrap(run_mod, "mux", a_mux)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--cached", action="store_true", help="reuse renders/<name>/query.npz")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser page")
    args, extra = ap.parse_known_args()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"error: video not found: {video}")
    out = Path(args.out).resolve() if args.out else video.with_name(video.stem + "_bgm.mp4")
    os.chdir(ROOT)

    from bgm.config import load_config
    from bgm.log import setup_logging
    setup_logging()
    cfg = load_config(args.config, OVERRIDES + extra)
    work = (Path(cfg.paths.renders) / out.stem).resolve()
    if not args.cached:
        for f in ("query.npz", "silent25.mp4"):
            (work / f).unlink(missing_ok=True)

    rep = Report(work / "steps", video, not args.no_open)
    print(f"Live page: {rep.page}", flush=True)
    install_hooks(rep)

    from bgm.render.run import render_video
    plan = render_video(cfg, str(video), str(out), args.run)
    rep.step("Done", "The output video has the music our system chose.", [f"video: {out}", f"plan: {plan}"])
    rep.write(done=True)
    print(f"\nAll pictures: {rep.dir}\nLive page:    {rep.page}", flush=True)


if __name__ == "__main__":
    main()
