"""Qualitative retrieval figure: video -> music and music -> video with the final system, plus distance histograms.

python -m bgm.analysis.retrieval_figure score   -> outputs/retrieval_figure/scores_fold{k}.npz
python -m bgm.analysis.retrieval_figure plot    -> outputs/retrieval_figure/{retrieval_figure.png,.pdf,cases.csv}

Final system (D75): 10-head ensemble (run seed 0 = heads 0-9), title split, every query a test clip scored by its own
fold's models. Gallery = the fold's test music files (video -> music) or test videos (music -> video). Score = mean
cosine of the heads; distance d = 1 - score. S[i, j] = score of test video i against test-fold music file j, so a row is
a video query and a column is a music query (the same matrix, read the other way). A hit is any item of the query's
track group (the evaluation protocol), so the ground-truth rank R is the rank of the first hit.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
OUT = Path("outputs/retrieval_figure")
TRAIN_OUT = Path("outputs/train_retrieval")    # training-clip queries (score_segments_train)
ROOT = Path("runs/titlesplit")
E = 0


def score() -> None:
    import torch
    from bgm.analysis.eval_suite import Scorers
    from bgm.config import load_config
    from bgm.eval.metrics import rank_of_first_hit
    from bgm.eval.store import FeatureStore, fold_ids

    OUT.mkdir(parents=True, exist_ok=True)
    store = None
    for k in range(5):
        cfg = load_config(ROOT / "pipeline_ens10" / f"fold{k}" / f"seed{E}" / "config.yaml")
        store = store or FeatureStore(cfg)
        f = fold_ids(cfg, k)
        test = list(f["test"])
        gal = list(dict.fromkeys(store.music_of[p] for p in test))
        sc = Scorers(cfg, store, k, E, f["train"]).sc
        with torch.no_grad():
            S = np.stack([np.asarray(sc.track(q, gal), float) for q in test])
        gtrack = np.array([store.music_track[m] for m in gal])
        ranks = np.array([rank_of_first_hit(S[i], gtrack == store.track[q]) for i, q in enumerate(test)])
        saved = np.load(ROOT / "semantic_ens" / "siglip-both-ens10" / f"fold{k}" / f"seed{E}" / "perquery.npz")
        assert np.array_equal(ranks, saved["track_test_fold"]), f"fold {k}: ranks differ from the saved ensemble run"
        np.savez(OUT / f"scores_fold{k}.npz", S=S, test=np.array(test), gallery=np.array(gal),
                 q_track=np.array([store.track[q] for q in test]), g_track=gtrack)
        log.info("fold %d: %d queries x %d files, R@10 %.3f", k, len(test), len(gal), (ranks <= 10).mean())
        del sc
        torch.cuda.empty_cache()


def score_segments() -> None:
    """Segment protocol (SPEC §5.12, 'all'): each test segment against every test segment's aligned music window."""
    import torch
    from bgm.analysis.eval_suite import Scorers
    from bgm.config import load_config
    from bgm.eval.protocol import segment_gallery, test_segments
    from bgm.eval.store import FeatureStore, fold_ids

    store = None
    for k in range(5):
        cfg = load_config(ROOT / "pipeline_ens10" / f"fold{k}" / f"seed{E}" / "config.yaml")
        store = store or FeatureStore(cfg)
        f = fold_ids(cfg, k)
        segs = test_segments(store, list(f["test"]))
        sc = Scorers(cfg, store, k, E, f["train"]).sc
        rows, ranks = [], []
        with torch.no_grad():
            for qs in segs:
                windows, hit = segment_gallery(store, segs, qs, "all")
                s = np.asarray(sc.window(qs[0], qs[2], qs[3], windows), float)
                hits = np.zeros(len(windows), bool)
                hits[hit] = True
                ranks.append(rank_of(s, hits))
                rows.append(dict(pair_id=qs[0], seg=qs[1], a=qs[2], b=qs[3], hit=hit, scores=s,
                                 windows=[(m, float(t), float(d)) for m, t, d in windows]))
        saved = np.load(ROOT / "semantic_ens" / "siglip-both-ens10" / f"fold{k}" / f"seed{E}" / "perquery.npz")
        assert np.array_equal(np.array(ranks), saved["segment_all"]), f"fold {k}: segment ranks differ from the saved run"
        pd.to_pickle(rows, OUT / f"segments_fold{k}.pkl")
        log.info("fold %d: %d segments, R@10 %.3f", k, len(rows), (np.array(ranks) <= 10).mean())
        del sc
        torch.cuda.empty_cache()


def score_segments_train(pair_id: str, k: int = 0) -> Path:
    """Segment protocol run on fold k's TRAIN split for one training video: each of its segments against every train
    segment's aligned music window. The heads were trained on this video and its music, so this pictures what the
    models fit, not how they generalise. -> outputs/train_retrieval/segments_train_<pair_id>.pkl"""
    import torch
    from bgm.analysis.eval_suite import Scorers
    from bgm.config import load_config
    from bgm.eval.protocol import segment_gallery, test_segments
    from bgm.eval.store import FeatureStore, fold_ids

    cfg = load_config(ROOT / "pipeline_ens10" / f"fold{k}" / f"seed{E}" / "config.yaml")
    store = FeatureStore(cfg)
    f = fold_ids(cfg, k)
    assert pair_id in set(f["train"]), f"{pair_id} is not in fold {k}'s train split"
    segs = test_segments(store, list(f["train"]))
    sc = Scorers(cfg, store, k, E, f["train"]).sc
    rows = []
    with torch.no_grad():
        for qs in [s for s in segs if s[0] == pair_id]:
            windows, hit = segment_gallery(store, segs, qs, "all")
            s = np.asarray(sc.window(qs[0], qs[2], qs[3], windows), float)
            rows.append(dict(pair_id=qs[0], seg=qs[1], a=qs[2], b=qs[3], hit=hit, scores=s, fold=k,
                             windows=[(m, float(t), float(d)) for m, t, d in windows]))
            hits = np.zeros(len(s), bool)
            hits[hit] = True
            log.info("segment %d: rank %d of %d", qs[1] + 1, rank_of(s, hits), len(s))
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    out = TRAIN_OUT / f"segments_train_{pair_id}.pkl"
    pd.to_pickle(rows, out)
    return out


def rank_of(s: np.ndarray, hits: np.ndarray) -> int:
    from bgm.eval.metrics import rank_of_first_hit
    return int(rank_of_first_hit(s, hits))


def load_folds() -> list[dict]:
    return [dict(np.load(OUT / f"scores_fold{k}.npz")) for k in range(5)]


def clean_title(t: str, max_len: int = 24) -> str:
    """Readable film title: 'Waqt.1965.WEB-DL.1080p' -> 'Waqt (1965)', '[compilation] Dhanush: 3, ...' -> 'Dhanush (comp.)'."""
    t = re.sub(r"^([^.]+)\.((?:19|20)\d\d)\..*$", r"\1 (\2)", t).replace("_", " ")
    t = re.sub(r"\s+-\s+.*$|\s+ESubs$", "", t)
    t = re.sub(r"^\[compilation\]\s*([^:]+):.*$", r"\1 (comp.)", t)
    return t if len(t) <= max_len else t[:max_len - 1].rstrip() + "…"


def names() -> tuple[dict, dict, dict]:
    """(music_id -> pair_id, pair_id -> music_id, pair_id -> 'Film · clip n'), n numbering a title's clips (unique)."""
    pairs = pd.read_csv("data/manifests/pairs.csv", dtype=str, keep_default_na=False)
    t = pd.read_csv("outputs/film_titles.csv", dtype=str)
    title = dict(zip(t.source + ":" + t.source_id, t.title))
    film = pd.Series([clean_title(title[f"{s}:{f}"]) for s, f in zip(pairs.meta_source, pairs.meta_film)], index=pairs.pair_id)
    n = film.sort_index().groupby(film.sort_index()).cumcount() + 1
    lab = {p: f"{film[p]} · clip {n[p]}" for p in pairs.pair_id}
    assert len(set(lab.values())) == len(lab), "labels are not unique"
    return dict(zip(pairs.music_id, pairs.pair_id)), dict(zip(pairs.pair_id, pairs.music_id)), lab


def cases_table() -> pd.DataFrame:
    """Every test query: ground-truth rank both ways and the margin of the true pair over the best other item."""
    _, p2m, lab = names()
    rows = []
    for k, z in enumerate(load_folds()):
        S, test, gal = z["S"], list(z["test"]), list(z["gallery"])
        for i, q in enumerate(test):
            j = gal.index(p2m[q])
            rows.append(dict(fold=k, pair_id=q, label=lab[q], rank_v2m=rank_of(S[i], z["g_track"] == z["q_track"][i]),
                             rank_m2v=rank_of(S[:, j], z["q_track"] == z["g_track"][j]),
                             margin_v2m=S[i, j] - np.delete(S[i], j).max(), margin_m2v=S[i, j] - np.delete(S[:, j], i).max()))
    return pd.DataFrame(rows)


def thumb(pair_id: str, cache: Path, t: float | None = None) -> np.ndarray:
    """Frame of the silent interim video at t s (default: the middle), [H, W, 3] uint8."""
    import matplotlib.image as mpimg
    path = cache / (f"{pair_id}.jpg" if t is None else f"{pair_id}@{t:.2f}.jpg")
    if not path.exists():
        src = Path("data/interim/video") / f"{pair_id}.mp4"
        if t is None:
            t = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                     check=True, capture_output=True, text=True).stdout) / 2
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=-2:270", "-q:v", "2", str(path)], check=True)
    return mpimg.imread(path)


def mel_db(music_id: str, cache: Path) -> tuple[np.ndarray, float]:
    """Log-mel spectrogram of the whole interim music file at 22,050 Hz: ([64 mels, frames] dB, duration s)."""
    import librosa
    path = cache / f"{music_id}.mel.npz"
    if not path.exists():
        y, sr = librosa.load(Path("data/interim/music") / f"{music_id}.flac", sr=22050, mono=True)
        assert sr == 22050
        M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64), ref=np.max)
        np.savez(path, M=M.astype(np.float32), dur=len(y) / sr)
    z = np.load(path)
    return z["M"], float(z["dur"])


GREEN, RED, GREY = "#2e7d32", "#d62839", "#6e7f8d"
BADGE = dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7)


def style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix"})


def framed(ax, img, ok: bool | None) -> None:
    """Image without ticks; a green (hit) or red (miss) frame, none for a query."""
    if img is not None:
        ax.imshow(img, aspect="auto")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(ok is not None)
        sp.set_color(GREEN if ok else RED)
        sp.set_linewidth(3)


def badge(ax, text: str, size: float = 8.5) -> None:
    ax.text(0.98, 0.04, text, color="w", fontsize=size, ha="right", va="bottom", transform=ax.transAxes, bbox=BADGE)


def gt_tag(ax, text: str = "✓ ground truth", size: float = 9) -> None:
    ax.text(0.02, 0.95, text, color="w", fontsize=size, fontweight="bold", ha="left", va="top", fontfamily="DejaVu Sans",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.25", fc=GREEN, ec="none"))


def spectrogram(ax, music_id: str, cache: Path, start: float = 0.0, dur: float | None = None) -> float:
    """Log-mel of [start, start + dur] s of a music file (default: all of it); returns the shown duration (s)."""
    M, total = mel_db(music_id, cache)
    dur = total - start if dur is None else dur
    fps = M.shape[1] / total
    a = int(round(start * fps))
    ax.imshow(M[:, a:max(a + 1, int(round((start + dur) * fps)))], origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
    return dur


def histogram(ax, d: np.ndarray, hits: np.ndarray, R: int, sym: str) -> None:
    """Distances to the whole gallery, the first hit's bin in red with its rank."""
    edges = np.linspace(d.min(), d.max(), 31)
    cnt, _ = np.histogram(d, edges)
    gt = min(int(np.searchsorted(edges, d[hits].min(), "right")) - 1, 29)
    ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge", color=[RED if b == gt else GREY for b in range(30)])
    ax.annotate(f"$R({sym}_q) = {R}$", xy=((edges[gt] + edges[gt + 1]) / 2, cnt[gt]), xytext=(10, 30), textcoords="offset points",
                ha="left", fontsize=10.5, arrowprops=dict(arrowstyle="-|>", color="black", lw=1),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xticks([edges[0], edges[-1]], [f"{edges[0]:.2f}", f"{edges[-1]:.2f}"], fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(0, cnt.max() * 1.45)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"N = {len(d)}", fontsize=9, labelpad=1)


def dividers(fig, g, first: int, last: int, cols: tuple[int, int, int, int] = (0, 2, 4, 5)) -> None:
    """Dashed lines between the query column and the results, and between the results and the histograms; cols =
    (query, first result, last result, histogram) grid columns."""
    import matplotlib.pyplot as plt
    top, bot = g[first, 0].get_position(fig), g[last, 0].get_position(fig)
    for a, b in (cols[:2], cols[2:]):
        x = (g[first, a].get_position(fig).x1 + g[first, b].get_position(fig).x0) / 2
        fig.add_artist(plt.Line2D([x, x], [bot.y0 - 0.02, top.y1 + 0.02], transform=fig.transFigure, ls=(0, (4, 4)), color="k", lw=1.5))


def plot(queries: list[str], both: bool = False, spec: bool = False, name: str = "retrieval_figure") -> None:
    """Video -> music rows for each query; with both, a music -> video section below (the same queries). A retrieved music
    file is pictured by a frame of its own video (with spec, by its log-mel spectrogram)."""
    import matplotlib.pyplot as plt

    style()
    m2p, p2m, lab = names()
    pairs = pd.read_csv("data/manifests/pairs.csv", dtype=str, keep_default_na=False)
    dur_of = dict(zip(pairs.pair_id, pairs.video_duration_s.astype(float)))
    folds = load_folds()
    cache = OUT / "thumbs"
    cache.mkdir(parents=True, exist_ok=True)
    where = {q: k for k, z in enumerate(folds) for q in z["test"]}
    n = len(queries)
    halves = 2 if both else 1
    fig = plt.figure(figsize=(14, halves * (2.25 * n + 0.9)))
    outer = fig.add_gridspec(halves, 1, hspace=0.16)

    for half, (title, qname, rname, hname) in enumerate([
            ("Video to Music retrieval", r"Video Query $v_q$", "Top 3 recommended music tracks", r"Histogram of $d(v_q, m_*)$"),
            ("Music to Video retrieval", r"Music Query $m_q$", "Top 3 recommended videos", r"Histogram of $d(m_q, v_*)$")][:halves]):
        g = outer[half].subgridspec(n + 1, 6, height_ratios=[0.32] + [1] * n, width_ratios=[1, 0.12, 1, 1, 1, 1.35],
                                    hspace=0.62, wspace=0.08)
        head = fig.add_subplot(g[0, :])
        head.axis("off")
        head.text(0.12, 1.25, title, fontsize=22, fontweight="bold", transform=head.transAxes)
        for x, s in [(0.08, qname), (0.47, rname), (0.89, hname)]:
            head.text(x, 0.0, s, fontsize=14, fontweight="bold", ha="center", transform=head.transAxes)
        for r, q in enumerate(queries):
            z = folds[where[q]]
            S, test, gal = z["S"], list(z["test"]), list(z["gallery"])
            i, j = test.index(q), gal.index(p2m[q])
            if half == 0:
                s, hits, items = S[i], z["g_track"] == z["q_track"][i], [m2p[m] for m in gal]
            else:
                s, hits, items = S[:, j], z["q_track"] == z["g_track"][j], test
            d = 1.0 - s
            order = np.argsort(d)
            R = rank_of(s, hits)
            ax = fig.add_subplot(g[r + 1, 0])
            framed(ax, thumb(q, cache), None)
            badge(ax, f"{dur_of[q]:.1f} s", 9.5)
            ax.set_xlabel(("♪ " if half else "") + lab[q], fontsize=12)
            for c in range(3):
                ax = fig.add_subplot(g[r + 1, 2 + c])
                ok = bool(hits[order[c]])
                if half == 0 and spec:
                    dur = spectrogram(ax, p2m[items[order[c]]], cache)
                    framed(ax, None, ok)
                    badge(ax, f"{dur:.1f} s")
                else:
                    framed(ax, thumb(items[order[c]], cache), ok)
                if ok:
                    gt_tag(ax)
                ax.set_title(f"d = {d[order[c]]:.3f}", fontsize=10.5)
                ax.set_xlabel(("" if half else "♪ ") + lab[items[order[c]]], fontsize=9)
            histogram(fig.add_subplot(g[r + 1, 5]), d, hits, R, "m" if half == 0 else "v")
        dividers(fig, g, 1, n)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    log.info("wrote %s", OUT / f"{name}.png")


def plot_segments(pair_id: str, spec: bool = False, top: int = 10, split: str = "test") -> Path:
    """One test video cut into its segments; per segment the top-`top` music windows of the segment protocol, 'other-video'
    variant (SPEC §5.12: the gallery drops the query video's own other windows, so no result is the query's own music at
    another time), and the histogram of distances to that gallery. Results are laid out 5 per row. A music window is
    pictured by the frame of its own video at the window's middle (music time 0 = first video frame), with spec by its
    log-mel spectrogram. split 'train': a training video scored by score_segments_train against the train-fold windows.
    -> outputs/{retrieval_figure,train_retrieval}/segments_[train_]<label>[_top<k>][_spec].png/.pdf"""
    import matplotlib.pyplot as plt

    style()
    m2p, p2m, lab = names()
    if split == "train":
        rows = pd.read_pickle(TRAIN_OUT / f"segments_train_{pair_id}.pkl")
        k = rows[0]["fold"]
    else:
        k = next(k for k in range(5) if any(r["pair_id"] == pair_id for r in pd.read_pickle(OUT / f"segments_fold{k}.pkl")))
        rows = [r for r in pd.read_pickle(OUT / f"segments_fold{k}.pkl") if r["pair_id"] == pair_id]
    for r in rows:
        keep = [i for i, (m, _, _) in enumerate(r["windows"]) if m != p2m[pair_id] or i == r["hit"]]
        r["scores"], r["windows"], r["hit"] = np.asarray(r["scores"])[keep], [r["windows"][i] for i in keep], keep.index(r["hit"])
    cache = OUT / "thumbs"
    cache.mkdir(parents=True, exist_ok=True)
    n, dur = len(rows), rows[-1]["b"]
    ncol, nsub = min(top, 5), -(-top // 5)
    small = top > 3                                        # small thumbnails: smaller labels
    wr = [1, 0.12] + [3 / ncol] * ncol + [1.35]
    sub_h = 2.25 if not small else 1.5
    fig = plt.figure(figsize=(14, sub_h * n * nsub + 1.9))
    g = fig.add_gridspec(2 + n * nsub, ncol + 3, height_ratios=[0.32 * 2.25 / sub_h, 0.36 * 2.25 / sub_h] + [1] * n * nsub,
                         width_ratios=wr, hspace=0.62 if not small else 0.75, wspace=0.08)
    head = fig.add_subplot(g[0, :])
    head.axis("off")
    head.text(0.0, 1.6, "Segment-wise Video to Music retrieval", fontsize=22, fontweight="bold", transform=head.transAxes)
    kind = "held-out test clip" if split == "test" else f"training clip of fold {k}, seen by its models"
    head.text(0.0, 0.95, f"Query video: {lab[pair_id]} ({dur:.1f} s, {kind}), cut into {n} segments.",
              fontsize=12.5, transform=head.transAxes)
    head.text(0.0, 0.3, f"Each segment searches {len(rows[0]['windows']):,} {split}-fold music windows, including its own "
              f"true window (green box); the query's music at other times is left out.", fontsize=12.5, transform=head.transAxes)
    hx = lambda a, b: (g[2, a].get_position(fig).x0 + g[2, b].get_position(fig).x1) / 2
    for x, t in [(hx(0, 0), r"Segment Query $v_q$"), (hx(2, ncol + 1), f"Top {top} recommended music windows"),
                 (hx(ncol + 2, ncol + 2), r"Histogram of $d(v_q, m_*)$")]:
        fig.text(x, head.get_position().y0 - 0.004, t, fontsize=14, fontweight="bold", ha="center", va="top")
    # timeline of the query video: one block per segment
    tl = fig.add_subplot(g[1, 0:ncol + 2])
    cols = plt.get_cmap("tab10").colors
    for r in rows:
        tl.barh(0, r["b"] - r["a"], left=r["a"], height=1, color=cols[r["seg"] % 10], alpha=0.85, ec="white", lw=2)
        tl.text((r["a"] + r["b"]) / 2, 0, f"S{r['seg'] + 1}", ha="center", va="center", color="w", fontsize=11, fontweight="bold")
    tl.set_xlim(0, dur)
    tl.set_ylim(-0.5, 0.5)
    tl.set_yticks([])
    tl.set_xticks([r["a"] for r in rows] + [dur], [f"{x:.1f} s" for x in [r["a"] for r in rows] + [dur]], fontsize=9)
    for sp in ("top", "right", "left"):
        tl.spines[sp].set_visible(False)
    for i, r in enumerate(rows):
        s = np.asarray(r["scores"], float)
        hits = np.zeros(len(s), bool)
        hits[r["hit"]] = True
        d = 1.0 - s
        order = np.argsort(d)
        R = rank_of(s, hits)
        r0 = 2 + i * nsub
        ax = fig.add_subplot(g[r0:r0 + nsub, 0])
        img = thumb(pair_id, cache, (r["a"] + r["b"]) / 2)
        framed(ax, img, None)
        ax.set_box_aspect(img.shape[0] / img.shape[1])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color(cols[r["seg"] % 10]), sp.set_linewidth(4)
        badge(ax, f"{r['a']:.1f}–{r['b']:.1f} s", 9.5)
        ax.set_xlabel(f"S{r['seg'] + 1} · {r['b'] - r['a']:.1f} s", fontsize=12)
        for c in range(top):
            m, t0, w = r["windows"][order[c]]
            ok = bool(hits[order[c]])
            ax = fig.add_subplot(g[r0 + c // ncol, 2 + c % ncol])
            if spec:
                spectrogram(ax, m, cache, t0, w)
                framed(ax, None, ok)
            else:
                framed(ax, thumb(m2p[m], cache, t0 + w / 2), ok)
            badge(ax, f"{t0:.1f}–{t0 + w:.1f} s", 7 if small else 8.5)
            if ok:
                gt_tag(ax, "✓ GT" if small else "✓ ground truth", 7.5 if small else 9)
            ax.set_title((f"#{c + 1}  " if small else "") + f"d = {d[order[c]]:.3f}", fontsize=8.5 if small else 10.5, pad=2)
            film, clip = lab[m2p[m]].rsplit(" · ", 1)
            ax.set_xlabel(f"♪ {clean_title(film, 15) if small else film} · {clip}", fontsize=7.5 if small else 10, labelpad=1.5)
        histogram(fig.add_subplot(g[r0:r0 + nsub, ncol + 2]), d, hits, R, "m")
    dividers(fig, g, 2, 1 + n * nsub, (0, 2, ncol + 1, ncol + 2))
    tag = (f"_top{top}" if top != 3 else "") + ("_spec" if spec else "")
    out = (TRAIN_OUT if split == "train" else OUT) / f"segments_{'train_' if split == 'train' else ''}{lab[pair_id].replace(' · ', '_').replace(' ', '_')}{tag}"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    log.info("wrote %s", out.with_suffix(".png"))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if sys.argv[1] == "score":
        score()
    elif sys.argv[1] == "score-segments":
        score_segments()
    elif sys.argv[1] in ("segments", "segments-train"):
        top = next((int(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("--top=")), 10)
        fold = next((int(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("--fold=")), 0)
        for q in [a for a in sys.argv[2:] if not a.startswith("--")]:
            if sys.argv[1] == "segments-train":
                score_segments_train(q, fold)
            plot_segments(q, spec="--spec" in sys.argv, top=top, split="test" if sys.argv[1] == "segments" else "train")
    elif sys.argv[1] == "cases":
        cases_table().to_csv(OUT / "cases.csv", index=False)
    else:
        flags = {"--both", "--spec"}
        name = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--name=")), "retrieval_figure")
        plot([a for a in sys.argv[2:] if a not in flags and not a.startswith("--name=")], both="--both" in sys.argv,
             spec="--spec" in sys.argv, name=name)
