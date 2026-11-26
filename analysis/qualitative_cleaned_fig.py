"""Failure-case figure on the cleaned 1,011 pairs. Cases are chosen by fixed rules from failures.csv:

(a) miss whose top-1 is another cue of the same film, with the highest file-mean PANNs score on a non-music sound class;
(b) the file that is top-1 for the most queries in one fold, shown with its six worst-ranked queries;
(c) the change predictor on the timeline case (qualitative_cleaned_timeline).

python -m bgm.analysis.qualitative_cleaned_fig -> outputs/qualitative_cleaned/{describe.json,failure_*.png/pdf,failures.png/pdf}
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from panns_inference import labels
from PIL import Image
from scipy.stats import spearmanr

OUT = Path("outputs/qualitative_cleaned")
FR = OUT / "frames"
L = np.array(list(labels))
SKIP = {"Music", "Speech", "Silence", "Inside, small room", "Inside, large room or hall", "Musical instrument"}
SFX = {"Vehicle", "Car", "Engine", "Accelerating, revving, vroom", "Train", "Rail transport", "Railroad car, train wagon", "Aircraft",
       "Telephone", "Telephone bell ringing", "Ringtone", "Gunshot, gunfire", "Explosion", "Door", "Dog", "Bird", "Siren", "Thunder",
       "Rain", "Footsteps", "Applause", "Crowd", "Glass", "Alarm", "Water", "Boat, Water vehicle", "Motorboat, speedboat"}
NOTES = "C C# D D# E F F# G G# A A# B".split()
BLUE, RED, GREEN, GREY = "#1f6fd1", "#c0392b", "#1a9e5a", "#9a9a9a"
pairs = pd.read_csv("cleaned_dataset/manifests/pairs.csv", dtype=str, keep_default_na=False).set_index("pair_id")
dur = pairs.video_duration_s.astype(float).to_dict()
p2m = pairs.music_id.to_dict()
aud = pd.read_csv("results/cleaned/bgm_presence_files.csv").set_index("music_id")
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def panns(p: str) -> np.ndarray:
    return np.load(f"cleaned_dataset/features/music_panns_hp150/{p2m[p]}.npz")["panns_probs_1hz"]


def desc(p: str) -> dict:
    m, t = p2m[p], panns(p).mean(0)
    f = np.load(f"cleaned_dataset/features/music/{m}.npz", allow_pickle=True)
    tags = [f"{L[i]} {t[i]:.2f}" for i in np.argsort(-t) if L[i] not in SKIP][:4]
    key = f"{NOTES[int(f['key_pc'])]} {'major' if int(f['key_mode']) == 1 else 'minor'}" if "key_pc" in f.files else "key n/a"
    return dict(pair=p, tags=", ".join(tags), tempo=round(float(f["tempo_bpm"]), 1), key=key, music_share=float(aud.loc[m, "bgm_share"]),
                verdict=aud.loc[m, "verdict"], db=float(aud.loc[m, "median_db_above_150hz"]))


def mdesc(d: dict) -> str:
    tags = ", ".join(t.rsplit(" ", 1)[0] for t in d["tags"].split(", ")[:3])
    return f"{tags}\n{d['tempo']:.0f} BPM, {d['key']}, {d['db']:.0f} dB\nmusic in {d['music_share']:.0%} of seconds"


def frame(p: str, frac: float = 0.5) -> np.ndarray:
    FR.mkdir(parents=True, exist_ok=True)
    o = FR / f"{p}_{int(frac * 100)}.jpg"
    if not o.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{dur[p] * frac:.2f}", "-i", f"cleaned_dataset/interim/video/{p}.mp4",
                        "-frames:v", "1", "-vf", "scale=320:-2", str(o)], check=True)
    return plt.imread(o)


def img(ax, p, frac, title, color="black") -> None:
    ax.imshow(frame(p, frac)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2)
    ax.set_title(title, fontsize=7, color=color)


def boot(df: pd.DataFrame, fn, n: int = 2000) -> list[float]:
    """95% CI, resampling film groups."""
    g = {k: v for k, v in df.groupby("film_group")}
    keys, rng = list(g), np.random.default_rng(0)
    vals = [fn(pd.concat([g[keys[i]] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(n)]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def pick_sfx(df: pd.DataFrame) -> pd.Series:
    idx = [i for i, l in enumerate(L) if l in SFX]
    m = df[(~df.hit10) & df.same_film].copy()
    m["sfx"] = [float(panns(t).mean(0)[idx].max()) for t in m.top1]
    return m.sort_values(["sfx", "pair_id"], ascending=[False, True]).iloc[0]


def centrality(df: pd.DataFrame, fold: int, hub: str) -> dict:
    d = df[df.fold == fold]
    gal = [p2m[p] for p in d.pair_id]
    out = {}
    n = np.array([(d.top1 == p).sum() for p in d.pair_id])
    for feat in ("sem_clap_1hz", "sem_mert_1hz"):
        X = np.stack([np.load(f"cleaned_dataset/features/music/{m}.npz")[feat].mean(0) for m in gal])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        c = ((X @ X.T).sum(1) - 1) / (len(gal) - 1)
        h = c[list(d.pair_id).index(hub)]
        r = spearmanr(c, n)
        out[feat] = dict(hub=float(h), pct=float((c < h).mean()), rho=float(r.statistic), p=float(r.pvalue))
    return out


def main() -> None:
    df = pd.read_csv(OUT / "failures.csv")
    fi = df.set_index("pair_id")
    ex = json.loads((OUT / "timeline/example.json").read_text())
    D: dict = {}

    # (a)
    r = pick_sfx(df); q, t1 = r.pair_id, r.top1
    D["a_query"], D["a_top1"] = desc(q), desc(t1)
    D["a_case"] = dict(fold=int(r.fold), film=r.film, rank=int(r["rank"]), N=int(r.N), sfx=float(r.sfx))
    fig = plt.figure(figsize=(13, 2.6))
    g = GridSpec(1, 8, figure=fig, width_ratios=[1, 1, 1, .12, 1, 1, 1, 1.45], wspace=.06, left=.01, right=.99, top=.78, bottom=.03)
    for i, f in enumerate((.2, .5, .8)):
        img(fig.add_subplot(g[i]), q, f, f"query, {f * dur[q]:.0f} s", BLUE)
        img(fig.add_subplot(g[4 + i]), t1, f, f"top-1's own scene, {f * dur[t1]:.0f} s", RED)
    ax = fig.add_subplot(g[7]); ax.axis("off")
    ax.text(.03, 1, f"True music: rank {r['rank']} of {r.N}", weight="bold", color=BLUE, va="top")
    ax.text(.03, .86, mdesc(D["a_query"]), va="top", fontsize=7)
    ax.text(.03, .47, "Top-1 (another cue, same film)", weight="bold", color=RED, va="top")
    ax.text(.03, .33, mdesc(D["a_top1"]), va="top", fontsize=7)
    fig.suptitle(f"(a) Non-music audio: a file of vehicle sound outranks the true music for an indoor scene  ({r.film}, fold {r.fold})", x=.01, ha="left", weight="bold", fontsize=10)
    fig.savefig(OUT / "failure_a.png", dpi=200); fig.savefig(OUT / "failure_a.pdf"); plt.close(fig)

    # (b)
    c = df.groupby(["fold", "top1"]).size().sort_values(ascending=False)
    (hf, hub), hn = c.index[0], int(c.iloc[0])
    hq = df[(df.fold == hf) & (df.top1 == hub) & (df.pair_id != hub)].sort_values(["rank", "pair_id"], ascending=[False, True])
    qs = list(hq.pair_id[:6])
    D["b_hub"] = desc(hub)
    D["b_case"] = dict(fold=int(hf), film=fi.loc[hub, "film"], n=hn, N=int(fi.loc[hub, "N"]), hits=int(hq.hit10.sum()), n_other=len(hq),
                       ranks=[int(x) for x in hq["rank"][:6]], films=list(hq.film[:6]), all_films=int(hq.film.nunique()))
    D["b_centrality"] = centrality(df, int(hf), hub)
    fig = plt.figure(figsize=(13, 4.6))
    g = GridSpec(2, 4, figure=fig, width_ratios=[1.25, 1, 1, 1], hspace=.25, wspace=.06, left=.01, right=.99, top=.88, bottom=.02)
    img(fig.add_subplot(g[0, 0]), hub, .5, f"hub music's own scene ({fi.loc[hub, 'film']})", RED)
    ax = fig.add_subplot(g[1, 0]); ax.axis("off")
    m = D["b_centrality"]["sem_mert_1hz"]
    ax.text(.03, 1, f"Hub: top-1 for {hn} of {fi.loc[hub, 'N']} fold-{hf} queries", weight="bold", color=RED, va="top")
    ax.text(.03, .84, mdesc(D["b_hub"]), va="top", fontsize=7)
    ax.text(.03, .45, f"Not a central file: MERT centrality above {m['pct']:.0%} of the gallery\ncentrality vs times top-1 (fold {hf}):\nSpearman {m['rho']:.2f}, p = {m['p']:.2f}", va="top", fontsize=7)
    for j, p in enumerate(qs):
        img(fig.add_subplot(g[j // 3, 1 + j % 3]), p, .5, f"query: {fi.loc[p, 'film']}, true rank {fi.loc[p, 'rank']}", BLUE)
    fig.suptitle(f"(b) Hub: a file of vehicle sound with no detected music is top-1 for {hn} queries from {D['b_case']['all_films']} films (fold {hf}; six worst-ranked shown)", x=.01, ha="left", weight="bold", fontsize=10)
    fig.savefig(OUT / "failure_b.png", dpi=200); fig.savefig(OUT / "failure_b.pdf"); plt.close(fig)

    # (c)
    P = np.array(ex["predictor"]["p"]); bt = np.array(ex["predictor"]["boundary_times_s"])
    tc = np.array(ex["case"]["true_change_times_s"])
    is_true = np.array([np.abs(tc - b).min() <= .05 for b in bt])
    ld = np.array([t["log_dur"] for t in ex["predictor"]["tokens"]])
    x = np.exp((ld[:-1] + ld[1:]) / 2)
    rho = spearmanr(x, P)
    fp, fn = int(np.argmax(np.where(~is_true, P, -1))), int(np.argmin(np.where(is_true, P, 2)))
    D["c_case"] = dict(n_boundaries=len(P), n_true=len(tc), true_on_boundary=int(is_true.sum()), rho=float(rho.statistic), p=float(rho.pvalue),
                       top_false_t=float(bt[fp]), top_false_p=float(P[fp]), low_true_t=float(bt[fn]), low_true_p=float(P[fn]),
                       auc_like_mean_true=float(P[is_true].mean()), mean_false=float(P[~is_true].mean()))
    cols = [GREEN if t else GREY for t in is_true]
    fig = plt.figure(figsize=(13, 3.2))
    g = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=.22, left=.06, right=.99, top=.84, bottom=.17)
    ax = fig.add_subplot(g[0]); ax.bar(range(len(P)), P, color=cols); ax.axhline(.5, ls=":", c="k", lw=.8)
    ax.set_xticks(range(len(P)), [f"{b:.0f}" for b in bt], fontsize=7); ax.set_xlabel("segment boundary (s)"); ax.set_ylabel("p(music changes)"); ax.set_ylim(0, 1.2)
    ax.annotate("no change", (fp, P[fp]), (fp - 1.6, 1.08), fontsize=7, arrowprops=dict(arrowstyle="->"))
    ax.annotate("true change, low p", (fn, P[fn]), (fn - 1, .62), fontsize=7, arrowprops=dict(arrowstyle="->"))
    ax.legend(handles=[Patch(color=GREEN, label="true music change"), Patch(color=GREY, label="no change")], frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax2 = fig.add_subplot(g[1]); ax2.scatter(x, P, c=cols, s=30); ax2.set_xlabel("geometric mean length of the two segments (s)"); ax2.set_ylabel("p(music changes)")
    ax2.set_title(f"p vs segment length (Spearman {rho.statistic:.2f}, n = {len(P)})", fontsize=8)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("(c) Stitching: the change predictor on the timeline case (fold 1, Sivappu Manjal Pachai)", x=.01, ha="left", weight="bold", fontsize=10)
    fig.savefig(OUT / "failure_c.png", dpi=200); fig.savefig(OUT / "failure_c.pdf"); plt.close(fig)

    ims = [Image.open(OUT / f"failure_{k}.png").convert("RGB") for k in "abc"]
    canvas = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims) + 80), "white")
    y = 0
    for i in ims:
        canvas.paste(i, (0, y)); y += i.height + 40
    canvas.save(OUT / "failures.png"); canvas.save(OUT / "failures.pdf", resolution=200)

    D["ci"] = {"top1_own_film_minus_chance": boot(df, lambda d: d.top1_own_film.mean() - d.chance_own_film.mean()),
               "little_bgm_top1_minus_gallery": boot(df, lambda d: (d.top1_verdict == "little BGM").mean() - (d.verdict == "little BGM").mean()),
               "R10": boot(df, lambda d: d.hit10.mean())}
    D["point"] = {"little_bgm_top1_minus_gallery": float((df.top1_verdict == "little BGM").mean() - (df.verdict == "little BGM").mean()),
                  "top1_own_film_minus_chance": float(df.top1_own_film.mean() - df.chance_own_film.mean()),
                  "n_films": int(df.film_group.nunique())}
    (OUT / "describe.json").write_text(json.dumps(D, indent=1, default=str))
    print(json.dumps(D, indent=1, default=str))


if __name__ == "__main__":
    main()
