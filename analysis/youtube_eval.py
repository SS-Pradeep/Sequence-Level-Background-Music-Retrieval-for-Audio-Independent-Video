"""Out-of-domain test: the shipped semantic heads on YouTube music videos (tools/download_yt_dataset.py pilot,
Demucs BGM via tools/separate_bgm.py).

Run: python -m bgm.analysis.youtube_eval --root yt_dataset   -> results/youtube_pilot.md, <root>/eval/{perquery.csv, top5.csv}

Queries: every downloaded video whose BGM stem separated cleanly (bgm_manifest status ok). Gallery: all of those BGM
stems; the hit is the video's own stem. No video here is in any training set, so every fold's heads are leak-free.
Features are made like the dataset's (rule 8: video from a silent 25 fps copy; rule 7: CLAP 48 kHz, MERT 24 kHz,
ImageBind 16 kHz, asserted by the encoders), cached at <root>/features/<id>.npz.

Systems (track retrieval, whole-clip pooling, Scorer.track's inputs):
* final: the 10-head ensembles of runs/cleaned/semantic/siglip-both (5 folds x 3 disjoint ensembles, D99 / D116);
* single heads (5 folds x seeds 0-2); ImageBind zero-shot; exact chance.
Mean ± std over the 15 runs; ensemble R@10 also with a 95% bootstrap CI over queries (each video is its own cluster).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soxr
import torch
from omegaconf import OmegaConf

from bgm.data.media import ffprobe
from bgm.data.prep import decode_mono
from bgm.eval.metrics import chance_summary, cluster_bootstrap, rank_of_first_hit, recall_summary
from bgm.eval.store import l2
from bgm.features.extract import SR, EncoderBank, centred_windows, frames_1hz
from bgm.features.timing import n_1hz
from bgm.render.query import silent_copy
from bgm.train.semantic import SemanticModel

log = logging.getLogger(__name__)
RUNS = Path("runs/cleaned/semantic/siglip-both")
KS = (1, 5, 10)


# ---------------------------------------------------------------- features
def ok_ids(root: Path) -> list[str]:
    """Videos with a finished download and a non-silent BGM stem (last manifest line per id wins)."""
    def last(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r["status"]
        return out
    dl, sep = last(root / "manifest.jsonl"), last(root / "bgm_manifest.jsonl")
    return sorted(i for i, s in dl.items() if s == "ok" and sep.get(i) == "ok"
                  and (root / "video" / f"{i}.mp4").exists() and bgm_path(root, i) is not None)


def bgm_path(root: Path, vid: str) -> Path | None:
    return next((p for e in ("wav", "flac") if (p := root / "bgm" / f"{vid}.{e}").exists()), None)


def featurize(vid: str, root: Path, cfg, bank: EncoderBank) -> dict:
    """{video_siglip, video_imagebind: [T1v, D]; music_clap, music_mert, music_imagebind: [T1m, D]} f16, cached."""
    cache = root / "features" / f"{vid}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    silent = root / "interim" / f"{vid}.mp4"
    silent_copy(root / "video" / f"{vid}.mp4", silent, cfg)
    info = ffprobe(silent)
    dur = info["video"]["duration_s"] if np.isfinite(info["video"]["duration_s"]) else info["duration_s"]
    frames = frames_1hz(silent, n_1hz(dur))
    out = {"video_siglip": bank.frames("siglip", frames), "video_imagebind": bank.frames("imagebind", frames)}

    sm, ib = cfg.features.sem_music, cfg.features.imagebind
    y48 = decode_mono(str(bgm_path(root, vid)), SR["clap"])
    T1 = n_1hz(len(y48) / SR["clap"])
    out["music_clap"] = bank.audio("clap", centred_windows(y48, SR["clap"], T1, sm.clap_window_s, pad=False), SR["clap"])
    mert = bank.mert_track(soxr.resample(y48, SR["clap"], SR["mert"]).astype(np.float32))
    out["music_mert"] = (mert[0] if isinstance(mert, tuple) else mert)[:T1]
    y16 = soxr.resample(y48, SR["clap"], SR["imagebind"]).astype(np.float32)
    out["music_imagebind"] = bank.audio("imagebind", centred_windows(y16, SR["imagebind"], T1, ib.window_s, pad=True), SR["imagebind"])
    assert len(out["music_mert"]) == T1 == len(out["music_clap"]), (vid, T1, len(out["music_mert"]), len(out["music_clap"]))
    out = {k: np.asarray(v, np.float16) for k, v in out.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def pooled(F: dict[str, dict], ids: list[str]) -> dict[str, np.ndarray]:
    """Whole-clip inputs as SemanticInputs builds them: video l2(mean SigLIP) [N, 1152]; music
    [l2(mean CLAP) | l2(mean MERT)] [N, 1536]; ImageBind l2(mean) on both sides [N, 1024]."""
    m = lambda i, k: l2(F[i][k].astype(np.float32).mean(0))  # noqa: E731
    return {"video": np.stack([m(i, "video_siglip") for i in ids]),
            "music": np.stack([np.concatenate([m(i, "music_clap"), m(i, "music_mert")]) for i in ids]),
            "ib_video": np.stack([m(i, "video_imagebind") for i in ids]),
            "ib_music": np.stack([m(i, "music_imagebind") for i in ids])}


# ---------------------------------------------------------------- scoring
def head_scores(run: Path, X: dict, device: str) -> np.ndarray:
    """[N_query, N_gallery] cosine scores of one trained head."""
    ck = torch.load(run / "ckpt.pt", map_location=device)
    model = SemanticModel(ck["dv"], ck["dm"], OmegaConf.load(run / "config.yaml"), device)
    model.load(ck["state"])
    model.eval()
    assert (ck["dv"], ck["dm"]) == (X["video"].shape[1], X["music"].shape[1]), (run, ck["dv"], ck["dm"])
    with torch.no_grad():
        return model.embed("video", X["video"]) @ model.embed("music", X["music"]).T


def ranks_of(S: np.ndarray) -> np.ndarray:
    """Rank of each query's own track (the diagonal)."""
    eye = np.eye(len(S), dtype=bool)
    return np.array([rank_of_first_hit(S[q], eye[q]) for q in range(len(S))])


def summarize(rank_sets: list[np.ndarray]) -> dict:
    per = [recall_summary(r, KS) for r in rank_sets]
    return {k: (float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))) for k in per[0] if k != "n"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="yt_dataset")
    ap.add_argument("--out", default="results/youtube_pilot.md")
    ap.add_argument("--title", default="YouTube music-video pilot")
    a = ap.parse_args()
    root = Path(a.root)
    cfg = OmegaConf.load(RUNS / "fold0" / "seed0" / "config.yaml")
    ids = ok_ids(root)
    log.info("%d queries", len(ids))

    bank = EncoderBank(cfg, fake=False)
    F = {}
    for n, i in enumerate(ids, 1):
        F[i] = featurize(i, root, cfg, bank)
        if n % 20 == 0:
            log.info("  features %d/%d", n, len(ids))
    del bank
    torch.cuda.empty_cache()
    X = pooled(F, ids)

    singles, ens, ens_S = [], [], []
    for k in range(5):
        S_seed = {s: head_scores(RUNS / f"fold{k}" / f"seed{s}", X, "cuda") for s in range(30)}
        singles += [ranks_of(S_seed[s]) for s in range(3)]
        for e in range(3):
            S = np.mean([S_seed[s] for s in range(10 * e, 10 * e + 10)], 0)
            ens.append(ranks_of(S))
            ens_S.append(S)
        log.info("fold %d done", k)
    ib = ranks_of(X["ib_video"] @ X["ib_music"].T)

    N = len(ids)
    chance = chance_summary([N] * N, [1] * N, KS)
    rows = [("random (exact chance)", "—", {k: (chance[k], 0.0) for k in chance if k != "n"} | {"MedR": ((N + 1) / 2, 0.0)}),
            ("ImageBind zero-shot", "1", summarize([ib])),
            ("heads SigLIP → CLAP+MERT (seeds 0–2)", "15", summarize(singles)),
            ("**final: 10-head ensemble**", "15", summarize(ens))]
    hit10 = np.mean([(r <= 10).astype(float) for r in ens], 0)
    ci = cluster_bootstrap(hit10, np.arange(N))
    d_ib = cluster_bootstrap(hit10 - (ib <= 10), np.arange(N))

    meta = {r["id"]: r for r in map(json.loads, open(root / "manifest.jsonl", encoding="utf-8")) if r["status"] == "ok"}
    mean_rank = np.mean(ens, 0)
    S_all = np.mean(ens_S, 0)
    ev = root / "eval"
    ev.mkdir(exist_ok=True)
    pd.DataFrame({"id": ids, "title": [meta[i].get("title") for i in ids], "duration_s": [meta[i].get("duration") for i in ids],
                  "ensemble_mean_rank": mean_rank, "ensemble_R@10_share": hit10, "imagebind_rank": ib}
                 ).sort_values("ensemble_mean_rank").to_csv(ev / "perquery.csv", index=False)
    top = [{"query": ids[q], "query_title": meta[ids[q]].get("title"), "own_rank_all15": int(rank_of_first_hit(S_all[q], np.eye(N, dtype=bool)[q])),
            **{f"top{j + 1}": meta[ids[t]].get("title") for j, t in enumerate(np.argsort(-S_all[q])[:5])}} for q in range(N)]
    pd.DataFrame(top).to_csv(ev / "top5.csv", index=False)

    fmt = lambda v: f"{v[0]:.3f} ± {v[1]:.3f}" if v[1] else f"{v[0]:.3f}"  # noqa: E731
    lines = [f"# Out-of-domain test: {a.title}",
             "",
             "Generated by `python -m bgm.analysis.youtube_eval` (src/bgm/analysis/youtube_eval.py). "
             f"**{N} YouTube videos** (random sample of tools/Youtube_ID.txt, seed 42; downloaded by tools/download_yt_dataset.py, "
             "vocals removed by Demucs htdemucs two-stem, tools/separate_bgm.py). Query = the silent video; gallery = all "
             f"{N} BGM stems; hit = the video's own stem. Heads trained on the cleaned Indian-film dataset "
             "(runs/cleaned/semantic/siglip-both), none of these videos seen in training, so every fold's heads are leak-free. "
             "Whole-clip track retrieval; mean ± std over the runs.",
             "",
             "| system | runs | track R@1 | track R@5 | track R@10 | MedR | MRR |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, runs, s in rows:
        lines.append(f"| {name} | {runs} | " + " | ".join(fmt(s[k]) if k in s else "—" for k in ("R@1", "R@5", "R@10", "MedR", "MRR")) + " |")
    lines += ["",
              f"- Final ensemble track R@10, 95% bootstrap CI over the {N} queries: {ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}].",
              f"- Ensemble − ImageBind, track R@10: {d_ib['mean']:+.3f} [{d_ib['lo']:+.3f}, {d_ib['hi']:+.3f}].",
              "- In-domain reference (results/cleaned/compare.md §1, cleaned test folds, gallery ≈ 145 files): final ensemble "
              "R@10 0.302 ± 0.029, chance 0.069.",
              "- Single random sample of YouTube; not a fold-level result. Per-query ranks: "
              f"{ev}/perquery.csv; top-5 titles per query (mean of all 15 ensembles): {ev}/top5.csv."]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
