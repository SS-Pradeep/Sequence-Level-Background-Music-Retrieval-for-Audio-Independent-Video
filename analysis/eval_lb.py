"""LanguageBind score for the final system (D117): LB and tw-LB on the natural test videos, title split.

The systems' soundtracks are the ones the evaluation suite (D111) already decoded: runs/titlesplit/eval_suite/
fold*/seed*/perquery.npz stores, per test segment, each system's chosen music file and window offset (original
music, final = Viterbi, greedy, random). Nothing is retrained, retrieved or tuned here; the segments are
store.segments in test order, and every saved 'original' choice is checked against the ground truth.

Per test video (982) and system:
  * LB    = cos(LanguageBind video embedding of the whole clip (8 frames spread over it),
                LanguageBind audio embedding of the whole rendered soundtrack (render_soundtrack, what `bgm render`
                plays, before beat sync)).
  * tw-LB = time-weighted segment score: Σ_k d_k cos(v_k, m_k) / Σ_k d_k, v_k = the segment's 8 frames, m_k = its
            chosen music window (span rule widened to >= 5 s, as the IB metric of D111), d_k = segment length.
  * seg-LB = the same without the weights (every segment counts once).
Higher = the music fits the picture more in LanguageBind's joint space.

Reference rows: LanguageBind zero-shot track retrieval (whole clip vs whole music file over the test fold's
gallery; a check that the ported encoders carry signal, next to ImageBind's 0.110 R@10) and the per-video
correlation of tw-LB with the suite's IB score.

Video embeddings are cached in data/features/video_languagebind/<pair_id>.npz with the config hash (rule 9).

Run: python -m bgm.analysis.eval_lb [run | report]   -> runs/titlesplit/eval_lb/, results/titlesplit/eval_lb.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bgm.analysis.eval_suite import FOLDS, ROOT, SEEDS, SYSTEMS
from bgm.analysis.eval_suite import EXP as SUITE
from bgm.config import load_config
from bgm.data.prep import interim_music_path, interim_video_path
from bgm.eval.store import FeatureStore, fold_ids
from bgm.features import languagebind as LB
from bgm.render.render import render_soundtrack, runs_from_path
from bgm.retrieval.stitch import Candidate
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)
EXP = "eval_lb"
OUT = Path("results/titlesplit/eval_lb.md")
LABEL = {"original": "original music (reference)", "viterbi": "**final (Viterbi)**", "greedy": "greedy (top window per segment)",
         "random": "random window per segment"}


# ---------------------------------------------------------------- video side (cached features)
def video_cache_path(cfg, pair_id: str) -> Path:
    return Path(cfg.paths.features) / "video_languagebind" / f"{pair_id}.npz"


def load_video_lb(cfg, pair_id: str) -> dict:
    with np.load(video_cache_path(cfg, pair_id)) as z:
        assert str(z["config_hash"]) == LB.CONFIG_HASH, f"{pair_id}: stale LanguageBind cache"
        return {"whole": z["lb_whole"].astype(np.float32), "segs": z["lb_segs"].astype(np.float32), "bounds": z["seg_bounds_s"]}


def extract_videos(cfg, store: FeatureStore, pair_ids: list[str]) -> None:
    """Whole-clip and per-segment LanguageBind video embeddings from the silent interim video (rule 8)."""
    from decord import VideoReader, cpu
    todo = [p for p in pair_ids if not video_cache_path(cfg, p).exists()
            or str(np.load(video_cache_path(cfg, p))["config_hash"]) != LB.CONFIG_HASH]
    log.info("LanguageBind video: %d of %d clips to encode (config %s)", len(todo), len(pair_ids), LB.CONFIG_HASH)
    if not todo:
        return
    tower = LB.load_tower("video")
    t0 = time.time()
    for n, p in enumerate(todo):
        vr = VideoReader(str(interim_video_path(cfg, p)), ctx=cpu(0))
        fps = vr.get_avg_fps()
        assert abs(fps - 25.0) < 0.01, f"{p}: interim video must be 25 fps, got {fps}"
        N = len(vr)
        segs = store.segments[p]
        idx = [LB.frame_indices(0.0, N / 25.0, n_total=N)] + [LB.frame_indices(a, b, n_total=N) for a, b in segs]
        frames = vr.get_batch(np.concatenate(idx).tolist()).asnumpy()
        clips = [frames[i * LB.N_FRAMES:(i + 1) * LB.N_FRAMES] for i in range(len(idx))]
        E = LB.encode_video_clips(tower, clips)
        out = video_cache_path(cfg, p)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, lb_whole=E[0].astype(np.float16), lb_segs=E[1:].astype(np.float16),
                 seg_bounds_s=np.array(segs, float), config_hash=LB.CONFIG_HASH, config=json.dumps(LB.CONFIG))
        if (n + 1) % 100 == 0:
            log.info("LanguageBind video %d/%d (%.0f s)", n + 1, len(todo), time.time() - t0)
    del tower


# ---------------------------------------------------------------- music side
class MusicLB:
    """16 kHz audio of the interim music files and memoized LanguageBind audio embeddings of windows."""

    def __init__(self, cfg, store: FeatureStore, tower):
        self.cfg, self.store, self.tower = cfg, store, tower
        self._y16: dict[str, np.ndarray] = {}
        self._emb: dict[tuple, np.ndarray] = {}

    def y16(self, m: str) -> np.ndarray:
        if m not in self._y16:
            y, sr = sf.read(interim_music_path(self.cfg, m), dtype="float32")
            self._y16[m] = LB.resample_48k_to_16k(y.mean(1) if y.ndim == 2 else y, sr)
        return self._y16[m]

    def windows(self, keys: list[tuple]) -> np.ndarray:
        """[n, 768] embeddings of (music_id, a, b) windows; new ones encoded in one batch."""
        new = list(dict.fromkeys(k for k in keys if k not in self._emb))
        if new:
            waves = [self.y16(m)[int(round(a * LB.SR)):int(round(b * LB.SR))] for m, a, b in new]
            for k, e in zip(new, LB.encode_audio_clips(self.tower, waves, LB.SR)):
                self._emb[k] = e
        return np.stack([self._emb[k] for k in keys])

    def soundtracks(self, mixes48: list[np.ndarray]) -> np.ndarray:
        return LB.encode_audio_clips(self.tower, [LB.resample_48k_to_16k(y, 48000) for y in mixes48], LB.SR)

    def files(self, ids: list[str]) -> np.ndarray:
        """Whole music files (three chunks when longer than 10.36 s)."""
        return LB.encode_audio_clips(self.tower, [self.y16(m) for m in ids], LB.SR)

    def drop_audio(self) -> None:
        self._y16.clear()


def span_key(store: FeatureStore, m: str, off: float, dur: float) -> tuple:
    a, b = store.span("music", m, off, off + dur)
    return (m, round(float(a), 3), round(float(b), 3))


# ---------------------------------------------------------------- one run
def evaluate_run(cfg, store, M: MusicLB, fold: int, e: int, zs: dict) -> None:
    t0 = time.time()
    src = ROOT / SUITE / f"fold{fold}" / f"seed{e}"
    with np.load(src / "perquery.npz") as z:
        S = {k: z[k] for k in z.files}
    test = list(S["align/pair"])
    assert test == list(fold_ids(cfg, fold)["test"]), "test order differs from the suite run"
    segs_of = {p: store.segments[p] for p in test}
    n_seg = sum(len(s) for s in segs_of.values())
    assert len(S["div/seg/original"]) == n_seg, (len(S["div/seg/original"]), n_seg)
    # check the saved original choices against the ground truth (own file, offset = segment start fitted to the file)
    k = 0
    for p in test:
        for a, b in segs_of[p]:
            m = store.music_of[p]
            assert S["div/seg/original"][k] == m, (p, k)
            want = max(0.0, min(a, max(0.0, store.duration("music", m) - (b - a))))
            assert abs(S["path/original/offset"][k] - want) < 1e-6, (p, a, S["path/original/offset"][k], want)
            k += 1
    per = {"align/pair": np.array(test)}
    for s in SYSTEMS:
        picks, offs = S[f"div/seg/{s}"], S[f"path/{s}/offset"]
        whole, tw, sm, k = [], [], [], 0
        mixes = []
        for p in test:
            V = load_video_lb(cfg, p)
            segs = segs_of[p]
            assert np.allclose(V["bounds"], np.array(segs, float)), p
            path = [Candidate(str(picks[k + i]), float(offs[k + i]), b - a, -1, float(store.music(str(picks[k + i]))["tempo_bpm"]), 0.0)
                    for i, (a, b) in enumerate(segs)]
            k += len(segs)
            Mw = M.windows([span_key(store, c.music_id, c.offset, c.dur) for c in path])
            sc = (Mw * V["segs"]).sum(1)
            d = np.array([b - a for a, b in segs])
            tw.append(LB.tw_score(sc, d))
            sm.append(float(sc.mean()))
            mix, _ = render_soundtrack(runs_from_path(path, segs), store.duration("video", p),
                                       lambda m: interim_music_path(cfg, m), cfg.render)
            mixes.append(mix)
            whole.append(V["whole"])
        Ms = M.soundtracks(mixes)
        per[f"lb/{s}/LB"] = (Ms * np.stack(whole)).sum(1).astype(float)
        per[f"lb/{s}/twLB"] = np.array(tw, float)
        per[f"lb/{s}/segLB"] = np.array(sm, float)
        per[f"lb/{s}/IB"] = S[f"align/{s}/IB"].astype(float)
    if e == 0:
        per.update({f"zs/{k}": v for k, v in zs.items()})
    d = run_dir(cfg, EXP, fold, e)
    np.savez(d / "perquery.npz", **per)
    summary = {k: float(np.nanmean(v)) for k, v in per.items() if v.dtype.kind == "f"}
    write_run(d, cfg, {"source_run": str(src), "config_hash": LB.CONFIG_HASH, "summary": summary}, e)
    log.info("LB fold %d seed %d (%.0f s): LB %s; tw-LB %s", fold, e, time.time() - t0,
             {s: round(summary[f"lb/{s}/LB"], 4) for s in SYSTEMS}, {s: round(summary[f"lb/{s}/twLB"], 4) for s in SYSTEMS})


def zero_shot(store, M: MusicLB, cfg, test: list[str]) -> dict:
    """LanguageBind zero-shot track retrieval: whole clip vs whole music file over the test fold's gallery."""
    gal = list(dict.fromkeys(store.music_of[p] for p in test))
    G = M.files(gal)
    gtrack = np.array([store.music_track[m] for m in gal])
    ranks = []
    for p in test:
        s = G @ load_video_lb(cfg, p)["whole"]
        hit = gtrack == store.track[p]
        ranks.append(int((s > s[hit].max()).sum()) + 1)
    return {"track_rank": np.array(ranks), "N": np.full(len(test), len(gal))}


def run_all() -> None:
    import torch
    cfg = load_config(ROOT / "pipeline_ens10" / "fold0" / "seed0" / "config.yaml")
    store = FeatureStore(cfg)
    tests = {k: list(fold_ids(cfg, k)["test"]) for k in FOLDS}
    extract_videos(cfg, store, [p for k in FOLDS for p in tests[k]])
    M = MusicLB(cfg, store, LB.load_tower("audio"))
    for k in FOLDS:
        zs = None
        for e in SEEDS:
            if (ROOT / EXP / f"fold{k}" / f"seed{e}" / "perquery.npz").exists():
                continue
            cfg_k = load_config(ROOT / "pipeline_ens10" / f"fold{k}" / f"seed{e}" / "config.yaml")
            if e == 0 and zs is None:
                zs = zero_shot(store, M, cfg, tests[k])
            evaluate_run(cfg_k, store, M, k, e, zs or {})
        M.drop_audio()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- report
def write_report() -> Path:
    from bgm.analysis.eval_suite_report import B, Pool, _film, ci, ms
    from bgm.eval.metrics import cluster_bootstrap
    R = {k: [] for k in FOLDS}
    for k in FOLDS:
        for e in SEEDS:
            with np.load(ROOT / EXP / f"fold{k}" / f"seed{e}" / "perquery.npz") as z:
                R[k].append({n: z[n] for n in z.files})
    P = Pool(R, _film())
    n_vid = sum(len(R[k][0]["align/pair"]) for k in FOLDS)
    L = ["# LanguageBind score of the final system (D117, title split)", "",
         "Generated by `python -m bgm.analysis.eval_lb report` (src/bgm/analysis/eval_lb.py; encoders in "
         "src/bgm/features/languagebind.py) from runs/titlesplit/eval_lb/. The soundtracks are the ones the evaluation "
         "suite (D111, results/titlesplit/eval_suite.md) decoded and stored per segment; nothing is retrained or tuned. "
         f"LanguageBind_Video_FT and LanguageBind_Audio_FT (frozen, fp16; config {LB.CONFIG_HASH}). {n_vid} natural test videos; "
         "tables: mean ± std over 15 runs (5 folds × 3 disjoint 10-head ensembles); [95% CI]: per-video values averaged "
         f"over seeds, cluster bootstrap over film groups, {B:,} resamples.", "",
         "* **LB**: cosine between the LanguageBind embedding of the whole video (8 frames spread over the clip) and of the "
         "whole rendered soundtrack (what `bgm render` plays, before beat sync).",
         "* **tw-LB** (time-weighted): each segment's cosine (its 8 frames vs its chosen music window, widened to ≥ 5 s as "
         "for IB), averaged with the segment lengths as weights.",
         "* seg-LB: the same average without the weights. IB: the suite's ImageBind score (duration-weighted), for comparison.", "",
         "Higher = the music suits the picture more in LanguageBind's joint space.", "",
         "## 1. Scores", "",
         "| system | LB [95% CI] | tw-LB [95% CI] | seg-LB | IB (D111) |", "| --- | --- | --- | --- | --- |"]
    for s in SYSTEMS:
        L.append(f"| {LABEL[s]} | {ms(P.per_run(f'lb/{s}/LB'), 4)} {ci(P.boot(f'lb/{s}/LB', 'video'), 4)} | "
                 f"{ms(P.per_run(f'lb/{s}/twLB'), 4)} {ci(P.boot(f'lb/{s}/twLB', 'video'), 4)} | "
                 f"{ms(P.per_run(f'lb/{s}/segLB'), 4)} | {ms(P.per_run(f'lb/{s}/IB'))} |")
    L += ["", "## 2. Paired differences (per video, seeds averaged)", "",
          "| comparison | Δ LB [95% CI] | Δ tw-LB [95% CI] | Δ IB [95% CI] |", "| --- | --- | --- | --- |"]
    for a, b in (("viterbi", "random"), ("viterbi", "greedy"), ("viterbi", "original"), ("original", "random")):
        cells = [ci(P.boot(f"lb/{a}/{m}", "video", minus=f"lb/{b}/{m}"), 4) for m in ("LB", "twLB", "IB")]
        dm = [float(np.mean(P.pooled(f"lb/{a}/{m}", "video")[0] - P.pooled(f"lb/{b}/{m}", "video")[0])) for m in ("LB", "twLB", "IB")]
        L.append(f"| {LABEL[a].strip('*')} − {LABEL[b].strip('*')} | " + " | ".join(f"{d:+.4f} {c}" for d, c in zip(dm, cells)) + " |")
    # zero-shot retrieval + correlation with IB
    rk = np.concatenate([R[k][0]["zs/track_rank"] for k in FOLDS])
    cl = np.concatenate([P.clusters(k, "video") for k in FOLDS])
    N = np.concatenate([R[k][0]["zs/N"] for k in FOLDS])
    r10 = [float((R[k][0]["zs/track_rank"] <= 10).mean()) for k in FOLDS]
    b10 = cluster_bootstrap((rk <= 10).astype(float), cl, B)
    tw, cl2 = P.pooled("lb/viterbi/twLB", "video")
    ib = P.pooled("lb/viterbi/IB", "video")[0]
    tw_o, ib_o = P.pooled("lb/original/twLB", "video")[0], P.pooled("lb/original/IB", "video")[0]
    L += ["", "## 3. Checks", "",
          f"LanguageBind zero-shot track retrieval (whole clip vs whole music file, test-fold gallery of {N.mean():.0f} files on "
          f"average; a hit is any file of the query's track group): R@1 {np.mean([float((R[k][0]['zs/track_rank'] <= 1).mean()) for k in FOLDS]):.3f}, "
          f"R@10 {ms(r10)} {ci(b10)}, MedR {np.mean([np.median(R[k][0]['zs/track_rank']) for k in FOLDS]):.1f} "
          "(chance R@10 0.058, ImageBind zero-shot 0.110, final system 0.295; results/titlesplit/eval_suite.md).", "",
          f"Per-video Pearson correlation of tw-LB with IB: final {np.corrcoef(tw, ib)[0, 1]:.3f}, original music "
          f"{np.corrcoef(tw_o, ib_o)[0, 1]:.3f}.", ""]
    OUT.write_text("\n".join(L))
    return OUT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what in ("run", "all"):
        run_all()
    if what in ("report", "all"):
        log.info("wrote %s", write_report())


if __name__ == "__main__":
    main()
