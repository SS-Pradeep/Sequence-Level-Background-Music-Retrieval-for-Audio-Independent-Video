"""L (mean cue length) and rho (reprise rate) from the training films' BGM only (per fold).

`.venv/bin/python -m stitching.cue_stats [--jobs 16]` -> stitching/results/cue_stats.json

Audio: the interim 48 kHz FLAC of each music file (the clip's BGM; the dataset's music is already the BGM), loaded
at 22,050 Hz, high-passed at 120 Hz (removes the Konjum DC / 100 Hz hum, memory note), mono.
Features: chroma_stft (12) + MFCC 2..13 (12), each z-scored per file, pooled to 0.5 s frames.
Boundaries: Foote novelty on the cosine self-similarity matrix (Gaussian-tapered checkerboard kernel, half-width 8 s),
peaks via librosa.util.peak_pick (local max within +-4 s, above the moving mean + 0.1 over +-8 s, >= 8 s apart).
Cues of a file = the pieces between its boundaries (and the file ends).
L   = total training music duration / number of cues.
rho = share of cues (>= 4 s) whose chroma cross-correlation with an EARLIER cue of the same film exceeds the 99th
      percentile of the same statistic over 20,000 random cue pairs from different films (film order = file order
      by name, which follows the film timeline for the numbered clips).
      Cross-correlation = max over lags (overlap >= 4 s) of the mean frame cosine of the 1 s chroma sequences.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
SR = 22050
HOP = 512
FR = 0.5                 # s per pooled frame
CACHE = Path("stitching/cache/cue_feats.npz")


def file_features(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    import librosa
    from scipy.signal import butter, sosfiltfilt
    y, sr = librosa.load(path, sr=SR, mono=True)
    assert sr == SR
    y = sosfiltfilt(butter(4, 120, "highpass", fs=SR, output="sos"), y).astype(np.float32)
    dur = len(y) / SR
    if dur < 1.0 or not np.any(y):
        return np.zeros((0, 24), np.float32), np.zeros((0, 12), np.float32), dur
    ch = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP)
    mf = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=14)[2:14]
    n = int(round(FR * SR / HOP))
    m = min(ch.shape[1], mf.shape[1]) // n
    pool = lambda X: X[:, :m * n].reshape(X.shape[0], m, n).mean(2).T  # noqa: E731
    C, M = pool(ch), pool(mf)
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-6)  # noqa: E731
    return np.hstack([z(C), z(M)]).astype(np.float32), C.astype(np.float32), dur


def foote_boundaries(X: np.ndarray, half_s: float = 8.0) -> list[int]:
    """Frame indices of novelty peaks (0.5 s frames)."""
    import librosa
    n = len(X)
    w = int(half_s / FR)
    if n < 2 * w // 2 + 4:
        return []
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * w)) ** 2)
    K = np.outer(g, g) * np.kron(np.array([[1, -1], [-1, 1]]), np.ones((w, w)))
    Sp = np.pad(S, w, mode="constant")
    nov = np.array([(Sp[i:i + 2 * w, i:i + 2 * w] * K).sum() for i in range(n)])
    nov = np.maximum(nov, 0)
    if nov.max() <= 0:
        return []
    nov = nov / nov.max()
    pk = librosa.util.peak_pick(nov, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.1, wait=16)
    return [int(p) for p in pk if 8 <= p <= n - 8]      # not within 4 s of a file edge


def xcorr(A: np.ndarray, B: np.ndarray, min_overlap: int = 4) -> float:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    M = An @ Bn.T
    best = -1.0
    for o in range(-(len(A) - min_overlap), len(B) - min_overlap + 1):
        d = np.diagonal(M, offset=o)
        if len(d) >= min_overlap:
            best = max(best, float(d.mean()))
    return best


def extract_all(jobs: int) -> dict:
    from joblib import Parallel, delayed
    from bgm.config import load_config
    from bgm.data import manifest
    from bgm.data.prep import interim_music_path
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    pairs = manifest.read_pairs(cfg)
    mids = sorted(set(pairs["music_id"]))
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["d"].item()
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(file_features)(str(interim_music_path(cfg, m))) for m in mids)
    d = {m: r for m, r in zip(mids, res)}
    np.savez(CACHE, d=np.array(d, dtype=object))
    return d


def cues_of(X: np.ndarray, C: np.ndarray) -> list[np.ndarray]:
    """Chroma (1 s frames) of each cue of a file."""
    b = [0] + foote_boundaries(X) + [len(X)]
    out = []
    for a, e in zip(b[:-1], b[1:]):
        c = C[a:e]
        m = len(c) // 2
        out.append(c[:2 * m].reshape(m, 2, 12).mean(1) if m else c)
    return out


def main(jobs: int) -> None:
    from bgm.config import load_config
    from bgm.data import manifest
    from bgm.eval.store import fold_ids
    feats = extract_all(jobs)
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json"])
    pairs = manifest.read_pairs(cfg).set_index("pair_id")
    fg = manifest.read_folds(cfg)["film_group"]
    rng = np.random.default_rng(0)
    out = {"method": __doc__, "folds": {}}
    for k in range(5):
        train = fold_ids(cfg, k)["train"]
        files = {}
        for p in train:
            files.setdefault(pairs.loc[p, "music_id"], fg[p])
        total, n_cues, film_cues = 0.0, 0, {}
        for m in sorted(files):
            X, C, dur = feats[m]
            if not len(X):
                continue
            cs = cues_of(X, C)
            total += dur
            n_cues += len(cs)
            film_cues.setdefault(files[m], []).extend([c for c in cs if len(c) >= 4])
        L = total / n_cues
        # null: random different-film cue pairs
        allc = [(f, c) for f, cs in film_cues.items() for c in cs]
        null = []
        for _ in range(20000):
            i, j = rng.integers(len(allc), size=2)
            if allc[i][0] != allc[j][0]:
                null.append(xcorr(allc[i][1], allc[j][1]))
        thr = float(np.quantile(null, 0.99))
        n_rep, n_elig = 0, 0
        for f, cs in film_cues.items():
            for i in range(len(cs)):
                n_elig += 1
                if i and max(xcorr(cs[h], cs[i]) for h in range(max(0, i - 60), i)) > thr:   # earlier cues (up to 60 back)
                    n_rep += 1
        rho = n_rep / max(n_elig, 1)
        out["folds"][str(k)] = {"L_s": L, "rho": rho, "n_cues": n_cues, "n_files": len(files), "music_minutes": total / 60,
                                "n_cues_ge4s": n_elig, "n_reprises": n_rep, "xcorr_threshold_p99_null": thr,
                                "n_films": len(film_cues)}
        log.info("fold %d: L %.1f s (%d cues, %.0f min, %d files), rho %.3f (%d / %d, thr %.3f)", k, L, n_cues, total / 60,
                 len(files), rho, n_rep, n_elig, thr)
    Path("stitching/results/cue_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    main(ap.parse_args().jobs)
