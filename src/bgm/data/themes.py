"""Theme groups (PROGRESS.md D66): music files of one film that are near-identical cues, for one-to-many targets.

Two files are linked when they belong to the same film group AND the cosines of their file-mean MERT and file-mean
CLAP embeddings both exceed the `pct`-th percentile of those cosines over different-film pairs (audio only; no video
feature and no model is involved). Files of one track_id are always linked. Union-find gives the groups. Train-only
films are left out of the audio links: Konjum salangai's 100 Hz hum makes all its clips look alike (D9).
The groups are labels, like track groups: they are defined on the whole library, test files included, and are NOT
verified by ear yet (reports/listening_sheet.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.features.extract import feature_path


def link_groups(n: int, links) -> np.ndarray:
    """Union-find over n items; links: iterable of (i, j). Returns a group label per item (the smallest member index)."""
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j in links:
        a, b = find(int(i)), find(int(j))
        if a != b:
            par[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def theme_links(mert: np.ndarray, clap: np.ndarray, film: np.ndarray, pct: float, skip: np.ndarray) -> tuple[np.ndarray, dict]:
    """mert, clap: [n, d] unit file means; film: [n] film group; skip: [n] bool (no audio links). Returns (bool [n, n]
    link matrix, {thresholds})."""
    sm, sc = mert @ mert.T, clap @ clap.T
    same = film[:, None] == film[None, :]
    iu = np.triu_indices(len(film), 1)
    diff = ~same[iu]
    tm, tc = float(np.percentile(sm[iu][diff], pct)), float(np.percentile(sc[iu][diff], pct))
    link = same & (sm > tm) & (sc > tc) & ~skip[:, None] & ~skip[None, :]
    np.fill_diagonal(link, False)
    return link, {"mert_min": tm, "clap_min": tc, "pct": pct}


def make_themes(cfg: DictConfig, pct: float = 99.0) -> dict:
    """{'theme': {music_id: theme_id}, thresholds, counts}; written to <manifests>/themes.json."""
    pairs = manifest.read_pairs(cfg)
    doc = manifest.read_folds(cfg)
    film_of, train_only = doc["film_group"], set(doc.get("train_only", []))
    mus = list(dict.fromkeys(pairs.music_id))
    mfilm = {m: film_of[p] for p, m in zip(pairs.pair_id, pairs.music_id)}
    mtrack = dict(zip(pairs.music_id, pairs.track_id))
    E = {"sem_mert_1hz": [], "sem_clap_1hz": []}
    for m in mus:
        with np.load(feature_path(cfg, "music", m)) as z:
            for k in E:
                v = z[k].astype(np.float32).mean(0)
                E[k].append(v / (np.linalg.norm(v) + 1e-12))
    film = np.array([mfilm[m] for m in mus])
    link, thr = theme_links(np.stack(E["sem_mert_1hz"]), np.stack(E["sem_clap_1hz"]), film, pct, np.isin(film, list(train_only)))
    first = {}
    track_links = [(first.setdefault(mtrack[m], i), i) for i, m in enumerate(mus)]
    g = link_groups(len(mus), list(zip(*np.nonzero(np.triu(link, 1)))) + track_links)
    theme = {m: f"theme{int(x):04d}" for m, x in zip(mus, g)}
    _, cnt = np.unique(g, return_counts=True)
    out = {"theme": theme, **thr, "n_files": len(mus), "n_track_groups": len(set(mtrack.values())), "n_themes": int(len(cnt)),
           "files_in_multi_file_themes": int(cnt[cnt > 1].sum()), "largest_theme": int(cnt.max()),
           "manifest_hash": doc["manifest_hash"], "verified_by_ear": False}
    themes_path(cfg).write_text(json.dumps(out, indent=1))
    return out


def themes_path(cfg: DictConfig) -> Path:
    """`train.semantic.theme_file` under the manifests dir: themes.json (same-film near-identical cues, D66) or
    music_clusters.json (many-to-many clusters over the whole library, D77)."""
    return manifest.manifests_dir(cfg) / cfg.train.semantic.get("theme_file", "themes.json")


def make_music_clusters(cfg: DictConfig, k: int = 100, seed: int = 0) -> dict:
    """Many-to-many groups (D77): k-means (cosine) over every music file's [unit CLAP mean | unit MERT mean] embedding,
    across all films, so n videos share m interchangeable pieces of music. Files of one track_id are forced into
    one cluster (the cluster of the track's first file). Train-only films are clustered like the rest. Audio only;
    not verified by ear. Written to <manifests>/music_clusters.json in the themes.json format."""
    from sklearn.cluster import KMeans
    pairs = manifest.read_pairs(cfg)
    doc = manifest.read_folds(cfg)
    mus = list(dict.fromkeys(pairs.music_id))
    mtrack = dict(zip(pairs.music_id, pairs.track_id))
    X = []
    for m in mus:
        with np.load(feature_path(cfg, "music", m)) as z:
            v = [z[key].astype(np.float32).mean(0) for key in ("sem_clap_1hz", "sem_mert_1hz")]
            X.append(np.concatenate([u / (np.linalg.norm(u) + 1e-12) for u in v]))
    X = np.stack(X)
    X = X - X.mean(0)                                        # centre: file-mean embeddings share a large common direction
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    first = {}
    lab = np.array([first.setdefault(mtrack[m], int(c)) for m, c in zip(mus, lab)])
    _, cnt = np.unique(lab, return_counts=True)
    out = {"theme": {m: f"cluster{int(c):03d}" for m, c in zip(mus, lab)}, "k": k, "n_files": len(mus), "n_themes": int(len(cnt)),
           "largest_theme": int(cnt.max()), "median_size": float(np.median(cnt)), "manifest_hash": doc["manifest_hash"], "verified_by_ear": False}
    (manifest.manifests_dir(cfg) / "music_clusters.json").write_text(json.dumps(out, indent=1))
    return out


def read_themes(cfg: DictConfig) -> dict[str, str]:
    return json.loads(themes_path(cfg).read_text())["theme"]
