"""Long-video retrieval analysis: whole-clip pooling vs chunked querying, single models vs ensembles.

Run: python -m bgm.analysis.long_video   -> reports/long_video.md
Query sets:
  * 1.mp4 (silent), its original soundtrack appended to the 1,248 library files as the only correct answer.
    Its music is in no library file (fingerprint check), so all 15 models are leak-free for it.
  * Test-fold clips >= `min_long_s`, test-fold gallery; ensembles use only that fold's seeds (leak-free).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from bgm.config import load_config
from bgm.eval.metrics import cluster_bootstrap, rank_of_first_hit, recall_summary
from bgm.eval.store import FeatureStore, Series1Hz, fold_ids, l2
from bgm.retrieval.chunked import chunked_track_scores, ensemble, equal_chunks, window_starts
from bgm.train.semantic import SemanticModel

log = logging.getLogger(__name__)
VARIANT = "siglip-both"


def load_models(cfg, fold: int | None = None) -> list[tuple[str, SemanticModel]]:
    out = []
    for ck in sorted((Path(cfg.paths.runs) / "semantic" / VARIANT).glob("fold*/seed*/ckpt.pt")):
        k = int(ck.parent.parent.name[4:])
        if fold is not None and k != fold:
            continue
        dev = "cuda" if torch.cuda.is_available() else "cpu"   # CPU fallback when the GPU is busy
        st = torch.load(ck, map_location=dev)
        m = SemanticModel(st["dv"], st["dm"], cfg, dev)
        m.load(st["state"])
        m.eval()
        out.append((f"fold{k}/{ck.parent.name}", m))
    return out


class Pooler:
    """Model-independent pooled inputs: video [D_v] unit, music concat(unit CLAP, unit MERT) [512 + 1024]."""

    def __init__(self, v_series: dict[str, Series1Hz], m_series: dict[str, tuple[Series1Hz, Series1Hz]], dur: dict):
        self.v, self.m, self.dur = v_series, m_series, dur

    def video(self, q, a, b):
        return l2(self.v[q].span_mean(a, b)).astype(np.float32)

    def music(self, t, a, b):
        c, r = self.m[t]
        return np.concatenate([l2(c.span_mean(a, b)), l2(r.span_mean(a, b))]).astype(np.float32)

    def whole_music(self, t):
        return self.music(t, 0.0, self.dur[t])

    def windows(self, t, window_s, hop_s):
        d = self.dur[t]
        return np.stack([self.music(t, s, min(s + window_s, d)) for s in window_starts(d, window_s, hop_s)])


def scores_for(model, pool: Pooler, q: str, q_dur: float, gallery: list[str], wcache: dict, chunk_s=20.0, window_s=20.0, hop_s=5.0):
    """(whole-clip scores [G], chunked scores [G]) for one query and one model."""
    v = model.embed("video", pool.video(q, 0.0, q_dur)[None])[0]
    M = model.embed("music", np.stack([pool.whole_music(t) for t in gallery]))
    C = model.embed("video", np.stack([pool.video(q, a, b) for a, b in equal_chunks(q_dur, chunk_s)]))
    Wraw = [wcache.setdefault(t, pool.windows(t, window_s, hop_s)) for t in gallery]
    sizes = np.cumsum([0] + [len(w) for w in Wraw])
    Wall = model.embed("music", np.concatenate(Wraw))
    W = [Wall[sizes[i]:sizes[i + 1]] for i in range(len(gallery))]
    return M @ v, chunked_track_scores(C, W)


def analyze_1mp4(cfg, store: FeatureStore) -> dict:
    q = Path("data/interim/query")
    v_sig = np.load(q / "1_feats.npz")["siglip"]
    gt = np.load(q / "1_BGM_Ori_feats.npz")
    lib = list(dict.fromkeys(store.pairs.music_id))
    GT = "__original__"
    m_series = {t: (store.series("music", t, "sem_clap_1hz"), store.series("music", t, "sem_mert_1hz")) for t in lib}
    m_series[GT] = (Series1Hz(gt["clap"]), Series1Hz(gt["mert"]))
    dur = {t: store.duration("music", t) for t in lib}
    dur[GT] = float(len(gt["clap"]))
    q_dur = float(len(v_sig))
    pool = Pooler({"1": Series1Hz(v_sig)}, m_series, dur)
    gallery = lib + [GT]
    hits = np.zeros(len(gallery), bool)
    hits[-1] = True
    wcache, per = {}, {"whole": [], "chunked": []}
    for name, model in load_models(cfg):
        w, c = scores_for(model, pool, "1", q_dur, gallery, wcache)
        per["whole"].append(w), per["chunked"].append(c)
    res = {}
    for mode in ("whole", "chunked"):
        ranks = np.array([rank_of_first_hit(s, hits) for s in per[mode]])
        res[mode] = {"single_ranks": ranks.tolist(), "single_median": float(np.median(ranks)),
                     "single_R@10": float((ranks <= 10).mean()), "single_R@50": float((ranks <= 50).mean()),
                     "ensemble_rank": int(rank_of_first_hit(ensemble(per[mode]), hits))}
    res["n_chunks"] = len(equal_chunks(q_dur))
    res["gallery"] = len(gallery)
    return res


def analyze_long_test(cfg, store: FeatureStore, min_long_s: float = 40.0) -> dict:
    m_series = {t: (store.series("music", t, "sem_clap_1hz"), store.series("music", t, "sem_mert_1hz")) for t in dict.fromkeys(store.pairs.music_id)}
    dur = {t: store.duration("music", t) for t in m_series}
    v_series = {p: store.series("video", p, "sem_siglip_1hz") for p in store.pairs.pair_id}
    pool = Pooler(v_series, m_series, dur)
    rows = {k: [] for k in ("whole_single", "chunked_single", "whole_ens", "chunked_ens")}
    clusters, n_long, wcache = [], 0, {}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        gallery = list(dict.fromkeys(store.music_of[p] for p in f["test"]))
        gtrack = np.array([store.music_track[t] for t in gallery])
        longs = [p for p in f["test"] if store.duration("video", p) >= min_long_s]
        models = load_models(cfg, k)
        for p in longs:
            hits = gtrack == store.track[p]
            sw, sc = zip(*[scores_for(m, pool, p, store.duration("video", p), gallery, wcache) for _, m in models])
            rows["whole_single"].append(np.mean([rank_of_first_hit(s, hits) for s in sw]))
            rows["chunked_single"].append(np.mean([rank_of_first_hit(s, hits) for s in sc]))
            rows["whole_ens"].append(rank_of_first_hit(ensemble(list(sw)), hits))
            rows["chunked_ens"].append(rank_of_first_hit(ensemble(list(sc)), hits))
            clusters.append(store.track[p])
        n_long += len(longs)
    out = {"n_queries": n_long, "min_long_s": min_long_s}
    for key, r in rows.items():
        r = np.array(r, float)
        out[key] = recall_summary(r) if "ens" in key else {"R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()),
                                                            "MedR": float(np.median(r)), "n": int(len(r)), "note": "rank averaged over 3 seeds"}
    for a, b in (("chunked_ens", "whole_ens"), ("whole_ens", "whole_single"), ("chunked_ens", "whole_single")):
        ra, rb = np.array(rows[a]), np.array(rows[b])
        out[f"delta_R@10 {a} - {b}"] = cluster_bootstrap((ra <= 10).astype(float) - (rb <= 10).astype(float), np.array(clusters), 10000)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    store = FeatureStore(cfg)
    one = analyze_1mp4(cfg, store)
    log.info("1.mp4 done")
    lt = analyze_long_test(cfg, store)
    import json
    Path("reports").mkdir(exist_ok=True)
    Path("reports/long_video.json").write_text(json.dumps({"1.mp4": one, "long_test": lt}, indent=1, default=float))
    print(json.dumps({"1.mp4": {k: v for k, v in one.items() if k != "whole" and k != "chunked"},
                      "whole": {k: v for k, v in one["whole"].items() if k != "single_ranks"},
                      "chunked": {k: v for k, v in one["chunked"].items() if k != "single_ranks"}}, indent=1))
    print(json.dumps(lt, indent=1, default=float))


if __name__ == "__main__":
    main()
