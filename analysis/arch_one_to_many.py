"""One-to-many retrieval (PROGRESS.md D74), 1,248-pair film split: weighted multi-positive Stage 1, a Stage 2
reranker with positive-aware hard negatives, and one-to-many evaluation.

Run: python -m bgm.analysis.arch_one_to_many   -> reports/arch_one_to_many.md + ledger rows

Positives P(v): the paired music (primary, weight 1) and the other files of its theme group (soft, weight 0.5;
data/manifests/themes.json: same film and near-identical audio, NOT annotated and not verified by ear).
Systems (3 seeds each, per-query values averaged over seeds):
  baseline    the single-positive heads (siglip-both);
  stage 1     heads trained with `losses.info_nce_weighted` (siglip-both-o2m);
  stage 1+2   the same, with the top 50 re-ordered by a reranker: an MLP on [zv, zm, zv*zm, |zv - zm|] of the frozen
              Stage 1 embeddings, trained on the train fold with BCE (one positive sampled from P(v) per step) plus a
              margin term, negatives = hard (top Stage 1 non-positives) and random, never from P(v); epoch chosen on val.
Metrics: R@10 primary (the heads' own rule), R@10 any-hit (any file of P(v)), mAP (binary, P(v)), nDCG@10 (gain 1
for the primary track, 0.5 for a theme mate); segments: R@10 primary and any-hit. Paired Δ vs baseline, 95% cluster
bootstrap over film groups.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch

from bgm.analysis.arch_phase0 import B, FILM_ROOT, film_cfg
from bgm.analysis.full_dataset import load_scorer
from bgm.data import manifest
from bgm.data.themes import read_themes
from bgm.eval.metrics import cluster_bootstrap
from bgm.eval.protocol import segment_gallery, test_segments
from bgm.eval.store import FeatureStore, fold_ids

log = logging.getLogger(__name__)
OUT = Path("reports/arch_one_to_many.md")
SEEDS, TOPK, W_SOFT = (0, 1, 2), 50, 0.5


class Reranker(torch.nn.Module):
    def __init__(self, d: int = 256, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4 * d, hidden), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(hidden, 1))

    def forward(self, zv, zm):                               # [..., d] each -> [...] logits
        return self.net(torch.cat([zv, zm, zv * zm, (zv - zm).abs()], -1)).squeeze(-1)


def pair_scores(rr: Reranker, v: np.ndarray, M: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        tv, tm = torch.from_numpy(v).to(device), torch.from_numpy(M).to(device)
        return rr(tv[None].expand(len(tm), -1), tm).cpu().numpy()


def rerank(s1: np.ndarray, rr_scores_fn, k: int = TOPK) -> np.ndarray:
    """Stage 1 scores [G] -> final scores: the top k re-ordered by the reranker and kept above everything else."""
    top = np.argsort(-s1)[:k]
    out = s1 - 1e6
    r = rr_scores_fn(top)
    out[top] = r - r.min() + 1.0
    return out


def train_reranker(V, M, pos, Vv, Mv, pos_v, seed: int, device: str, epochs: int = 40, n_hard: int = 4, n_rand: int = 4, margin: float = 0.2):
    """V [N, d], M [G, d] train embeddings; pos [N, G] bool positive sets. Epoch chosen by val primary MRR after reranking."""
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 5])
    rr = Reranker(V.shape[1]).to(device)
    opt = torch.optim.AdamW(rr.parameters(), 1e-3, weight_decay=0.05)
    s1 = V @ M.T
    hard = [np.argsort(-np.where(pos[i], -np.inf, s1[i]))[:20] for i in range(len(V))]          # never from P(v)
    negs = [np.flatnonzero(~pos[i]) for i in range(len(V))]
    tV, tM = torch.from_numpy(V).to(device), torch.from_numpy(M).to(device)
    best, best_state = -1.0, None
    for ep in range(epochs):
        rr.train()
        order = rng.permutation(len(V))
        for b in range(0, len(order), 128):
            ii = order[b:b + 128]
            p = np.array([rng.choice(np.flatnonzero(pos[i])) for i in ii])
            n = np.stack([np.r_[rng.choice(hard[i], n_hard, replace=False), rng.choice(negs[i], n_rand, replace=False)] for i in ii])
            zv = tV[torch.from_numpy(ii).to(device)]
            sp, sn = rr(zv, tM[torch.from_numpy(p).to(device)]), rr(zv[:, None].expand(-1, n.shape[1], -1), tM[torch.from_numpy(n).to(device)])
            bce = torch.nn.functional.binary_cross_entropy_with_logits
            loss = bce(sp, torch.ones_like(sp)) + bce(sn, torch.zeros_like(sn)) + torch.relu(margin - sp[:, None] + sn).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        rr.eval()
        sv = Vv @ Mv.T
        mrr = np.mean([1.0 / (1 + int((f[~pos_v[i]] > f[pos_v[i]].max()).sum()))
                       for i in range(len(Vv)) for f in [rerank(sv[i], lambda top, i=i: pair_scores(rr, Vv[i], Mv[top], device))]])
        if mrr > best:
            best, best_state = mrr, {k: v.detach().clone() for k, v in rr.state_dict().items()}
    rr.load_state_dict(best_state)
    rr.eval()
    return rr


def track_metrics(scores: np.ndarray, primary: np.ndarray, anyhit: np.ndarray) -> dict:
    order = np.argsort(-scores)
    rel, pri = anyhit[order], primary[order]
    hits = np.flatnonzero(rel)
    ap = float(np.mean([(j + 1) / (r + 1) for j, r in enumerate(hits)]))
    gain = np.where(pri, 1.0, np.where(rel, W_SOFT, 0.0))
    disc = 1.0 / np.log2(np.arange(2, 12))
    ideal = np.sort(gain)[::-1][:10]
    return {"R@10 primary": float(pri[:10].any()), "R@10 any-hit": float(rel[:10].any()), "mAP": ap,
            "nDCG@10": float((gain[:10] * disc[:len(gain[:10])]).sum() / max((ideal * disc[:len(ideal)]).sum(), 1e-12))}


def embed_set(sc, store, vids, mus):
    inp, st = sc.inp, store
    V = np.stack([sc._emb(("tv", p), "video", lambda p=p: inp.video(p, 0.0, st.duration("video", p))) for p in vids])
    M = np.stack([sc._emb(("tm", m), "music", lambda m=m: inp.music(m, 0.0, st.duration("music", m))) for m in mus])
    return V.astype(np.float32), M.astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = film_cfg()
    film, theme = manifest.read_folds(cfg)["film_group"], read_themes(cfg)
    store = FeatureStore(cfg)
    dev = "cuda"
    SYS = ("baseline", "stage 1", "stage 1+2")
    TK = ("R@10 primary", "R@10 any-hit", "mAP", "nDCG@10")
    SK = ("seg R@10 primary", "seg R@10 any-hit")
    acc = {s: {k: [] for k in TK + SK} for s in SYS}
    CL = {"t": [], "s": []}
    for k in range(cfg.splits.n_folds):
        f = fold_ids(cfg, k)
        uniq = lambda ids: list(dict.fromkeys(store.music_of[p] for p in ids))  # noqa: E731
        gtr, gva, gte = uniq(f["train"]), uniq(f["val"]), uniq(f["test"])
        th = lambda ms: np.array([theme[m] for m in ms])  # noqa: E731
        pos = lambda vids, ms: np.stack([th(ms) == theme[store.music_of[p]] for p in vids])  # noqa: E731
        any_te = pos(f["test"], gte)
        pri_te = np.stack([np.array([store.music_track[m] for m in gte]) == store.track[p] for p in f["test"]])
        segs = test_segments(store, f["test"])
        gal = [segment_gallery(store, segs, q, "all") for q in segs]
        fold = {s: {m: [] for m in TK + SK} for s in SYS}
        for seed in SEEDS:
            for name, variant in (("baseline", "siglip-both"), ("stage 1", "siglip-both-o2m")):
                sc = load_scorer(FILM_ROOT / "semantic" / variant / f"fold{k}" / f"seed{seed}", store, dev)
                Vte, Mte = embed_set(sc, store, f["test"], gte)
                S = Vte @ Mte.T
                finals = {name: (lambda i, s1: s1, None)}
                if name == "stage 1":
                    Vtr, Mtr = embed_set(sc, store, f["train"], gtr)
                    Vva, Mva = embed_set(sc, store, f["val"], gva)
                    rr = train_reranker(Vtr, Mtr, pos(f["train"], gtr), Vva, Mva, pos(f["val"], gva), seed, dev)
                    finals["stage 1+2"] = (lambda i, s1: rerank(s1, lambda top: pair_scores(rr, Vte[i], Mte[top], dev)), rr)
                for sysn, (fn, rrm) in finals.items():
                    tm = [track_metrics(fn(i, S[i]), pri_te[i], any_te[i]) for i in range(len(S))]
                    for m in TK:
                        fold[sysn][m].append(np.array([x[m] for x in tm]))
                    sp, sa = [], []
                    for q, (windows, hit) in zip(segs, gal):
                        s1 = np.asarray(sc.window(q[0], q[2], q[3], windows))
                        if rrm is not None:
                            zv = sc._cache[("wv", q[0], round(q[2], 3), round(q[3], 3))].astype(np.float32)
                            W = np.stack([sc._cache[("mw", m, round(x, 3), round(d, 3))] for m, x, d in windows]).astype(np.float32)
                            s1 = rerank(s1, lambda top: pair_scores(rrm, zv, W[top], dev))
                        top10 = np.argsort(-s1)[:10]
                        mate = np.array([theme[m] == theme[store.music_of[q[0]]] for m, _, _ in windows])
                        sp.append(float(hit in top10)), sa.append(float(mate[top10].any()))
                    fold[sysn]["seg R@10 primary"].append(np.array(sp)), fold[sysn]["seg R@10 any-hit"].append(np.array(sa))
        for s in SYS:
            for m in TK + SK:
                acc[s][m].append(np.mean(fold[s][m], 0))
        CL["t"].append(np.array([film[p] for p in f["test"]])), CL["s"].append(np.array([film[q[0]] for q in segs]))
        log.info("fold %d: " + "; ".join(f"{s} primary {acc[s]['R@10 primary'][-1].mean():.3f} any {acc[s]['R@10 any-hit'][-1].mean():.3f}" for s in SYS), k)
    V = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in acc.items()}
    cl = {m: np.concatenate(CL["s" if m.startswith("seg") else "t"]) for m in TK + SK}
    fmt = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"  # noqa: E731
    L = ["# One-to-many retrieval: weighted multi-positive Stage 1, Stage 2 reranker, one-to-many metrics (D74)", "",
         "1,248-pair film split, test-fold gallery 196 (195–199), 3 seeds per system. Positives: the paired music (weight 1) and its theme "
         "mates (weight 0.5; same film and near-identical audio, not annotated). Each cell: value (paired Δ vs baseline [95% CI], film-group "
         f"bootstrap, {B:,} resamples).", "", "| system | " + " | ".join(TK + SK) + " |", "| --- | " + " | ".join("---" for _ in TK + SK) + " |"]
    rows = []
    for s in SYS:
        cells = []
        for m in TK + SK:
            d = None if s == "baseline" else cluster_bootstrap(V[s][m] - V["baseline"][m], cl[m], B)
            cells.append(f"{V[s][m].mean():.3f}" + ("" if d is None else f" ({fmt(d)})"))
            if d is not None and m == "R@10 primary":
                dp = d
            if d is not None and m == "R@10 any-hit":
                da = d
        L.append(f"| {s} | " + " | ".join(cells) + " |")
        if s != "baseline":
            rows.append(f"| {date.today().isoformat()} | One-to-many: {s} (weighted multi-positive InfoNCE{' + reranker' if '2' in s else ''}; themes not annotated) | film | 196 (195–199) | — | "
                        f"primary {V[s]['R@10 primary'].mean():.3f}; any-hit {V[s]['R@10 any-hit'].mean():.3f} | — | primary **{dp['mean']:+.3f}** [{dp['lo']:+.3f}, {dp['hi']:+.3f}]; "
                        f"any-hit **{da['mean']:+.3f}** [{da['lo']:+.3f}, {da['hi']:+.3f}] | {'KEEP' if dp['lo'] > 0 else ('REJECT (worse)' if dp['hi'] < 0 else 'NULL')} (primary rule) |")
    OUT.write_text("\n".join(L) + "\n")
    with open("results/arch_ablation.md", "a") as fh:
        fh.write("".join(r + "\n" for r in rows))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
