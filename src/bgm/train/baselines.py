"""`bgm train baseline --variant cfd|utcmvmr|mvpt`: published video-music retrieval methods, reimplemented on our
frozen features and trained / evaluated exactly like the semantic heads (PROGRESS.md D43).

They are reimplementations of each method's idea on our features, not the authors' code or numbers:

* cfd     dual path (content + emotion; CFD / "Video-Music Retrieval: A Dual-Path Cross-Modal Network"): a content
          path on SigLIP -> CLAP+MERT sequences and an emotion path on the mood-logit sequences of both sides
          (standardized with train-split statistics). One temporal encoder per path and side; the score is the mean
          of the two paths' cosines (the fused unit vector is [z_content, z_emotion] / sqrt 2).
* utcmvmr rhythm and optical flow (UT-CMVMR, Mao et al. 2024): video tokens = SigLIP + per-second optical-flow
          statistics, music tokens = CLAP+MERT + per-second rhythm statistics (mean and max of the 5 normalized
          dynamics channels over each second). The "unified target": a tag head on each side predicts the pair's
          calibrated music mood distribution (CLAP's zero-shot mood tagger stands in for the paper's internal tagger).
* mvpt    temporal context (MVPt, Suris et al. 2022): a transformer over the 1 Hz sequence of the segment plus
          `mvpt_context_s` seconds of context on each side; only the segment's own tokens are pooled.

Shared with the semantic heads (bgm.train.semantic): segment positives with the span rule, one segment per video per
batch, same-track negatives masked (info_nce), frame drop and music jitter, the per-film epoch cap, cosine schedule,
early stopping on val segment R@10, and the M5 test protocol (`evaluate`).
Run outputs: runs/<runs>/baselines/<variant>/fold<k>/seed<s>/.
"""
from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.eval.attributes import fit_calibration
from bgm.eval.metrics import recall_summary
from bgm.eval.protocol import segment_retrieval
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.runs import run_dir, write_run
from bgm.train.losses import info_nce
from bgm.train.semantic import cap_per_film, cosine_warmup, epoch_batches, evaluate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Path_:
    name: str
    video: tuple[str, ...]
    music: tuple[str, ...]
    layers: int


VARIANTS = {
    "cfd": [Path_("content", ("siglip",), ("clap", "mert"), 1), Path_("emotion", ("attr",), ("attr",), 1)],
    "utcmvmr": [Path_("main", ("siglip", "flow"), ("clap", "mert", "rhythm"), 2)],
    "mvpt": [Path_("main", ("siglip",), ("clap", "mert"), 2)],
}


# ---------------------------------------------------------------- inputs
def per_second_stats(dyn: np.ndarray, T1: int) -> np.ndarray:
    """[T25, C] normalized 25 Hz curves -> [T1, 2C]: mean and max over each second [k, k + 1)."""
    x = dyn[: 25 * T1].reshape(T1, 25, -1) if len(dyn) >= 25 * T1 else np.pad(dyn, ((0, 25 * T1 - len(dyn)), (0, 0)), mode="edge").reshape(T1, 25, -1)
    return np.concatenate([x.mean(1), x.max(1)], 1).astype(np.float32)


def span_indices(T: int, a: float, b: float) -> np.ndarray:
    """1 Hz samples centred in [a, b) (sample k is centred at k + 0.5 s); the nearest one if none."""
    k = np.arange(T)
    idx = k[(k + 0.5 >= a) & (k + 0.5 < b)]
    return idx if len(idx) else np.array([int(np.clip(np.floor((a + b) / 2), 0, T - 1))])


class SeqFeatures:
    """Per-item 1 Hz token arrays for every feature name, cached; emotion / tag calibration fit on train only."""

    def __init__(self, store: FeatureStore, cfg: DictConfig, train_ids: list[str]):
        self.store, self._c = store, {}
        self.cal = fit_calibration(store, train_ids, "attr_logits_1hz", list(cfg.features.attributes.temps), "baselines")

    def full(self, side: str, item: str, feat: str) -> np.ndarray:
        key = (side, item, feat)
        if key not in self._c:
            st = self.store
            d = st.video(item) if side == "video" else st.music(item)
            if feat == "siglip":
                x = d["sem_siglip_1hz"].astype(np.float32)
            elif feat in ("clap", "mert"):
                x = l2(d[f"sem_{feat}_1hz"].astype(np.float32))
            elif feat == "attr":
                x = (d["attr_logits_1hz"] - self.cal.mu[side]) / self.cal.sd[side]
            elif feat in ("flow", "rhythm"):
                x = per_second_stats(st.dyn(side, item), len(d["sem_siglip_1hz" if side == "video" else "sem_clap_1hz"]))
            else:
                raise KeyError(feat)
            self._c[key] = np.asarray(x, np.float32)
        return self._c[key]

    def tokens(self, side: str, item: str, feats: tuple[str, ...]) -> np.ndarray:
        parts = [self.full(side, item, f) for f in feats]
        T = min(len(p) for p in parts)
        assert max(len(p) for p in parts) - T <= 1, (side, item, [len(p) for p in parts])
        return np.concatenate([p[:T] for p in parts], 1)

    def tag_target(self, m: str, a: float, b: float) -> np.ndarray:
        """Calibrated music mood distribution averaged over the span (UT-CMVMR's unified target stand-in)."""
        x = self.cal.transform("music")(self.store.music(m)["attr_logits_1hz"])
        return x[span_indices(len(x), a, b)].mean(0).astype(np.float32)


def seq_item(feats: SeqFeatures, side: str, item: str, a: float, b: float, names: tuple[str, ...], context_s: float,
             drop: float = 0.0, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Tokens of [a - context, b + context) and a pool mask marking the span [a, b) itself; `drop` removes that share of
    the span's tokens (never all). a, b already have the span rule applied."""
    x = feats.tokens(side, item, names)
    span = span_indices(len(x), a, b)
    if drop and rng is not None and len(span) > 1:
        keep = span[rng.random(len(span)) >= drop]
        span = keep if len(keep) else span[[rng.integers(len(span))]]
    if context_s > 0:
        ctx = span_indices(len(x), a - context_s, b + context_s)
        idx = np.union1d(ctx, span) if not drop else np.union1d(np.setdiff1d(ctx, span_indices(len(x), a, b)), span)
    else:
        idx = span
    return x[idx], np.isin(idx, span)


def pad_batch(items: list[tuple[np.ndarray, np.ndarray]], device: str):
    """[(tokens [n, D], pool [n])] -> x [B, T, D], valid [B, T], pool [B, T] (padded at the end)."""
    T = max(len(x) for x, _ in items)
    D = items[0][0].shape[1]
    X = np.zeros((len(items), T, D), np.float32)
    V = np.zeros((len(items), T), bool)
    P = np.zeros((len(items), T), bool)
    for i, (x, p) in enumerate(items):
        X[i, :len(x)], V[i, :len(x)], P[i, :len(x)] = x, True, p
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(X), t(V), t(P)


# ---------------------------------------------------------------- model
class SeqEncoder(nn.Module):
    """[B, T, d_in] tokens -> [B, d_out] unit vector: input projection, learned positions, a pre-norm transformer
    (padding masked), attention pooling over the pool-mask tokens."""

    def __init__(self, d_in: int, d_model: int = 256, layers: int = 2, heads: int = 4, p: float = 0.1,
                 d_out: int = 256, max_len: int = 256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model))
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, p, batch_first=True, norm_first=True),
                                         layers, enable_nested_tensor=False) if layers else None
        self.att = nn.Linear(d_model, 1)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_out))
        self.max_len = max_len

    def forward(self, x, valid, pool):
        T = x.shape[1]
        assert T <= self.max_len, f"sequence of {T} tokens > max_len {self.max_len}"
        h = self.inp(x) + self.pos[:T]
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=~valid)
        w = self.att(h).squeeze(-1).masked_fill(~pool, float("-inf")).softmax(-1)
        return F.normalize(self.out((w.unsqueeze(-1) * h).sum(1)), dim=-1)


class BaselineModel(nn.Module):
    def __init__(self, variant: str, dims: dict, c: DictConfig, n_tags: int):
        super().__init__()
        self.paths = VARIANTS[variant]
        mk = lambda d, L: SeqEncoder(d, c.d_model, L, c.heads, c.dropout, c.out_dim, c.max_len)  # noqa: E731
        self.v = nn.ModuleDict({p.name: mk(dims[("video", p.name)], p.layers) for p in self.paths})
        self.m = nn.ModuleDict({p.name: mk(dims[("music", p.name)], p.layers) for p in self.paths})
        self.tags = nn.ModuleDict({"video": nn.Linear(c.out_dim, n_tags), "music": nn.Linear(c.out_dim, n_tags)}) if variant == "utcmvmr" else None
        self.log_tau = nn.Parameter(torch.tensor(math.log(c.tau_init)))

    def embed(self, side: str, batches: dict) -> tuple[torch.Tensor, dict]:
        """batches: {path: (x, valid, pool)} -> fused unit vector [B, P * d_out] (mean of path cosines) and per-path z."""
        enc = self.v if side == "video" else self.m
        z = {p.name: enc[p.name](*batches[p.name]) for p in self.paths}
        return torch.cat([z[p.name] for p in self.paths], -1) / math.sqrt(len(self.paths)), z


# ---------------------------------------------------------------- scoring
class BaselineScorer:
    """track(q, gallery) / window(q, a, b, windows) for the M5 protocols; embeddings cached, windows batched."""

    def __init__(self, model: BaselineModel, feats: SeqFeatures, store: FeatureStore, context_s: float, device: str, batch: int = 512):
        self.model, self.f, self.st, self.ctx, self.dev, self.bs = model, feats, store, context_s, device, batch
        self._cache = {}

    @torch.no_grad()
    def _embed(self, side: str, specs: list[tuple[str, float, float]]) -> np.ndarray:
        """specs: [(item, a, b)] with the span rule applied -> [N, D]; cached per (side, item, a, b)."""
        keys = [(side, i, round(a, 3), round(b, 3)) for i, a, b in specs]
        todo = [(k, s) for k, s in zip(keys, specs) if k not in self._cache]
        for j in range(0, len(todo), self.bs):
            chunk = todo[j:j + self.bs]
            b = {p.name: pad_batch([seq_item(self.f, side, i, a, bb, getattr(p, side), self.ctx) for _, (i, a, bb) in chunk], self.dev)
                 for p in self.model.paths}
            z = self.model.embed(side, b)[0].float().cpu().numpy()
            for (k, _), v in zip(chunk, z):
                self._cache[k] = v
        return np.stack([self._cache[k] for k in keys])

    def _video_span(self, q, a, b):
        a, b = self.st.span("video", q, a, b)
        return q, a, b

    def _music_span(self, m, a, b):
        a, b = apply_span_rule(a, b, self.st.duration("music", m), self.st.min_span)
        return m, a, b

    def track(self, q, gallery):
        v = self._embed("video", [self._video_span(q, 0.0, self.st.duration("video", q))])[0]
        M = self._embed("music", [self._music_span(m, 0.0, self.st.duration("music", m)) for m in gallery])
        return M @ v

    def window(self, q, a, b, windows):
        v = self._embed("video", [self._video_span(q, a, b)])[0]
        M = self._embed("music", [self._music_span(m, s, s + d) for m, s, d in windows])
        return M @ v

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------- training
def train_baseline(cfg: DictConfig, store: FeatureStore, fold: int, seed: int, variant: str, device: str) -> dict:
    c, cs = cfg.train.baselines, cfg.train.semantic
    f = fold_ids(cfg, fold)
    train_ids, val_ids, test_ids = f["train"], f["val"], f["test"]
    feats = SeqFeatures(store, cfg, train_ids)
    ctx = c.mvpt_context_s if variant == "mvpt" else 0.0
    p0 = store.pairs.pair_id.iloc[0]
    dims = {(side, p.name): feats.tokens(side, p0 if side == "video" else store.music_of[p0], getattr(p, side)).shape[1]
            for p in VARIANTS[variant] for side in ("video", "music")}
    torch.manual_seed(seed)
    model = BaselineModel(variant, dims, c, len(cfg.features.attributes.moods)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    cap = cs.get("max_clips_per_film_per_epoch")
    film = manifest.read_film_groups(cfg) if cap else None
    train_film = [film[p] for p in train_ids] if cap else None
    n_epoch = len(cap_per_film(train_film, cap, np.random.default_rng(0))) if cap else len(train_ids)
    total = c.max_epochs * math.ceil(n_epoch / c.batch)
    track_index = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    out = run_dir(cfg, f"baselines/{variant}", fold, seed)
    logf = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(logf)
    wr.writerow(["epoch", "loss", "tau", "val_seg_R@10", "seconds"])
    best, best_state, best_epoch, bad, step = -1.0, None, -1, 0, 0
    for epoch in range(c.max_epochs):
        t0 = time.time()
        model.train()
        rng = np.random.default_rng([seed, epoch])
        if cap:
            sel = cap_per_film(train_film, cap, np.random.default_rng([seed, epoch, 1]))
            order = sel[rng.permutation(len(sel))]
        else:
            order = rng.permutation(len(train_ids))
        losses = []
        for bi, ids in enumerate(epoch_batches(train_ids, order, c.batch)):
            V = {p.name: [] for p in model.paths}
            M = {p.name: [] for p in model.paths}
            T, tags = [], []
            for j, pid in enumerate(ids):
                irng = np.random.default_rng([seed, epoch, int(order[bi * c.batch + j])])
                a, b = store.segments[pid][irng.integers(len(store.segments[pid]))]
                va, vb = store.span("video", pid, a, b)
                m = store.music_of[pid]
                dur = store.duration("music", m)
                ma, mb = apply_span_rule(a, b, dur, store.min_span)
                jit = float(np.clip(irng.uniform(-c.music_jitter_s, c.music_jitter_s), -ma, dur - mb))
                ma, mb = ma + jit, mb + jit
                for p in model.paths:
                    V[p.name].append(seq_item(feats, "video", pid, va, vb, p.video, ctx, c.frame_drop, irng))
                    M[p.name].append(seq_item(feats, "music", m, ma, mb, p.music, ctx))
                T.append(track_index[store.track[pid]])
                if model.tags is not None:
                    tags.append(feats.tag_target(m, ma, mb))
            for g in opt.param_groups:
                g["lr"] = c.lr * cosine_warmup(step, total, cfg.train.warmup_frac)
            zv, pv = model.embed("video", {k: pad_batch(v, device) for k, v in V.items()})
            zm, pm = model.embed("music", {k: pad_batch(v, device) for k, v in M.items()})
            tid = torch.tensor(T, device=device)
            loss = info_nce(zv, zm, tid, model.log_tau)
            if len(model.paths) > 1:                             # dual path: each path is trained on its own too
                loss = loss + sum(info_nce(pv[p.name], pm[p.name], tid, model.log_tau) for p in model.paths) / len(model.paths)
            if model.tags is not None:
                tgt = torch.from_numpy(np.stack(tags)).to(device)
                loss = loss + c.tag_weight * sum(F.kl_div(model.tags[s](z).log_softmax(-1), tgt, reduction="batchmean")
                                                 for s, z in (("video", zv), ("music", zm)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        val = recall_summary(segment_retrieval(store, val_ids, BaselineScorer(model, feats, store, ctx, device).window, "all").ranks)["R@10"]
        wr.writerow([epoch, np.mean(losses), model.log_tau.exp().item(), val, round(time.time() - t0, 2)])
        logf.flush()
        if val > best:
            best, best_state, best_epoch, bad = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            bad += 1
            if bad >= c.patience:
                break
    logf.close()
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state": best_state, "dims": dims, "variant": variant}, out / "ckpt.pt")
    metrics = {"val": {"segment_R@10": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}}
    metrics["test"], per = evaluate(cfg, store, BaselineScorer(model, feats, store, ctx, device), test_ids)
    np.savez(out / "perquery.npz", **per)
    write_run(out, cfg, metrics, seed)
    log.info("baselines/%s fold %d seed %d: val seg R@10 %.3f (epoch %d) | test track R@10 %.3f, seg R@10 %.3f", variant, fold, seed,
             best, best_epoch, metrics["test"]["track_test_fold"]["R@10"], metrics["test"]["segment_all"]["R@10"])
    return metrics


def run_train_baseline(cfg: DictConfig, variant: str, folds=None, seeds=None) -> list[dict]:
    assert variant in VARIANTS, f"unknown baseline {variant}; known: {list(VARIANTS)}"
    device = cfg.features.device if torch.cuda.is_available() else "cpu"
    store = FeatureStore(cfg)
    folds = range(cfg.splits.n_folds) if folds is None else folds
    seeds = list(cfg.train.seeds) if seeds is None else seeds
    return [train_baseline(cfg, store, k, s, variant, device) for k in folds for s in seeds]
