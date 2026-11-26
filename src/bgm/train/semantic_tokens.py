"""Phase 1b (D59): temporal tokens with late interaction, behind `train.semantic.temporal_tokens`.

Per side: Linear(d_in -> tok_dim) -> a tok_layers-deep TransformerEncoder (<= 0.5M parameters in total across both
sides, asserted) -> per-token L2. Tokens are the stored 1 Hz features over the span (video: SigLIP rows; music:
[l2 CLAP | l2 MERT] per second). Score = ColBERT max-sim: each video token takes its best music token's cosine,
averaged over video tokens. Both training and retrieval encode the exact span (no whole-file slicing), so
train-time and eval-time inputs match. Loss, batching (incl. the film cap), schedule, early stopping (val segment
R@10), run outputs and perquery format all mirror bgm.train.semantic, so arch_paired evaluates the runs unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.eval.metrics import recall_summary
from bgm.eval.protocol import segment_retrieval, track_retrieval
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.runs import run_dir, write_run
from bgm.train.semantic import cap_per_film, cosine_warmup, epoch_batches, evaluate, film_stratified_order

log = logging.getLogger(__name__)


class TokenInputs:
    """1 Hz token sequences over the span (Series1Hz index rule; at least one token)."""

    def __init__(self, store: FeatureStore, video_feature: str):
        self.store, self.vf = store, video_feature

    @property
    def dims(self) -> tuple[int, int]:
        p = self.store.pairs.pair_id.iloc[0]
        md = self.store.music(self.store.music_of[p])
        return self.store.video(p)[self.vf].shape[1], md["sem_clap_1hz"].shape[1] + md["sem_mert_1hz"].shape[1]

    @staticmethod
    def _span_idx(a: float, b: float, T: int) -> np.ndarray:
        lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T))
        hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T))
        if hi <= lo:
            return np.array([int(np.clip(np.floor((a + b) / 2), 0, T - 1))])
        return np.arange(lo, hi)

    def video(self, p: str, a: float, b: float, drop: float = 0.0, rng=None) -> np.ndarray:
        x = self.store.video(p)[self.vf].astype(np.float32)
        idx = self._span_idx(a, b, len(x))
        if drop and rng is not None and len(idx) > 1:
            keep = idx[rng.random(len(idx)) >= drop]
            idx = keep if len(keep) else idx[[rng.integers(len(idx))]]
        return l2(x[idx])

    def music(self, m: str, a: float, b: float, jitter: float = 0.0, rng=None) -> np.ndarray:
        d = self.store.music(m)
        dur = self.store.duration("music", m)
        a, b = apply_span_rule(a, b, dur, self.store.min_span)
        if jitter and rng is not None:
            j = float(np.clip(rng.uniform(-jitter, jitter), -a, dur - b))
            a, b = a + j, b + j
        c = d["sem_clap_1hz"].astype(np.float32)
        t = d["sem_mert_1hz"].astype(np.float32)
        idx = self._span_idx(a, b, min(len(c), len(t)))
        return np.concatenate([l2(c[idx]), l2(t[idx])], 1)

    def music_window(self, m: str, s: float, dd: float) -> np.ndarray:
        return self.music(m, s, s + dd)


class TokenEncoder(torch.nn.Module):
    def __init__(self, d_in: int, d: int, layers: int, heads: int, ff: int, p: float):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d)
        enc = torch.nn.TransformerEncoderLayer(d, heads, ff, dropout=p, batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(enc, layers)

    def forward(self, x, pad):                               # x [B, T, d_in]; pad [B, T] True where padding
        h = self.enc(self.proj(x), src_key_padding_mask=pad)
        return F.normalize(h, dim=-1)


class TokenModel:
    def __init__(self, dv: int, dm: int, cfg: DictConfig, device: str):
        c = cfg.train.semantic
        d, lay, heads, ff = int(c.tok_dim), int(c.tok_layers), int(c.tok_heads), int(c.tok_ff)
        self.ev = TokenEncoder(dv, d, lay, heads, ff, c.dropout).to(device)
        self.em = TokenEncoder(dm, d, lay, heads, ff, c.dropout).to(device)
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(c.tau_init), device=device))
        self.device = device
        self.n_params = sum(x.numel() for m in (self.ev, self.em) for x in m.parameters())
        assert self.n_params <= 500_000, f"1b budget: {self.n_params} parameters > 0.5M"

    def params(self):
        return list(self.ev.parameters()) + list(self.em.parameters()) + [self.log_tau]

    def train(self):
        self.ev.train(), self.em.train()

    def eval(self):
        self.ev.eval(), self.em.eval()

    def state(self) -> dict:
        return {"ev": self.ev.state_dict(), "em": self.em.state_dict(), "log_tau": self.log_tau.detach().cpu()}

    def load(self, st: dict) -> None:
        self.ev.load_state_dict(st["ev"]), self.em.load_state_dict(st["em"])
        self.log_tau.data.copy_(st["log_tau"].to(self.device))


def pad_stack(seqs: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, seqs[0].shape[1], device=device)
    pad = torch.ones(len(seqs), T, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.from_numpy(s).to(device)
        pad[i, : len(s)] = False
    return x, pad


def maxsim(zv, vpad, zm, mpad) -> torch.Tensor:
    """[B, C]: mean over valid video tokens of the max over valid music tokens of the cosine."""
    s = torch.einsum("bvd,cmd->bcvm", zv, zm)
    s = s.masked_fill(mpad[None, :, None, :], float("-inf"))   # every music row has >= 1 valid token
    s = s.max(-1).values                                      # [B, C, V]
    s = s.masked_fill(vpad[:, None, :], 0.0)
    return s.sum(-1) / (~vpad).sum(-1).clamp_min(1)[:, None]


class TokenScorer:
    """Protocol score functions; caches encoded token sequences per item / window."""

    def __init__(self, model: TokenModel, inp: TokenInputs):
        self.model, self.inp, self._c = model, inp, {}

    @torch.no_grad()
    def _enc(self, side: str, key, fn):
        if key not in self._c:
            x = fn()
            t, pad = pad_stack([x], self.model.device)
            z = (self.model.ev if side == "video" else self.model.em)(t, pad)[0, : len(x)]
            self._c[key] = z
        return self._c[key]

    @torch.no_grad()
    def _score_one(self, zv, zms: list[torch.Tensor]) -> np.ndarray:
        T = max(len(z) for z in zms)
        zm = torch.zeros(len(zms), T, zv.shape[1], device=zv.device)
        mpad = torch.ones(len(zms), T, dtype=torch.bool, device=zv.device)
        for i, z in enumerate(zms):
            zm[i, : len(z)] = z
            mpad[i, : len(z)] = False
        s = maxsim(zv[None], torch.zeros(1, len(zv), dtype=torch.bool, device=zv.device), zm, mpad)
        return s[0].cpu().numpy()

    def track(self, q, gallery):
        st = self.inp.store
        zv = self._enc("video", ("tv", q), lambda: self.inp.video(q, 0.0, st.duration("video", q)))
        zms = [self._enc("music", ("tm", m), lambda m=m: self.inp.music(m, 0.0, st.duration("music", m))) for m in gallery]
        return self._score_one(zv, zms)

    def window(self, q, a, b, windows):
        zv = self._enc("video", ("wv", q, round(a, 3), round(b, 3)), lambda: self.inp.video(q, a, b))
        zms = [self._enc("music", ("mw", m, round(s, 3), round(d, 3)), lambda m=m, s=s, d=d: self.inp.music_window(m, s, d))
               for m, s, d in windows]
        return self._score_one(zv, zms)


def train_one_tokens(cfg: DictConfig, store: FeatureStore, fold: int, seed: int, variant: str, device: str,
                     init: dict | None = None) -> dict:
    assert init is None, "1b has no pretrained init"
    c = cfg.train.semantic
    f = fold_ids(cfg, fold)
    train_ids, val_ids, test_ids = f["train"], f["val"], f["test"]
    cap = c.get("max_clips_per_film_per_epoch")
    strat = bool(c.get("film_stratified_batches", False))
    film = manifest.read_film_groups(cfg) if (cap or strat) else None
    inp = TokenInputs(store, c.video_feature)
    dv, dm = inp.dims
    torch.manual_seed(seed)
    model = TokenModel(dv, dm, cfg, device)
    opt = torch.optim.AdamW(model.params(), lr=c.lr_scratch, weight_decay=c.weight_decay)
    train_film = [film[p] for p in train_ids] if cap else None
    n_epoch = len(cap_per_film(train_film, cap, np.random.default_rng(0))) if cap else len(train_ids)
    total = c.max_epochs * math.ceil(n_epoch / c.batch)
    track_index = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    out = run_dir(cfg, f"semantic/{variant}", fold, seed)
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
        if strat:
            order = film_stratified_order(order, [film[p] for p in train_ids], int(c.get("film_batch_k", 4)),
                                          np.random.default_rng([seed, epoch, 2]))
        losses = []
        for bi, ids in enumerate(epoch_batches(train_ids, order, c.batch)):
            V, M, T = [], [], []
            for j, p in enumerate(ids):
                irng = np.random.default_rng([seed, epoch, int(order[bi * c.batch + j])])
                a, b = store.segments[p][irng.integers(len(store.segments[p]))]
                V.append(inp.video(p, a, b, c.frame_drop, irng))
                M.append(inp.music(store.music_of[p], a, b, c.music_jitter_s, irng))
                T.append(track_index[store.track[p]])
            for g in opt.param_groups:
                g["lr"] = c.lr_scratch * cosine_warmup(step, total, cfg.train.warmup_frac)
            xv, vpad = pad_stack(V, device)
            xm, mpad = pad_stack(M, device)
            s = maxsim(model.ev(xv, vpad), vpad, model.em(xm, mpad), mpad)
            tau = model.log_tau.exp().clamp(min=c.tau_min)
            tt = torch.tensor(T, device=device)
            same = (tt[:, None] == tt[None, :]) & ~torch.eye(len(ids), dtype=torch.bool, device=device)
            logits = (s / tau).masked_fill(same, float("-inf"))
            y = torch.arange(len(ids), device=device)
            loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        val = recall_summary(segment_retrieval(store, val_ids, TokenScorer(model, inp).window, "all").ranks)["R@10"]
        wr.writerow([epoch, np.mean(losses), model.log_tau.exp().item(), val, round(time.time() - t0, 2)])
        logf.flush()
        if val > best:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: ({n: t.detach().clone() for n, t in v.items()} if isinstance(v, dict) else v.clone())
                          for k, v in model.state().items()}
        else:
            bad += 1
            if bad >= c.patience:
                break
    logf.close()
    model.load(best_state)
    model.eval()
    torch.save({"state": best_state, "dv": dv, "dm": dm, "variant": variant}, out / "ckpt.pt")
    metrics = {"val": {"segment_R@10": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}, "params": model.n_params}
    metrics["test"], per = evaluate(cfg, store, TokenScorer(model, inp), test_ids)
    np.savez(out / "perquery.npz", **per)
    write_run(out, cfg, metrics, seed)
    log.info("semantic/%s fold %d seed %d (%d params): val seg R@10 %.3f (epoch %d) | test track R@10 %.3f, seg R@10 %.3f",
             variant, fold, seed, model.n_params, best, best_epoch, metrics["test"]["track_test_fold"]["R@10"],
             metrics["test"]["segment_all"]["R@10"])
    return metrics
