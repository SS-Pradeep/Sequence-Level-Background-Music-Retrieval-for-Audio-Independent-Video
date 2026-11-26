"""`bgm train dynamics` (SPEC M7): per-step dynamics encoders trained for video-music synchronization.

* Windows (§3.5): 8 s = 200 steps, starts uniform in [margin, T25 - margin - 200] (margin 1.25 s), one per 10 s of
  clip per epoch (at least one). Clips too short for a window are skipped and counted.
* Context encoding (§5.6), so no kept step sees padding that whole-file retrieval would not: video over
  [start - C, start + T + C], music over [start - 4 s - C, start + T + 4 s + C] (C = 30), both clamped to the file;
  the aligned window and every shifted negative are sliced from that one music encoding.
* Negatives: the batch's other music windows (same track masked) + up to K shifted copies of the item's own music
  from `sample_negative_shifts`, seeded per item. Never time-shift inputs as augmentation (CLAUDE.md rule 5).
* Augmentation: gain x0.8-1.2 per continuous channel, Gaussian noise 0.05, zero one channel with p 0.2.
* Early stopping on val sync accuracy (grounding: every 25 Hz offset of the pair's own music file, SPEC §5.12).
Run outputs: runs/dynamics/<variant>/fold<k>/seed<s>/.
"""
from __future__ import annotations

import csv
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bgm.eval.protocol import grounding, grounding_legacy
from bgm.eval.store import FeatureStore, fold_ids
from bgm.features.dynamics import IMPULSE_CHANNELS, M_CHANNELS, V_CHANNELS
from bgm.models.heads import DynamicsEncoder
from bgm.runs import run_dir, write_run
from bgm.train.losses import dynamics_loss
from bgm.train.semantic import cosine_warmup
from bgm.train.shifts import sample_negative_shifts

log = logging.getLogger(__name__)
RATE = 25
V_CONT = [i for i, c in enumerate(V_CHANNELS) if c not in IMPULSE_CHANNELS]
M_CONT = [i for i, c in enumerate(M_CHANNELS) if c not in IMPULSE_CHANNELS]
ONSET = M_CHANNELS.index("log_onset")


def window_length(n25: int, c: DictConfig, buckets: bool = True) -> int | None:
    """Longest window (steps) that fits the clip with its edge margins: 8 s, else the 6 s / 4 s buckets."""
    margin = int(math.ceil(c.edge_margin_s * RATE))
    lengths = [c.window_s] + (list(c.get("length_buckets_s", [])) if buckets else [])
    for L in sorted(lengths, reverse=True):
        T = int(round(L * RATE))
        if n25 - 2 * margin >= T:
            return T
    return None


def window_starts(n25: int, c: DictConfig, rng: np.random.Generator, buckets: bool = False) -> list[int]:
    """Start frames for one clip in one epoch; [] when the clip is too short for a window."""
    T, margin = window_length(n25, c, buckets), int(math.ceil(c.edge_margin_s * RATE))
    if T is None:
        return []
    lo, hi = margin, n25 - margin - T
    k = max(1, int((n25 / RATE) // 10 * c.windows_per_10s))
    return [int(s) for s in rng.integers(lo, hi + 1, k)]


class DynModel:
    def __init__(self, cfg: DictConfig, device: str):
        c = cfg.train.dynamics
        self.ev = DynamicsEncoder(c.get("c_in_video", None) or c.c_in, c.channels, c.kernel, tuple(c.dilations)).to(device)
        self.em = DynamicsEncoder(c.c_in, c.channels, c.kernel, tuple(c.dilations)).to(device)
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(c.tau_init), device=device))
        self.device = device

    def params(self):
        return list(self.ev.parameters()) + list(self.em.parameters()) + [self.log_tau]

    def train(self):
        self.ev.train(), self.em.train()

    def eval(self):
        self.ev.eval(), self.em.eval()

    def state(self) -> dict:
        return {"ev": {k: v.detach().clone() for k, v in self.ev.state_dict().items()},
                "em": {k: v.detach().clone() for k, v in self.em.state_dict().items()}, "log_tau": self.log_tau.detach().clone()}

    def load(self, st: dict) -> None:
        self.ev.load_state_dict(st["ev"]), self.em.load_state_dict(st["em"])
        self.log_tau.data.copy_(st["log_tau"].to(self.device))

    @torch.no_grad()
    def encode(self, side: str, x: np.ndarray) -> torch.Tensor:
        """Whole-file encoding [T25, 64] of normalized curves [T25, 5]."""
        enc = self.ev if side == "video" else self.em
        return enc(torch.from_numpy(np.ascontiguousarray(x.T, np.float32))[None].to(self.device))[0]


def augment(x: np.ndarray, cont: list[int], c: DictConfig, rng: np.random.Generator) -> np.ndarray:
    """[L, 5] curves: gain per continuous channel, noise on continuous channels, maybe zero one channel."""
    x = x.copy()
    lo, hi = c.gain_range
    x[:, cont] = x[:, cont] * rng.uniform(lo, hi, len(cont)) + rng.normal(0, c.noise_sigma, (len(x), len(cont)))
    if rng.random() < c.channel_drop_p:
        x[:, rng.integers(x.shape[1])] = 0.0
    return x.astype(np.float32)


def encode_grouped(enc: DynamicsEncoder, seqs: list[np.ndarray], device: str) -> list[torch.Tensor]:
    """Encode variable-length [L, 5] sequences without padding (batched by exact length) -> [L, 64] each."""
    out = [None] * len(seqs)
    groups = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[len(s)].append(i)
    for idx in groups.values():
        x = torch.from_numpy(np.stack([seqs[i].T for i in idx])).to(device)
        for i, h in zip(idx, enc(x)):
            out[i] = h
    return out


def make_batch(store: FeatureStore, items, cfg: DictConfig, seed: int, epoch: int, augment_on: bool, k_shift: int, T: int):
    """items: [(pair_id, start, item_idx)], all with window length T -> inputs for `dynamics_loss`."""
    c = cfg.train.dynamics
    C, S = c.context_steps, int(round(c.shift_max_s * RATE))
    v_in, m_in, v_off, m_off, shifts = [], [], [], [], []
    for p, s, idx in items:
        rng = np.random.default_rng([seed, epoch, idx])
        v, m = store.dyn("video", p), store.dyn("music", store.music_of[p])
        a, b = max(0, s - C), min(len(v), s + T + C)
        ma, mb = max(0, s - S - C), min(len(m), s + T + S + C)
        vx, mx = v[a:b], m[ma:mb]
        if augment_on:
            vx = augment(vx, V_CONT if vx.shape[1] == len(V_CHANNELS) else list(range(vx.shape[1])), c, rng)
            mx = augment(mx, M_CONT, c, rng)
        v_in.append(vx), m_in.append(mx), v_off.append(s - a), m_off.append(s - ma)
        beats = np.asarray(store.music(store.music_of[p])["beat_times_s"], float)
        sh = sample_negative_shifts(m[:, ONSET], beats, s, T, rng, k=k_shift, fps=RATE, min_s=c.shift_min_s, max_s=c.shift_max_s,
                                    guard_s=c.guard_s, max_ac=c.max_ac, ctx=c.ac_ctx_steps) if k_shift else []
        shifts.append([x - ma for x in sh if ma <= x and x + T <= mb])      # starts inside the encoded span
    return v_in, m_in, v_off, m_off, shifts, T


def batch_loss(model: DynModel, store: FeatureStore, items, cfg, seed, epoch, k_shift, track_index, T, train=True):
    dev = model.device
    v_in, m_in, v_off, m_off, shifts, T = make_batch(store, items, cfg, seed, epoch, train, k_shift, T)
    hv_all = encode_grouped(model.ev, v_in, dev)
    hm_all = encode_grouped(model.em, m_in, dev)
    hv = torch.stack([h[o:o + T] for h, o in zip(hv_all, v_off)])
    hm = torch.stack([h[o:o + T] for h, o in zip(hm_all, m_off)])
    K = max(k_shift, 1)
    hm_shift = torch.zeros(len(items), K, T, hv.shape[-1], device=dev)
    valid = torch.zeros(len(items), K, dtype=torch.bool, device=dev)
    for i, sh in enumerate(shifts):
        for j, o in enumerate(sh[:K]):
            hm_shift[i, j], valid[i, j] = hm_all[i][o:o + T], True
    tr = torch.tensor([track_index[store.track[p]] for p, _, _ in items], device=dev)
    return dynamics_loss(hv, hm, hm_shift, valid, tr, model.log_tau)


def offset_scorer(model: DynModel, store: FeatureStore):
    """offset_score(pair, a, b, music) for `grounding`: mean per-step cosine at every 25 Hz offset (sliding_dyn_scores)."""
    cache = {}

    def enc(side, i):
        if (side, i) not in cache:
            cache[(side, i)] = model.encode(side, store.dyn(side, i))
        return cache[(side, i)]

    def score(p, a, b, m):
        hv_full, hm = enc("video", p), enc("music", m)
        f0, L = int(round(a * RATE)), int(round((b - a) * RATE))
        hv = hv_full[f0:f0 + L]
        return (F.conv1d(hm.T[None], hv.T[None])[0, 0] / len(hv)).cpu().numpy()
    return score


def train_dynamics(cfg: DictConfig, store: FeatureStore, train_ids, val_ids, seed: int, device: str, k_shift: int,
                   out_dir=None, max_epochs: int | None = None, init: dict | None = None, lr: float | None = None,
                   stop_on: str = "sync", patience: int | None = None):
    """stop_on: 'sync' (val grounding sync accuracy, SPEC stage 2) or 'loss' (held-out loss, stage 1 pretraining).
    init: a DynModel state to start from (fine-tuning after pretraining); lr defaults to lr_scratch."""
    c = cfg.train.dynamics
    torch.manual_seed(seed)
    model = DynModel(cfg, device)
    if init is not None:
        model.load(init)
    lr = c.lr_scratch if lr is None else lr
    patience = c.patience if patience is None else patience
    opt = torch.optim.AdamW(model.params(), lr=lr, weight_decay=c.weight_decay)
    max_epochs = max_epochs or c.max_epochs
    n_frames = {p: min(len(store.video(p)["dyn_25hz"]), len(store.music(store.music_of[p])["dyn_25hz"])) for p in list(train_ids) + list(val_ids)}
    skipped_8s = [p for p in train_ids if window_length(n_frames[p], c, buckets=False) is None]
    use_buckets = len(skipped_8s) > 0.1 * len(train_ids) and bool(c.get("length_buckets_s"))   # SPEC §3.5
    skipped = [p for p in train_ids if window_length(n_frames[p], c, use_buckets) is None]
    per_epoch = sum(len(window_starts(n_frames[p], c, np.random.default_rng(0), use_buckets)) for p in train_ids)
    total = max_epochs * max(1, math.ceil(per_epoch / c.batch))
    track_index = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    logf = open(out_dir / "train_log.csv", "w", newline="") if out_dir else None
    wr = csv.writer(logf) if logf else None
    if wr:
        wr.writerow(["epoch", "loss", "tau", "val_sync", "val_strict", "seconds"])
    # -inf, not -1: with stop_on="loss" the criterion is -loss, which is below -1 for any loss > 1 (M8 pretraining)
    best, best_state, best_epoch, bad, step = -math.inf, None, -1, 0, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        rng = np.random.default_rng([seed, epoch, 10_000_000])
        items = [(p, s, window_length(n_frames[p], c, use_buckets)) for i, p in enumerate(train_ids)
                 for s in window_starts(n_frames[p], c, np.random.default_rng([seed, epoch, i]), use_buckets)]
        order = rng.permutation(len(items))
        by_len = defaultdict(list)                                  # batches never mix window lengths
        for i in order:
            by_len[items[i][2]].append(int(i))
        batches = [(T, idx[b0:b0 + c.batch]) for T, idx in by_len.items() for b0 in range(0, len(idx), c.batch)]
        losses = []
        for bi in rng.permutation(len(batches)):
            T, idx = batches[bi]
            batch = [(items[i][0], items[i][1], i) for i in idx]
            for g in opt.param_groups:
                g["lr"] = lr * cosine_warmup(step, total, cfg.train.warmup_frac)
            loss = batch_loss(model, store, batch, cfg, seed, epoch, k_shift, track_index, T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        if stop_on == "loss":
            sync, strict = -held_out_loss(model, store, val_ids, n_frames, cfg, seed, k_shift, track_index), float("nan")
        else:
            # early stopping keeps the pre-chance-level criterion (D31): every model selects epochs the same way
            g = grounding_legacy(store, val_ids, offset_scorer(model, store), cfg)
            sync, strict = float(g.sync.mean()), float(g.strict.mean())
        if wr:
            wr.writerow([epoch, np.mean(losses), model.log_tau.exp().item(), sync, strict, round(time.time() - t0, 2)])
            logf.flush()
        if sync > best:
            best, best_state, best_epoch, bad = sync, model.state(), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if logf:
        logf.close()
    model.load(best_state)
    model.eval()
    return model, {"val_sync": best, "best_epoch": best_epoch, "epochs_run": epoch + 1, "skipped_clips": len(skipped),
                   "too_short_for_8s": len(skipped_8s), "length_buckets": use_buckets,
                   "train_clips": len(train_ids), "windows_per_epoch": per_epoch}


@torch.no_grad()
def held_out_loss(model: DynModel, store: FeatureStore, val_ids, n_frames, cfg, seed, k_shift, track_index) -> float:
    """Mean contrastive loss on fixed (seeded, unaugmented) windows of the held-out clips."""
    c = cfg.train.dynamics
    items = [(p, s, window_length(n_frames[p], c, True)) for i, p in enumerate(val_ids)
             for s in window_starts(n_frames[p], c, np.random.default_rng([seed, 999, i]), True)]
    by_len = defaultdict(list)
    for k, it in enumerate(items):
        by_len[it[2]].append(k)
    losses, weights = [], []
    for T, idx in by_len.items():
        for b0 in range(0, len(idx), c.batch):
            batch = [(items[k][0], items[k][1], k) for k in idx[b0:b0 + c.batch]]
            if len(batch) > 1:
                losses.append(batch_loss(model, store, batch, cfg, seed, 999, k_shift, track_index, T, train=False).item())
                weights.append(len(batch))
    return float(np.average(losses, weights=weights)) if losses else float("nan")


def grounding_summary(g) -> dict:
    """Accuracy and chance per rule (strict = within 0.5 s), plus the median absolute error."""
    out = {"median_abs_err_s": float(np.median(np.abs(g.err_s))), "n": int(len(g.err_s))}
    for r in g.hits:
        out[r] = float(g.hits[r].mean())
        out[f"{r}_chance"] = float(g.chance[r].mean())
    return out


def grounding_arrays(g) -> dict:
    """Per-query arrays saved with a run (perquery.npz): hits and chance per rule, errors, offsets, clusters, and the
    number of offsets tied at the top score."""
    out = {"err_s": g.err_s, "true": g.true, "pred": g.pred, "clusters": g.clusters, "n_tied": g.n_tied}
    for r in g.hits:
        out[r] = g.hits[r].astype(float)
        out[f"{r}_chance"] = g.chance[r]
    return out


def run_train_dynamics(cfg: DictConfig, folds=None, seeds=None, variant: str | None = None, init_dir: str | None = None) -> list[dict]:
    """init_dir: runs/pretrain/dynamics -> each seed fine-tunes from seed<s>/ckpt.pt at lr_finetune."""
    c = cfg.train.dynamics
    k_shift = int(c.n_shift_neg)
    variant = variant or ("shiftneg" if k_shift else "noshift")
    device = cfg.features.device if torch.cuda.is_available() else "cpu"
    store = FeatureStore(cfg)
    out = []
    for k in (range(cfg.splits.n_folds) if folds is None else folds):
        f = fold_ids(cfg, k)
        for s in (list(cfg.train.seeds) if seeds is None else seeds):
            d = run_dir(cfg, f"dynamics/{variant}", k, s)
            init = torch.load(Path(init_dir) / f"seed{s}" / "ckpt.pt", map_location=device)["state"] if init_dir else None
            model, val = train_dynamics(cfg, store, f["train"], f["val"], s, device, k_shift, d, init=init,
                                        lr=c.lr_finetune if init is not None else None)
            g = grounding(store, f["test"], offset_scorer(model, store), cfg)
            metrics = {"val": val, "test": {"grounding": grounding_summary(g)}}
            np.savez(d / "perquery.npz", **grounding_arrays(g))
            torch.save({"state": model.state()}, d / "ckpt.pt")
            write_run(d, cfg, metrics, s)
            log.info("dynamics/%s fold %d seed %d: val sync %.3f (epoch %d; %d clips too short for 8 s, buckets %s, %d skipped)"
                     " | test sync %.3f strict %.3f", variant, k, s, val["val_sync"], val["best_epoch"], val["too_short_for_8s"],
                     val["length_buckets"], val["skipped_clips"], metrics["test"]["grounding"]["sync"],
                     metrics["test"]["grounding"]["strict"])
            out.append(metrics)
    return out
