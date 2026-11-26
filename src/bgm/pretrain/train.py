"""M8 stage 1 (SPEC §5.7): self-supervised pretraining on the unlabeled clips.

Semantic heads: symmetric InfoNCE on random 5-10 s aligned spans (they stand in for segments), batch 256, lr 1e-3,
the M6 augmentations (drop 25% of 1 Hz frames, jitter the music span ±1 s); clips of one source video share a
track_id and are masked as negatives. Dynamics encoders: the M7 trainer with shifted negatives. Both stop when the
loss on a 5% held-out slice (by source video) has not fallen for 10 epochs.
Outputs: runs/pretrain/{semantic,dynamics}/seed<s>/{ckpt.pt, metrics.json, config.yaml, train_log.csv, ...}.
"""
from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from bgm.eval.store import FeatureStore
from bgm.pretrain.unlabeled import unlabeled_config
from bgm.runs import write_run
from bgm.train.dynamics import train_dynamics
from bgm.train.losses import info_nce
from bgm.train.semantic import SemanticInputs, SemanticModel, cosine_warmup

log = logging.getLogger(__name__)


def split_by_source(store: FeatureStore, frac: float, seed: int) -> tuple[list[str], list[str]]:
    srcs = sorted(set(store.track.values()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(srcs, max(1, int(round(frac * len(srcs)))), replace=False))
    ids = list(store.pairs.pair_id)
    return [p for p in ids if store.track[p] not in held], [p for p in ids if store.track[p] in held]


def random_span(dur: float, rng, lo: float = 5.0, hi: float = 10.0) -> tuple[float, float]:
    L = min(dur, rng.uniform(lo, hi))
    a = rng.uniform(0, max(0.0, dur - L))
    return a, a + L


def pretrain_semantic(cfg: DictConfig, seed: int, device: str) -> dict:
    ucfg = unlabeled_config(cfg)
    c = cfg.train.semantic
    store = FeatureStore(ucfg)
    tr, va = split_by_source(store, 0.05, seed)
    inp = SemanticInputs(store, c.video_feature, c.music_feature)
    dv, dm = inp.dims
    torch.manual_seed(seed)
    model = SemanticModel(dv, dm, cfg, device)
    opt = torch.optim.AdamW(model.params(), lr=c.lr_pretrain, weight_decay=c.weight_decay)
    B = c.batch_pretrain
    steps = math.ceil(len(tr) / B)
    total = c.max_epochs * steps
    tix = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    out = Path(cfg.paths.runs) / "pretrain" / "semantic" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    def batch(ids, rng, train):
        V, M, T = [], [], []
        for p in ids:
            a, b = random_span(store.duration("video", p), rng)
            V.append(inp.video(p, a, b, c.frame_drop if train else 0.0, rng))
            M.append(inp.music(store.music_of[p], a, b, c.music_jitter_s if train else 0.0, rng))
            T.append(tix[store.track[p]])
        return (torch.from_numpy(np.stack(V)).to(device), torch.from_numpy(np.stack(M)).to(device), torch.tensor(T, device=device))

    @torch.no_grad()
    def val_loss():
        rng = np.random.default_rng([seed, 999])
        ls = []
        for b0 in range(0, len(va), B):
            v, m, t = batch(va[b0:b0 + B], rng, False)
            if len(t) > 1:
                ls.append(info_nce(model.hv(v), model.hm(m), t, model.log_tau).item())
        return float(np.mean(ls))

    best, best_state, best_ep, bad, step = np.inf, None, -1, 0, 0
    logf = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(logf)
    wr.writerow(["epoch", "loss", "val_loss", "seconds"])
    for ep in range(c.max_epochs):
        t0 = time.time()
        model.train()
        order = np.random.default_rng([seed, ep]).permutation(len(tr))
        losses = []
        for b0 in range(0, len(order), B):
            v, m, t = batch([tr[i] for i in order[b0:b0 + B]], np.random.default_rng([seed, ep, b0]), True)
            for g in opt.param_groups:
                g["lr"] = c.lr_pretrain * cosine_warmup(step, total, cfg.train.warmup_frac)
            loss = info_nce(model.hv(v), model.hm(m), t, model.log_tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        vl = val_loss()
        wr.writerow([ep, np.mean(losses), vl, round(time.time() - t0, 1)])
        logf.flush()
        if vl < best - 1e-4:
            best, best_state, best_ep, bad = vl, {k: ({n: x.detach().clone() for n, x in v.items()} if isinstance(v, dict) else v.clone())
                                                  for k, v in model.state().items()}, ep, 0
        else:
            bad += 1
            if bad >= 10:
                break
    logf.close()
    torch.save({"state": best_state, "dv": dv, "dm": dm}, out / "ckpt.pt")
    metrics = {"val_loss": best, "best_epoch": best_ep, "epochs_run": ep + 1, "train_clips": len(tr), "held_out_clips": len(va)}
    write_run(out, cfg, metrics, seed)
    log.info("pretrain semantic seed %d: held-out loss %.3f (epoch %d of %d), %d train / %d held-out clips", seed, best, best_ep, ep + 1, len(tr), len(va))
    return metrics


def pretrain_dynamics(cfg: DictConfig, seed: int, device: str) -> dict:
    ucfg = unlabeled_config(cfg)
    store = FeatureStore(ucfg)
    tr, va = split_by_source(store, 0.05, seed)
    out = Path(cfg.paths.runs) / "pretrain" / "dynamics" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model, val = train_dynamics(ucfg, store, tr, va, seed, device, int(cfg.train.dynamics.n_shift_neg), out,
                                lr=cfg.train.dynamics.lr_pretrain, stop_on="loss", patience=10)
    torch.save({"state": model.state()}, out / "ckpt.pt")
    metrics = {"val_loss": -val["val_sync"], **{k: v for k, v in val.items() if k != "val_sync"}}
    write_run(out, cfg, metrics, seed)
    log.info("pretrain dynamics seed %d: held-out loss %.3f (epoch %d)", seed, metrics["val_loss"], val["best_epoch"])
    return metrics
