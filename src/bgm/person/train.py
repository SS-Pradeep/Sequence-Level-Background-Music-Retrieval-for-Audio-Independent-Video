"""Person-branch heads (Step 3, docs/person_branch_plan.md), trained like the semantic heads.

Run: python -m bgm.person.train --mode person|early --variant NAME [--folds 0,1] [--seeds 0,1,2] [dotted overrides]
     -> runs/<paths.runs>/semantic/<variant>/fold<k>/seed<s>/ (config.yaml, metrics.json, ckpt.pt, train_log.csv, ...)

Modes (video side; the music side is CLAP+MERT exactly as in siglip-both):
  person  P2  input [person mean (1152) | mask]: the span's person seconds, mean of their crop vectors, l2;
  early   P3  input [SigLIP frame mean (1152) | person mean (1152) | mask].
A span without any person second has mask 0, and the head replaces its person part with a learned "no person"
vector. Protocol = `bgm.train.semantic.train_one` (train.semantic settings, CLAUDE.md rule 4, film cap, frame drop on
both parts with the same dropped seconds, music jitter, cosine warm-up, early stopping on val segment R@10, M5 test
metrics) without the options this task does not use; `train.semantic.distill_from_seeds` with `--teachers <variant>`
distils that variant's same-fold heads (D76).
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from bgm.data import manifest
from bgm.eval.metrics import recall_summary
from bgm.eval.protocol import segment_retrieval
from bgm.eval.store import FeatureStore, fold_ids, l2
from bgm.models.heads import SemanticHead
from bgm.person import features
from bgm.runs import run_dir, write_run
from bgm.train.losses import distill_kl, info_nce
from bgm.train.semantic import Scorer, SemanticInputs, SemanticModel, cap_per_film, cosine_warmup, epoch_batches, evaluate

log = logging.getLogger(__name__)
D_PERSON = 1152
MODES = ("person", "early")


class PersonStore:
    """Per-pair person arrays (data/features/video_person) as float32 embeddings and a has-person mask."""

    def __init__(self, cfg: DictConfig):
        self.cfg, self._c = cfg, {}

    def add(self, key: str, arrays: dict) -> None:
        """Register arrays for a video that is not a dataset pair (render inputs, concatenations)."""
        self._c[key] = (arrays["person_siglip_1hz"].astype(np.float32), arrays["person_kind_1hz"] > 0)

    def get(self, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
        if pair_id not in self._c:
            self.add(pair_id, features.load(features.person_path(self.cfg, pair_id)))
        return self._c[pair_id]


def span_index(T: int, a: float, b: float) -> np.ndarray:
    """1 Hz samples centred in [a, b) (the nearest one if none): the index rule of `Series1Hz.span_mean`."""
    lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T))
    hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T))
    if hi <= lo:
        return np.array([int(np.clip(np.floor((a + b) / 2), 0, T - 1))])
    return np.arange(lo, hi)


def person_vector(emb: np.ndarray, has: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[1153]: l2 mean of the person seconds among idx, and the mask flag (zeros and 0 when there is none)."""
    sel = idx[has[idx]]
    if not len(sel):
        return np.zeros(D_PERSON + 1, np.float32)
    return np.r_[l2(emb[sel].mean(0)), 1.0].astype(np.float32)


def video_input(mode: str, siglip: np.ndarray, emb: np.ndarray, has: np.ndarray, idx: np.ndarray) -> np.ndarray:
    pv = person_vector(emb, has, idx)
    if mode == "person":
        return pv
    return np.r_[l2(siglip[idx].mean(0)), pv].astype(np.float32)


def drop_seconds(idx: np.ndarray, drop: float, rng) -> np.ndarray:
    """SemanticInputs' frame drop: keep each second with probability 1 - drop, at least one."""
    if len(idx) > 1:
        keep = idx[rng.random(len(idx)) >= drop]
        return keep if len(keep) else idx[[rng.integers(len(idx))]]
    return idx


class PersonInputs:
    """`SemanticInputs` interface (video, music, music_window, dims, store) with person video inputs."""

    def __init__(self, store: FeatureStore, pstore: PersonStore, mode: str):
        assert mode in MODES, mode
        self.store, self.pstore, self.mode = store, pstore, mode
        self.music_inp = SemanticInputs(store, "sem_siglip_1hz", "both")

    @property
    def dims(self) -> tuple[int, int]:
        return (D_PERSON + 1 if self.mode == "person" else 2 * D_PERSON + 1), self.music_inp.dims[1]

    def video(self, p: str, a: float, b: float, drop: float = 0.0, rng=None) -> np.ndarray:
        a, b = self.store.span("video", p, a, b)
        emb, has = self.pstore.get(p)
        siglip = self.store.series("video", p, "sem_siglip_1hz").x
        idx = span_index(len(siglip), a, b)
        if drop and rng is not None:
            idx = drop_seconds(idx, drop, rng)
        return video_input(self.mode, siglip, emb, has, idx)

    def music(self, m, a, b, jitter=0.0, rng=None):
        return self.music_inp.music(m, a, b, jitter, rng)

    def music_window(self, m, s, d):
        return self.music_inp.music_window(m, s, d)


class PersonVideoHead(nn.Module):
    """[B, d_prefix + 1152 + 1] -> [B, d_out] unit vectors. Where the mask (last input) is 0, the person part is
    replaced by the learned no-person vector before the SemanticHead MLP."""

    def __init__(self, d_in: int, hidden: int, d_out: int, p: float, n_hidden: int = 1):
        super().__init__()
        self.d_prefix = d_in - D_PERSON - 1
        assert self.d_prefix in (0, D_PERSON), d_in
        self.null = nn.Parameter(0.02 * torch.randn(D_PERSON))
        self.head = SemanticHead(d_in, hidden, d_out, p, n_hidden)

    def forward(self, x):
        pre, per, flag = x[:, :self.d_prefix], x[:, self.d_prefix:-1], x[:, -1:]
        per = torch.where(flag > 0.5, per, self.null.expand_as(per))
        return self.head(torch.cat([pre, per, flag], 1))


def person_model(dv: int, dm: int, cfg: DictConfig, device: str) -> SemanticModel:
    """A SemanticModel whose video head is a PersonVideoHead (state / load / embed / params work unchanged)."""
    c = cfg.train.semantic
    m = SemanticModel(dv, dm, cfg, device)
    m.hv = PersonVideoHead(dv, c.hidden, c.out_dim, c.dropout, int(c.get("hidden_layers", 1))).to(device)
    return m


def load_person_model(run: Path, cfg: DictConfig, device: str) -> SemanticModel:
    ck = torch.load(Path(run) / "ckpt.pt", map_location=device)
    m = person_model(ck["dv"], ck["dm"], cfg, device)
    m.load(ck["state"])
    m.eval()
    return m


def train_one(cfg: DictConfig, store: FeatureStore, pstore: PersonStore, fold: int, seed: int, variant: str, mode: str, device: str,
              teachers_variant: str | None = None) -> dict:
    c = cfg.train.semantic
    f = fold_ids(cfg, fold)
    train_ids, val_ids, test_ids = f["train"], f["val"], f["test"]
    cap = c.get("max_clips_per_film_per_epoch")
    film = manifest.read_film_groups(cfg)
    inp = PersonInputs(store, pstore, mode)
    dv, dm = inp.dims
    torch.manual_seed(seed)
    model = person_model(dv, dm, cfg, device)
    lr = c.lr_scratch
    opt = torch.optim.AdamW(model.params(), lr=lr, weight_decay=c.weight_decay)
    teachers = []
    if c.get("distill_from_seeds", None):
        assert teachers_variant, "distill_from_seeds needs --teachers <variant>"
        tdir = Path(cfg.paths.runs) / "semantic" / teachers_variant / f"fold{fold}"
        teachers = [load_person_model(tdir / f"seed{ts}", cfg, device) for ts in c.distill_from_seeds]
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
                g["lr"] = lr * cosine_warmup(step, total, cfg.train.warmup_frac)
            zv = model.hv(torch.from_numpy(np.stack(V)).to(device))
            zm = model.hm(torch.from_numpy(np.stack(M)).to(device))
            tt = torch.tensor(T, device=device)
            loss = info_nce(zv, zm, tt, model.log_tau)
            if teachers:
                xv, xm = np.stack(V), np.stack(M)
                tsim = torch.from_numpy(np.mean([t.embed("video", xv) @ t.embed("music", xm).T for t in teachers], 0)).to(device)
                loss = loss + float(c.get("distill_weight", 1.0)) * distill_kl(zv, zm, tsim, tt, model.log_tau, float(c.get("distill_tau", 0.07)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        val = recall_summary(segment_retrieval(store, val_ids, Scorer(model, inp).window, "all").ranks)["R@10"]
        wr.writerow([epoch, np.mean(losses), model.log_tau.exp().item(), val, round(time.time() - t0, 2)])
        logf.flush()
        if val > best:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: {n: t.detach().clone() for n, t in v.items()} if isinstance(v, dict) else v.clone() for k, v in model.state().items()}
        else:
            bad += 1
            if bad >= c.patience:
                break
    logf.close()
    model.load(best_state)
    model.eval()
    torch.save({"state": best_state, "dv": dv, "dm": dm, "variant": variant, "mode": mode,
                "person_config_hash": features.CONFIG_HASH}, out / "ckpt.pt")
    metrics = {"val": {"segment_R@10": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}, "mode": mode, "teachers": teachers_variant}
    metrics["test"], per = evaluate(cfg, store, Scorer(model, inp), test_ids)
    np.savez(out / "perquery.npz", **per)
    write_run(out, cfg, metrics, seed)
    log.info("semantic/%s fold %d seed %d: val seg R@10 %.3f (epoch %d) | test track R@10 %.3f, seg R@10 %.3f", variant, fold, seed,
             best, best_epoch, metrics["test"]["track_test_fold"]["R@10"], metrics["test"]["segment_all"]["R@10"])
    return metrics


def main() -> None:
    from bgm.person.diagnose import film_cfg
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--teachers", default=None, help="variant whose same-fold heads are the distillation teachers")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = film_cfg(a.overrides)
    device = cfg.features.device if torch.cuda.is_available() else "cpu"
    store, pstore = FeatureStore(cfg), PersonStore(cfg)
    for k in map(int, a.folds.split(",")):
        for s in map(int, a.seeds.split(",")):
            train_one(cfg, store, pstore, k, s, a.variant, a.mode, device, a.teachers)


if __name__ == "__main__":
    main()
