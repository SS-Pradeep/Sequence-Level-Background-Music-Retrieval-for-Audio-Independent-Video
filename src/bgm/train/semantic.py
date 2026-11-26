"""`bgm train semantic` (SPEC M6): semantic heads with symmetric InfoNCE on segment (or track) positives.

* Inputs are span means of frozen 1 Hz embeddings, with the span rule on both sides (SPEC §3.5).
  Video: `train.semantic.video_feature`; music: `sem_clap_1hz` | `sem_mert_1hz` | `both` (concatenated unit vectors).
* A batch holds at most one segment per video (one random segment per training video per epoch); pairs sharing a
  track_id are masked out of the negatives (CLAUDE.md rule 4).
* Augmentation: drop 25% of the video's 1 Hz frames before pooling; jitter the music span by +/- 1 s.
* Early stopping on val segment-level R@10; the best-on-val checkpoint is kept. Test metrics use the M5 protocol.
* D134 options (defaults = the baseline): head_type mlp | linear, tau_learnable, video_pool mean | attn, and
  log_train_recall (per-epoch track R@1/10 on a size-matched train subset and on the test fold; logged only,
  never used for early stopping or the checkpoint).
* D135 regularisers (default off): input_dropout (dropout on the frozen input features, both sides), input_noise (D138) and ema_decay
  (validation, early stopping and the checkpoint use an exponential moving average of the trained weights).
Run outputs: runs/semantic/<variant>/fold<k>/seed<s>/ (config.yaml, metrics.json, ckpt.pt, train_log.csv, ...).
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

from bgm.data import manifest
from bgm.eval.metrics import recall_summary
from bgm.eval.protocol import segment_retrieval, track_retrieval
from bgm.eval.store import FeatureStore, apply_span_rule, fold_ids, l2
from bgm.models.heads import AttnPool, SemanticHead, pad_tokens
from bgm.runs import run_dir, write_run
from bgm.train.losses import dcl, distill_kl, hn_nce, info_nce, info_nce_multi, info_nce_weighted, sigmoid_loss, vse_max

log = logging.getLogger(__name__)


class SemanticInputs:
    """Pooled raw inputs (unit-normalized span means) for either side. With `mert_layer_mix` (1a, D53) the music
    vector is [l2 CLAP span mean | 25 per-layer MERT span means, flattened]; the model mixes the layers."""

    def __init__(self, store: FeatureStore, video_feature: str, music_feature: str, mert_layer_mix: bool = False,
                 video_tokens: bool = False):
        self.store, self.vf, self.mf = store, video_feature, music_feature
        self.mix, self.tokens = mert_layer_mix, video_tokens
        if mert_layer_mix:
            assert music_feature == "both", "mert_layer_mix needs music_feature=both"
        self.m_keys = ["sem_clap_1hz", "sem_mert_1hz"] if music_feature == "both" else [music_feature]

    def _layers_span(self, m: str, a: float, b: float) -> np.ndarray:
        """[25, 1024] span means of the stored per-second, per-layer unit vectors (Series1Hz index rule)."""
        L = self.store.music(m)["sem_mert_layers_1hz"]
        T = L.shape[1]
        lo = int(np.clip(np.ceil(a - 0.5 - 1e-9), 0, T))
        hi = int(np.clip(np.ceil(b - 0.5 - 1e-9), 0, T))
        if hi <= lo:
            k = int(np.clip(np.floor((a + b) / 2), 0, T - 1))
            return L[:, k].astype(np.float32)
        return L[:, lo:hi].astype(np.float32).mean(1)

    @property
    def dims(self) -> tuple[int, int]:
        p = self.store.pairs.pair_id.iloc[0]
        dv = self.store.video(p)[self.vf].shape[1]
        md = self.store.music(self.store.music_of[p])
        if self.mix:
            lay = md["sem_mert_layers_1hz"]
            return dv, md["sem_clap_1hz"].shape[1] + lay.shape[0] * lay.shape[2]
        return dv, sum(md[k].shape[1] for k in self.m_keys)

    def video(self, p: str, a: float, b: float, drop: float = 0.0, rng=None) -> np.ndarray:
        a, b = self.store.span("video", p, a, b)
        if self.tokens:
            return self._video_tokens(p, a, b, drop, rng)
        if drop and rng is not None:
            x = self.store.series("video", p, self.vf).x
            k = np.arange(len(x))
            idx = k[(k + 0.5 >= a) & (k + 0.5 < b)]
            if len(idx) > 1:
                keep = idx[rng.random(len(idx)) >= drop]
                idx = keep if len(keep) else idx[[rng.integers(len(idx))]]
            if len(idx):
                return l2(x[idx].mean(0)).astype(np.float32)
        return l2(self.store.series("video", p, self.vf).span_mean(a, b)).astype(np.float32)

    def _video_tokens(self, p: str, a: float, b: float, drop: float, rng) -> np.ndarray:
        """video_pool=attn: [n, dv] 1 Hz tokens of the (span-ruled) span, with the same frame drop and rng draws as
        video(); their mean, re-normalized, is the baseline's pooled input."""
        s = self.store.series("video", p, self.vf)
        k = np.arange(len(s.x))
        idx = k[(k + 0.5 >= a) & (k + 0.5 < b)]
        if drop and rng is not None and len(idx) > 1:
            keep = idx[rng.random(len(idx)) >= drop]
            idx = keep if len(keep) else idx[[rng.integers(len(idx))]]
        return s.x[idx] if len(idx) else s.span_mean(a, b)[None].astype(np.float32)

    def music(self, m: str, a: float, b: float, jitter: float = 0.0, rng=None) -> np.ndarray:
        dur = self.store.duration("music", m)
        a, b = apply_span_rule(a, b, dur, self.store.min_span)
        if jitter and rng is not None:
            j = float(np.clip(rng.uniform(-jitter, jitter), -a, dur - b))
            a, b = a + j, b + j
        if self.mix:
            clap = l2(self.store.series("music", m, "sem_clap_1hz").span_mean(a, b))
            return np.concatenate([clap, self._layers_span(m, a, b).ravel()]).astype(np.float32)
        return np.concatenate([l2(self.store.series("music", m, k).span_mean(a, b)) for k in self.m_keys]).astype(np.float32)

    def music_window(self, m: str, s: float, d: float) -> np.ndarray:
        return self.music(m, s, s + d)


class MertLayerMix(torch.nn.Module):
    """1a (D53): softmax weights over the 25 MERT layers, uniform at init. Input [B, d_clap + 25*1024] from
    SemanticInputs with mert_layer_mix; output [B, d_clap + 1024] with the mixed MERT vector re-normalized."""

    def __init__(self, d_clap: int, n_layers: int = 25, d: int = 1024):
        super().__init__()
        self.d_clap, self.n, self.d = d_clap, n_layers, d
        self.logits = torch.nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        clap, lay = x[:, : self.d_clap], x[:, self.d_clap:].reshape(len(x), self.n, self.d)
        z = torch.einsum("l,bld->bd", torch.softmax(self.logits, 0), lay)
        return torch.cat([clap, torch.nn.functional.normalize(z, dim=-1)], -1)


class SemanticModel:
    def __init__(self, dv: int, dm: int, cfg: DictConfig, device: str):
        c = cfg.train.semantic
        self.mix = None
        if c.get("mert_layer_mix", False):
            self.mix = MertLayerMix(dm - 25 * 1024).to(device)
            dm = self.mix.d_clap + 1024
        head_type = c.get("head_type", "mlp")                   # D134: mlp (SPEC head) | linear
        assert head_type in ("mlp", "linear"), head_type
        nh = int(c.get("hidden_layers", 1)) if head_type == "mlp" else 0
        p_in = float(c.get("input_dropout", 0.0))            # D135
        noise = float(c.get("input_noise", 0.0))             # D138
        self.hv = SemanticHead(dv, c.hidden, c.out_dim, c.dropout, nh, p_in, noise).to(device)
        self.hm = SemanticHead(dm, c.hidden, c.out_dim, c.dropout, nh, p_in, noise).to(device)
        self.tau_learn = bool(c.get("tau_learnable", True))   # D134: false = tau fixed at tau_init
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(c.tau_init), device=device), requires_grad=self.tau_learn)
        pool = c.get("video_pool", "mean")                    # D134: mean | attn (learned pooling over 1 Hz tokens)
        assert pool in ("mean", "attn"), pool
        self.pool = AttnPool(dv).to(device) if pool == "attn" else None
        self.device = device

    def music_in(self, x):
        return self.mix(x) if self.mix is not None else x

    def params(self):
        extra = list(self.mix.parameters()) if self.mix is not None else []
        extra += list(self.pool.parameters()) if self.pool is not None else []
        return list(self.hv.parameters()) + list(self.hm.parameters()) + extra + ([self.log_tau] if self.tau_learn else [])

    def eval(self):
        self.hv.eval(), self.hm.eval()
        for m in (self.mix, self.pool):
            if m is not None:
                m.eval()

    def train(self):
        self.hv.train(), self.hm.train()
        for m in (self.mix, self.pool):
            if m is not None:
                m.train()

    def video_forward(self, V) -> torch.Tensor:
        """Training-time video embeddings [B, out_dim]: V is a list of [dv] pooled inputs, or of [n_i, dv] tokens (attn)."""
        if self.pool is None:
            return self.hv(torch.from_numpy(np.stack(V)).to(self.device))
        return self.hv(self.pool(*pad_tokens(V, self.device)))

    @torch.no_grad()
    def embed(self, side: str, x: np.ndarray) -> np.ndarray:
        """x: [N, d] pooled inputs; with video_pool=attn the video side takes [N, n, dv] tokens of equal length."""
        t = torch.from_numpy(np.asarray(x, np.float32)).to(self.device)
        if side == "video":
            if self.pool is not None:
                t = self.pool(t, torch.ones(t.shape[:2], dtype=torch.bool, device=self.device))
            return self.hv(t).cpu().numpy()
        return self.hm(self.music_in(t)).cpu().numpy()

    def state(self) -> dict:
        st = {"hv": self.hv.state_dict(), "hm": self.hm.state_dict(), "log_tau": self.log_tau.detach().cpu()}
        if self.mix is not None:
            st["mix"] = self.mix.state_dict()
        if self.pool is not None:
            st["pool"] = self.pool.state_dict()
        return st

    def load(self, st: dict) -> None:
        self.hv.load_state_dict(st["hv"]), self.hm.load_state_dict(st["hm"])
        if self.mix is not None:
            self.mix.load_state_dict(st["mix"])
        if self.pool is not None:
            self.pool.load_state_dict(st["pool"])
        self.log_tau.data.copy_(st["log_tau"].to(self.device))


class Scorer:
    """Track / window score functions for the M5 protocols, with embedding caches."""

    def __init__(self, model: SemanticModel, inputs: SemanticInputs):
        self.model, self.inp, self._cache = model, inputs, {}

    def _emb(self, key, side, fn):
        if key not in self._cache:
            self._cache[key] = self.model.embed(side, fn()[None])[0]
        return self._cache[key]

    def track(self, q, gallery):
        st = self.inp.store
        v = self._emb(("tv", q), "video", lambda: self.inp.video(q, 0.0, st.duration("video", q)))
        M = np.stack([self._emb(("tm", m), "music", lambda m=m: self.inp.music(m, 0.0, st.duration("music", m))) for m in gallery])
        return M @ v

    def window(self, q, a, b, windows):
        v = self._emb(("wv", q, round(a, 3), round(b, 3)), "video", lambda: self.inp.video(q, a, b))
        X = np.stack([self.inp.music_window(m, s, d) for m, s, d in windows])
        return self.model.embed("music", X) @ v


def epoch_batches(ids: list[str], order: np.ndarray, batch: int) -> list[list[str]]:
    """One segment per video per epoch: each video appears in exactly one batch, once (CLAUDE.md rule 4)."""
    return [[ids[i] for i in order[b:b + batch]] for b in range(0, len(order), batch)]


def cap_per_film(film: list[str], cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices (sorted) of a random subset keeping at most `cap` items per film; redraw every epoch (PROGRESS.md D36)."""
    film = np.asarray(film)
    keep = [ix if len(ix) <= cap else rng.choice(ix, cap, replace=False) for ix in (np.flatnonzero(film == f) for f in np.unique(film))]
    return np.sort(np.concatenate(keep))


def film_stratified_order(idx: np.ndarray, film: list[str], k: int, rng: np.random.Generator) -> np.ndarray:
    """1c (D57): reorder the epoch's selected indices so same-film clips sit in shuffled chunks of up to `k`;
    consecutive batches then hold ~batch/k films x k clips (same-film hard negatives). Each index appears exactly
    once, so a batch still holds at most one segment per video (rule 4), and same-track pairs inside a batch stay
    masked out of the negatives by info_nce."""
    film = np.asarray(film)
    chunks = []
    for f in np.unique(film[idx]):
        ix = idx[film[idx] == f]
        ix = ix[rng.permutation(len(ix))]
        chunks += [ix[i:i + k] for i in range(0, len(ix), k)]
    return np.concatenate([chunks[j] for j in rng.permutation(len(chunks))])


def film_subset(train_ids: list[str], film_group: dict[str, str], fraction: float, keep_groups: set[str], seed) -> list[str]:
    """Learning curve (step 3): the training clips of the first ceil(fraction x n) film groups in a fixed random order
    of the training film groups (so subsets are nested across fractions), plus every clip of `keep_groups`."""
    films = sorted({film_group[p] for p in train_ids} - keep_groups)
    order = np.random.default_rng(seed).permutation(len(films))
    chosen = {films[i] for i in order[:math.ceil(fraction * len(films))]} | keep_groups
    return [p for p in train_ids if film_group[p] in chosen]


def clone_state(st: dict) -> dict:
    """Detached copy of SemanticModel.state() (nested state dicts and the log_tau tensor)."""
    return {k: {n: t.detach().clone() for n, t in v.items()} if isinstance(v, dict) else v.detach().clone() for k, v in st.items()}


def ema_update(ema: dict, st: dict, decay: float) -> None:
    """In place: ema <- decay * ema + (1 - decay) * st, over every tensor of a SemanticModel.state() (D135)."""
    for k, v in st.items():
        for n, t in (v.items() if isinstance(v, dict) else [(None, v)]):
            e = ema[k][n] if n is not None else ema[k]
            if e.is_floating_point():
                e.mul_(decay).add_(t.detach().to(e.device), alpha=1 - decay)
            else:
                e.copy_(t)


def cosine_warmup(step: int, total: int, warmup_frac: float) -> float:
    w = max(1, int(total * warmup_frac))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def train_one(cfg: DictConfig, store: FeatureStore, fold: int, seed: int, variant: str, device: str, init: dict | None = None) -> dict:
    """init: a SemanticModel state from pretraining (M8); then the learning rate is lr_finetune instead of lr_scratch."""
    c = cfg.train.semantic
    f = fold_ids(cfg, fold)
    train_ids, val_ids, test_ids = f["train"], f["val"], f["test"]
    cap, frac = c.get("max_clips_per_film_per_epoch"), c.get("train_film_fraction", 1.0)
    strat = bool(c.get("film_stratified_batches", False))
    film = manifest.read_film_groups(cfg) if (cap or frac < 1.0 or strat) else None
    assert film is not None or not (cap or frac < 1.0 or strat), \
        "film cap / film subsets / stratified batches need the film split (splits.folds_file=folds_film.json)"
    if frac < 1.0:
        n_all = len(train_ids)
        train_ids = film_subset(train_ids, film, frac, set(manifest.read_folds(cfg).get("train_only", [])), [c.get("subset_seed", 0), fold])
        log.info("fold %d: %.0f%% of the training films -> %d of %d training clips", fold, 100 * frac, len(train_ids), n_all)
    attn = c.get("video_pool", "mean") == "attn"
    inp = SemanticInputs(store, c.video_feature, c.music_feature, bool(c.get("mert_layer_mix", False)), attn)
    dv, dm = inp.dims
    torch.manual_seed(seed)
    model = SemanticModel(dv, dm, cfg, device)
    if init is not None:
        model.load(init)
    lr = c.lr_finetune if init is not None else c.lr_scratch
    lam = float(c.get("attr_distill_lambda", 0.0))
    loss_name = c.get("loss", "info_nce")                    # D124: info_nce | sigmoid | hn_nce | dcl | vse_max
    assert loss_name in ("info_nce", "sigmoid", "hn_nce", "dcl", "vse_max"), loss_name
    sig_bias = None
    if loss_name == "sigmoid":                               # training-only; a constant shift does not change any ranking
        sig_bias = torch.nn.Parameter(torch.tensor(float(c.get("sigmoid_bias_init", -10.0)), device=device))
    aux, targets = None, None
    if lam > 0:                                              # 3.1 (D62): the video head also predicts the music's profile
        from bgm.train.attr_distill import ProfileTargets, distill_loss
        targets = ProfileTargets(store, train_ids)
        aux = torch.nn.Linear(c.out_dim, targets.dim).to(device)
    opt = torch.optim.AdamW(model.params() + (list(aux.parameters()) if aux is not None else []) + ([sig_bias] if sig_bias is not None else []),
                            lr=lr, weight_decay=c.weight_decay)
    teachers = []                                            # D76: distil the ensemble of these seeds' heads (same fold) into one model
    if c.get("distill_from_seeds", None):
        assert not attn, "distillation needs pooled video inputs"
        tdir = Path(cfg.paths.runs) / "semantic" / c.get("distill_from_variant", "siglip-both") / f"fold{fold}"
        for ts in c.distill_from_seeds:
            ck = torch.load(tdir / f"seed{ts}" / "ckpt.pt", map_location=device)
            tm = SemanticModel(ck["dv"], ck["dm"], cfg, device)
            tm.load(ck["state"])
            tm.eval()
            teachers.append(tm)
    train_film = [film[p] for p in train_ids] if cap else None
    n_epoch = len(cap_per_film(train_film, cap, np.random.default_rng(0))) if cap else len(train_ids)
    steps_per_epoch = math.ceil(n_epoch / c.batch)
    total = c.max_epochs * steps_per_epoch
    track_index = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    theme_mode = c.get("theme_targets", None)                # D66: null | mask | multi (one-to-many targets over theme groups)
    if theme_mode:
        from bgm.data.themes import read_themes
        assert theme_mode in ("mask", "multi", "weighted"), theme_mode
        th = read_themes(cfg)
        tix = {t: i for i, t in enumerate(sorted(set(th.values())))}
        group_of = {p: tix[th[store.music_of[p]]] for p in train_ids}
    out = run_dir(cfg, f"semantic/{variant}", fold, seed)
    logf = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(logf)
    diag = bool(c.get("log_train_recall", False))
    if diag:                                                 # D134 diagnostics; the same train subset in every seed and config
        diag_train = sorted(np.random.default_rng([fold, 134]).choice(train_ids, min(len(test_ids), len(train_ids)), replace=False))
        gal_tr = list(dict.fromkeys(store.music_of[p] for p in diag_train))
        gal_te = list(dict.fromkeys(store.music_of[p] for p in test_ids))
    wr.writerow(["epoch", "loss", "tau", "val_seg_R@10", "seconds"]
                + (["train_R@1", "train_R@10", "train_MRR", "test_R@1", "test_R@10", "test_MRR"] if diag else []))
    best, best_state, best_epoch, bad, step = -1.0, None, -1, 0, 0
    ema_decay = float(c.get("ema_decay", 0.0))              # D135: evaluate and keep the EMA weights
    ema = clone_state(model.state()) if ema_decay > 0 else None
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
            V, M, T, A = [], [], [], []
            for j, p in enumerate(ids):
                irng = np.random.default_rng([seed, epoch, int(order[bi * c.batch + j])])
                if c.positive == "track":
                    a, b = 0.0, store.duration("video", p)
                else:
                    a, b = store.segments[p][irng.integers(len(store.segments[p]))]
                V.append(inp.video(p, a, b, c.frame_drop, irng))
                M.append(inp.music(store.music_of[p], a, b if c.positive != "track" else store.duration("music", store.music_of[p]),
                                   c.music_jitter_s, irng))
                T.append(group_of[p] if theme_mode else track_index[store.track[p]])
                if targets is not None:
                    A.append(targets(p, a, b if c.positive != "track" else store.duration("music", store.music_of[p])))
            for g in opt.param_groups:
                g["lr"] = lr * cosine_warmup(step, total, cfg.train.warmup_frac)
            zv = model.video_forward(V)
            zm = model.hm(model.music_in(torch.from_numpy(np.stack(M)).to(device)))
            tt = torch.tensor(T, device=device)
            if loss_name != "info_nce":
                assert not theme_mode, "D124 losses use track masking only"
                if loss_name == "sigmoid":
                    loss = sigmoid_loss(zv, zm, tt, model.log_tau, sig_bias)
                elif loss_name == "hn_nce":
                    loss = hn_nce(zv, zm, tt, model.log_tau, float(c.get("hn_beta", 0.5)))
                elif loss_name == "dcl":
                    loss = dcl(zv, zm, tt, model.log_tau)
                else:
                    loss = vse_max(zv, zm, tt, float(c.get("vse_margin", 0.2)))
            elif theme_mode == "weighted":
                loss = info_nce_weighted(zv, zm, tt, model.log_tau, float(c.get("theme_soft_weight", 0.5)))
            else:
                loss = (info_nce_multi if theme_mode == "multi" else info_nce)(zv, zm, tt, model.log_tau, float(c.get("tau_min", 0.01)))
            if aux is not None:
                loss = loss + lam * distill_loss(aux, zv, torch.from_numpy(np.stack(A)).to(device))
            if teachers:
                xv, xm = np.stack(V), np.stack(M)
                tsim = torch.from_numpy(np.mean([t.embed("video", xv) @ t.embed("music", xm).T for t in teachers], 0)).to(device)
                loss = loss + float(c.get("distill_weight", 1.0)) * distill_kl(zv, zm, tsim, tt if theme_mode else torch.tensor(T, device=device),
                                                                             model.log_tau, float(c.get("distill_tau", 0.07)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:
                ema_update(ema, model.state(), ema_decay)
            step += 1
            losses.append(loss.item())
        model.eval()
        if ema is not None:
            live = clone_state(model.state())
            model.load(ema)
        val = recall_summary(segment_retrieval(store, val_ids, Scorer(model, inp).window, "all").ranks)["R@10"]
        row = [epoch, np.mean(losses), model.log_tau.exp().item(), val, round(time.time() - t0, 2)]
        if diag:
            sc = Scorer(model, inp)
            for q, g in ((diag_train, gal_tr), (test_ids, gal_te)):
                r = recall_summary(track_retrieval(store, q, g, sc.track).ranks)
                row += [r["R@1"], r["R@10"], r["MRR"]]
        wr.writerow(row)
        logf.flush()
        if val > best:
            best, best_state, best_epoch, bad = val, {k: {n: t.detach().clone() for n, t in v.items()} if isinstance(v, dict) else v.clone()
                                                      for k, v in model.state().items()}, epoch, 0
        else:
            bad += 1
        if ema is not None:
            model.load(live)
        if bad >= c.patience:
            break
    logf.close()
    model.load(best_state)
    model.eval()
    torch.save({"state": best_state, "dv": dv, "dm": dm, "variant": variant}, out / "ckpt.pt")
    metrics = {"val": {"segment_R@10": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}}
    if model.mix is not None:
        metrics["mert_layer_weights"] = [round(float(w), 5) for w in torch.softmax(model.mix.logits.detach().cpu(), 0)]
    metrics["test"], per = evaluate(cfg, store, Scorer(model, inp), test_ids)
    if diag:
        metrics["train_track_size_matched"] = recall_summary(track_retrieval(store, diag_train, gal_tr, Scorer(model, inp).track).ranks,
                                                             cfg.eval.ks) | {"gallery": len(gal_tr), "test_gallery": len(gal_te)}
    np.savez(out / "perquery.npz", **per)
    write_run(out, cfg, metrics, seed)
    log.info("semantic/%s fold %d seed %d: val seg R@10 %.3f (epoch %d) | test track R@10 %.3f, seg R@10 %.3f", variant, fold, seed,
             best, best_epoch, metrics["test"]["track_test_fold"]["R@10"], metrics["test"]["segment_all"]["R@10"])
    return metrics


def evaluate(cfg: DictConfig, store: FeatureStore, sc: Scorer, test_ids: list[str]) -> tuple[dict, dict]:
    gal_test = list(dict.fromkeys(store.music_of[p] for p in test_ids))
    gal_all = list(dict.fromkeys(store.pairs.music_id))
    res, per = {}, {}
    for key, gal in (("track_test_fold", gal_test), ("track_all", gal_all)):
        r = track_retrieval(store, test_ids, gal, sc.track)
        res[key], per[key], per[f"{key}/clusters"] = recall_summary(r.ranks, cfg.eval.ks), r.ranks, r.clusters
    for key, variant in (("segment_all", "all"), ("segment_other", "other")):
        r = segment_retrieval(store, test_ids, sc.window, variant)
        res[key], per[key], per[f"{key}/clusters"] = recall_summary(r.ranks, cfg.eval.ks), r.ranks, r.clusters
    return res, per


def run_train_semantic(cfg: DictConfig, folds=None, seeds=None, variant: str | None = None, init_dir: str | None = None) -> list[dict]:
    """init_dir: runs/pretrain/semantic -> each seed fine-tunes from seed<s>/ckpt.pt."""
    c = cfg.train.semantic
    variant = variant or f"{c.video_feature.replace('sem_', '').replace('_1hz', '')}-{c.music_feature.replace('sem_', '').replace('_1hz', '')}"
    device = cfg.features.device if torch.cuda.is_available() else "cpu"
    store = FeatureStore(cfg)
    folds = range(cfg.splits.n_folds) if folds is None else folds
    seeds = list(cfg.train.seeds) if seeds is None else seeds
    from pathlib import Path
    init = (lambda s: torch.load(Path(init_dir) / f"seed{s}" / "ckpt.pt", map_location=device)["state"]) if init_dir else (lambda s: None)
    if c.get("temporal_tokens", False):
        from bgm.train.semantic_tokens import train_one_tokens as fn
    else:
        fn = train_one
    return [fn(cfg, store, k, s, variant, device, init(s)) for k in folds for s in seeds]
