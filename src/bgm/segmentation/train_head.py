"""Train the boundary head on frozen BaSSL embeddings to predict music changes (D140).

Per fold k (title split, folds_title.json) and seed s:
  * training sequences: 256 per epoch, drawn from fold k's train clips with per-item seeds [seed, k, epoch, i]
    (bgm.cue.sequences.sample_sequence, blocked mode), so they are redrawn every epoch;
  * BCE with pos_weight = neg / pos counted on 2,048 draws; AdamW lr 1e-3, weight decay 0.3; batches of 32 sequences;
  * early stopping on AP over fold k's frozen val sequences (5 partitions of the val clips), patience 30, max 150.
Variants (SPECS): 'emb_p' = BaSSL embedding + logit of BaSSL's own scene probability (proposed); 'emb_p_dur' adds log
shot duration and mean log motion of the two adjacent shots (ablation; duration can mark clip edits, PROGRESS.md D79).
D142 variants change one thing each against emb_p: the loss (focal: focal BCE, gamma 2; rank: BCE + pairwise logistic
ranking loss over the batch's change / no-change pairs), the inputs (contrast: + left / right cosine contrast, ResNet
shots at 1, 2, 4, 8 shots and SigLIP over 2, 4 s), or the layers (gru: a BiGRU over the sequence's boundaries before the
classifier; no time or duration input, so it sees boundary order only). Test clips are never loaded here.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from bgm.cue.sequences import by_film, clip_table, sample_sequence
from bgm.segmentation.evaluate import average_precision
from bgm.segmentation.head import BoundaryHead, SequenceHead
from bgm.segmentation.sequences import Chain, ClipShots, chain, scalar_features, shot_contrast, time_contrast

log = logging.getLogger(__name__)

VARIANTS = ("emb_p", "emb_p_dur")
SPECS = {"emb_p": {"feats": "emb_p", "loss": "bce", "arch": "mlp"},
         "emb_p_dur": {"feats": "emb_p_dur", "loss": "bce", "arch": "mlp"},
         "focal": {"feats": "emb_p", "loss": "focal", "arch": "mlp"},
         "rank": {"feats": "emb_p", "loss": "bce_rank", "arch": "mlp"},
         "contrast": {"feats": "emb_p_c", "loss": "bce", "arch": "mlp"},
         "gru": {"feats": "emb_p", "loss": "bce", "arch": "gru"},
         "combo": {"feats": "emb_p_c", "loss": "bce", "arch": "gru"}}   # D142 C: the parts that raised val AP


@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 0.3
    seqs_per_epoch: int = 256
    batch_seqs: int = 32
    max_epochs: int = 150
    patience: int = 30
    pos_weight_seqs: int = 2048
    seq_min_s: float = 60.0
    seq_max_s: float = 180.0
    block_max: int = 8


class SegData:
    """Everything the study needs per clip: shots of both detectors, BaSSL shot features, motion, SigLIP 1 Hz."""

    def __init__(self, cfg, need_siglip: bool = False):
        from bgm.data import manifest
        from bgm.features.extract import feature_path, load_features
        from bgm.segmentation.run_all import SEQ_FILE, bassl_cache_path, out_dir
        self.cfg = cfg
        self.pairs = manifest.read_pairs(cfg).set_index("pair_id", drop=False)
        self.folds = manifest.read_folds(cfg)
        assert self.folds.get("split") == "film" and cfg.splits.folds_file == "folds_title.json"
        self.fg = self.folds["film_group"]
        self.track = {p: str(t) for p, t in self.pairs["track_id"].items()}
        self.sets = json.loads((manifest.manifests_dir(cfg) / SEQ_FILE).read_text())
        assert self.sets["folds_file"] == "folds_title.json", self.sets["folds_file"]
        d = out_dir(cfg)
        sh = {k: json.loads((d / f"shots_{k}.json").read_text()) for k in ("transnetv2", "pyscenedetect")}
        feats = np.load(bassl_cache_path(cfg))
        self.clips: dict[str, dict[str, ClipShots]] = {"transnetv2": {}, "pyscenedetect": {}}
        self.siglip: dict[str, np.ndarray] = {}
        for p in self.pairs.index:
            v = load_features(feature_path(cfg, "video", p))
            mot = np.asarray(v["dyn_25hz"], np.float32)[:, 0]
            if need_siglip:
                self.siglip[p] = np.asarray(v["sem_siglip_1hz"], np.float32)
            for det, s in sh.items():
                a = np.array([[x[2], x[3]] for x in s[p]], float)
                flow = np.array([mot[int(round(t0 * 25)):max(int(round(t1 * 25)), int(round(t0 * 25)) + 1)].mean()
                                 for t0, t1 in a], np.float32)
                self.clips[det][p] = ClipShots(p, a, feats[p].astype(np.float32) if det == "transnetv2" else None, flow)

    def chain(self, parts: list[str], det: str = "transnetv2") -> Chain:
        ch = chain(parts, self.clips[det], self.track, self.fg)
        if self.siglip:
            off, e, c = 0.0, [], []
            for p in parts:
                e.append(self.siglip[p])
                c.append(np.arange(len(self.siglip[p])) + 0.5 + off)
                off += float(self.clips[det][p].shots[-1, 1])
            ch.extra["sig"], ch.extra["cent"] = np.concatenate(e), np.concatenate(c)
        return ch

    def split_ids(self, k: int, split: str) -> list[str]:
        return list(self.folds["folds"][str(k)][split])

    def frozen(self, k: int, split: str) -> list[list[str]]:
        return self.sets["sets"][str(k)][split]["blocked"]


def check_no_leak(data: SegData, k: int) -> None:
    f = data.folds["folds"][str(k)]
    tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
    assert not tr & va and not tr & te and not va & te, "clip in two splits"
    films = lambda ids: {data.fg[p] for p in ids}  # noqa: E731
    assert not films(tr) & films(te), "film in train and test"
    assert not films(va) & films(te), "film in val and test"
    assert not films(tr) & films(va), "film in train and val"


def features(ch: Chain, ctx: dict, variant: str, scaler: tuple | None = None) -> np.ndarray:
    feats = SPECS[variant]["feats"]
    p = np.clip(ctx["p"], 1e-4, 1 - 1e-4)
    x = [ctx["emb"], np.log(p / (1 - p))[:, None]]
    if feats == "emb_p_dur":
        s = scalar_features(ch)
        x.append((s - scaler[0]) / scaler[1])
    if feats == "emb_p_c":
        x.append(shot_contrast(ch.feats))
        x.append(time_contrast(ch.extra["sig"], ch.extra["cent"], ch.btimes))
    return np.concatenate(x, 1).astype(np.float32)


def make_head(spec: dict, d_in: int) -> torch.nn.Module:
    return SequenceHead(d_in) if spec["arch"] == "gru" else BoundaryHead(d_in)


def forward(head: torch.nn.Module, xs: list[torch.Tensor]) -> torch.Tensor:
    """Logits of all boundaries of a list of sequences, concatenated in order."""
    return head(xs) if isinstance(head, SequenceHead) else head(torch.cat(xs))


def loss_fn(kind: str, z: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor, gamma: float = 2.0, lam: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight, reduction="none")
    if kind == "focal":
        pt = torch.where(y > 0.5, torch.sigmoid(z), 1 - torch.sigmoid(z))
        return ((1 - pt) ** gamma * bce).mean()
    loss = bce.mean()
    if kind == "bce_rank" and (y > 0.5).any() and (y < 0.5).any():
        loss = loss + lam * F.softplus(z[y < 0.5][None, :] - z[y > 0.5][:, None]).mean()
    return loss


def train_one(data: SegData, bassl, k: int, seed: int, variant: str, tc: TrainCfg, device: str = "cuda") -> dict:
    check_no_leak(data, k)
    spec = SPECS[variant]
    films = by_film(clip_table(data.pairs, data.fg, data.split_ids(k, "train")))
    limited = frozenset(data.folds.get("train_only", []))
    draw = lambda rng: sample_sequence(films, rng, tc.seq_min_s, tc.seq_max_s, "blocked", tc.block_max, limited)  # noqa: E731
    t0 = time.time()
    pw_chains = [data.chain(draw(np.random.default_rng([seed, k, 10**6, i]))) for i in range(tc.pos_weight_seqs)]
    y_all = np.concatenate([c.labels for c in pw_chains])
    pos_weight = float((y_all == 0).sum() / max((y_all == 1).sum(), 1))
    sc = np.concatenate([scalar_features(c) for c in pw_chains])
    scaler = (sc.mean(0), sc.std(0) + 1e-6)
    val_chains = [data.chain(q) for q in data.frozen(k, "val")]
    val_x = [features(c, x, variant, scaler) for c, x in zip(val_chains, bassl.context_many([c.feats for c in val_chains]))]
    val_y = np.concatenate([c.labels for c in val_chains])
    vx = [torch.from_numpy(x).to(device) for x in val_x]

    torch.manual_seed(int(np.random.default_rng([seed, k, 7]).integers(2**31)))
    head = make_head(spec, vx[0].shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    pwt = torch.tensor(pos_weight, device=device)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(tc.max_epochs):
        chains = [data.chain(draw(np.random.default_rng([seed, k, ep, i]))) for i in range(tc.seqs_per_epoch)]
        ctxs = bassl.context_many([c.feats for c in chains])
        xs = [torch.from_numpy(features(c, x, variant, scaler)) for c, x in zip(chains, ctxs)]
        ys = [torch.from_numpy(c.labels.astype(np.float32)) for c in chains]
        order = np.random.default_rng([seed, k, ep, 10**7]).permutation(len(chains))
        head.train()
        losses = []
        for i in range(0, len(order), tc.batch_seqs):
            b = order[i:i + tc.batch_seqs]
            y = torch.cat([ys[j] for j in b]).to(device)
            loss = loss_fn(spec["loss"], forward(head, [xs[j].to(device) for j in b]), y, pwt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            vz = forward(head, vx).float().cpu().numpy()
        ap = average_precision(vz, val_y)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_ap": ap})
        if ap > best:
            best, bad = ap, 0
            best_state = {n: t.detach().cpu().clone() for n, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.patience:
                break
    head.load_state_dict(best_state)
    return {"head": head.eval(), "scaler": scaler, "d_in": int(vx[0].shape[1]), "pos_weight": pos_weight, "val_ap": best,
            "best_epoch": int(np.argmax([h["val_ap"] for h in hist])), "epochs_run": len(hist), "history": hist,
            "seconds": time.time() - t0}


def run_train(cfg, folds=range(5), seeds=(0, 1, 2), variants=VARIANTS) -> None:
    from omegaconf import OmegaConf
    from bgm.runs import run_dir, write_run
    from bgm.segmentation.bassl_features import BaSSL
    data = SegData(cfg, need_siglip=any(SPECS[v]["feats"] == "emb_p_c" for v in variants))
    bassl = BaSSL()
    tc = TrainCfg()
    for v in variants:
        for k in folds:
            for s in seeds:
                d = run_dir(cfg, f"segmentation/{v}", k, s)
                if (d / "head.pt").exists():
                    log.info("skip %s (done)", d)
                    continue
                r = train_one(data, bassl, k, s, v, tc)
                torch.save({"state": r["head"].state_dict(), "scaler": [a.tolist() for a in r["scaler"]], "d_in": r["d_in"],
                            "spec": SPECS[v]}, d / "head.pt")
                rcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                rcfg.segmentation = {"variant": v, "spec": SPECS[v], "fold": k, "seed": s, "train": dict(tc.__dict__)}
                write_run(d, rcfg, {k2: r[k2] for k2 in ("val_ap", "best_epoch", "epochs_run", "pos_weight", "seconds", "history")}, s)
                log.info("%s fold %d seed %d: val AP %.3f (epoch %d / %d, %.0f s)", v, k, s, r["val_ap"], r["best_epoch"], r["epochs_run"], r["seconds"])


def load_head(cfg, variant: str, k: int, s: int, device: str = "cuda"):
    from pathlib import Path
    z = torch.load(Path(cfg.paths.runs) / "segmentation" / variant / f"fold{k}" / f"seed{s}" / "head.pt", map_location="cpu")
    h = make_head(z.get("spec", SPECS[variant]), z["d_in"])
    h.load_state_dict(z["state"])
    return h.eval().to(device), tuple(np.array(a, np.float32) for a in z["scaler"])
