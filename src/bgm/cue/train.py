"""`bgm cue train` (Step 2): the cue-switch predictor per fold x seed x variant.

Variants (runs/<paths.runs>/<cue.exp>/<variant>/fold<k>/seed<s>/):
  pair       R3: pair-only head on the MLP tokens, blocked training sequences
  ctx        R4: [CTX] + Transformer, blocked training sequences
  ctx_cross  R5: as ctx, trained on cross_film sequences (shortcut check)
  ctx_nodur  diagnostic: as ctx without the log-duration feature (segment lengths mark clip joins, D79)
  pair_splice / ctx_splice   R3s / R4s (D94): as pair / ctx, plus same-music splices in training (see `splice_clip`)
  pair_audio / pair_splice_audio   R3a / R3sa (D98): as pair / pair_splice, but every training boundary's target is the
             audio teacher's p (audio_teacher.py: the music's own change at that boundary) instead of the join label

Training sequences come from the fold's TRAIN clips only and are redrawn every epoch (rng [seed, fold, epoch, item]);
Konjum (train-only) at most once per sequence. Loss: BCE with pos_weight = neg / pos counted on `pos_weight_seqs`
sequences drawn once. Early stopping on the average precision of the boundary logits over the frozen val (blocked)
sequences, which come from the fold's val clips (film-disjoint from train and test, D54).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from bgm.cue.model import CueModel, collate, predict
from bgm.cue.sequences import boundary_labels, by_film, clip_table, read_sets, sample_sequence, with_timeline
from bgm.cue.tokens import Scaler, build_tokens, sequence_tokens
from bgm.data import manifest
from bgm.eval.store import FeatureStore
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)
VARIANTS = {"pair": (False, "blocked", ()), "ctx": (True, "blocked", ()), "ctx_cross": (True, "cross_film", ()),
            "ctx_nodur": (True, "blocked", ("log_dur",)), "pair_splice": (False, "blocked", ()), "ctx_splice": (True, "blocked", ()),
            "pair_audio": (False, "blocked", ()), "pair_splice_audio": (False, "blocked", ())}
SPLICE = frozenset({"pair_splice", "ctx_splice", "pair_splice_audio"})
AUDIO = frozenset({"pair_audio", "pair_splice_audio"})          # D98: audio-teacher targets
SEP = "\u00a7"                      # synthetic part ids: <pair_id>§<t1>-<t2>A / B


def splice_clip(sem_1hz: np.ndarray, dyn_25hz: np.ndarray, cuts_s, rng: np.random.Generator, seg_cfg, min_part_s: float,
                gap_s: tuple[float, float]) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    """Same-music splice (D94): remove [t1, t2) from one clip and keep the two ends as two parts joined in a sequence.
    The join looks like a clip join (a visual jump, segments cut by the clip ends) but the music is one cue, so its
    label is 0. t1, t2 are whole seconds so the parts keep the 1 Hz SigLIP grid; each part is re-segmented with the
    dataset's own `make_segments` and tokenised like a real clip (frame 0 of each part left out of the motion stats).
    -> (t1, t2, raw tokens of part A, raw tokens of part B), or None if the clip is too short."""
    from bgm.cue.tokens import RATE, clip_segments, segment_features
    from bgm.data.segments import make_segments
    n25 = len(dyn_25hz)
    L = int(np.floor(n25 / RATE))
    g_hi = int(min(gap_s[1], L - 2 * min_part_s))
    if g_hi < gap_s[0]:
        return None
    g = int(rng.integers(int(gap_s[0]), g_hi + 1))
    t1 = int(rng.integers(int(min_part_s), int(L - min_part_s - g) + 1))
    t2 = t1 + g
    lm = np.asarray(dyn_25hz)[:, 0]
    cuts = np.asarray(cuts_s, float)
    out = []
    for lo, hi in ((0, t1), (t2, None)):
        sem = sem_1hz[lo:hi]
        m = lm[lo * RATE:None if hi is None else hi * RATE]
        dur = len(m) / RATE
        c = cuts[(cuts > lo) & (cuts < lo + dur)] - lo
        segs = make_segments(c, dur, sem, seg_cfg.merge_cosine, seg_cfg.max_len_s, seg_cfg.min_shot_s)
        out.append(segment_features(sem, m, clip_segments([(a, b) for a, b, _ in segs], len(m))))
    return t1, t2, out[0], out[1]


class FoldData:
    """Everything a fold's cue model sees: train clips by film, the train-fold scaler, tokens, labels."""

    def __init__(self, cfg: DictConfig, fold: int, store: FeatureStore | None = None, tokens: dict | None = None,
                 drop: tuple[str, ...] = ()):
        self.cfg, self.fold = cfg, fold
        self.store = store or FeatureStore(cfg)
        self.folds = manifest.read_folds(cfg)
        assert self.folds.get("split") == "film", "the cue predictor uses the film split (folds_film.json)"
        f = self.folds["folds"][str(fold)]
        self.ids = f
        self.fg = self.folds["film_group"]
        self.tokens = tokens if tokens is not None else build_tokens(self.store)
        self.train = by_film(clip_table(with_timeline(self.store.pairs, cfg), self.fg, f["train"]))
        self.limited = frozenset(self.folds.get("train_only", []))
        self.scaler = Scaler.fit([self.tokens[p] for p in f["train"]], drop)
        self.splice_p = 0.0                         # set by train_cue for the *_splice variants (D94)
        self.syn: dict[str, np.ndarray] = {}
        self.teacher = None                         # set by train_cue for the *_audio variants (D98); training draws only
        self.cache: dict = {}                       # fold-level objects shared by the variants (the D98 teacher)

    def xy(self, parts: list[str], teacher: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Scaled tokens, labels (the join labels; the audio teacher's p when `teacher` and one is set) and types."""
        base = {p: p.split(SEP)[0] for p in parts}
        raw, seg_part = sequence_tokens({p: self.syn[p] if SEP in p else self.tokens[p] for p in parts}, parts)
        y, typ = boundary_labels(seg_part, parts, {p: self.store.track[b] for p, b in base.items()}, {p: self.fg[b] for p, b in base.items()})
        if teacher and self.teacher is not None:
            y = self.teacher.targets(parts, raw, seg_part)
        return self.scaler(raw), y, typ

    def draw(self, rng: np.random.Generator, mode: str) -> list[str]:
        c = self.cfg.cue
        parts = sample_sequence(self.train, rng, c.seq_min_s, c.seq_max_s, mode, c.block_max_clips, self.limited)
        if self.splice_p <= 0:
            return parts
        self.syn.clear()                            # every caller converts the drawn sequence (xy) before the next draw
        out = []
        for p in parts:
            sp = None
            if rng.random() < self.splice_p:
                v = self.store.video(p)
                sp = splice_clip(v["sem_siglip_1hz"].astype(np.float32), np.asarray(v["dyn_25hz"]), v["cut_times_s"], rng,
                                 self.cfg.segments, float(c.splice.min_part_s), tuple(c.splice.gap_s))
            if sp is None:
                out.append(p)
                continue
            t1, t2, a, b = sp
            ka, kb = f"{p}{SEP}{t1}-{t2}A", f"{p}{SEP}{t1}-{t2}B"
            self.syn[ka], self.syn[kb] = a, b
            out += [ka, kb]
        return out


def fit_teacher(data: FoldData, n_seqs: int, half_s: int):
    """D98: the audio teacher of a fold, its calibration fit on `n_seqs` training sequences drawn without splices (rng
    [fold, 4242, i], the same for every seed and variant): change joins = 1, within-clip boundaries = 0."""
    from bgm.cue.audio_teacher import AudioTeacher, Calibration
    t = AudioTeacher(data.store, half_s)
    sp, data.splice_p = data.splice_p, 0.0
    ds, ys = [], []
    for i in range(n_seqs):
        parts = data.draw(np.random.default_rng([data.fold, 4242, i]), "blocked")
        raw, seg_part = sequence_tokens(data.tokens, parts)
        y, typ = boundary_labels(seg_part, parts, data.store.track, data.fg)
        d = t.distance(parts, raw, seg_part)
        keep = np.array([k == "clip" or (k == "film" and v == 1) for k, v in zip(typ, y)], bool)
        ds.append(d[keep])
        ys.append(y[keep])
    data.splice_p = sp
    t.calib = Calibration.fit(np.concatenate(ds), np.concatenate(ys))
    return t


def boundary_scores(y: np.ndarray, z: np.ndarray) -> dict:
    ok = len(np.unique(y)) == 2
    return {"ap": float(average_precision_score(y, z)) if ok else float("nan"),
            "auroc": float(roc_auc_score(y, z)) if ok else float("nan"), "n": int(len(y)), "pos": int(y.sum())}


def train_cue(cfg: DictConfig, fold: int, seed: int, variant: str, data: FoldData | None = None) -> dict:
    t0 = time.time()
    context, mode, drop = VARIANTS[variant]
    c, tc = cfg.cue, cfg.cue.train
    data = data or FoldData(cfg, fold)
    if data.scaler.drop != drop:
        data = FoldData(cfg, fold, data.store, data.tokens, drop)
    data.splice_p = float(c.splice.p) if variant in SPLICE else 0.0      # D94; the scaler stays fit on real train clips
    data.syn.clear()
    data.teacher = None
    if variant in AUDIO:                                                 # D98
        key = ("teacher", int(c.audio.half_s), int(c.audio.calib_seqs))
        if key not in data.cache:
            data.cache[key] = fit_teacher(data, int(c.audio.calib_seqs), int(c.audio.half_s))
        data.teacher = data.cache[key]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(np.random.default_rng([seed, fold, 7]).integers(2**31)))
    # neg / pos on sequences drawn once
    ys = [data.xy(data.draw(np.random.default_rng([seed, fold, 10**6, i]), mode), True)[1] for i in range(tc.pos_weight_seqs)]
    pos = float(sum(y.sum() for y in ys))
    neg = float(sum(len(y) for y in ys)) - pos
    pos_weight = neg / pos
    val = [data.xy(q) for q in read_sets(cfg)["sets"][str(fold)]["val"]["blocked"]]
    vx, vy = [x for x, _, _ in val], np.concatenate([y for _, y, _ in val])
    d_in = vx[0].shape[1]
    m = c.model
    model = CueModel(d_in, m.d_model, m.heads, m.layers, m.ff, m.dropout, context, m.get("pos", "none")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev), reduction="none")
    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    for ep in range(tc.max_epochs):
        model.train()
        seqs = [data.xy(data.draw(np.random.default_rng([seed, fold, ep, i]), mode), True) for i in range(tc.seqs_per_epoch)]
        order = np.random.default_rng([seed, fold, ep, 10**7]).permutation(len(seqs))
        tot, nb = 0.0, 0
        for k in range(0, len(order), tc.batch):
            b = [seqs[i] for i in order[k:k + tc.batch]]
            x, valid, y, bvalid = collate([s[0] for s in b], [s[1] for s in b], dev)
            z = model(x, valid)
            loss = (lossf(z, y) * bvalid).sum() / bvalid.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        vz = np.concatenate(predict(model, vx, dev))
        sc = boundary_scores(vy, vz)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_ap": sc["ap"], "val_auroc": sc["auroc"]})
        if sc["ap"] > best:
            best, best_ep, bad = sc["ap"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.patience:
                break
    model.load_state_dict(best_state)
    vz = np.concatenate(predict(model, vx, dev))
    d = run_dir(cfg, f"{c.exp}/{variant}", fold, seed)
    torch.save({"state": best_state, "d_in": d_in, "context": context, "mode": mode, "scaler": data.scaler.to_dict(),
                "model": dict(m), "variant": variant}, d / "ckpt.pt")
    metrics = {"variant": variant, "mode": mode, "fold": fold, "seed": seed, "pos_weight": pos_weight, "best_epoch": best_ep,
               "splice": OmegaConf.to_container(c.splice) if variant in SPLICE else None,
               "audio_teacher": ({"half_s": data.teacher.half_s, "calib": data.teacher.calib.to_dict()} if data.teacher else None),
               "val": boundary_scores(vy, vz), "epochs_run": len(hist), "history": hist, "seconds": time.time() - t0,
               "n_params": int(sum(p.numel() for p in model.parameters()))}
    write_run(d, cfg, metrics, seed)
    log.info("cue train %s fold %d seed %d: pos_weight %.2f, best epoch %d, val AP %.3f AUROC %.3f [%.0f s]", variant, fold, seed,
             pos_weight, best_ep, metrics["val"]["ap"], metrics["val"]["auroc"], metrics["seconds"])
    return metrics


def load_cue(cfg: DictConfig, variant: str, fold: int, seed: int, device: str = "cpu") -> tuple[CueModel, Scaler]:
    from pathlib import Path
    st = torch.load(Path(cfg.paths.runs) / cfg.cue.exp / variant / f"fold{fold}" / f"seed{seed}" / "ckpt.pt", map_location=device)
    m = st["model"]
    model = CueModel(st["d_in"], m["d_model"], m["heads"], m["layers"], m["ff"], m["dropout"], st["context"], m.get("pos", "none")).to(device)
    model.load_state_dict(st["state"])
    model.eval()
    return model, Scaler.from_dict(st["scaler"])


def run_train(cfg: DictConfig, variants, folds, seeds) -> list[dict]:
    out = []
    store = FeatureStore(cfg)
    tokens = build_tokens(store)
    for k in folds:
        data = FoldData(cfg, k, store, tokens)
        for v in variants:
            for s in seeds:
                out.append(train_cue(cfg, k, s, v, data))
    return out
