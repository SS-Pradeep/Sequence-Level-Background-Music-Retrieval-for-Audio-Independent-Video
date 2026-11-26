"""Train and evaluate the end-to-end soundtrack model (D141), cleaned title split, per (fold, ensemble e).

Loss per batch of training sequences: CRF negative log-likelihood of the gold track group's paths (offsets latent,
crf_marginal_nll) per segment, plus
LAMBDA_NCE x the symmetric InfoNCE of the heads on one segment per clip of the batch (rule 4; same-track pairs
masked), so per-clip retrieval cannot drift while the sequence loss reshapes it. Early stopping on the val
concatenations' tune-stitch objective (part track accuracy + switch F1, decoded with the shipped beam); epoch 0 is
the shipped decoder. Test once with the best epoch:
  A  the shipped candidate lattices, re-scored and decoded by the model
  B  (head-training variants) lattices rebuilt by the full pipeline with the fine-tuned heads, decoded by the model
Variants:
  fixed        no training: the shipped penalties (reproduction check)
  trans        heads frozen; emission scale + video and music transitions learned
  joint_v      heads + emission scale + video-only transitions
  joint        heads + emission scale + video and music transitions   (main)
  joint_edge   joint + file-edge music features (leak upper bound, not a system)
  joint_nodur  joint without the segment-duration video features (clip-join shortcut check)

Run: python -m bgm.e2e.train [data|train] [variants=...] [folds=...] [ens=...]
  -> runs/cleaned/e2e/data/fold<k>_e<e>.pkl, runs/cleaned/e2e/<variant>/fold<k>/seed<e>/{metrics.json, perseq.npz, ...}
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bgm.e2e.crf import crf_marginal_nll, keytempo, structure, transition, viterbi_beam
from bgm.e2e.data import Seq, build_fold, fold_models
from bgm.e2e.model import E2EModel
from bgm.retrieval.stitch_eval import switch_f1

log = logging.getLogger(__name__)
VARIANTS = {
    "fixed": None,
    "trans": dict(heads=False, music=True, edge=False, dur=True),
    "joint_v": dict(heads=True, music=False, edge=False, dur=True),
    "joint": dict(heads=True, music=True, edge=False, dur=True),
    "joint_edge": dict(heads=True, music=True, edge=True, dur=True),
    "joint_nodur": dict(heads=True, music=True, edge=False, dur=False),
}
EPOCHS, PATIENCE, BATCH = 12, 3, 8
LR_HEADS, LR_REST, WD_HEADS = 3e-5, 1e-2, 0.01
LAMBDA_NCE, TAU = 0.5, 0.05
TOL_S = 0.04
IN_FILM = ("clip", "film")


def _cfg():
    from bgm.analysis.cleaned_compare import CLEAN, make_cfg
    cfg = make_cfg(CLEAN)
    cfg.pipeline.exp = "pipeline_ens10"
    cfg.pipeline.dyn_variant = None
    return cfg


def out_root(cfg) -> Path:
    return Path(cfg.paths.runs) / "e2e"


# ---------------------------------------------------------------- per-sequence tensors
class Prepared:
    """Static arrays of one sequence: pooled music inputs, seam features, lattice structure (computed once)."""

    def __init__(self, s: Seq, music, n_music: int, device: str):
        mid, off, dur = (np.concatenate(x) for x in (s.mid, s.off, s.dur))
        self.sizes = [len(m) for m in s.mid]
        self.bounds = np.r_[0, np.cumsum(self.sizes)]
        self.xm = torch.from_numpy(music.pooled(mid, off, dur)).to(device)
        self.seg_of = torch.from_numpy(np.repeat(np.arange(len(s.mid)), self.sizes)).to(device)
        self.xv = torch.from_numpy(s.xv).to(device)
        self.exit = torch.from_numpy(music.exit_feats(mid, off, dur)[:, :n_music]).to(device)
        self.entry = torch.from_numpy(music.entry_feats(mid, off)[:, :n_music]).to(device)
        self.vfeat = torch.from_numpy(s.vfeat).to(device)
        self.zmu = torch.tensor(s.zmu, dtype=torch.float32, device=device)
        self.zsd = torch.tensor(s.zsd, dtype=torch.float32, device=device)
        self.masks = [torch.from_numpy(np.asarray(m, bool)).to(device) for m in s.gold_mask]
        g = s.gold_win
        self.xg = torch.from_numpy(music.pooled(g[:, 0].astype(int), g[:, 1], g[:, 2])).to(device)
        self.same, self.cont, self.kt = [], [], []
        for i in range(1, len(s.mid)):
            sm, ct = structure(s.mid[i - 1], s.off[i - 1], s.dur[i - 1], s.mid[i], s.off[i], TOL_S)
            k = keytempo(music.key[s.mid[i - 1]], music.tempo[s.mid[i - 1]], music.key[s.mid[i]], music.tempo[s.mid[i]])
            self.same.append(torch.from_numpy(sm).to(device)), self.cont.append(torch.from_numpy(ct).to(device))
            self.kt.append(torch.from_numpy(k.astype(np.float32)).to(device))


def energies(model: E2EModel, p: Prepared) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """(unary costs per layer, transition matrices per boundary) of one sequence."""
    cos = model.cosines(p.xv, p.seg_of, p.xm)
    a = model.log_a.exp()
    u = [-a * (cos[p.bounds[i]:p.bounds[i + 1]] - p.zmu[i]) / p.zsd[i] for i in range(len(p.sizes))]
    base, h = model.boundary(p.vfeat)
    wo, wi = model.seam_weights(h)
    T = []
    for i in range(1, len(p.sizes)):
        A = p.exit[p.bounds[i - 1]:p.bounds[i]] @ wo[i - 1]
        B = p.entry[p.bounds[i]:p.bounds[i + 1]] @ wi[i - 1]
        T.append(transition(p.same[i - 1], p.cont[i - 1], p.kt[i - 1], model.b_sw + base[i - 1], model.b_jp + base[i - 1], A, B, model.mu))
    return u, T


@torch.no_grad()
def decode(model: E2EModel, s: Seq, p: Prepared, beam: int) -> list[int]:
    model.head_mode(False)
    u, T = energies(model, p)
    costs = [x.double().cpu().numpy() for x in u]
    Tn = [x.double().cpu().numpy() for x in T]
    return viterbi_beam(costs, s.n_top, s.cont_of, lambda i, a, c: Tn[i - 1][np.ix_(a, c)], beam)


# ---------------------------------------------------------------- metrics
def match(pred: list[float], true: list[float], tol: float = 1.0) -> tuple[int, int, int]:
    used, tp = set(), 0
    for t in true:
        c = [(abs(q - t), k) for k, q in enumerate(pred) if k not in used and abs(q - t) <= tol]
        if c:
            used.add(min(c)[1])
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def seq_metrics(s: Seq, picks: list[int]) -> dict:
    tr = np.array([ct[j] for ct, j in zip(s.cand_track, picks)])
    mids = np.array([m[j] for m, j in zip(s.mid, picks)])
    sw = mids[1:] != mids[:-1]
    bt = s.btimes
    out = {"part_acc": float(np.average(tr == s.seg_track, weights=s.w)),
           "switch_f1": switch_f1(bt[sw].tolist(), bt[s.labels == 1].tolist()),
           "switches_per_min": float(sw.sum() / (s.seg[-1, 1] - s.seg[0, 0]) * 60),
           "true_per_min": float((s.labels == 1).sum() / (s.seg[-1, 1] - s.seg[0, 0]) * 60),
           "gold_in_lattice": float(np.mean(s.gold >= 0)),
           "gold_track_in_lattice": float(np.mean([np.any(m) for m in s.gold_mask]))}
    infilm = np.isin(np.array(s.types), IN_FILM)
    tp, fp, fn = match(bt[sw & infilm].tolist(), bt[(s.labels == 1) & infilm].tolist())
    out.update({"infilm_tp": tp, "infilm_fp": fp, "infilm_fn": fn})
    return out


def objective(rows: list[dict]) -> float:
    return float(np.mean([r["part_acc"] + r["switch_f1"] for r in rows]))


# ---------------------------------------------------------------- training
def info_nce_batch(model: E2EModel, batch: list[tuple[Seq, Prepared]], rng: np.random.Generator) -> torch.Tensor:
    """One segment per clip in the batch (its gold window as the positive), symmetric InfoNCE on the ensemble cosine."""
    xv, xm, tracks, seen = [], [], [], set()
    for s, p in batch:
        part = part_of(s)
        for j, clip in enumerate(s.parts):
            segs = np.flatnonzero(part == j)
            if clip in seen or not len(segs):
                continue
            seen.add(clip)
            i = int(segs[rng.integers(len(segs))])
            xv.append(p.xv[i]), xm.append(p.xg[i]), tracks.append(s.seg_track[i])
    if len(xv) < 2:
        return torch.zeros((), device=model.device)
    xv, xm = torch.stack(xv), torch.stack(xm)
    cos = 0.0
    for hv, hm in zip(model.hv, model.hm):
        cos = cos + hv(xv) @ hm(xm).T
    logits = cos / len(model.hv) / TAU
    t = np.array(tracks)
    same = torch.from_numpy((t[:, None] == t[None, :]) & ~np.eye(len(t), dtype=bool)).to(model.device)
    logits = logits.masked_fill(same, float("-inf"))
    y = torch.arange(len(t), device=model.device)
    ce = torch.nn.functional.cross_entropy
    return 0.5 * (ce(logits, y) + ce(logits.T, y))


def part_of(s: Seq) -> np.ndarray:
    """[N] index into s.parts of each segment's clip (segments tile the parts in order; a new part starts at every
    non-'clip' boundary)."""
    return np.r_[0, np.cumsum([t != "clip" for t in s.types])].astype(int)


def train_variant(cfg, store, data: dict, variant: str, e: int, device: str = "cuda") -> tuple[E2EModel | None, dict]:
    pen = data["stitch"]["penalties"]
    beam = int(data["stitch"]["beam"])
    spec = VARIANTS[variant]
    sem = fold_models(cfg, store, data["fold"], e).sem.models
    allv = np.concatenate([s.vfeat for s in data["train"]])
    vstats = (allv.mean(0), allv.std(0) + 1e-6)
    s0 = spec or dict(heads=False, music=False, edge=False, dur=True)
    model = E2EModel(sem, pen, s0["music"], s0["edge"], s0["dur"], s0["heads"], vstats, device)
    music = data["music"]
    prep = {k: [Prepared(s, music, model.n_music, device) for s in data[k]] for k in ("train", "val")}
    curve = []

    def val_obj():
        return objective([seq_metrics(s, decode(model, s, p, beam)) for s, p in zip(data["val"], prep["val"])])

    best, best_state, bad = val_obj(), copy.deepcopy(model.state_dict()), 0
    curve.append({"epoch": 0, "val": best})
    log.info("%s fold %d e %d: epoch 0 (shipped decoder) val objective %.4f (stitch.json %.4f)", variant, data["fold"], e,
             best, data["stitch"]["val_objective"])
    if spec is None:
        return model, {"curve": curve, "best_epoch": 0, "val_objective": best}
    heads = [q for q in model.parameters() if q.requires_grad and any(q is h for h in list(model.hv.parameters()) + list(model.hm.parameters()))]
    rest = [q for q in model.parameters() if q.requires_grad and not any(q is h for h in heads)]
    groups = [{"params": rest, "lr": LR_REST, "weight_decay": 0.0}] + ([{"params": heads, "lr": LR_HEADS, "weight_decay": WD_HEADS}] if heads else [])
    opt = torch.optim.AdamW(groups)
    order_rng = np.random.default_rng([data["fold"], e, 7])
    best_epoch = 0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        idx = order_rng.permutation(len(data["train"]))
        losses = []
        for b in range(0, len(idx), BATCH):
            batch = [(data["train"][i], prep["train"][i]) for i in idx[b:b + BATCH]]
            model.head_mode(True)
            nll = 0.0
            for s, p in batch:
                u, T = energies(model, p)
                nll = nll + crf_marginal_nll(u, T, p.masks) / len(u)
            loss = nll / len(batch)
            if model.train_heads:
                loss = loss + LAMBDA_NCE * info_nce_batch(model, batch, np.random.default_rng([data["fold"], e, ep, b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        v = val_obj()
        curve.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": v, "s": time.time() - t0, **model.describe()})
        log.info("%s fold %d e %d: epoch %d loss %.4f val %.4f (best %.4f) [%.0f s]", variant, data["fold"], e, ep,
                 np.mean(losses), v, best, time.time() - t0)
        if v > best + 1e-9:
            best, best_state, bad, best_epoch = v, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"curve": curve, "best_epoch": best_epoch, "val_objective": best}


def rebuilt(cfg, store, data: dict, model: E2EModel, sem_models: list, e: int) -> dict[str, list[Seq]]:
    """Test lattices rebuilt by the full pipeline with the model's (fine-tuned) heads."""
    from bgm.cue.sequences import read_sets
    from bgm.data import manifest
    from bgm.e2e.data import build_seq
    from bgm.eval.store import fold_ids
    from bgm.retrieval.index import EnsembleSemantic, FoldModels, MusicIndex
    from bgm.retrieval.stitch_eval import read_concat_sets
    fold = data["fold"]
    f = fold_ids(cfg, fold)
    base = fold_models(cfg, store, fold, e)
    for m, hv, hm in zip(sem_models, model.hv, model.hm):
        m.hv, m.hm = hv, hm
        m.eval()
    models = FoldModels(fold, e, EnsembleSemantic(sem_models), None, base.cal, base.device)
    index = MusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    fg = manifest.read_folds(cfg)["film_group"]
    cfg.cue.sequences_file = "cue_sequences_title.json"
    sets = {"test": read_concat_sets(cfg)[str(fold)]["test"], "infilm": read_sets(cfg)["sets"][str(fold)]["test"]["blocked"]}
    return {k: [build_seq(store, index, models, data["fusion"], cfg, data["music"], fg, parts, "concat" if k == "test" else "infilm",
                          "test", i, False) for i, parts in enumerate(v)] for k, v in sets.items()}


def evaluate(model: E2EModel, data: dict, seqs: dict[str, list[Seq]], device: str) -> dict[str, list[dict]]:
    beam = int(data["stitch"]["beam"])
    return {k: [seq_metrics(s, decode(model, s, Prepared(s, data["music"], model.n_music, device), beam)) for s in v]
            for k, v in seqs.items()}


def run_one(cfg, store, fold: int, e: int, variant: str, device: str = "cuda") -> dict:
    from bgm.runs import run_dir, write_run
    d = run_dir(cfg, f"e2e/{variant}", fold, e)
    if (d / "metrics.json").exists():
        return json.loads((d / "metrics.json").read_text())
    data = build_fold(cfg, store, fold, e, out_root(cfg) / "data" / f"fold{fold}_e{e}.pkl")
    data["music"].store = store
    sem_models = fold_models(cfg, store, fold, e).sem.models
    model, info = train_variant(cfg, store, data, variant, e, device)
    res = {"A": evaluate(model, data, {"test": data["test"], "infilm": data["infilm"]}, device)}
    spec = VARIANTS[variant]
    if spec is not None and spec["heads"]:
        res["B"] = evaluate(model, data, rebuilt(cfg, store, data, model, sem_models, e), device)
    per = {f"{lat}/{k}/{m}": np.array([r[m] for r in rows], float) for lat, v in res.items() for k, rows in v.items() for m in rows[0]}
    np.savez(d / "perseq.npz", **per)
    torch.save({k: v for k, v in model.state_dict().items() if not (k.startswith("hv.") or k.startswith("hm."))}, d / "transitions.pt")
    if variant == "joint":
        torch.save({k: v.half() for k, v in model.state_dict().items() if k.startswith(("hv.", "hm."))}, d / "heads_fp16.pt")
    summ = {k: float(np.mean(v)) for k, v in per.items()}
    metrics = {"variant": variant, "fold": fold, "ensemble": e, "test": summ, **info, "params": model.describe()}
    write_run(d, cfg, metrics, e)
    log.info("%s fold %d e %d: test A concat part acc %.3f switch F1 %.3f | in-film part acc %.3f", variant, fold, e,
             summ["A/test/part_acc"], summ["A/test/switch_f1"], summ["A/infilm/part_acc"])
    return metrics


def main(argv: list[str]) -> None:
    from bgm.eval.store import FeatureStore
    kw = dict(a.split("=", 1) for a in argv if "=" in a)
    cmd = next((a for a in argv if "=" not in a), "train")
    variants = kw.get("variants", ",".join(VARIANTS)).split(",")
    folds = [int(x) for x in kw.get("folds", "0,1,2,3,4").split(",")]
    ens = [int(x) for x in kw.get("ens", "0,1,2").split(",")]
    cfg = _cfg()
    store = FeatureStore(cfg)
    for k in folds:
        for e in ens:
            if cmd == "data":
                build_fold(cfg, store, k, e, out_root(cfg) / "data" / f"fold{k}_e{e}.pkl")
                continue
            for v in variants:
                run_one(cfg, store, k, e, v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main(sys.argv[1:])
