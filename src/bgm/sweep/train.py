"""One sweep run: the current trainer (`bgm.train.semantic.train_one`, siglip-both settings) with the loss swapped.

Identical to train_one in everything but the loss (user decision 5): SemanticInputs / SemanticModel (LayerNorm ->
512 -> GELU -> Dropout 0.2 -> 256), AdamW lr 1e-3 wd 0.05, cosine warm-up, batch 128, one segment per video per
epoch, <= 30 clips per film per epoch, 25% frame drop, +/- 1 s music jitter, same-track masking, the same per-item RNG
streams, early stopping on v->m segment R@10 of the early-stopping films (patience 20, max 200 epochs).
Differences: only the loss. drop_last is off (user, 2026-11-16, after decision 8 moved infonce +0.026); with it off `infonce` reproduces train_one bit for bit
tests in run.py smoke), and the loss from bgm.losses.registry.

mode "fold": train on folds[k].train, early-stop on folds[k].val, report on folds[k].test (the fold's films).
mode "final": train on every non-TEST clip for a fixed number of epochs, report on heldout_test (Stage 5 only).
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bgm.losses.registry import GradNorm, SweepLoss
from bgm.sweep import eval as sweep_eval
from bgm.train.semantic import (SemanticInputs, SemanticModel, cap_per_film, clone_state, cosine_warmup, ema_update,
                                epoch_batches)

log = logging.getLogger(__name__)


class Diverged(Exception):
    pass


def _loss_param_groups(loss: SweepLoss, wd: float) -> list[dict]:
    mats = [p for p in loss.model_params() if p.ndim >= 2]
    rest = [p for p in loss.model_params() if p.ndim < 2]
    return [g for g in ({"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}) if g["params"]]


def run(cfg, store, spec: dict, split: dict, fold: int, seed: int, device: str, tags=None, mode: str = "fold",
        fixed_epochs: int | None = None, max_epochs: int | None = None, drop_last: bool = False,
        curve_png: Path | None = None, return_model: bool = False) -> dict:
    """spec: {name, main, aux: {term: lambda}, weighting}. Returns metrics (status ok | diverged)."""
    if spec.get("music_feature") is not None:        # per-encoder heads (D145): sem_clap_1hz | sem_mert_1hz
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.train.semantic.music_feature = spec["music_feature"]
    c = cfg.train.semantic
    assert float(c.get("input_dropout", 0)) == 0 and float(c.get("input_noise", 0)) == 0 and c.get("video_pool", "mean") == "mean"
    t_start = time.time()
    if mode == "fold":
        f = split["folds"][str(fold)]
        train_ids, es_ids, rep_ids = f["train"], f["val"], f["test"]
    else:
        test = set(split["heldout_test"])
        train_ids = sorted(p for p in split["film_group"] if p not in test)
        es_ids, rep_ids = [], split["heldout_test"]
    film = split["film_group"]
    cap = c.get("max_clips_per_film_per_epoch")
    inp = SemanticInputs(store, c.video_feature, c.music_feature)
    dv, dm = inp.dims
    torch.manual_seed(seed)
    if spec.get("dropout") is not None:              # per-config head dropout (D135's dropout 0.5), trainer default otherwise
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.train.semantic.dropout = float(spec["dropout"])
    if spec.get("tau_learnable") is not None:        # adaptive-temperature ablation (fixed tau = tau_init)
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.train.semantic.tau_learnable = bool(spec["tau_learnable"])
    model = SemanticModel(dv, dm, cfg, device)
    ema_decay = float(spec.get("ema_decay", 0.0))     # D135: EMA of the weights per step; val, best and the result use it
    ema = clone_state(model.state()) if ema_decay > 0 else None
    hv_last, hm_last = model.hv.net[-1], model.hm.net[-1]
    ctx = {"log_tau": model.log_tau, "d": int(c.out_dim), "dv": dv, "dm": dm,
           "last_layers": (hv_last.weight, hm_last.weight)}
    if tags is not None:
        ctx |= {"group_slices": tags.group_slices, "n_genres": len(tags.genre_slice)}
    loss = SweepLoss(spec["main"], dict(spec.get("aux", {})), spec.get("weighting", "fixed"), ctx,
                     spec.get("main_kw")).to(device)
    needs_tags = bool(loss.needs & {"tag_teacher", "tag_heads", "supcon", "center"})
    assert tags is not None or not needs_tags, "tag terms need tag vectors"
    views = bool({"intra_nce", "supcon_music", "vv_nce"} & set(loss.aux_names)) or spec["main"] == "multipos"
    assert "trans" not in loss.aux_names, "trans was dropped from the sweep (within-video pairs: 1.2% change at 0.8)"
    lr = c.lr_scratch
    opt = torch.optim.AdamW([{"params": model.params()}] + _loss_param_groups(loss, c.weight_decay), lr=lr,
                            weight_decay=c.weight_decay)
    gn = GradNorm(loss, [hv_last.weight, hm_last.weight]) if loss.weighting == "gradnorm" else None
    train_film = [film[p] for p in train_ids] if cap else None
    n_epoch = len(cap_per_film(train_film, cap, np.random.default_rng(0))) if cap else len(train_ids)
    steps_per_epoch = (n_epoch // c.batch) if drop_last else math.ceil(n_epoch / c.batch)
    E = fixed_epochs if fixed_epochs is not None else (max_epochs or c.max_epochs)
    total = E * steps_per_epoch
    track_index = {t: i for i, t in enumerate(sorted(set(store.track.values())))}
    diag_ids = sorted(np.random.default_rng([fold, 134]).choice(train_ids, max(1, len(es_ids) or len(rep_ids)), replace=False))
    curve, best, best_state, best_epoch, bad, step = [], -1.0, None, -1, 0, 0
    first_mon, status, sigma_log = None, "ok", []
    try:
        for epoch in range(E):
            t0 = time.time()
            model.train()
            loss.train()
            rng = np.random.default_rng([seed, epoch])
            if cap:
                sel = cap_per_film(train_film, cap, np.random.default_rng([seed, epoch, 1]))
                order = sel[rng.permutation(len(sel))]
            else:
                order = rng.permutation(len(train_ids))
            batches = epoch_batches(train_ids, order, c.batch)
            if drop_last:
                batches = [b for b in batches if len(b) == c.batch]
            mons = []
            for bi, ids in enumerate(batches):
                V, M, T, TS, TM, G = [], [], [], [], [], []
                VW, MW = ([], []), ([], [])
                for j, p in enumerate(ids):
                    irng = np.random.default_rng([seed, epoch, int(order[bi * c.batch + j])])
                    k = int(irng.integers(len(store.segments[p])))
                    a, b = store.segments[p][k]
                    V.append(inp.video(p, a, b, c.frame_drop, irng))
                    M.append(inp.music(store.music_of[p], a, b, c.music_jitter_s, irng))
                    T.append(track_index[store.track[p]])
                    if views:                        # own RNG stream: the main draws above stay unchanged
                        vrng = np.random.default_rng([seed, epoch, int(order[bi * c.batch + j]), 7])
                        for w in range(2):
                            VW[w].append(inp.video(p, a, b, c.frame_drop, vrng))
                            MW[w].append(inp.music(store.music_of[p], a, b, c.music_jitter_s, vrng))
                    if needs_tags:
                        tm = tags.music(store.music_of[p], a, b)
                        TS.append(tags.scene(p, k)), TM.append(tm), G.append(tags.genre(tm))
                for g in opt.param_groups:
                    g["lr"] = lr * cosine_warmup(step, total, cfg.train.warmup_frac)
                xv = torch.from_numpy(np.stack(V)).to(device)
                xm = torch.from_numpy(np.stack(M)).to(device)
                v_pre, m_pre = model.hv.net(xv), model.hm.net(xm)
                v, m = F.normalize(v_pre, dim=-1), F.normalize(m_pre, dim=-1)
                batch = {"v_raw": xv, "m_raw": xm, "v_pre": v_pre, "m_pre": m_pre, "epoch": epoch,
                         "track": torch.tensor(T, device=device)}
                if needs_tags:
                    batch |= {"t_scene": torch.from_numpy(np.stack(TS)).to(device),
                              "t_music": torch.from_numpy(np.stack(TM)).to(device),
                              "genre": torch.tensor(G, device=device)}
                if views:
                    for w in range(2):
                        batch[f"v_view{w + 1}"] = F.normalize(model.hv.net(torch.from_numpy(np.stack(VW[w])).to(device)), dim=-1)
                        batch[f"m_view{w + 1}"] = F.normalize(model.hm.net(torch.from_numpy(np.stack(MW[w])).to(device)), dim=-1)
                if "rdrop" in loss.aux_names:
                    batch["v2"] = F.normalize(model.hv.net(xv), dim=-1)
                    batch["m2"] = F.normalize(model.hm.net(xm), dim=-1)
                tot, parts, mon = loss(v, m, batch)
                if not torch.isfinite(tot) or not torch.isfinite(mon):
                    raise Diverged(f"non-finite loss at epoch {epoch} step {bi}")
                opt.zero_grad()
                if gn is not None:
                    gn.step()
                tot.backward()
                opt.step()
                if ema is not None:
                    ema_update(ema, model.state(), ema_decay)
                step += 1
                mons.append(float(mon))
            loss.end_epoch()
            mon_epoch = float(np.mean(mons)) if mons else float("nan")
            # vsepp switches from InfoNCE (~4.4) to a summed hinge (~50) after its warm-up: restart the baseline there
            if first_mon is None or (spec["main"] == "vsepp" and epoch == getattr(loss.main, "warm", -1)):
                first_mon = mon_epoch
            if not math.isfinite(mon_epoch) or (first_mon > 0 and mon_epoch > 10 * first_mon):
                raise Diverged(f"loss {mon_epoch:.3g} vs start {first_mon:.3g} at epoch {epoch}")
            if loss.weighting in ("uncert", "dwa", "gradnorm"):
                sigma_log.append({"epoch": epoch} | (loss.sigmas() if loss.weighting == "uncert" else loss.weights_now()))
            model.eval()
            if ema is not None:
                live = clone_state(model.state())
                model.load(ema)
            tr10 = sweep_eval.r10_v2m(store, model, inp, diag_ids)
            if mode == "fold":
                val = sweep_eval.r10_v2m(store, model, inp, es_ids)
                curve.append((epoch, mon_epoch, tr10, val, round(time.time() - t0, 2)))
                if val > best:
                    best, best_state, best_epoch, bad = val, clone_state(model.state()), epoch, 0
                else:
                    bad += 1
            else:
                curve.append((epoch, mon_epoch, tr10, float("nan"), round(time.time() - t0, 2)))
            if ema is not None:
                if mode != "fold" and epoch == E - 1:
                    continue                          # final mode: keep the EMA weights loaded for the result
                model.load(live)
            if mode == "fold" and bad >= c.patience:
                break
    except Diverged as e:
        status = "diverged"
        log.warning("%s fold %s seed %s: diverged (%s)", spec["name"], fold, seed, e)
    out = {"status": status, "epochs": len(curve), "curve": curve, "sigma_log": sigma_log,
           "loss_weights_final": loss.weights_now(), "sigmas_final": loss.sigmas()}
    if status == "ok":
        if mode == "fold":
            model.load(best_state)
            out["best_epoch"], out["es_R10"] = best_epoch, best
        else:
            out["best_epoch"] = E - 1
        model.eval()
        ev = sweep_eval.evaluate(store, model, inp, rep_ids)
        out |= {"v2m": ev["v2m"], "m2v": ev["m2v"], "ranks_v2m": ev["ranks_v2m"], "ranks_m2v": ev["ranks_m2v"],
                "train_R10": float(sweep_eval.r10_v2m(store, model, inp, diag_ids))}
    out["wall_s"] = round(time.time() - t_start, 1)
    if return_model:
        out["model"], out["inp"] = model, inp
    if curve_png is not None and curve:
        _plot_curve(curve, curve_png, f"{spec['name']} fold {fold}", out.get("best_epoch"))
    return out


def _plot_curve(curve, path: Path, title: str, best_epoch) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r[0] for r in curve]
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    ax.plot(ep, [r[2] for r in curve], label="train R@10 (size-matched subset)", color="#4C78A8")
    if not all(math.isnan(r[3]) for r in curve):
        ax.plot(ep, [r[3] for r in curve], label="early-stop R@10", color="#F58518")
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("epoch"), ax.set_ylabel("v→m segment R@10"), ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
