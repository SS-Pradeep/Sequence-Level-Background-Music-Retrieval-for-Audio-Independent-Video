"""Planned losses on the title split (user request 2026-11-16; the loss-sweep split is paused, its results kept).

    python -m bgm.sweep.titlesplit run --seeds 0      # every defined config, seed 0, 5 folds
    python -m bgm.sweep.titlesplit run --seeds 1 2
    python -m bgm.sweep.titlesplit table

Protocol = results/titlesplit/: data/manifests/folds_title.json (5 folds over title film groups; per fold `val` = 10%
of the training film groups, used for early stopping; `test` = the fold's reported films), the shipped siglip-both
trainer settings (CLAP+MERT music, <= 30 clips per film per epoch, one segment per video per epoch, same-track
masking, frame drop 0.25, +/- 1 s jitter, patience 20), the M5 test protocol of `bgm train semantic` (`evaluate`:
segment_all and track_test_fold). Defaults exactly, nothing tuned.

REF = `bgm train semantic` unchanged (train_one). The others = bgm.sweep.train.run (bit-identical to train_one for
plain infonce) with the loss from bgm.losses.registry.
Outputs: runs/titlesplit/planned_losses/, results/titlesplit/planned_losses{_runs.csv,.md,.tex}.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bgm.sweep.titlesplit")
# BGM_SWEEP_DATA=cleaned (user 2026-11-21, N.1: the D144 bank on the cleaned dataset): same folds file name, but
# manifests / interim / features from cleaned_dataset/, runs under runs/cleaned/loss_bank, tables under
# results/cleaned/loss_bank/. Unset = the old 1,248-pair data exactly as D144 ran it.
DATA = os.environ.get("BGM_SWEEP_DATA", "old")
assert DATA in ("old", "cleaned"), DATA
CLEANED = DATA == "cleaned"
OUT = Path("results/cleaned/loss_bank" if CLEANED else "results/titlesplit")
RUNS_ROOT = Path("runs/cleaned/loss_bank" if CLEANED else "runs/titlesplit/planned_losses")
FOLDS = Path(("cleaned_dataset/manifests" if CLEANED else "data/manifests") + "/folds_title.json")
RUNS_CSV = OUT / "planned_losses_runs.csv"
FIELDS = ["config", "fold", "seed", "status", "epochs", "best_epoch", "R1", "R5", "R10", "MedR", "MRR", "track_R10",
          "track_MRR", "wall_s"]
N_FOLDS = 5

# name -> registry spec (None = REF, the current trainer itself).
# ASAC = ScoreAwareLoss without trans (user 2026-11-16): soft targets (eps 0.2, teacher tau 0.05, learnable tau) +
# 0.5 * intra. Ablations as proposed and confirmed: no_tau = tau fixed at 0.07; no_eps = eps 0 (one-hot targets);
# no_weights = fixed lambda instead of the learned (uncert) weights.
CONFIGS: dict[str, dict | None] = {
    "REF": None,
    "siglip": {"main": "siglip", "aux": {}},
    "asac_full": {"main": "soft", "aux": {"intra": 0.5}, "weighting": "uncert"},
    "asac_no_tau": {"main": "soft", "aux": {"intra": 0.5}, "weighting": "uncert", "main_kw": {"learn_tau": False}},
    "asac_no_eps": {"main": "soft", "aux": {"intra": 0.5}, "weighting": "uncert", "main_kw": {"eps": 0.0}},
    "asac_no_weights": {"main": "soft", "aux": {"intra": 0.5}, "weighting": "fixed"},
    "inter_intra_nce": {"main": "infonce", "aux": {"intra_nce": 0.5}},
    "infonce_intra": {"main": "infonce", "aux": {"intra": 0.5}},
    "infonce_rdrop": {"main": "infonce", "aux": {"rdrop": 1.0}},
}
# Improvement round (user 2026-11-16 "try to improve it"): defaults only, nothing tuned. D135's two test-positive
# regularisers (EMA 0.95, head dropout 0.5; cleaned dataset) on REF and on InfoNCE + intra, and two term combinations.
# Chosen after seeing the seed-0 planned-loss table, so this round is exploratory.
CONFIGS |= {
    "REF_ema": {"main": "infonce", "aux": {}, "ema_decay": 0.95},
    "REF_do05": {"main": "infonce", "aux": {}, "dropout": 0.5},
    "infonce_intra_ema": {"main": "infonce", "aux": {"intra": 0.5}, "ema_decay": 0.95},
    "infonce_intra_do05": {"main": "infonce", "aux": {"intra": 0.5}, "dropout": 0.5},
    "infonce_intra_rdrop": {"main": "infonce", "aux": {"intra": 0.5, "rdrop": 1.0}},
    "siglip_intra": {"main": "siglip", "aux": {"intra": 0.5}},
}
# Loss bank (user 2026-11-20, 10-row loss table; "don't use the tags"): every term is tag-free, positives come from
# track_id only. Inter = video<->music, intra = within a modality, adaptive = temperature / loss weights.
CONFIGS |= {
    "multipos": {"main": "multipos", "aux": {}},
    "debiased": {"main": "debiased", "aux": {}},
    "vsepp": {"main": "vsepp", "aux": {}},
    "infonce_supcon_m": {"main": "infonce", "aux": {"supcon_music": 0.5}},
    "infonce_vv": {"main": "infonce", "aux": {"vv_nce": 0.5}},
    "infonce_fixtau": {"main": "infonce", "aux": {}, "tau_learnable": False},
    "pq_tau": {"main": "pq_tau", "aux": {}},
    "recipe_fixed": {"main": "multipos", "aux": {"supcon_music": 0.5, "intra": 0.5}},
    "recipe_uncert": {"main": "multipos", "aux": {"supcon_music": 0.5, "intra": 0.5}, "weighting": "uncert"},
    "all_uncert": {"main": "multipos", "aux": {"supcon_music": 0.5, "vv_nce": 0.5, "intra": 0.5},
                   "weighting": "uncert"},
}
BANK_NEW = ["multipos", "debiased", "vsepp", "infonce_supcon_m", "infonce_vv", "infonce_fixtau", "pq_tau",
            "recipe_fixed", "recipe_uncert", "all_uncert"]
# Per-encoder heads for similarity fusion (user 2026-11-20): InfoNCE (REF loss) with only one music encoder.
CONFIGS |= {
    "REF_clap": {"main": "infonce", "aux": {}, "music_feature": "sem_clap_1hz"},
    "REF_mert": {"main": "infonce", "aux": {}, "music_feature": "sem_mert_1hz"},
}
BANK_ORDER = ["REF", "siglip", "inter_intra_nce", "infonce_intra", "asac_full"] + BANK_NEW
IMPROVE_ORDER = ["REF", "infonce_intra", "REF_ema", "REF_do05", "infonce_intra_ema", "infonce_intra_do05",
                 "infonce_intra_rdrop", "siglip_intra"]
ORDER = ["REF", "siglip", "asac_full", "asac_no_tau", "asac_no_eps", "asac_no_weights", "inter_intra_nce",
         "infonce_intra", "infonce_rdrop"]


def load_cfg():
    from bgm.config import load_config
    return load_config("configs/default.yaml", [
        "splits.folds_file=folds_title.json", "eval.concat_sets_file=concat_sets_title.json",
        "train.semantic.music_feature=both", "train.semantic.max_clips_per_film_per_epoch=30",
        f"paths.runs={RUNS_ROOT}"] + ([
        "paths.manifests=cleaned_dataset/manifests", "paths.interim=cleaned_dataset/interim",
        "paths.features=cleaned_dataset/features", "paths.results=results/cleaned",
        "paths.reports=reports/cleaned"] if CLEANED else []))


def done_rows() -> pd.DataFrame:
    return pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame(columns=FIELDS)


def append(row: dict) -> None:
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_one(cfg, store, split, name: str, fold: int, seed: int, device: str) -> dict:
    import torch
    from bgm.runs import run_dir, write_run
    from bgm.train.semantic import Scorer, evaluate, train_one
    t0 = time.time()
    spec = CONFIGS[name]
    if spec is None:
        m = train_one(cfg, store, fold, seed, "REF", device)
        res, status, epochs, best = m["test"], "ok", m["val"]["epochs_run"], m["val"]["best_epoch"]
    else:
        from bgm.sweep.train import run
        s = {"name": name, "weighting": "fixed"} | spec
        r = run(cfg, store, s, split, fold, seed, device, None, "fold", return_model=True)
        status, epochs, best = r["status"], r["epochs"], r.get("best_epoch", "")
        res = None
        out = run_dir(cfg, f"semantic/{name}", fold, seed)
        if status == "ok":
            res, per = evaluate(cfg, store, Scorer(r["model"], r["inp"]), split["folds"][str(fold)]["test"])
            np.savez(out / "perquery.npz", **per)
            torch.save({"state": r["model"].state(), "spec": s}, out / "ckpt.pt")
        write_run(out, cfg, {"spec": s, "status": status, "epochs": epochs, "best_epoch": best,
                             "es_R10": r.get("es_R10"), "test": res, "curve": r["curve"]}, seed)
    row = {"config": name, "fold": fold, "seed": seed, "status": status, "epochs": epochs, "best_epoch": best,
           "wall_s": round(time.time() - t0, 1)}
    if res is not None:
        sa, tr = res["segment_all"], res["track_test_fold"]
        row |= {"R1": sa["R@1"], "R5": sa["R@5"], "R10": sa["R@10"], "MedR": sa["MedR"], "MRR": sa["MRR"],
                "track_R10": tr["R@10"], "track_MRR": tr["MRR"]}
    append(row)
    print(f"  {name:<16} fold {fold} seed {seed}: {status:<8} seg R@10 {row.get('R10', float('nan')):.3f} "
          f"track R@10 {row.get('track_R10', float('nan')):.3f} epochs {epochs} [{row['wall_s']:.0f} s]", flush=True)
    return row


def cmd_run(seeds: list[int], names: list[str] | None = None) -> None:
    import torch
    from bgm.eval.store import FeatureStore
    cfg = load_cfg()
    store = FeatureStore(cfg)
    split = json.loads(FOLDS.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = names or [n for n in ORDER if n in CONFIGS]
    improve = not set(names) <= set(ORDER)
    for seed in seeds:
        for name in names:
            for k in range(N_FOLDS):
                d = done_rows()
                if len(d[(d.config == name) & (d.fold == k) & (d.seed == seed)]):
                    continue
                run_one(cfg, store, split, name, k, seed, device)
        if not improve:
            table()


def _ranks(name: str, fold: int, seed: int) -> np.ndarray:
    sub = "REF" if name == "REF" else name
    return np.load((RUNS_ROOT / "semantic") / sub / f"fold{fold}" / f"seed{seed}" / "perquery.npz")["segment_all"]


def _clusters() -> np.ndarray:
    """Film group of every test segment, folds 0..4 in order (the order of segment_retrieval)."""
    from bgm.eval.protocol import test_segments
    from bgm.eval.store import FeatureStore
    cfg = load_cfg()
    store = FeatureStore(cfg)
    split = json.loads(FOLDS.read_text())
    return np.array([split["film_group"][p] for k in range(N_FOLDS)
                     for p, *_ in test_segments(store, split["folds"][str(k)]["test"])])


def table(order: list[str] | None = None, stem: str = "planned_losses",
          title: str = "Planned losses on the title split") -> str:
    order = order or ORDER
    from bgm.eval.metrics import cluster_bootstrap
    d = done_rows()
    d = d[d.status == "ok"]
    seeds = sorted(d[d.config == "REF"].seed.unique())
    full_seeds = [s for s in seeds if all(len(d[(d.config == n) & (d.seed == s)]) == N_FOLDS for n in order)]
    rows, boot = [], len(full_seeds) > 1
    clusters = _clusters() if boot else None
    for name in order:
        x = d[(d.config == name) & d.seed.isin(full_seeds or seeds)]
        if name not in CONFIGS:
            rows.append({"config": name, "missing": "not defined yet"})
            continue
        r = {"config": name, "runs": len(x)}
        for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10"):
            v = x[c].astype(float)
            r[c], r[c + "_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan")
        ref = d[(d.config == "REF") & d.seed.isin(full_seeds or seeds)].set_index(["fold", "seed"]).R10
        xi = x.set_index(["fold", "seed"]).R10
        common = [i for i in xi.index if i in ref.index]
        dd = np.array([xi[i] - ref[i] for i in common])
        r["dR10"], r["dR10_sd"] = (dd.mean(), dd.std(ddof=1)) if len(dd) > 1 else (float("nan"), float("nan"))
        if boot and name != "REF" and len(x) == N_FOLDS * len(full_seeds):
            hit = lambda n: np.mean([np.concatenate([_ranks(n, k, s) for k in range(N_FOLDS)]) <= 10 for s in full_seeds], 0)
            b = cluster_bootstrap(hit(name) - hit("REF"), clusters, 10000, 0)
            r["ci"] = (b["mean"], b["lo"], b["hi"])
        rows.append(r)
    t = pd.DataFrame(rows)
    defined = t[t.get("missing").isna()] if "missing" in t else t
    defined = defined.sort_values("R10", ascending=False)
    t = pd.concat([defined, t[t.get("missing").notna()]] if "missing" in t else [defined])
    md = _md(t, full_seeds or seeds, boot, title)
    (OUT / f"{stem}.md").write_text(md)
    (OUT / f"{stem}.tex").write_text(_tex(t, boot))
    print("\n" + md, flush=True)
    return md


def _md(t: pd.DataFrame, seeds, boot: bool, title: str = "Planned losses on the title split") -> str:
    pm = lambda r, c, p=3: f"{r[c]:.{p}f} ± {r[c + '_sd']:.{p}f}" if np.isfinite(r.get(c, np.nan)) else "—"
    head = ["config", "runs", "seg R@1", "seg R@5", "seg R@10", "seg MedR ↓", "seg MRR", "track R@10", "Δ seg R@10 vs REF (paired)"]
    if boot:
        head.append("Δ seg R@10 [95% CI]")
    L = [f"# {title}", "",
         f"{'Cleaned dataset (cleaned_dataset/), ' if CLEANED else ''}Title split ({FOLDS}), 5 folds × seeds {', '.join(map(str, seeds))}; mean ± std over "
         "folds × seeds. Shipped siglip-both trainer settings for every row (CLAP+MERT music, ≤ 30 clips per film per "
         "epoch, one segment per video per epoch, same-track masking, 25% frame drop, ±1 s jitter); early stopping on "
         "each fold's `val` (10% of its training films, v→m segment R@10, patience 20). Test = M5 protocol of "
         "`bgm train semantic`: segment = segment_all (gallery: every test segment's aligned window), track = "
         "track_test_fold. Defaults exactly, nothing tuned. Sorted by segment R@10."
         + (" CI: per-query hit@10 averaged over seeds, paired against REF, film-group cluster bootstrap "
            "(10,000 resamples); * = excludes 0." if boot else " Single seed: development numbers."), "",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t.iterrows():
        if isinstance(r.get("missing"), str):
            L.append(f"| {r.config} | — | " + " | ".join(["not defined yet"] + ["—"] * (len(head) - 3)) + " |")
            continue
        cells = [r.config, str(int(r.runs)), pm(r, "R1"), pm(r, "R5"), pm(r, "R10"), pm(r, "MedR", 1), pm(r, "MRR"),
                 pm(r, "track_R10"), "—" if r.config == "REF" else f"{r.dR10:+.3f} ± {r.dR10_sd:.3f}"]
        if boot:
            ci = r.get("ci")
            cells.append("—" if not isinstance(ci, tuple) else
                         f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]" + ("*" if ci[1] > 0 or ci[2] < 0 else ""))
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "Definitions: siglip = pairwise sigmoid loss (t = e^t', t' init log 10, b init −10); inter_intra_nce = "
          "InfoNCE(v, m) + 0.5·[InfoNCE(v′, v″) + InfoNCE(m′, m″)] with two 25% frame-drop views of the video and two "
          "±1 s jitter crops of the music; infonce_intra = InfoNCE + 0.5 · intra (rows of the frozen vs projected "
          "similarity, both sides); infonce_rdrop = InfoNCE + 1.0 · R-Drop (symmetric KL of two dropout passes' "
          "similarity softmaxes)."]
    return "\n".join(L) + "\n"


def _tex(t: pd.DataFrame, boot: bool) -> str:
    d = t[t.get("missing").isna()] if "missing" in t else t
    cols = [("R1", "R@1", True), ("R5", "R@5", True), ("R10", "R@10", True), ("MedR", "MedR$\\downarrow$", False),
            ("MRR", "MRR", True), ("track_R10", "Track R@10", True)]
    best = {c: (d[c].max() if hi else d[c].min()) for c, _, hi in cols}
    out = [r"\begin{tabular}{l" + "c" * (len(cols) + (1 if boot else 0)) + "}", r"\toprule",
           "Loss & " + " & ".join(h for _, h, _ in cols) + (r" & $\Delta$R@10 [95\% CI]" if boot else "") + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        name = r.config.replace("_", r"\_")
        if isinstance(r.get("missing"), str):
            continue
        cells = []
        for c, _, _ in cols:
            p = 1 if c == "MedR" else 3
            s = f"{r[c]:.{p}f} $\\pm$ {r[c + '_sd']:.{p}f}"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(r[c], best[c]) else s)
        if boot:
            ci = r.get("ci")
            cells.append("--" if not isinstance(ci, tuple) else f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]")
        out.append(name + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table", "ensemble"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--set", choices=["planned", "improve", "bank"], default="planned")
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.seeds, a.configs), "table": lambda: table() if a.set == "planned" else (table(BANK_ORDER, "loss_bank",
                                                           "Tag-free loss bank on the title split (inter / intra / adaptive)") if a.set == "bank" else table(IMPROVE_ORDER, "improve_losses",
                                                           "Improvement round on the title split (exploratory)")), "ensemble": lambda: cmd_ensemble(a.configs)}[a.cmd]()


# ---------------------------------------------------------------- 10-seed ensembles (user 2026-11-16)
ENS_SEEDS = tuple(range(10))
ENS_CSV = OUT / "planned_losses_ens_runs.csv"


def load_scorer(cfg, store, name: str, fold: int, seed: int, device: str):
    import torch
    from bgm.train.semantic import Scorer, SemanticInputs, SemanticModel
    spec = CONFIGS.get(name) or {}
    if spec.get("music_feature"):
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.train.semantic.music_feature = spec["music_feature"]
    c = cfg.train.semantic
    inp = SemanticInputs(store, c.video_feature, c.music_feature)
    dv, dm = inp.dims
    ck = torch.load(Path(cfg.paths.runs) / "semantic" / name / f"fold{fold}" / f"seed{seed}" / "ckpt.pt", map_location=device)
    model = SemanticModel(dv, dm, cfg, device)
    model.load(ck["state"])
    model.eval()
    return Scorer(model, inp)


def cmd_ensemble(names: list[str] | None = None) -> None:
    """Mean-cosine ensemble (MeanScorer, the shipped D65/D75 rule) of seeds 0-9 per config and fold, scored with
    `evaluate`; per-query ranks to runs/titlesplit/planned_losses/semantic_ens/<name>/fold<k>/."""
    import torch
    from bgm.eval.store import FeatureStore
    from bgm.person.evaluate import MeanScorer
    from bgm.runs import run_dir, write_run
    from bgm.train.semantic import evaluate
    cfg = load_cfg()
    store = FeatureStore(cfg)
    split = json.loads(FOLDS.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = done_rows()
    e = pd.read_csv(ENS_CSV) if ENS_CSV.exists() else pd.DataFrame(columns=["config", "fold"])
    for name in names or [n for n in ORDER if n in CONFIGS]:
        for k in range(N_FOLDS):
            if len(e[(e.config == name) & (e.fold == k)]):
                continue
            ok = d[(d.config == name) & (d.fold == k) & (d.status == "ok") & d.seed.isin(ENS_SEEDS)]
            if len(ok) < len(ENS_SEEDS):
                print(f"  {name} fold {k}: {len(ok)}/{len(ENS_SEEDS)} members ok, ensemble skipped", flush=True)
                continue
            members = [load_scorer(cfg, store, name, k, s, device) for s in ENS_SEEDS]
            with torch.no_grad():
                res, per = evaluate(cfg, store, MeanScorer(members), split["folds"][str(k)]["test"])
            out = run_dir(cfg, f"semantic_ens/{name}", k, 0)
            np.savez(out / "perquery.npz", **per)
            write_run(out, cfg, {"test": res, "members": list(ENS_SEEDS)}, 0)
            sa, tr = res["segment_all"], res["track_test_fold"]
            row = {"config": name, "fold": k, "R1": sa["R@1"], "R5": sa["R@5"], "R10": sa["R@10"], "MedR": sa["MedR"],
                   "MRR": sa["MRR"], "track_R10": tr["R@10"], "track_MRR": tr["MRR"]}
            new = not ENS_CSV.exists()
            with open(ENS_CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  {name:<16} fold {k} ens10: seg R@10 {row['R10']:.3f} track R@10 {row['track_R10']:.3f}", flush=True)
    ens_table()


def ens_table() -> str:
    """Final table: ens10 per config, mean ± std over the 5 folds, paired Δ vs REF ens10 with a film-group cluster
    bootstrap CI on per-query hit@10 (and MRR); plus the single heads over folds × seeds 0-9."""
    from bgm.eval.metrics import cluster_bootstrap
    e = pd.read_csv(ENS_CSV)
    clusters = _clusters()
    seg = lambda n: np.concatenate([np.load((RUNS_ROOT / "semantic_ens") / n / f"fold{k}" / "seed0"
                                            / "perquery.npz")["segment_all"] for k in range(N_FOLDS)])
    trk = lambda n: np.concatenate([np.load((RUNS_ROOT / "semantic_ens") / n / f"fold{k}" / "seed0"
                                            / "perquery.npz")["track_test_fold"] for k in range(N_FOLDS)])
    full = [n for n in ORDER if n in CONFIGS and (e.config == n).sum() == N_FOLDS]
    rows = []
    for n in full:
        x = e[e.config == n]
        r = {"config": n}
        for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10"):
            r[c], r[c + "_sd"] = x[c].mean(), x[c].std(ddof=1)
        if n != "REF" and "REF" in full:
            r["ci"] = cluster_bootstrap((seg(n) <= 10).astype(float) - (seg("REF") <= 10), clusters, 10000, 0)
            r["ci_mrr"] = cluster_bootstrap(1 / seg(n) - 1 / seg("REF"), clusters, 10000, 0)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("R10", ascending=False)
    d = done_rows()
    d = d[(d.status == "ok") & d.seed.isin(ENS_SEEDS)]
    pm = lambda r, c, p=3: f"{r[c]:.{p}f} ± {r[c + '_sd']:.{p}f}"
    ci = lambda b: "—" if not isinstance(b, dict) else (f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"
                                                         + ("*" if b["lo"] > 0 or b["hi"] < 0 else ""))
    L = ["# Planned losses on the title split: 10-seed ensembles", "",
         "Title split (data/manifests/folds_title.json), 5 folds. Each config: seeds 0–9 trained per fold with the shipped "
         "siglip-both trainer settings (CLAP+MERT music, ≤ 30 clips per film per epoch, one segment per video per epoch, "
         "same-track masking, 25% frame drop, ±1 s jitter, early stopping on each fold's `val` = 10% of its training "
         "films, patience 20; defaults exactly, nothing tuned), then ensembled by mean cosine (MeanScorer, the shipped "
         "rule) and scored with `bgm train semantic`'s test protocol (segment = segment_all, track = track_test_fold). "
         "Mean ± std over the 5 folds. Δ = per-query difference vs the REF ensemble on the same 2,552 test segments, "
         "95% film-group cluster bootstrap (10,000 resamples); * = CI excludes 0.", "",
         "| config | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF [95% CI] | Δ seg MRR vs REF [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r.config} | {pm(r, 'R1')} | {pm(r, 'R5')} | {pm(r, 'R10')} | {pm(r, 'MedR', 1)} | {pm(r, 'MRR')} | "
                 f"{pm(r, 'track_R10')} | {ci(r.get('ci'))} | {ci(r.get('ci_mrr'))} |")
    missing = [n for n in ORDER if n not in full]
    if missing:
        L += ["", "Not complete yet: " + ", ".join(missing) + "."]
    L += ["", "## Single heads (the ensemble members)", "",
          "Mean ± std over 5 folds × seeds 0–9 (50 runs per config).", "",
          "| config | runs | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF (paired) |",
          "|---|---|---|---|---|---|---|---|---|"]
    srows = []
    ref = d[d.config == "REF"].set_index(["fold", "seed"]).R10
    for n in [n for n in ORDER if n in CONFIGS]:
        x = d[d.config == n]
        if not len(x):
            continue
        r = {"config": n, "runs": len(x)}
        for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10"):
            r[c], r[c + "_sd"] = x[c].astype(float).mean(), x[c].astype(float).std(ddof=1)
        xi = x.set_index(["fold", "seed"]).R10
        dd = np.array([xi[i] - ref[i] for i in xi.index if i in ref.index])
        r["dd"] = f"{dd.mean():+.3f} ± {dd.std(ddof=1):.3f}" if n != "REF" and len(dd) > 1 else "—"
        srows.append(r)
    for r in sorted(srows, key=lambda r: -r["R10"]):
        L.append(f"| {r['config']} | {r['runs']} | {pm(r, 'R1')} | {pm(r, 'R5')} | {pm(r, 'R10')} | {pm(r, 'MedR', 1)} | "
                 f"{pm(r, 'MRR')} | {pm(r, 'track_R10')} | {r['dd']} |")
    L += ["", "Definitions: REF = `bgm train semantic` unchanged (symmetric InfoNCE, learnable τ). siglip = pairwise "
          "sigmoid loss (t = e^t′, t′ init log 10, b init −10). ASAC = ScoreAwareLoss without the transition term: soft "
          "targets from the frozen music–music similarity (eps 0.2, teacher τ 0.05, learnable τ) + intra (λ 0.5, row "
          "cosine of frozen vs projected similarity, both sides), aux weight learned (uncertainty weighting, σ "
          "initialised at λ); no_tau = τ fixed at 0.07; no_eps = eps 0 (one-hot targets); no_weights = fixed λ 0.5. "
          "inter_intra_nce = InfoNCE(v, m) + 0.5·[InfoNCE(v′, v″) + InfoNCE(m′, m″)], two 25% frame-drop views of the "
          "video and two ±1 s jitter crops of the music. infonce_intra = InfoNCE + 0.5·intra. infonce_rdrop = InfoNCE + "
          "1.0·R-Drop."]
    md = "\n".join(L) + "\n"
    (OUT / "planned_losses.md").write_text(md)
    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r"Loss (ens10) & R@1 & R@5 & R@10 & MedR$\downarrow$ & MRR & Track R@10 & $\Delta$R@10 [95\% CI] \\", r"\midrule"]
    best = {c: (t[c].min() if c == "MedR" else t[c].max()) for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10")}
    for _, r in t.iterrows():
        cells = []
        for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10"):
            p = 1 if c == "MedR" else 3
            sv = f"{r[c]:.{p}f} $\\pm$ {r[c + '_sd']:.{p}f}"
            cells.append(f"\\textbf{{{sv}}}" if np.isclose(r[c], best[c]) else sv)
        b = r.get("ci")
        cells.append("--" if not isinstance(b, dict) else f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
        tex.append(r.config.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "planned_losses.tex").write_text("\n".join(tex) + "\n")
    print("\n" + md, flush=True)
    return md


def seed_table(seed: int, order: list[str] | None = None, stem: str = "planned") -> str:
    """One seed's table: mean ± std over the 5 folds, Δ seg R@10 vs REF on the same seed (paired per fold)."""
    order = order or ORDER
    d = done_rows()
    d = d[(d.seed == seed) & (d.status == "ok")]
    ref = d[d.config == "REF"].set_index("fold").R10
    rows = []
    for n in order:
        x = d[d.config == n]
        if not len(x):
            continue
        r = {"config": n, "folds": len(x)}
        for c in ("R1", "R5", "R10", "MedR", "MRR", "track_R10"):
            v = x[c].astype(float)
            r[c], r[c + "_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan")
        xi = x.set_index("fold").R10
        dd = np.array([xi[f] - ref[f] for f in xi.index if f in ref.index])
        r["d"] = "—" if n == "REF" or len(dd) < 2 else f"{dd.mean():+.3f} ± {dd.std(ddof=1):.3f}"
        r["per_fold"] = " / ".join(f"{xi.get(f, float('nan')):.3f}" for f in range(N_FOLDS))
        rows.append(r)
    rows.sort(key=lambda r: -r["R10"])
    pm = lambda r, c, p=3: f"{r[c]:.{p}f} ± {r[c + '_sd']:.{p}f}"
    L = [f"# Title split, {stem} configs, seed {seed} only", "",
         "Mean ± std over the 5 folds; Δ = paired per-fold difference vs REF (same seed). Single seed: development numbers.", "",
         "| config | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF | seg R@10 per fold (f0 / … / f4) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {pm(r, 'R1')} | {pm(r, 'R5')} | {pm(r, 'R10')} | {pm(r, 'MedR', 1)} | {pm(r, 'MRR')} | "
                 f"{pm(r, 'track_R10')} | {r['d']} | {r['per_fold']} |")
    md = "\n".join(L) + "\n"
    out = OUT / "per_seed"
    out.mkdir(exist_ok=True)
    (out / f"{stem}_seed{seed}.md").write_text(md)
    print(md, flush=True)
    return md
