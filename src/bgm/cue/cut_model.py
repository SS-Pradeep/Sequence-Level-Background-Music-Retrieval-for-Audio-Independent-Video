"""D103: train and evaluate the music-change-at-cut classifier (bgm.cue.cuts) on the title-grouped film split.

Per fold k: labels calibrated on fold k's TRAIN clips; features standardized and PCA-reduced on TRAIN only; models
fit on TRAIN; decision thresholds chosen on VAL; TEST touched once. Time since the music started = time since the clip
start (no reset at earlier changes: resetting at true changes would feed test labels into test inputs).

Systems: baselines (every cut; time since start; visual distance), and models on feature groups (logistic regression
with balanced class weights; small MLP with pos_weight BCE, mean of 3 seeds). Metrics: AP, AUROC, F1 at the VAL
threshold, boundary F1 with a +-1 shot tolerance; paired cluster bootstrap over film groups for the deltas.

    python -m bgm.cue.cut_model            -> results/titlesplit/cut_changes.md, runs/titlesplit/cut_changes/
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from bgm.cue.cuts import GROUPS, calibrate, cut_features, label_cuts

log = logging.getLogger(__name__)
VEC = ("sig_before", "sig_after", "sig_diff", "mert_before")


def build_fold(cfg, store, fold: int) -> tuple[pd.DataFrame, dict, dict]:
    from bgm.eval.store import fold_ids
    c = cfg.cue.cuts
    f = fold_ids(cfg, fold)
    cal, info = calibrate(store, f["train"], c.half_s, c.n_joins, fold)
    split = {p: s for s in ("train", "val", "test") for p in f[s]}
    df = label_cuts(store, [p for p in store.pairs.pair_id if p in split], cal, c)
    df["split"] = df.pair.map(split)
    feats = [cut_features(store, r, c.n_shots, c.music_s, r.t, c.tol_s) for r in df.itertuples()]
    vecs = {k: np.stack([x[k] for x in feats]) for k in ("sig_before", "sig_after", "mert_before")}
    vecs["sig_diff"] = np.abs(vecs["sig_after"] - vecs["sig_before"])
    for k in feats[0]:
        if k not in vecs:
            df[k] = [x[k] for x in feats]
    return df, vecs, info


def design(df: pd.DataFrame, vecs: dict, scalars: tuple, vec_keys: tuple, n_pca: int, tr: np.ndarray) -> np.ndarray:
    """Columns: standardized scalars + PCA(n_pca) of each vector group, every transform fit on rows `tr`."""
    cols = []
    if scalars:
        x = df[list(scalars)].to_numpy(float)
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        cols.append((x - mu) / sd)
    for k in vec_keys:
        p = PCA(n_pca, random_state=0).fit(vecs[k][tr])
        z = p.transform(vecs[k])
        cols.append(z / (z[tr].std(0) + 1e-9))
    return np.concatenate(cols, 1)


def fit_lr(X, y, tr):
    return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr]).predict_proba(X)[:, 1]


def fit_mlp(X, y, tr, seeds=(0, 1, 2), epochs: int = 60, hidden: int = 64):
    out = []
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-6))
    for s in seeds:
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        idx = np.flatnonzero(tr)
        g = np.random.default_rng(s)
        for _ in range(epochs):
            net.train()
            for b in np.array_split(g.permutation(idx), max(1, len(idx) // 128)):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b], pos_weight=pw)
                opt.zero_grad(), loss.backward(), opt.step()
        net.eval()
        with torch.no_grad():
            out.append(torch.sigmoid(net(Xt)[:, 0]).numpy())
    return np.mean(out, 0)


def inner_groups(df: pd.DataFrame, tr: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Inner fold id (0..n-1) per TRAIN row, by film group (Konjum, train-only, is always in the fitting part: -1)."""
    g = df.group.to_numpy()
    ug = sorted({x for x, t in zip(g, tr) if t and not str(x).endswith("Konjum salangai")})
    rng = np.random.default_rng(seed)
    fold_of = {x: int(i) % n for i, x in zip(range(len(ug)), rng.permutation(ug))}
    return np.array([fold_of.get(x, -1) if t else -2 for x, t in zip(g, tr)])


GBM_GRID = [{"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": leaf}
            for d in (3, 5) for lr in (0.03, 0.1) for it in (200, 400) for leaf in (20, 50)]


def fit_gbm(X, y, tr, inner: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gradient-boosted trees (sklearn HistGradientBoosting, balanced classes); hyperparameters by inner group CV AP."""
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    best, arg = -1.0, GBM_GRID[0]
    for hp in GBM_GRID:
        aps = []
        for f in range(inner.max() + 1):
            fit, ev = tr & (inner != f), inner == f
            if y[ev].sum() == 0:
                continue
            m = H(class_weight="balanced", l2_regularization=1.0, random_state=0, **hp).fit(X[fit], y[fit])
            aps.append(average_precision_score(y[ev], m.predict_proba(X[ev])[:, 1]))
        if np.mean(aps) > best:
            best, arg = float(np.mean(aps)), hp
    m = H(class_weight="balanced", l2_regularization=1.0, random_state=0, **arg).fit(X[tr], y[tr])
    return m.predict_proba(X)[:, 1], {**arg, "inner_ap": best}


def fit_deep_mlp(X, y, tr, inner: np.ndarray, seeds=(0, 1, 2), hidden: int = 256, max_epochs: int = 200) -> tuple[np.ndarray, dict]:
    """3-layer residual MLP (LayerNorm, GELU, dropout 0.3, AdamW wd 1e-2, pos_weight BCE). Epochs: early stopping on
    inner fold 0 (a held-out part of the TRAIN film groups), then refit on all train for that many epochs."""
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    class Net(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.inp = torch.nn.Linear(d, hidden)
            self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, hidden),
                                                                   torch.nn.GELU(), torch.nn.Dropout(0.3)) for _ in range(3)])
            self.out = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, 1))

        def forward(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = h + b(h)
            return self.out(h)[:, 0]

    def train(rows, epochs, s, ev=None):
        torch.manual_seed(s)
        net = Net(X.shape[1])
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
        pw = torch.tensor((1 - y[rows].mean()) / max(y[rows].mean(), 1e-6))
        idx, g, curve = np.flatnonzero(rows), np.random.default_rng(s), []
        for _ in range(epochs):
            net.train()
            for b in np.array_split(g.permutation(idx), max(1, len(idx) // 128)):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), yt[b], pos_weight=pw)
                opt.zero_grad(), loss.backward(), opt.step()
            if ev is not None:
                net.eval()
                with torch.no_grad():
                    curve.append(average_precision_score(y[ev], net(Xt[ev]).numpy()))
        return net, curve

    ev = inner == 0
    _, curve = train(tr & ~ev, max_epochs, 0, ev)
    k = int(np.argmax(np.convolve(curve, np.ones(5) / 5, "same"))) + 1
    out = []
    for s in seeds:
        net, _ = train(tr, k, s)
        net.eval()
        with torch.no_grad():
            out.append(torch.sigmoid(net(Xt)).numpy())
    return np.mean(out, 0), {"epochs": k, "inner_ap": float(max(curve))}


def fit_seq(df: pd.DataFrame, Xv: np.ndarray, Xm: np.ndarray, y, tr, inner: np.ndarray, seeds=(0, 1, 2), d: int = 64,
            max_epochs: int = 150) -> tuple[np.ndarray, dict]:
    """2-layer bidirectional transformer over the cuts of a clip. Tokens: VIDEO features only (+ cut time); the head
    adds the cut's OWN music features. Later cuts' music features are never shared: they contain the music after
    the current cut, i.e. the label."""
    clips = [np.flatnonzero((df.pair == p).to_numpy()) for p in dict.fromkeys(df.pair)]
    tpos = np.log1p(df.t.to_numpy())[:, None]
    V = torch.tensor(np.c_[Xv, tpos], dtype=torch.float32)
    M, Y = torch.tensor(Xm, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = torch.nn.Linear(V.shape[1], d)
            layer = torch.nn.TransformerEncoderLayer(d, 4, 2 * d, 0.3, batch_first=True, norm_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = torch.nn.Sequential(torch.nn.Linear(d + M.shape[1], d), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(d, 1))

        def forward(self, rows_list):
            L = max(len(r) for r in rows_list)
            x = torch.zeros(len(rows_list), L, V.shape[1])
            pad = torch.ones(len(rows_list), L, dtype=torch.bool)
            for i, r in enumerate(rows_list):
                x[i, :len(r)], pad[i, :len(r)] = V[r], False
            h = self.enc(self.tok(x), src_key_padding_mask=pad)
            hs = torch.cat([h[i, :len(r)] for i, r in enumerate(rows_list)])
            idx = np.concatenate(rows_list)
            return self.head(torch.cat([hs, M[idx]], 1))[:, 0], idx

    def train(mask, epochs, s, ev=None):
        torch.manual_seed(s)
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
        cl = [c for c in clips if mask[c[0]]]
        pw = torch.tensor((1 - y[mask].mean()) / max(y[mask].mean(), 1e-6))
        g, curve = np.random.default_rng(s), []
        for _ in range(epochs):
            net.train()
            order = g.permutation(len(cl))
            for b in np.array_split(order, max(1, len(cl) // 32)):
                z, idx = net([cl[i] for i in b])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y[idx], pos_weight=pw)
                opt.zero_grad(), loss.backward(), opt.step()
            if ev is not None:
                curve.append(average_precision_score(y[ev], predict(net, ev)))
        return net, curve

    def predict(net, mask):
        net.eval()
        out = np.zeros(len(df))
        with torch.no_grad():
            cl = [c for c in clips if mask[c[0]]]
            for b in range(0, len(cl), 64):
                z, idx = net(cl[b:b + 64])
                out[idx] = torch.sigmoid(z).numpy()
        return out[mask]

    ev = inner == 0
    _, curve = train(tr & ~ev, max_epochs, 0, ev)
    k = int(np.argmax(np.convolve(curve, np.ones(5) / 5, "same"))) + 1
    out = []
    allm = np.ones(len(df), bool)
    for s in seeds:
        net, _ = train(tr, k, s)
        out.append(predict(net, allm))
    return np.mean(out, 0), {"epochs": k, "inner_ap": float(max(curve))}


def best_threshold(s: np.ndarray, y: np.ndarray) -> float:
    cand = np.unique(s)
    best, th = -1.0, float(cand[-1]) if len(cand) else 0.5
    for t in np.quantile(cand, np.linspace(0, 1, 101)):
        p = s >= t
        tp = (p & (y == 1)).sum()
        f = 2 * tp / max(p.sum() + y.sum(), 1)
        if f > best:
            best, th = f, float(t)
    return th


def boundary_f1(df: pd.DataFrame, pred: np.ndarray, tol: int = 1) -> tuple[int, int, int]:
    """One-to-one matching of predicted and true change cuts within +-tol cut indices of the same clip."""
    tp = fp = fn = 0
    for _, g in df.assign(pred=pred).groupby("pair"):
        P, T = list(g.idx[g.pred == 1]), list(g.idx[g.label == 1])
        used = set()
        for i in P:
            m = [j for j in T if abs(j - i) <= tol and j not in used]
            if m:
                used.add(min(m, key=lambda j: abs(j - i)))
                tp += 1
            else:
                fp += 1
        fn += len(T) - len(used)
    return tp, fp, fn


def systems(df: pd.DataFrame, vecs: dict, n_pca: int) -> dict[str, np.ndarray]:
    y = df.label.to_numpy()
    tr = (df.split == "train").to_numpy()
    out = {"every cut": np.ones(len(df)), "time since start (rule)": df.since_start.to_numpy(),
           "visual distance (rule)": df.vis_dist_3.to_numpy()}
    specs = {"LR video": (GROUPS["video"], ("sig_before", "sig_after", "sig_diff")),
             "LR time": (GROUPS["time"], ()),
             "LR music": (GROUPS["music"], ("mert_before",)),
             "LR video + time": (GROUPS["video"] + GROUPS["time"], ("sig_before", "sig_after", "sig_diff")),
             "LR video + music": (GROUPS["video"] + GROUPS["music"], VEC),
             "LR all": (GROUPS["video"] + GROUPS["time"] + GROUPS["music"], VEC)}
    for name, (sc, vk) in specs.items():
        out[name] = fit_lr(design(df, vecs, sc, vk, n_pca, tr), y, tr)
    Xall = design(df, vecs, GROUPS["video"] + GROUPS["time"] + GROUPS["music"], VEC, n_pca, tr)
    out["MLP all"] = fit_mlp(Xall, y, tr)
    out["MLP video"] = fit_mlp(design(df, vecs, GROUPS["video"], ("sig_before", "sig_after", "sig_diff"), n_pca, tr), y, tr)
    inner = inner_groups(df, tr, 3)
    hp = {}
    out["GBM all"], hp["GBM all"] = fit_gbm(Xall, y, tr, inner)
    out["deep MLP all"], hp["deep MLP all"] = fit_deep_mlp(Xall, y, tr, inner)
    Xv = design(df, vecs, GROUPS["video"], ("sig_before", "sig_after", "sig_diff"), n_pca, tr)
    Xm = design(df, vecs, GROUPS["music"] + GROUPS["time"], ("mert_before",), n_pca, tr)
    out["transformer (clip sequence)"], hp["transformer (clip sequence)"] = fit_seq(df, Xv, Xm, y, tr, inner)
    rank = lambda v: pd.Series(v).rank(pct=True).to_numpy()
    out["ensemble (GBM + deep MLP + transformer)"] = (rank(out["GBM all"]) + rank(out["deep MLP all"]) + rank(out["transformer (clip sequence)"])) / 3
    log.info("hyperparameters: %s", hp)
    return out


def cluster_ci(rows: pd.DataFrame, a: str, b: str, B: int, metric) -> tuple[float, float, float]:
    """Paired bootstrap over film groups of metric(a) - metric(b) on pooled test rows."""
    g = rows.group.to_numpy()
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(0)
    y = rows.label.to_numpy()
    d0 = metric(y, rows[a].to_numpy()) - metric(y, rows[b].to_numpy())
    ds = []
    for _ in range(B):
        sel = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        if y[sel].min() == y[sel].max():
            continue
        ds.append(metric(y[sel], rows[a].to_numpy()[sel]) - metric(y[sel], rows[b].to_numpy()[sel]))
    return d0, float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))


def run(cfg, n_pca: int = 16, B: int = 2000) -> Path:
    from bgm.eval.store import FeatureStore
    from bgm.data import manifest
    store = FeatureStore(cfg)
    group = manifest.read_folds(cfg)["film_group"]
    root = Path(cfg.paths.runs) / "cut_changes"
    root.mkdir(parents=True, exist_ok=True)
    tests, infos, f1rows = [], {}, []
    for k in range(cfg.splits.n_folds):
        df, vecs, info = build_fold(cfg, store, k)
        infos[k] = {**info, "cuts": len(df), "pos": int(df.label.sum()),
                    **{f"{s}_cuts": int((df.split == s).sum()) for s in ("train", "val", "test")},
                    **{f"{s}_pos": int(df.label[df.split == s].sum()) for s in ("train", "val", "test")}}
        df["group"] = df.pair.map(group)
        S = systems(df, vecs, n_pca)
        va, te = (df.split == "val").to_numpy(), (df.split == "test").to_numpy()
        y = df.label.to_numpy()
        for name, s in S.items():
            th = best_threshold(s[va], y[va]) if name != "every cut" else 0.5
            tp, fp, fn = boundary_f1(df[te], (s[te] >= th).astype(int))
            f1rows.append({"fold": k, "system": name, "tp": tp, "fp": fp, "fn": fn})
        t = df[te].copy()
        for name, s in S.items():
            t[name] = s[te]
        t["fold"] = k
        tests.append(t)
        log.info("fold %d: %d cuts (%d pos), test %d (%d pos)", k, len(df), int(y.sum()), int(te.sum()), int(y[te].sum()))
    T = pd.concat(tests, ignore_index=True)
    T.drop(columns=[c for c in T.columns if c.startswith("sig_") or c == "mert_before"], errors="ignore").to_csv(root / "test_predictions.csv", index=False)
    F = pd.DataFrame(f1rows).groupby("system")[["tp", "fp", "fn"]].sum()
    names = [c for c in S]
    y = T.label.to_numpy()
    L = ["# Music change at shot cuts (D103)", "",
         "Title-grouped film split (`folds_title.json`), 5 folds. Every measurable shot cut **inside** a clip; label 1 if the clip's own "
         "music changes within ±1 s (MERT jump, calibrated per fold on train clips only, or a drop to silence). Test cuts pooled over folds. "
         "Thresholds for F1 chosen on val (small: about 150 cuts per fold). Boundary F1: ±1 cut, one-to-one. "
         f"CIs: paired cluster bootstrap over film groups, {B:,} resamples.", "",
         f"Test cuts {len(T)}, changes {int(y.sum())} ({100 * y.mean():.1f}%); chance AP = {y.mean():.3f}.", "",
         "| system | AP | AUROC | boundary F1 | precision | recall | Δ AP vs visual rule [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        ap = average_precision_score(y, T[n]) if n != "every cut" else y.mean()
        au = roc_auc_score(y, T[n]) if n != "every cut" else 0.5
        tp, fp, fn = F.loc[n]
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        ci = "—"
        if n not in ("every cut", "visual distance (rule)"):
            d, lo, hi = cluster_ci(T, n, "visual distance (rule)", B, average_precision_score)
            ci = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| {n} | {ap:.3f} | {au:.3f} | {f1:.3f} | {pr:.3f} | {rc:.3f} | {ci} |")
    L += ["", "Ablations (paired Δ AP [95% CI]):", ""]
    for a, b in (("LR video + music", "LR video"), ("LR video + time", "LR video"), ("LR all", "LR video"), ("LR all", "LR music"),
                 ("MLP all", "LR all"), ("LR video", "time since start (rule)"), ("GBM all", "MLP all"), ("deep MLP all", "MLP all"),
                 ("transformer (clip sequence)", "MLP all"), ("ensemble (GBM + deep MLP + transformer)", "MLP all")):
        d, lo, hi = cluster_ci(T, a, b, B, average_precision_score)
        L.append(f"- {a} − {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    L += ["", "Per fold (labels): " + "; ".join(f"fold {k}: {v['test_pos']}/{v['test_cuts']} test, {v['val_pos']}/{v['val_cuts']} val, "
                                                 f"d at p=0.5 {v['d_at_p50']:.3f}" for k, v in infos.items())]
    (root / "labels.json").write_text(json.dumps(infos, indent=1))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (root / "metrics.json").write_text(json.dumps({"commit": commit, "n_test": len(T), "pos": int(y.sum()),
                                                   "ap": {n: float(average_precision_score(y, T[n])) for n in names if n != "every cut"}}, indent=1))
    out = Path(cfg.paths.results) / "cut_changes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from bgm.config import load_config
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit",
                                                "paths.results=results/titlesplit", *sys.argv[1:]])
    print(run(cfg))
