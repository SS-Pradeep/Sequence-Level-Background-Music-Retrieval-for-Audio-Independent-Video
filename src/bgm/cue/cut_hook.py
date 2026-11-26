"""D104: the music-change-at-cut classifier (D103) inside the Viterbi stitcher, and its end-to-end evaluation.

At the boundary between segments i - 1 and i (a shot cut at time t), for every candidate window w of segment i - 1
(music file m, offset o, length d), p_w = P(change | video around the cut, the music of w up to o + d - gap). Transition
costs from w:
    continue w seamlessly (same file, contiguous):  base + alpha * -log(1 - p_w)
    anything else (switch file, or jump in a file): base + alpha * -log(p_w)
where base is the usual fixed-penalty transition (lam_switch, lam_jump, mu). p_w uses the CANDIDATE'S OWN music, so
the stitcher can prefer to leave a cue where it sounds finished.

End-to-end test (title split, per fold): held-out clips with >= 2 segments; real changes = the D103 labels of the
segment-boundary cuts (the clip's own music changes there); the stitched soundtrack "changes" at a boundary when the
chosen windows switch file or jump. Penalties / alpha tuned on the fold's VAL clips, test touched once.

    python -m bgm.cue.cut_hook          -> results/titlesplit/cut_viterbi.md, runs/titlesplit/cut_viterbi/
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bgm.cue.cut_model import build_fold, fit_gbm, fit_mlp, inner_groups
from bgm.cue.cuts import GROUPS, music_features, video_features
from bgm.retrieval.stitch_eval import transition_matrix

log = logging.getLogger(__name__)
VIDEO_VECS = ("sig_before", "sig_after", "sig_diff")
MUSIC_VECS = ("mert_before",)


# ---------------------------------------------------------------- a fitted scorer that can score new rows
@dataclass
class Transform:
    scalars: tuple
    vec_keys: tuple
    mu: np.ndarray
    sd: np.ndarray
    pcas: dict
    vsd: dict

    @classmethod
    def fit(cls, rows: pd.DataFrame, vecs: dict, scalars: tuple, vec_keys: tuple, n_pca: int) -> "Transform":
        from sklearn.decomposition import PCA
        x = rows[list(scalars)].to_numpy(float)
        pcas, vsd = {}, {}
        for k in vec_keys:
            pcas[k] = PCA(n_pca, random_state=0).fit(vecs[k])
            vsd[k] = pcas[k].transform(vecs[k]).std(0) + 1e-9
        return cls(tuple(scalars), tuple(vec_keys), x.mean(0), x.std(0) + 1e-9, pcas, vsd)

    def __call__(self, scal: pd.DataFrame, vecs: dict) -> np.ndarray:
        cols = [(scal[list(self.scalars)].to_numpy(float) - self.mu) / self.sd]
        cols += [self.pcas[k].transform(vecs[k]) / self.vsd[k] for k in self.vec_keys]
        return np.concatenate(cols, 1)


class CutScorer:
    """kind: 'mlp' (64-unit MLP, 3 seeds) or 'gbm' (inner-CV HistGradientBoosting); music=False: video features only."""

    def __init__(self, kind: str = "gbm", music: bool = True, n_pca: int = 16):
        self.kind, self.music, self.n_pca = kind, music, n_pca
        self.scalars = GROUPS["video"] + (GROUPS["music"] if music else ())
        self.vec_keys = VIDEO_VECS + (MUSIC_VECS if music else ())

    def fit(self, df: pd.DataFrame, vecs: dict, tr: np.ndarray) -> "CutScorer":
        self.tf = Transform.fit(df[tr], {k: v[tr] for k, v in vecs.items()}, self.scalars, self.vec_keys, self.n_pca)
        X, y = self.tf(df, vecs), df.label.to_numpy()
        if self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier as H
            _, hp = fit_gbm(X, y, tr, inner_groups(df, tr, 3))
            hp.pop("inner_ap")
            self.model = H(class_weight="balanced", l2_regularization=1.0, random_state=0, **hp).fit(X[tr], y[tr])
            self.hp = hp
        else:
            import torch
            self.nets = []
            for s in (0, 1, 2):
                torch.manual_seed(s)
                self.nets.append(_train_mlp(X[tr], y[tr], s))
        return self

    def predict(self, scal: pd.DataFrame, vecs: dict) -> np.ndarray:
        X = self.tf(scal, vecs)
        if self.kind == "gbm":
            return self.model.predict_proba(X)[:, 1]
        import torch
        with torch.no_grad():
            return np.mean([torch.sigmoid(n(torch.tensor(X, dtype=torch.float32))[:, 0]).numpy() for n in self.nets], 0)


def _train_mlp(X, y, seed, epochs: int = 60, hidden: int = 64):
    import torch
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))
    g = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(g.permutation(len(X)), max(1, len(X) // 128)):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b], pos_weight=pw)
            opt.zero_grad(), loss.backward(), opt.step()
    net.eval()
    return net


# ---------------------------------------------------------------- per-boundary, per-candidate probabilities
def boundary_probs(scorer: CutScorer, store, index, layers, seg_bounds: list[tuple[float, float]], cuts_s: np.ndarray, dur_s: float,
                   sig_1hz: np.ndarray, log_motion_25hz: np.ndarray, n_shots: int, music_s: int, gap_s: float) -> list[np.ndarray | None]:
    """P[i] [len(layers[i - 1].mid)]: p(change) at the boundary before segment i for each candidate of segment i - 1
    (None at i = 0, or when the boundary is not a shot cut)."""
    out = [None]
    mus = {}
    for i in range(1, len(layers)):
        t = seg_bounds[i][0]
        j = int(np.argmin(np.abs(cuts_s - t))) if len(cuts_s) else -1
        if j < 0 or abs(cuts_s[j] - t) > 0.05:
            out.append(None)
            continue
        vf = video_features(cuts_s, dur_s, sig_1hz, log_motion_25hz, j, n_shots)
        P = layers[i - 1]
        rows, mv = [], []
        for m, o, d in zip(P.mid, P.off, P.dur):
            mid = index.ids[int(m)]
            if mid not in mus:
                mus[mid] = store.music(mid)
            mf = music_features(mus[mid]["sem_mert_1hz"], mus[mid]["dyn_25hz"], float(o + d) - gap_s, music_s)
            rows.append({**{k: v for k, v in vf.items() if not k.startswith("sig_")}, **{k: v for k, v in mf.items() if k != "mert_before"}})
            mv.append(mf["mert_before"])
        scal = pd.DataFrame(rows)
        n = len(rows)
        vecs = {"sig_before": np.repeat(vf["sig_before"][None], n, 0), "sig_after": np.repeat(vf["sig_after"][None], n, 0),
                "sig_diff": np.repeat(np.abs(vf["sig_after"] - vf["sig_before"])[None], n, 0), "mert_before": np.stack(mv)}
        out.append(np.clip(scorer.predict(scal, vecs), 1e-4, 1 - 1e-4))
    return out


def decode_learned(layers, pen: dict, probs: list, alpha: float, beam: int | None, tol_s: float = 0.04) -> list[int]:
    """decode_lattice with the learned per-candidate change costs added to every transition."""
    act = np.arange(len(layers[0].mid))
    D = layers[0].cost[act].copy()
    acts, backs = [act], []
    for i in range(1, len(layers)):
        L, Pfull = layers[i], layers[i - 1]
        conts = L.cont_of[acts[-1]]
        cur = np.unique(np.r_[np.arange(L.n_top), conts[conts >= 0]])
        Ps, Cs = Pfull.subset(acts[-1]), L.subset(cur)
        T = transition_matrix(Ps, Cs, pen, tol_s)
        if probs[i] is not None and alpha > 0:
            p = probs[i][acts[-1]]
            stay = (Ps.mid[:, None] == Cs.mid[None, :]) & (np.abs(Cs.off[None, :] - (Ps.off + Ps.dur)[:, None]) <= tol_s)
            T = T + alpha * np.where(stay, -np.log(1 - p)[:, None], -np.log(p)[:, None])
        tot = D[:, None] + T
        bk = tot.argmin(0)
        Dc = tot.min(0) + L.cost[cur]
        if beam is not None and len(cur) > beam:
            keep = np.sort(np.argsort(Dc)[:beam])
            cur, Dc, bk = cur[keep], Dc[keep], bk[keep]
        acts.append(cur), backs.append(bk)
        D = Dc
    j = int(np.argmin(D))
    path = [j]
    for bk in reversed(backs):
        j = int(bk[j])
        path.append(j)
    path = path[::-1]
    return [int(acts[i][path[i]]) for i in range(len(layers))]


def path_changes(layers, picks: list[int], tol_s: float = 0.04) -> np.ndarray:
    """[n - 1] 1 where the soundtrack changes between segments i - 1 and i (other file, or a jump in the file)."""
    out = []
    for i in range(1, len(layers)):
        a, b = layers[i - 1], layers[i]
        ja, jb = picks[i - 1], picks[i]
        cont = a.mid[ja] == b.mid[jb] and abs(b.off[jb] - (a.off[ja] + a.dur[ja])) <= tol_s
        out.append(0 if cont else 1)
    return np.array(out, int)


# ---------------------------------------------------------------- end-to-end evaluation
@dataclass
class ClipCase:
    pair: str
    split: str
    layers: list
    labels: np.ndarray          # [n - 1] 1 / 0, -1 = boundary not measurable
    probs: dict                 # scorer name -> per-boundary candidate probabilities
    tracks: list                # true track per segment (the clip's own)
    chosen_track: callable = None


def clip_cases(cfg, store, fold: int, df: pd.DataFrame, scorers: dict, index, models, splits=("val", "test")) -> list[ClipCase]:
    from bgm.retrieval.index import pair_query
    from bgm.retrieval.pipeline import run_video
    from bgm.retrieval.stitch_eval import build_lattice
    c = cfg.cue.cuts
    lab = {(r.pair, round(r.t, 2)): int(r.label) for r in df.itertuples()}
    cases = []
    for s in splits:
        for p in sorted(set(df.pair[df.split == s])):
            segs = list(store.segments[p])
            if len(segs) < 2:
                continue
            vq = pair_query(store, p)
            vq.prepare(models, store.min_span)
            res = run_video(index, vq, {"sem": 1.0, "attr": 0.0}, int(cfg.retrieval.coarse_top_k), float(cfg.retrieval.window_hop_s))
            layers = build_lattice(index, vq, res, {"sem": 1.0, "dyn": 0.0, "attr": 0.0}, int(cfg.retrieval.fine_top_j))
            v = store.video(p)
            cuts = np.asarray(v["cut_times_s"], float)
            labels = np.array([lab.get((p, round(float(cuts[np.argmin(np.abs(cuts - a))]) if len(cuts) else -1, 2)), -1)
                               if len(cuts) and np.min(np.abs(cuts - a)) <= 0.05 else -1 for a, _ in segs[1:]])
            probs = {n: boundary_probs(sc, store, index, layers, segs, cuts, store.duration("video", p), v["sem_siglip_1hz"],
                                       v["dyn_25hz"][:, 0], c.n_shots, c.music_s, c.tol_s) for n, sc in scorers.items()}
            cases.append(ClipCase(p, s, layers, labels, probs, [store.track[p]] * len(segs)))
    return cases


def score(cases: list[ClipCase], picks_of, index, store) -> dict:
    tp = fp = fn = 0
    n_change = n_bound = 0
    track_ok = track_n = 0
    for c in cases:
        picks = picks_of(c)
        ch = path_changes(c.layers, picks)
        m = c.labels >= 0
        tp += int(((ch == 1) & (c.labels == 1) & m).sum())
        fp += int(((ch == 1) & (c.labels == 0) & m).sum())
        fn += int(((ch == 0) & (c.labels == 1) & m).sum())
        n_change += int(ch.sum())
        n_bound += len(ch)
        for L, j, t in zip(c.layers, picks, c.tracks):
            track_ok += int(store.music_track[index.ids[int(L.mid[j])]] == t)
            track_n += 1
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"f1": f1, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn,
            "change_rate": n_change / max(n_bound, 1), "track_acc": track_ok / max(track_n, 1)}


PEN_GRID = [{"lam_switch": ls, "lam_jump": lj, "mu": 0.0} for ls in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for lj in (0.25, 1.0)]
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def tune(val: list[ClipCase], decode, grid: list[dict], index, store) -> tuple[dict, float]:
    best, arg = -1.0, grid[0]
    for g in grid:
        f = score(val, lambda c: decode(c, g), index, store)["f1"]
        if f > best:
            best, arg = f, g
    return arg, best


def run(cfg, B: int = 2000) -> Path:
    import torch
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.retrieval.index import FoldModels, MusicIndex
    store = FeatureStore(cfg)
    group = manifest.read_folds(cfg)["film_group"]
    beam = int(cfg.stitch.beam_factor) * int(cfg.retrieval.fine_top_j)    # as tune_stitch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg.paths.runs) / "cut_viterbi"
    root.mkdir(parents=True, exist_ok=True)
    rows, tuned = [], {}
    for k in range(cfg.splits.n_folds):
        df, vecs, _ = build_fold(cfg, store, k)
        df["group"] = df.pair.map(group)
        tr = (df.split == "train").to_numpy()
        scorers = {"GBM video + music": CutScorer("gbm", True).fit(df, vecs, tr),
                   "MLP video + music": CutScorer("mlp", True).fit(df, vecs, tr),
                   "GBM video only": CutScorer("gbm", False).fit(df, vecs, tr)}
        f = fold_ids(cfg, k)
        models = FoldModels.load(cfg, store, k, 0, cfg.pipeline.sem_variant, None, dev, train_ids=f["train"])
        index = MusicIndex(store, models, list(dict.fromkeys(store.pairs.music_id)))
        cases = clip_cases(cfg, store, k, df, scorers, index, models)
        val, test = [c for c in cases if c.split == "val"], [c for c in cases if c.split == "test"]
        from bgm.retrieval.stitch_eval import decode_lattice
        fixed = lambda c, g: decode_lattice(c.layers, g, beam)
        pen, vf = tune(val, fixed, PEN_GRID, index, store)
        systems = {"fixed penalties (tuned on val clips)": lambda c, pen=pen: fixed(c, pen)}
        tuned[k] = {"fixed": {**pen, "val_f1": vf}}
        for name in scorers:
            grid = [{**g, "alpha": a} for g in PEN_GRID for a in ALPHAS]
            dec = lambda c, g, name=name: decode_learned(c.layers, {x: g[x] for x in ("lam_switch", "lam_jump", "mu")}, c.probs[name], g["alpha"], beam)
            g, vf2 = tune(val, dec, grid, index, store)
            systems[f"learned: {name}"] = lambda c, g=g, dec=dec: dec(c, g)
            tuned[k][name] = {**g, "val_f1": vf2}
        for name, dec in systems.items():
            for c in test:
                sc = score([c], dec, index, store)
                rows.append({"fold": k, "system": name, "pair": c.pair, "group": group[c.pair], **sc})
        log.info("fold %d: %d val clips, %d test clips; tuned %s", k, len(val), len(test), tuned[k])
    R = pd.DataFrame(rows)
    R.to_csv(root / "test_rows.csv", index=False)
    (root / "tuned.json").write_text(json.dumps(tuned, indent=1, default=float))
    return write_report(cfg, R, B)


def write_report(cfg, R: pd.DataFrame, B: int) -> Path:
    def agg(d):
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "change_rate": d.change_rate.mean(), "track_acc": d.track_acc.mean()}
    base = "fixed penalties (tuned on val clips)"
    rng = np.random.default_rng(0)
    groups = R.group.unique()
    L = ["# Learned change costs in the Viterbi stitcher (D104)", "",
         "Title split, 5 folds. Held-out clips with >= 2 segments; true changes = the clip's own music changes at segment-boundary cuts "
         "(D103 labels). The soundtrack changes at a boundary when the chosen windows switch file or jump. Penalties, alpha and the "
         "scorers' thresholds-free costs tuned on each fold's VAL clips; retrieval = the title-split 10-head ensemble (seeds 0–9), "
         f"semantic only. CIs: paired cluster bootstrap over film groups, {B:,} resamples.", "",
         "| system | boundary F1 | precision | recall | soundtrack changes per boundary | segments on the clip's own track | Δ F1 vs fixed [95% CI] |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in R.system.unique():
        d = R[R.system == name]
        a = agg(d)
        ci = "—"
        if name != base:
            x = d.set_index("pair")
            y = R[R.system == base].set_index("pair")
            d0 = a["f1"] - agg(y)["f1"]
            ds = []
            for _ in range(B):
                gs = rng.choice(groups, len(groups))
                sel = np.concatenate([x.index[x.group == g].to_numpy() for g in gs])
                ds.append(agg(x.loc[sel])["f1"] - agg(y.loc[sel])["f1"])
            ci = f"{d0:+.3f} [{np.quantile(ds, .025):+.3f}, {np.quantile(ds, .975):+.3f}]"
        L.append(f"| {name} | {a['f1']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} | {a['change_rate']:.3f} | {a['track_acc']:.3f} | {ci} |")
    n_pairs = R[R.system == base].pair.nunique()
    L += ["", f"Test clips: {n_pairs}; scored boundaries with a change: {int(R[R.system == base][['tp', 'fn']].sum().sum())}."]
    out = Path(cfg.paths.results) / "cut_viterbi.md"
    out.write_text("\n".join(L) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (Path(cfg.paths.runs) / "cut_viterbi" / "metrics.json").write_text(json.dumps({"commit": commit, "n_test_clips": int(n_pairs)}, indent=1))
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from bgm.config import load_config
    cfg = load_config("configs/default.yaml", ["splits.folds_file=folds_title.json", "paths.runs=runs/titlesplit", "paths.results=results/titlesplit",
                                                "pipeline.dyn_variant=null", "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]", *sys.argv[1:]])
    print(run(cfg))


# ---------------------------------------------------------------- render hook
def fitted_scorer(cfg, fold: int, kind: str, music: bool) -> CutScorer:
    """The fold's scorer, trained on the TITLE split's train clips; cached in runs/titlesplit/cut_viterbi/."""
    import pickle
    from bgm.config import load_config
    from bgm.data import manifest
    from bgm.eval.store import FeatureStore
    cc = cfg.cue.cuts_render
    path = Path(cc.runs) / "cut_viterbi" / f"scorer_{kind}_{'vm' if music else 'v'}_fold{fold}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    tcfg = load_config("configs/default.yaml", [f"splits.folds_file={cc.folds_file}", f"paths.runs={cc.runs}"])
    store = FeatureStore(tcfg)
    df, vecs, _ = build_fold(tcfg, store, fold)
    df["group"] = df.pair.map(manifest.read_folds(tcfg)["film_group"])
    sc = CutScorer(kind, music).fit(df, vecs, (df.split == "train").to_numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(sc))
    return sc


def cut_decode(cfg, store, index, layers, feat: dict, results, fold: int, beam: int) -> tuple[list[int], dict]:
    """Render hook (`cue.cuts_render.enabled`): Viterbi with learned per-candidate change costs."""
    cc, c = cfg.cue.cuts_render, cfg.cue.cuts
    sc = fitted_scorer(cfg, fold, cc.kind, bool(cc.music))
    segs = [(r.a, r.b) for r in results]
    cuts = np.asarray(feat["cut_times_s"], float)
    dur = len(feat["dyn_25hz"]) / 25.0
    probs = boundary_probs(sc, store, index, layers, segs, cuts, dur, feat["sem_siglip_1hz"].astype(np.float32),
                           np.asarray(feat["dyn_25hz"])[:, 0], c.n_shots, c.music_s, c.tol_s)
    pen = {"lam_switch": float(cc.lam_switch), "lam_jump": float(cc.lam_jump), "mu": float(cc.mu)}
    picks = decode_learned(layers, pen, probs, float(cc.alpha), beam)
    chosen_p = [None] + [float(p[picks[i - 1]]) if p is not None else None for i, p in enumerate(probs) if i > 0]
    return picks, {"kind": cc.kind, "music": bool(cc.music), "fold": fold, "penalties": pen, "alpha": float(cc.alpha),
                   "p_change_of_chosen": chosen_p}
