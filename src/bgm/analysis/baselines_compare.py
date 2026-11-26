"""All baselines and the final system under one evaluator on the cleaned title-split test (D120).

Each method's test-fold score matrix S[video, music] (rows and columns = the fold's test pairs; every pair has its own
music file) is rebuilt from what its project saved, then scored by the same code:

  final       10-head siglip-both ensembles (runs/cleaned/semantic, 3 disjoint ens10 per fold), MeanScorer.track
  imagebind   ImageBind zero-shot (whole-clip 1 Hz means, as D118)
  languagebind  LanguageBind zero-shot (whole-clip vectors of ~/VM-NET/cbvmr_cleaned/features)
  mmtrail     CoCa frame captions -> CLAP text vs CLAP music (~/MMTrail/mmtrail_cleaned/out embeddings)
  vmnet       VM-NET (CBVMR), retrained with its own run.py (5 folds x 3 seeds; ranks checked against the saved runs)
  emomv_*     EmoMV retrieval models, single- and multi-task (~/EmoMV/DS4_cleaned/fold*/models), as ds4_tools/recall.py
  deccm*      DecCM student and its teacher (~/DecCM/mydata/runs/fold*/exp/<tag>), scored on the full test gallery
              instead of DecCM's 1 positive + 100 sampled negatives; tag = best mean validation score of the grid

Per query: track and exact-pair ranks in both directions (track hit = any file of the query's track_id), track
pairwise accuracy (own file vs every other-track file, ties 1/2), and the ImageBind / LanguageBind cosine between the
query video and its top-1 music (whole-clip vectors of the VM-NET features, the common judge). Per-query values are
averaged over a fold's runs; CIs are film-group cluster bootstraps.

Run: python -m bgm.analysis.baselines_compare [dump|report|all]  -> results/cleaned/baselines.md
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HOME = Path.home()
VMNET = HOME / "VM-NET" / "cbvmr_cleaned"
MMTRAIL = HOME / "MMTrail" / "mmtrail_cleaned" / "out"
EMOMV = HOME / "EmoMV"
DECCM = HOME / "DecCM"
MANIFESTS = Path("cleaned_dataset/manifests").resolve()
HEADS = Path("runs/cleaned/semantic/siglip-both").resolve()
OUT_RUNS = Path("runs/cleaned/baselines")
OUT = Path("results/cleaned/baselines.md")
FOLDS, ENS_SIZE, ENS_RUNS, VMNET_SEEDS = range(5), 10, (0, 1, 2), (0, 1, 2)
KS = (1, 5, 10)
NAMES = {"final": "**Final system (10-head ensemble)**", "imagebind": "ImageBind zero-shot",
         "languagebind": "LanguageBind zero-shot", "deccm_teacher": "DecCM teacher (no deconfounding)",
         "deccm": "DecCM", "vmnet": "VM-NET (CBVMR)", "mmtrail": "MMTrail captions → CLAP",
         "emomv_single": "EmoMV, single-task", "emomv_multi": "EmoMV, multi-task"}
TRAINED = {"final", "vmnet", "emomv_single", "emomv_multi", "deccm", "deccm_teacher"}


# ---------------------------------------------------------------- metrics (pure)
def first_hit_ranks(S: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """S, hit: [queries, gallery]. 1-based rank of the best-scored hit: 1 + non-hits scored above it
    (bgm.eval.metrics.rank_of_first_hit, row-wise)."""
    best = np.where(hit, S, -np.inf).max(1)
    return 1 + ((S > best[:, None]) & ~hit).sum(1)


def pairwise_rows(S: np.ndarray, mismatch: np.ndarray) -> np.ndarray:
    """Per row: share of mismatched items scored below the diagonal item, ties 1/2. S, mismatch: [n, n]."""
    own = np.diag(S)[:, None]
    win = (S < own) + 0.5 * (S == own)
    return (win * mismatch).sum(1) / mismatch.sum(1)


def evaluate_matrix(S: np.ndarray, track: np.ndarray, IB: np.ndarray, LB: np.ndarray) -> dict:
    """S [n, n]: score of test video i with the music of test pair j; track [n]; IB, LB [n, n] judge cosines.
    Returns per-query arrays."""
    same = track[:, None] == track[None, :]
    eye = np.eye(len(S), dtype=bool)
    top1 = S.argmax(1)
    rows = np.arange(len(S))
    out = {"pairwise": pairwise_rows(S, ~same), "IB@1": IB[rows, top1], "LB@1": LB[rows, top1]}
    for d, M, H, E in (("v2m", S, same, eye), ("m2v", S.T, same.T, eye)):
        out[f"{d}/track_rank"] = first_hit_ranks(M, H)
        out[f"{d}/exact_rank"] = first_hit_ranks(M, E)
    return out


def reference_rows(track: np.ndarray, IB: np.ndarray, LB: np.ndarray) -> dict:
    """Chance (exact expectation per query), the original music and a random gallery file."""
    from bgm.eval.metrics import chance_recall
    n = len(track)
    h = (track[:, None] == track[None, :]).sum(1)
    out = {f"chance/track_R@{k}": np.array([chance_recall(n, hi, k) for hi in h]) for k in KS}
    out.update({f"chance/exact_R@{k}": np.full(n, min(k / n, 1.0)) for k in KS})
    out["chance/track_rank"] = (n + 1) / (h + 1)
    for tag, M in (("IB", IB), ("LB", LB)):
        out[f"original/{tag}"] = np.diag(M).copy()
        out[f"random/{tag}"] = M.mean(1)
    return out


# ---------------------------------------------------------------- score matrices from each project
def _folds() -> dict:
    return json.loads((MANIFESTS / "folds_title.json").read_text())


def _test_ids(k: int) -> list[str]:
    return list(_folds()["folds"][str(k)]["test"])


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _whole(kind: str) -> tuple[dict, np.ndarray, np.ndarray]:
    ids = (VMNET / "features" / "ids.txt").read_text().splitlines()
    z = np.load(VMNET / "features" / f"{kind}_whole.npz")
    return {p: i for i, p in enumerate(ids)}, z["video"], z["music"]


def scores_final(device: str = "cuda"):
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.analysis.mema_protocols import ImageBindScorer, make_cfg
    from bgm.eval.store import FeatureStore, fold_ids
    from bgm.person.evaluate import MeanScorer
    cfg = make_cfg("clean")
    store = FeatureStore(cfg)
    final, ib = {}, {}
    for k in FOLDS:
        test = fold_ids(cfg, k)["test"]
        assert sorted(test) == sorted(_test_ids(k))
        gal = [store.music_of[p] for p in test]
        assert len(set(gal)) == len(gal), "a music file is shared by two test pairs"
        ib[k, 0] = (test, np.stack([ImageBindScorer(store).track(q, gal) for q in test]))
        for e in ENS_RUNS:
            scorer = MeanScorer([load_scorer(HEADS / f"fold{k}" / f"seed{m}", store, device)
                                 for m in range(ENS_SIZE * e, ENS_SIZE * (e + 1))])
            with torch.no_grad():
                S = np.stack([np.asarray(scorer.track(q, gal), float) for q in test])
            with np.load(Path("runs/mema/clean/ens10") / f"fold{k}" / f"seed{e}" / "perquery.npz") as z:
                assert list(z["track_q"]) == test
                track = np.array([store.track[p] for p in test])
                assert np.array_equal(first_hit_ranks(S, track[:, None] == track[None, :]), z["track_rank"]), (k, e)
            final[k, e] = (test, S)
            torch.cuda.empty_cache()
        log.info("final: fold %d", k)
    return {"final": final, "imagebind": ib}


def scores_languagebind():
    pos, V, A = _whole("languagebind")
    out = {}
    for k in FOLDS:
        t = _test_ids(k)
        i = [pos[p] for p in t]
        out[k, 0] = (t, V[i] @ A[i].T)
    return out


def scores_mmtrail():
    import pandas as pd
    ids = pd.read_csv(MANIFESTS / "pairs.csv").pair_id.tolist()
    pos = {p: i for i, p in enumerate(ids)}
    V, A = np.load(MMTRAIL / "clap_video_from_captions.npy"), np.load(MMTRAIL / "clap_music.npy")
    out = {}
    for k in FOLDS:
        t = _test_ids(k)
        i = [pos[p] for p in t]
        out[k, 0] = (t, V[i] @ A[i].T)
    return out


def scores_vmnet():
    """Retrain with VM-NET's own train_one (seeded); the test ranks must match its saved perquery.npz."""
    sys.path.insert(0, str(VMNET))
    run = _load_module("vmnet_run", VMNET / "run.py")
    D = run.Data()
    out, same = {}, []
    for k in FOLDS:
        X, Y = D.fold_inputs(k)
        te = D.folds[k]["test"]
        t = [D.ids[i] for i in te]
        for s in VMNET_SEEDS:
            S, _, _ = run.train_one(D, k, s, X, Y)
            with np.load(VMNET / "results" / f"fold{k}" / f"seed{s}" / "perquery.npz") as z:
                assert list(z["pair_id"]) == t
                tr = D.track[te]
                same.append(float((first_hit_ranks(S, tr[:, None] == tr[None, :]) == z["vmnet/v2m/rank"]).mean()))
            out[k, s] = (t, S)
        log.info("vmnet: fold %d", k)
    log.info("vmnet retrain reproduces saved v2m ranks for %.1f%% of queries", 100 * np.mean(same))
    return out, float(np.mean(same))


def scores_emomv(variant: str):
    """Run _emomv_matrices in EmoMV's own environment (it has h5py; this one does not) and read its output."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "S.npz"
        subprocess.run([str(EMOMV / ".venv" / "bin" / "python"), "-c",
                        f"from bgm.analysis.baselines_compare import _emomv_matrices; _emomv_matrices({variant!r}, {str(f)!r})"],
                       check=True, env={**os.environ, "PYTHONPATH": str(Path("src").resolve())})
        z = np.load(f)
        return {(k, 0): (list(z[f"{k}_ids"]), z[f"{k}_S"]) for k in FOLDS}


def _emomv_matrices(variant: str, out_path: str) -> None:
    """ds4_tools/recall.py's retrieval(): StandardScaler fit on the training features, projection heads, cosine."""
    import h5py
    import torch
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(EMOMV))
    mod = importlib.import_module(f"DS1_model_retrieval_{variant}")

    def h5(p):
        with h5py.File(p, "r") as h:
            return np.asarray(h[list(h.keys())[0]], dtype=np.float32)

    out = {}
    for k in FOLDS:
        d = EMOMV / "DS4_cleaned" / f"fold{k}"
        ids = [r[1] for r in csv.reader(open(d / "for_retrieval" / "DS4_TEST_labels_for_retrieval.csv"))]
        V = h5(d / "for_retrieval" / "SlowFast_DS4_TEST_for_retrieval.h5")
        A = h5(d / "for_retrieval" / "VGGish_DS4_TEST_for_retrieval.h5")
        V = StandardScaler().fit(h5(d / "extracted_features" / "SlowFast_DS4_TRAIN_MATCH_MISMATCH.h5")).transform(V)
        A = StandardScaler().fit(h5(d / "extracted_features" / "VGGish_DS4_TRAIN_MATCH_MISMATCH.h5")).transform(A)
        m = mod.embedding_network()
        m.load_state_dict(torch.load(d / "models" / f"_DS1_model_retrieval_{variant}.pth", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            ve = torch.nn.functional.normalize(m.video_projection(m.video_br(torch.from_numpy(V).float())), dim=1)
            ae = torch.nn.functional.normalize(m.audio_projection(m.audio_br(torch.from_numpy(A).float())), dim=1)
        assert sorted(ids) == sorted(_test_ids(k)), k
        out[f"{k}_ids"], out[f"{k}_S"] = np.array(ids), (ve @ ae.T).numpy()
    np.savez(out_path, **out)


def deccm_tag() -> str:
    """Grid configuration with the best mean validation score (DecCM's own selection metric); ties -> first in
    grid.sh order (lr 1e-4 before 1e-3, then epochs 50, 200, 500)."""
    order = [f"t{lr}_e{ep}" for lr in ("1e-4", "1e-3") for ep in (50, 200, 500)]
    val = {}
    for t in order:
        r = json.loads((DECCM / "mydata" / "runs" / f"results_{t}.json").read_text())
        val[t] = np.mean([f["val IPS R@25 (selection)"] for f in r.values()])
    return max(order, key=lambda t: (val[t], -order.index(t)))


def scores_deccm(device: str = "cuda"):
    """report.py's scoring (student: v_mu . linear([confounder, m_mu]); teacher: v_mu . m_mu), full test gallery."""
    import pandas as pd
    import torch
    sys.path.insert(0, str(DECCM))
    from Model import s_net
    from Model_teacher import t_net
    tag = deccm_tag()
    ids = pd.read_csv(DECCM / "cleaned_dataset" / "pairs.csv").pair_id.tolist()
    pos = {p: i for i, p in enumerate(ids)}
    student, teacher = {}, {}
    cwd = os.getcwd()
    for k in FOLDS:
        fold_dir = DECCM / "mydata" / "runs" / f"fold{k}"
        os.chdir(fold_dir)                                   # s_net loads model/pgc/net_params.pkl from the CWD
        try:
            p = "data_matching/"
            v_feat = np.load(p + "visual_features_resnet_V.npy")
            m_feat = np.concatenate([np.load(p + "music_features.npy"), np.load(p + "new_music_smile_features_norm.npy")], 1)
            prior = np.load(p + "user_prior_mean.npy")
            test = np.load(p + "100neg/test.npy", allow_pickle=True)
            t = _test_ids(k)
            i = np.array([pos[q] for q in t])
            assert set(test[:, 0].astype(int)) == set(i) and (test[:, 0].astype(int) == test[:, 1].astype(int)).all()
            s = s_net(device, 256, 2048, 100, 128, 6, 0.2, prior).to(device)
            s.load_state_dict(torch.load(fold_dir / "exp" / tag / "student.pkl", map_location=device))
            s.eval()
            tn = t_net(device, 256, 2048, 768, 128, 0.2).to(device)
            tn.load_state_dict(torch.load(fold_dir / "exp" / tag / "teacher.pkl", map_location=device))
            tn.eval()
            with torch.no_grad():
                mf = torch.tensor(m_feat[i], dtype=torch.float32, device=device)
                vf = torch.tensor(v_feat[i], dtype=torch.float32, device=device)
                m_mu, _ = s.m_encoder(mf)
                v_mu, _ = s.v_encoder(vf)
                cp = torch.tensor(prior, dtype=torch.float32, device=device).unsqueeze(-1)
                conf = torch.sum(s.confounder_emb(torch.arange(6, device=device).repeat(len(m_mu), 1)) * cp, 1)
                student[k, 0] = (t, v_mu.mm(s.linear(torch.cat([conf, m_mu], -1)).t()).cpu().numpy())
                tm, _ = tn.m_encoder(mf)
                tv, _ = tn.v_encoder(vf)
                teacher[k, 0] = (t, tv.mm(tm.t()).cpu().numpy())
        finally:
            os.chdir(cwd)
    return {"deccm": student, "deccm_teacher": teacher}, tag


def _canonical(t: list[str], S: np.ndarray, k: int) -> np.ndarray:
    """Reorder rows and columns to the fold's folds_title.json test order."""
    ref = _test_ids(k)
    idx = [t.index(p) for p in ref]
    return S[np.ix_(idx, idx)]


def dump(device: str = "cuda") -> None:
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    notes = {}
    jobs = {"languagebind": scores_languagebind, "mmtrail": scores_mmtrail,
            "emomv_single": lambda: scores_emomv("single_task"), "emomv_multi": lambda: scores_emomv("multi_task")}
    allS = {m: f() for m, f in jobs.items()}
    allS.update(scores_final(device))
    allS["vmnet"], notes["vmnet_reproduced"] = scores_vmnet()
    d, notes["deccm_tag"] = scores_deccm(device)
    allS.update(d)
    for m, runs in allS.items():
        np.savez_compressed(OUT_RUNS / f"{m}.npz", **{f"{k}_{r}": _canonical(t, S, k) for (k, r), (t, S) in runs.items()})
    (OUT_RUNS / "notes.json").write_text(json.dumps(notes, indent=1))


# ---------------------------------------------------------------- report
def _fmt(b: dict, nd: int = 3, sign: bool = False) -> str:
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(b['mean'])} [{f.format(b['lo'])}, {f.format(b['hi'])}]"


def report() -> Path:
    import pandas as pd
    from bgm.eval.metrics import cluster_bootstrap
    folds = _folds()
    film = folds["film_group"]
    track_of = dict(zip(*pd.read_csv(MANIFESTS / "pairs.csv")[["pair_id", "track_id"]].T.values))
    ipos, iv, im = _whole("imagebind")
    lpos, lv, lm = _whole("languagebind")
    notes = json.loads((OUT_RUNS / "notes.json").read_text())

    # per method: per-query values averaged over a fold's runs, all folds concatenated; per-run summaries
    pooled, per_run, refs, clusters, order = {}, {}, {}, [], []
    for k in FOLDS:
        t = _test_ids(k)
        tr = np.array([track_of[p] for p in t])
        IB = iv[[ipos[p] for p in t]] @ im[[ipos[p] for p in t]].T
        LB = lv[[lpos[p] for p in t]] @ lm[[lpos[p] for p in t]].T
        for key, v in reference_rows(tr, IB, LB).items():
            refs.setdefault(key, []).append(v)
        clusters += [film[p] for p in t]
        order += t
        for m in NAMES:
            with np.load(OUT_RUNS / f"{m}.npz") as z:
                runs = [evaluate_matrix(z[n], tr, IB, LB) for n in sorted(z.files) if n.startswith(f"{k}_")]
            for r in runs:
                summ = {f"{d}/{h}_R@{kk}": float((r[f"{d}/{h}_rank"] <= kk).mean())
                        for d in ("v2m", "m2v") for h in ("track", "exact") for kk in KS}
                summ.update({f"{d}/MedR": float(np.median(r[f"{d}/track_rank"])) for d in ("v2m", "m2v")})
                per_run.setdefault(m, []).append(summ)
            for key in runs[0]:
                vals = np.mean([r[key].astype(float) for r in runs], 0)
                pooled.setdefault(m, {}).setdefault(key, []).append(vals)
    clusters = np.array(clusters)
    P = {m: {key: np.concatenate(v) for key, v in d.items()} for m, d in pooled.items()}
    R = {key: np.concatenate(v) for key, v in refs.items()}
    n_q = len(order)

    def _hit(m, d, h, kk):  # per-query hit rate averaged over a fold's runs
        return np.concatenate(pooled_hits[m][d, h, kk])

    pooled_hits = {}
    for k in FOLDS:
        t = _test_ids(k)
        tr = np.array([track_of[p] for p in t])
        same = tr[:, None] == tr[None, :]
        eye = np.eye(len(t), dtype=bool)
        for m in NAMES:
            with np.load(OUT_RUNS / f"{m}.npz") as z:
                mats = [z[n] for n in sorted(z.files) if n.startswith(f"{k}_")]
            for d in ("v2m", "m2v"):
                for h, H in (("track", same), ("exact", eye)):
                    ranks = [first_hit_ranks(S if d == "v2m" else S.T, H) for S in mats]
                    for kk in KS:
                        pooled_hits.setdefault(m, {}).setdefault((d, h, kk), []).append(
                            np.mean([(r <= kk).astype(float) for r in ranks], 0))
            pooled_hits[m].setdefault(("mrr",), []).append(
                np.mean([1.0 / first_hit_ranks(S, same) for S in mats], 0))

    def ms(m, key):
        v = [r[key] for r in per_run[m]]
        return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

    def cb(v, nd=3, sign=False):
        return _fmt(cluster_bootstrap(v, clusters), nd, sign)

    n_runs = {m: len(per_run[m]) for m in NAMES}
    ch = {kk: R[f"chance/track_R@{kk}"].mean() for kk in KS}
    out = ["# All methods under one evaluator (cleaned title-split test, D120)", "",
           f"{n_q} test videos, 5 title folds; gallery = the fold's test music files ({n_q / 5:.0f} on average, one "
           "per test pair). Every method's test score matrix is rebuilt from its own saved models or embeddings and "
           "scored by src/bgm/analysis/baselines_compare.py. [95% CI] = film-group cluster bootstrap over per-video "
           "values averaged over a fold's runs; ± = std over runs.", "",
           "Runs per method: " + ", ".join(f"{NAMES[m].strip('*')} {n_runs[m]}" for m in NAMES)
           + f". VM-NET retrained with its own seeded code reproduces the saved v2m ranks for "
             f"{100 * notes['vmnet_reproduced']:.1f}% of queries. DecCM grid configuration (best mean validation "
             f"score): {notes['deccm_tag']}; scored on the full test gallery, not 1 positive + 100 sampled negatives.",
           "", "## Video → music retrieval (track: any file of the right song counts)", "",
           "| method | trained | R@1 | R@5 | R@10 [95% CI] | median rank | MRR | pairwise acc. [95% CI] |",
           "|---|---|---|---|---|---|---|---|"]
    rows = sorted(NAMES, key=lambda m: -np.mean(_hit(m, "v2m", "track", 10)))
    for m in rows:
        out.append(f"| {NAMES[m]} | {'yes' if m in TRAINED else 'no'} | {ms(m, 'v2m/track_R@1')} | "
                   f"{ms(m, 'v2m/track_R@5')} | {cb(_hit(m, 'v2m', 'track', 10))} | "
                   f"{np.mean([r['v2m/MedR'] for r in per_run[m]]):.1f} | {np.concatenate(pooled_hits[m][('mrr',)]).mean():.3f} | "
                   f"{cb(P[m]['pairwise'])} |")
    out.append(f"| chance | – | {ch[1]:.3f} | {ch[5]:.3f} | {ch[10]:.3f} | {np.median(R['chance/track_rank']):.1f} | – | 0.500 |")

    out += ["", "## Both directions and the exact-pair rule", "",
            "Exact pair: only the video's own music file (or, music → video, the music's own video) counts.", "",
            "| method | v2m track R@10 | m2v track R@10 | v2m exact R@10 | m2v exact R@10 | m2v median rank |",
            "|---|---|---|---|---|---|"]
    for m in rows:
        out.append(f"| {NAMES[m]} | {ms(m, 'v2m/track_R@10')} | {ms(m, 'm2v/track_R@10')} | "
                   f"{ms(m, 'v2m/exact_R@10')} | {ms(m, 'm2v/exact_R@10')} | "
                   f"{np.mean([r['m2v/MedR'] for r in per_run[m]]):.1f} |")
    out.append(f"| chance | {ch[10]:.3f} | {ch[10]:.3f} | {R['chance/exact_R@10'].mean():.3f} | "
               f"{R['chance/exact_R@10'].mean():.3f} | {np.median(R['chance/track_rank']):.1f} |")

    out += ["", "## How well the top-1 pick fits the video (common judge)", "",
            "Cosine between the query video and its top-1 music file, whole-clip ImageBind and LanguageBind vectors. "
            "ImageBind / LanguageBind zero-shot pick the gallery file their own judge scores highest, so their rows "
            "in their own column are an upper bound, not a competitor.", "",
            "| music used | ImageBind score [95% CI] | LanguageBind score [95% CI] |", "|---|---|---|",
            f"| original music | {cb(R['original/IB'], 4)} | {cb(R['original/LB'], 4)} |"]
    for m in sorted(NAMES, key=lambda m: -P[m]["IB@1"].mean()):
        star = {"imagebind": (" (upper bound)", ""), "languagebind": ("", " (upper bound)")}.get(m, ("", ""))
        out.append(f"| {NAMES[m]} top-1 | {cb(P[m]['IB@1'], 4)}{star[0]} | {cb(P[m]['LB@1'], 4)}{star[1]} |")
    out.append(f"| random gallery file (expected) | {cb(R['random/IB'], 4)} | {cb(R['random/LB'], 4)} |")

    out += ["", "## Paired differences: final system − each method (same test videos)", "",
            "| method | Δ track R@10 | Δ pairwise acc. | Δ ImageBind score | Δ LanguageBind score |",
            "|---|---|---|---|---|"]
    for m in rows:
        if m == "final":
            continue
        out.append(f"| {NAMES[m]} | {cb(_hit('final', 'v2m', 'track', 10) - _hit(m, 'v2m', 'track', 10), 3, True)} | "
                   f"{cb(P['final']['pairwise'] - P[m]['pairwise'], 3, True)} | "
                   f"{cb(P['final']['IB@1'] - P[m]['IB@1'], 4, True)} | {cb(P['final']['LB@1'] - P[m]['LB@1'], 4, True)} |")
    out += ["", "Each method − random file: "
            + "; ".join(f"{NAMES[m].strip('*')} IB {cb(P[m]['IB@1'] - R['random/IB'], 4, True)}, "
                        f"LB {cb(P[m]['LB@1'] - R['random/LB'], 4, True)}" for m in rows) + "."]

    out += ["", "## Per fold: video → music track R@10 (mean over the fold's runs)", "",
            "| method | " + " | ".join(f"fold {k}" for k in FOLDS) + " |", "|---|" + "---|" * len(FOLDS)]
    for m in rows + ["chance"]:
        if m == "chance":
            v = np.split(R["chance/track_R@10"], np.cumsum([len(_test_ids(k)) for k in FOLDS])[:-1])
        else:
            v = pooled_hits[m]["v2m", "track", 10]
        out.append(f"| {NAMES.get(m, m)} | " + " | ".join(f"{x.mean():.3f}" for x in v) + " |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    log.info("wrote %s", OUT)
    return OUT


def main(what: str = "all") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if what in ("dump", "all"):
        dump()
    if what in ("report", "all"):
        report()


if __name__ == "__main__":
    main(*sys.argv[1:])
