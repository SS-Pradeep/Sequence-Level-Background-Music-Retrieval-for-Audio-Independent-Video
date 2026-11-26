"""Step 3: evaluate one system on fold x seed -> runs/filmsplit/person/eval/<system>/fold<k>/seed<s>/.

Run: python -m bgm.person.run --system P2 [--folds 0,1,2,3,4] [--seeds 0,1,2]

Per run: fusion.json (w), stitch.json (penalties), perquery.npz (retrieval ranks, pipeline per-segment hits,
stitching per concatenation / natural video), metrics.json, config.yaml, git commit, seed (CLAUDE.md rule 10).
Late fusion: score = z(s_frame) + w z(s_branch), w per (fold, seed) on that fold's val films only (user decision
2026-10-20): grid W_GRID, objective = mean of val track R@10 (val gallery) and val segment R@10, ties keep the
smaller w. The same w is used in retrieval, in the pipeline's coarse and fine stages and in Viterbi.
Reference systems are checked against their saved runs before their numbers are used: P0 against
runs/filmsplit/pipeline_ens10 (pipeline and stitching), P0s against runs/filmsplit/pipeline (D60).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from bgm.analysis.full_dataset import CachedScorer, load_scorer
from bgm.eval.metrics import rank_of_first_hit
from bgm.eval.store import FeatureStore, fold_ids
from bgm.person.diagnose import film_cfg
from bgm.person.evaluate import MeanScorer, assert_reproduces, pipeline_fold, retrieval_ranks, retrieval_scores, retrieval_truth, zq
from bgm.person.pipeline import PersonMusicIndex, load_models, person_query
from bgm.person.stitch import stitch_fold
from bgm.person.train import PersonInputs, PersonStore, load_person_model
from bgm.runs import run_dir, write_run

log = logging.getLogger(__name__)
ROOT = Path("runs/filmsplit")
W_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
TEN = list(range(10))


def systems(seed: int) -> dict[str, dict]:
    """name -> spec for run seed `seed` (ensembles ignore it and run once per fold as seed 0)."""
    one = [seed]
    return {
        "P0": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": TEN}, "ref": "pipeline_ens10"},
        "P0s": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": one}, "ref": "pipeline"},
        "P1": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": one}, "branch": {"kind": "face", "variant": "face-both", "seeds": one}},
        "P2": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": one}, "branch": {"kind": "person", "variant": "person-both", "seeds": one}},
        "P3": {"mode": "early", "frame": {"variant": "siglipperson-both", "seeds": one}},
        # P4 (user decision 2026-10-20): the winner of P1-P3 as a 10-seed ensemble of heads with weight decay 0.3 +
        # dropout 0.4 (D76), plus a distillation row (one student per seed, the 10 regularised heads as teachers)
        "P4-P2": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": TEN}, "branch": {"kind": "person", "variant": "person-both-reg", "seeds": TEN}},
        "P4-P2-kd": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": TEN}, "branch": {"kind": "person", "variant": "person-both-reg-kd", "seeds": one}},
        "P4-P3": {"mode": "early", "frame": {"variant": "siglipperson-both-reg", "seeds": TEN}},
        "P4-P3-kd": {"mode": "early", "frame": {"variant": "siglipperson-both-reg-kd", "seeds": one}},
        "P4-P1": {"mode": "late", "frame": {"variant": "siglip-both", "seeds": TEN}, "branch": {"kind": "face", "variant": "face-both", "seeds": [0, 1, 2]}},
    }


def scorer(cfg, store, pstore, part: dict, kind: str, fold: int, device: str):
    """kind: 'semantic' (siglip-both / face-both heads) | 'person' | 'early' (person-head models)."""
    runs = [ROOT / "semantic" / part["variant"] / f"fold{fold}" / f"seed{s}" for s in part["seeds"]]
    if kind == "semantic":
        ms = [load_scorer(r, store, device) for r in runs]
    else:
        inp = PersonInputs(store, pstore, kind)
        ms = [CachedScorer(load_person_model(r, cfg, device), inp) for r in runs]
    return ms[0] if len(ms) == 1 else MeanScorer(ms)


def fused_ranks(Sf: dict, Sb: dict | None, truth: dict, w: float) -> dict:
    if Sb is None or w == 0:
        return retrieval_ranks(Sf, truth)
    mix = {"track": [zq(a) + w * zq(b) for a, b in zip(Sf["track"], Sb["track"])],
           "track_all": [zq(a) + w * zq(b) for a, b in zip(Sf["track_all"], Sb["track_all"])],
           "segment": [zq(a) + w * zq(b) for a, b in zip(Sf["segment"], Sb["segment"])]}
    return retrieval_ranks(mix, truth)


def tune_w(Sf: dict, Sb: dict, truth: dict) -> tuple[float, list]:
    """Val: w maximizing mean(track R@10, segment R@10); ties keep the smaller w."""
    best, best_w, curve = -1.0, 0.0, []
    for w in W_GRID:
        r = fused_ranks(Sf, Sb, truth, w)
        obj = 0.5 * (float((r["track_test_fold"] <= 10).mean()) + float((r["segment_all"] <= 10).mean()))
        curve.append([w, obj])
        if obj > best + 1e-12:
            best, best_w = obj, w
    return best_w, curve


def run_one(cfg, store, pstore, name: str, fold: int, seed: int, device: str = "cuda") -> Path:
    spec = systems(seed)[name]
    out = run_dir(cfg, f"person/eval/{name}", fold, seed)
    if (out / "metrics.json").exists():
        log.info("%s fold %d seed %d: done already", name, fold, seed)
        return out
    t0 = time.time()
    f = fold_ids(cfg, fold)
    fkind = "early" if spec["mode"] == "early" else "semantic"
    br = spec.get("branch")
    bkind = None if br is None else ("semantic" if br["kind"] == "face" else "person")
    frame = scorer(cfg, store, pstore, spec["frame"], fkind, fold, device)
    branch = None if br is None else scorer(cfg, store, pstore, br, bkind, fold, device)
    # ---- w on val
    w, curve = 0.0, []
    if branch is not None:
        vt = retrieval_truth(store, f["val"])
        w, curve = tune_w(retrieval_scores(store, frame, f["val"]), retrieval_scores(store, branch, f["val"]), vt)
    # ---- test retrieval
    truth = retrieval_truth(store, f["test"])
    Sf = retrieval_scores(store, frame, f["test"])
    Sb = None if branch is None else retrieval_scores(store, branch, f["test"])
    ranks = fused_ranks(Sf, Sb, truth, w)
    per = {f"ret/{k}": v for k, v in ranks.items()}
    if Sb is not None:
        per.update({f"ret/branch/{k}": v for k, v in retrieval_ranks(Sb, truth).items()})
    per.update({"ret/pair": truth["pair"], "ret/seg_pair": truth["seg_pair"], "ret/seg_idx": truth["seg_idx"]})
    del frame, branch
    # ---- pipeline
    ref = json.loads((ROOT / spec.get("ref", "pipeline_ens10") / f"fold{fold}" / f"seed{seed if spec.get('ref') == 'pipeline' else 0}" / "fusion.json").read_text())
    fusion = {"w_coarse": dict(ref["w_coarse"]), "w_fine": dict(ref["w_fine"]), "K": ref["K"], "J": ref["J"], "window_hop_s": ref["window_hop_s"]}
    if br is not None:
        fusion["w_coarse"]["person"] = fusion["w_fine"]["person"] = w
    fusion["val_curve"] = curve
    models = load_models(cfg, store, fold, spec, device, train_ids=f["train"])
    index = PersonMusicIndex(store, models, list(dict.fromkeys(store.music_of[p] for p in f["test"])))
    queries = {}
    for p in f["test"]:
        vq = person_query(store, pstore, [p])
        vq.prepare(models, store.min_span)
        queries[p] = vq
    pipe = pipeline_fold(store, index, queries, fusion, chance=False)
    per.update({f"pipe/{k}": v for k, v in pipe.items()})
    # ---- stitching
    stitch, sper = stitch_fold(cfg, store, pstore, models, fold, seed, fusion, f)
    per.update({f"stitch/{k}": v for k, v in sper.items()})
    if "ref" in spec:
        check_reference(spec["ref"], fold, seed, pipe, stitch, sper)
    (out / "fusion.json").write_text(json.dumps(fusion, indent=1))
    (out / "stitch.json").write_text(json.dumps(stitch, indent=1))
    np.savez(out / "perquery.npz", **per)
    summ = {"w": w, "track_R@10": float((ranks["track_test_fold"] <= 10).mean()), "segment_R@10": float((ranks["segment_all"] <= 10).mean()),
            **{f"pipe/{k}": float(np.nanmean(v)) for k, v in pipe.items() if v.dtype.kind == "f"},
            **{f"stitch/{k}": float(np.nanmean(v)) for k, v in sper.items() if v.dtype.kind == "f"}, "seconds": time.time() - t0}
    write_run(out, cfg, {"system": name, "spec": spec, "fusion": fusion, "stitch": stitch, "test": summ}, seed)
    log.info("%s fold %d seed %d: w %.2f | track R@10 %.3f seg R@10 %.3f | track@1 %.3f window@1 %.3f | switch F1 %.3f part acc %.3f [%.0f s]",
             name, fold, seed, w, summ["track_R@10"], summ["segment_R@10"], summ["pipe/track@1"], summ["pipe/window@1"],
             summ["stitch/concat/viterbi/switch_f1"], summ["stitch/concat/viterbi/part_track_acc"], summ["seconds"])
    return out


def check_reference(ref: str, fold: int, seed: int, pipe: dict, stitch: dict, sper: dict) -> None:
    d = ROOT / ref / f"fold{fold}" / f"seed{seed if ref == 'pipeline' else 0}"
    with np.load(d / "perquery.npz") as z:
        assert_reproduces(z, pipe, f"{ref} fold {fold} seed {seed}")
    assert json.loads((d / "stitch.json").read_text())["penalties"] == stitch["penalties"], (ref, fold, "penalties")
    # Under heavy GPU load cuBLAS may pick other kernels; last-bit differences can then flip an exact tie in a Viterbi
    # path (seen once: fold 2 seed 2 of D60 under 5 parallel evaluations and a render; alone it reproduces exactly).
    # Allowed: at most 2 differing concatenations / natural videos, logged.
    with np.load(d / "stitch_perquery.npz") as z:
        for k in ("concat/viterbi/switch_f1", "concat/viterbi/part_track_acc", "concat/greedy/switch_f1", "natural/viterbi/track"):
            bad = np.where(~np.isclose(z[k], sper[k], equal_nan=True))[0]
            assert len(bad) <= 2, (ref, fold, seed, k, bad)
            if len(bad):
                log.warning("%s fold %d seed %d %s: %d item(s) differ from the saved run: %s saved %s now %s", ref, fold, seed, k,
                            len(bad), bad.tolist(), z[k][bad].tolist(), sper[k][bad].tolist())
    log.info("%s fold %d: reproduces the saved pipeline and stitching", ref, fold)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cfg = film_cfg([])
    store, pstore = FeatureStore(cfg), PersonStore(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for k in map(int, a.folds.split(",")):
        for s in map(int, a.seeds.split(",")):
            run_one(cfg, store, pstore, a.system, k, s, device)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
