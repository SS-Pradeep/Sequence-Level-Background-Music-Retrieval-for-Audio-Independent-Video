"""D138 Step 6: choose the combined setting on validation, then evaluate it on test once.

python -m bgm.analysis.final_d138 heads   -> val track R@10 / MRR of the candidate 10-head ensembles (seeds 0-9) on each
                                             fold's val clips (gallery = their music); prints the chosen variant
python -m bgm.analysis.final_d138 report  -> results/d138_final/{final,cue}.csv, summary.md

Choice rules (validation only; the test fold never chooses anything):
  heads      the Step 3 winner's 10-head ensemble vs the shipped one, on val track MRR averaged over folds (D136's rule:
             44-69 val clips per fold); ties keep the shipped heads. Step 2 hubness is left out (no non-transductive
             gain on val or test).
  candidates the Step 4 cell with the best val stitching objective (K 10, C 40, window step 2 s).
  penalties  lam_switch, lam_jump, mu re-tuned on val by tune-stitch (Step 5); alpha, lam_jump of the cue (change
             predictor) system re-tuned on the cue val sequences by `bgm cue eval` (mu from that tune-stitch run).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from bgm.analysis.baselines_compare import FOLDS, MANIFESTS
from bgm.analysis.zs_blend import VAL, track_ranks

log = logging.getLogger(__name__)

OUT = Path("results/d138_final")
RUNS = Path("runs/cleaned")
SEL = Path("reports/cleaned/d138/final_selection.json")
SHIPPED = "siglip-both"
CAND = (10, 40, 2.0)
FINAL_EXP = "d138_final"


def _val_ids(k: int) -> list[str]:
    return list(json.loads((MANIFESTS / "folds_title.json").read_text())["folds"][str(k)]["val"])


def val_matrices(variant: str, seeds: list[int], device: str = "cuda") -> dict[int, np.ndarray]:
    """{fold: [val clips, val music] mean cosine of the variant's heads `seeds`}."""
    import torch
    from bgm.analysis.full_dataset import load_scorer
    from bgm.analysis.mema_protocols import make_cfg
    from bgm.eval.store import FeatureStore
    from bgm.person.evaluate import MeanScorer
    store = FeatureStore(make_cfg("clean"))
    out = {}
    for k in FOLDS:
        val = _val_ids(k)
        gal = [store.music_of[p] for p in val]
        sc = MeanScorer([load_scorer(RUNS / "semantic" / variant / f"fold{k}" / f"seed{s}", store, device) for s in seeds])
        with torch.no_grad():
            out[k] = np.stack([np.asarray(sc.track(q, gal), float) for q in val])
    return out


def choose_heads(winner: str) -> str:
    V = np.load(VAL)
    rows = {}
    mats = {SHIPPED: {k: V[f"{k}_heads0"] for k in FOLDS}}          # D136: shipped heads seeds 0-9 on the val clips
    if winner != "d134-base":
        mats[winner] = val_matrices(winner, list(range(10)))
    for name, M in mats.items():
        r = [track_ranks(M[k], V[f"{k}_track"]) for k in FOLDS]
        rows[name] = {"val_MRR": float(np.mean([(1 / x).mean() for x in r])), "val_R@10": float(np.mean([(x <= 10).mean() for x in r])),
                      "val_R@1": float(np.mean([(x <= 1).mean() for x in r]))}
    best = max(rows, key=lambda n: (round(rows[n]["val_MRR"], 4), n == SHIPPED))
    SEL.parent.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps({"val": rows, "heads": best, "candidates": CAND}, indent=1))
    log.info("heads on val: %s -> %s", rows, best)
    return best


def report() -> None:
    import pandas as pd
    from bgm.eval.metrics import cluster_bootstrap
    sel = json.loads(SEL.read_text())
    systems = {"shipped (K 10, C 20, 0.5 s)": "pipeline_ens10", "best combined (val-chosen)": FINAL_EXP}
    rows, per = [], {}
    for lab, e in systems.items():
        vals = {}
        for k in FOLDS:
            for s in range(3):
                d = RUNS / e / f"fold{k}" / f"seed{s}"
                st = json.loads((d / "stitch_metrics.json").read_text())
                pq = np.load(d / "stitch_perquery.npz")
                for m, key in (("switch_f1", "concat/viterbi/switch_f1"), ("song_acc", "concat/viterbi/part_track_acc"),
                               ("switches_per_min", "concat/viterbi/switches_per_min"), ("natural_track", "natural/viterbi/track")):
                    vals.setdefault(m, {}).setdefault(k, []).append(st["test"][key])
                    per.setdefault((lab, m), {}).setdefault(k, []).append(pq[key])
                vals.setdefault("penalties", {}).setdefault(k, []).append(st["penalties"])
        r = {"system": lab, "exp": e}
        for m in ("switch_f1", "song_acc", "switches_per_min", "natural_track"):
            F = np.array([np.mean(vals[m][k]) for k in FOLDS])
            r[m], r[m + "_std"] = float(F.mean()), float(F.std(ddof=1))
        r["penalties_per_fold"] = json.dumps({k: vals["penalties"][k] for k in FOLDS})
        rows.append(r)
    base, new = list(systems)
    for m in ("switch_f1", "song_acc", "switches_per_min"):
        diff = np.concatenate([np.mean(per[(new, m)][k], 0) - np.mean(per[(base, m)][k], 0) for k in FOLDS])
        cl = np.concatenate([[f"{k}_{i}" for i in range(len(per[(new, m)][k][0]))] for k in FOLDS])
        ci = cluster_bootstrap(diff, cl)
        rows[-1] |= {f"d_{m}": ci["mean"], f"d_{m}_lo": ci["lo"], f"d_{m}_hi": ci["hi"]}
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "final.csv", index=False)
    L = ["# Final: best combined setting on test (D138, Step 6)", "",
         f"Chosen on validation only: heads = **{sel['heads']}** (val track MRR {json.dumps({n: round(v['val_MRR'], 4) for n, v in sel['val'].items()})}); "
         f"candidates K = {CAND[0]}, C = {CAND[1]}, window step {CAND[2]:g} s (best val stitching objective, Step 4); "
         "fixed penalties re-tuned on val per fold and run seed (Step 5). Cleaned title split, 50 test concatenations per "
         "fold, 5 folds × 3 run seeds (disjoint 10-head ensembles); ± = std over fold means; Δ = per sequence, run seeds "
         "averaged, 95% cluster bootstrap over sequences. Test evaluated once.", "",
         "| system | switch F1 | switches / min | song acc. | natural-video track acc. | Δ switch F1 [95% CI] | Δ song acc. [95% CI] |",
         "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        d1 = "—" if r.exp == "pipeline_ens10" else f"{r.d_switch_f1:+.3f} [{r.d_switch_f1_lo:+.3f}, {r.d_switch_f1_hi:+.3f}]"
        d2 = "—" if r.exp == "pipeline_ens10" else f"{r.d_song_acc:+.3f} [{r.d_song_acc_lo:+.3f}, {r.d_song_acc_hi:+.3f}]"
        L.append(f"| {r.system} | {r.switch_f1:.3f} ± {r.switch_f1_std:.3f} | {r.switches_per_min:.2f} | {r.song_acc:.3f} ± {r.song_acc_std:.3f} | "
                 f"{r.natural_track:.3f} | {d1} | {d2} |")
    L.append("")
    cue = cue_rows()
    if cue is not None:
        cue.to_csv(OUT / "cue.csv", index=False)
        L += ["## With the change predictor (R4, α and λ_jump re-tuned on the cue val sequences)", "",
              "Blocked test sequences of `bgm cue eval` (emission run seed 0 = one 10-head ensemble; cue seeds 0–2 averaged).", "",
              "| emission | system | α per fold | λ_jump per fold | μ per fold | switch F1 (all joins) | in-film F1 (paper headline) | song acc. |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in cue.iterrows():
            L.append(f"| {r.emission} | {r.system} | {r.alpha} | {r.lam_jump} | {r.mu} | {r.switch_f1:.3f} | {r.in_film_f1:.3f} | {r.song_acc:.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", OUT / "summary.md")


def cue_rows():
    """R0_tuned and R4 (cue seeds averaged) on the blocked test sequences, shipped emission vs the D138 one."""
    import pandas as pd
    from bgm.cue.report import f1_of
    out = []
    for lab, exp in (("shipped", "cue"), ("best combined", "cue_d138")):
        ev = [RUNS / exp / "eval" / f"fold{k}" / "eval.json" for k in FOLDS]
        if not all(p.exists() for p in ev):
            return None
        E = [json.loads(p.read_text()) for p in ev]
        for sysname in ("R0_tuned", "R4"):
            f1, inf, acc, par = [], [], [], []
            for e in E:
                names = [n for n in e["systems"] if n == sysname or n.startswith(sysname + "_s")]
                vs = [e["systems"][n] for n in names]
                f1.append(np.mean([test_metric(v, "switch_f1") for v in vs]))
                inf.append(np.mean([[r["in_film/1"] for r in v["rows"] if r["mode"] == "blocked"] for v in vs], 0))
                acc.append(np.mean([test_metric(v, "part_track_acc") for v in vs]))
                par.append([v["params"] for v in vs])
            fmt = lambda key: "; ".join("/".join(f"{p.get(key, float('nan')):g}" for p in ps) for ps in par)  # noqa: E731
            out.append({"emission": lab, "system": sysname, "alpha": fmt("alpha") if sysname == "R4" else "—",
                        "lam_jump": fmt("lam_jump"), "mu": fmt("mu"), "switch_f1": float(np.mean(f1)), "in_film_f1": f1_of(np.concatenate(inf)),
                        "song_acc": float(np.mean(acc))})
    return pd.DataFrame(out)


def test_metric(sysrec: dict, key: str) -> float:
    """Mean of a per-sequence metric over the blocked test sequences of one system record of eval.json."""
    rows = [r for r in sysrec["rows"] if r["mode"] == "blocked"]
    return float(np.mean([r[key] for r in rows]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1]
    if cmd == "heads":
        print(choose_heads(sys.argv[2]))
    else:
        report()
