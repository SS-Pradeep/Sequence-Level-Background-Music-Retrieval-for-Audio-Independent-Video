"""Oracle-p table on the cleaned title split (D145): constant duration prior (runs/cleaned/cue_hazard, L picked per fold on
val) vs fixed penalties, the learned predictor R4 and oracle p (runs/cleaned/cue, D122). Asserts identical lattices and
reproduced R0_tuned / oracle counts, then writes runs/cleaned/cue_hazard/metrics.json. Run: .venv/bin/python -m bgm.analysis.cue_hazard_table"""
import json
from pathlib import Path
import numpy as np
from bgm.cue.report import Table, boot_f1, fmt_ci
old = [json.loads(Path(f"runs/cleaned/cue/eval/fold{k}/eval.json").read_text()) for k in range(5)]
new = [json.loads(Path(f"runs/cleaned/cue_hazard/eval/fold{k}/eval.json").read_text()) for k in range(5)]
out = {"per_fold": []}
for o, n in zip(old, new):
    assert [c["digest"] for c in o["cases"]] == [c["digest"] for c in n["cases"]], o["fold"]
    for s in ("R0_tuned", "oracle"):
        a = [r["in_film/1"] for r in o["systems"][s]["rows"]]; b = [r["in_film/1"] for r in n["systems"][s]["rows"]]
        assert a == b, (o["fold"], s)
    hz = {k: v for k, v in n["systems"].items() if k.startswith("Rhaz")}
    best = max(hz, key=lambda k: hz[k]["val_objective"])
    out["per_fold"].append({"fold": n["fold"], "picked": best, "params": hz[best]["params"],
                            "val": {k: round(v["val_objective"], 4) for k, v in hz.items()}})
    o["systems"]["Rhaz"] = hz[best]
    o["systems"]["Rhaz"]["seed"] = None
import bgm.cue.report as R
R.SEEDLESS = tuple(R.SEEDLESS) + ("Rhaz",)
T = Table(old); cl = T.clusters("blocked")
def c(s): return T.counts(s, "blocked", "in_film/1")
res = {}
for s in ("Rhaz", "R0_tuned", "R4", "oracle"):
    tp, fp, fn = c(s).sum(0); b = boot_f1(c(s), cl)
    res[s] = {"f1": b, "precision": tp / (tp + fp), "recall": tp / (tp + fn)}
    print(f"{s:9s} F1 {fmt_ci(b)}  P {tp/(tp+fp):.3f}  R {tp/(tp+fn):.3f}")
for a, b in [("R4", "Rhaz"), ("oracle", "Rhaz"), ("Rhaz", "R0_tuned"), ("oracle", "R4")]:
    d = boot_f1(c(a), cl, c(b)); res[f"{a}-{b}"] = d; print(f"{a} - {b}: {fmt_ci(d, True)}")
for p in out["per_fold"]: print(p["fold"], p["picked"], p["params"], p["val"])
out["results"] = res; out["n_sequences"] = len(cl); out["n_clusters"] = len(set(cl))
Path("runs/cleaned/cue_hazard/metrics.json").write_text(json.dumps(out, indent=1, default=float))
