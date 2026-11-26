#!/bin/bash
# After the 100 nested heads: cross-fitted episodes -> gold-in-shortlist check -> CRF retrain (only if check passes).
cd "$(dirname "$0")/.."
P=.venv/bin/python
L=stitching/results/logs
until [ $(ls runs/titlesplit_nested/semantic/siglip-both/fold*/seed*/ckpt.pt 2>/dev/null | wc -l) -ge 100 ] && ! pgrep -f "python -m stitching.nested train" >/dev/null; do sleep 30; done
echo "== heads done $(date)"
for k in 0 1 2 3 4; do $P -m stitching.crossfit --fold $k > $L/crossfit_fold$k.log 2>&1 & done; wait
echo "== crossfit episodes done $(date)"
for k in 0 1 2 3 4; do $P -m stitching.crf_crossfit --fold $k --check-only > $L/xfcheck_fold$k.log 2>&1; done
$P - <<'PY'
import json, sys
r = [json.load(open(f"stitching/results/fold{k}_xf.json")) for k in range(5)]
for x in r:
    g = x["gold_stats"]
    print(f"fold {x['fold']}: gold-in-candidates cross-fitted train {g['xtrain']['gold_in_candidates']:.3f} | in-sample train {g['train_insample']['gold_in_candidates']:.3f} | val {g['val']['gold_in_candidates']:.3f} | test {g['test']['gold_in_candidates']:.3f} | ratio {x['xtrain_over_test_ratio']:.2f}")
bad = [x["fold"] for x in r if x["xtrain_over_test_ratio"] > 2]
open("stitching/results/xf_check.txt", "w").write("FAIL " + str(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
PY
if [ $? -ne 0 ]; then echo "== CHECK FAILED: cross-fitted gold rate > 2x test; CRF not retrained $(date)"; exit 1; fi
for k in 0 1 2 3 4; do $P -m stitching.crf_crossfit --fold $k > $L/crf_xf_fold$k.log 2>&1 & done; wait
echo "== PHASE 1 DONE $(date)"
