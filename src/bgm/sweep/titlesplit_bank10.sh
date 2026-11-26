#!/usr/bin/env bash
# Loss bank to 10 seeds (user 2026-11-20: "3 seeds is not enough"): seeds 3-9 for all 15 bank configs, 5 workers.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=results/titlesplit/loss_bank.log
S="3 4 5 6 7 8 9"
while pgrep -f "bgm.sweep.train_vs_test" >/dev/null; do sleep 20; done
echo "=== bank10 start $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit run --seeds $S --configs REF multipos debiased >> results/titlesplit/c1.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs siglip recipe_fixed infonce_fixtau >> results/titlesplit/c2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs inter_intra_nce recipe_uncert vsepp >> results/titlesplit/c3.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs infonce_intra all_uncert pq_tau >> results/titlesplit/c4.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs asac_full infonce_supcon_m infonce_vv >> results/titlesplit/c5.log 2>&1 &
wait
echo "=== bank10 training done $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit table --set bank >> $LOG 2>&1
$PY -m bgm.sweep.train_vs_test > results/titlesplit/train_vs_test.log 2>&1
echo "=== bank10 tables written $(date -Iseconds)" >> $LOG
