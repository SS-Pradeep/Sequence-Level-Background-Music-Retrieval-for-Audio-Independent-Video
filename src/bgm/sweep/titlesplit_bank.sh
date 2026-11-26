#!/usr/bin/env bash
# Tag-free loss bank (user 2026-11-20): 10 new configs x 5 folds x seeds 0-2 on the title split, 3 workers.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=results/titlesplit/loss_bank.log
echo "=== bank start $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs multipos recipe_fixed debiased infonce_fixtau >> results/titlesplit/b1.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs infonce_supcon_m recipe_uncert vsepp >> results/titlesplit/b2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs infonce_vv all_uncert pq_tau >> results/titlesplit/b3.log 2>&1 &
wait
$PY -m bgm.sweep.titlesplit table --set bank >> $LOG 2>&1
echo "=== bank table written $(date -Iseconds)" >> $LOG
