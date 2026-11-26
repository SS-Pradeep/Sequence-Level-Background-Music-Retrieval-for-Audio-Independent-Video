#!/usr/bin/env bash
# Title-split planned losses (user 2026-11-16): seed 0 for all 9 configs -> table; seeds 1, 2 -> table with bootstrap CIs.
# 3 parallel workers per seed group; resume-safe (finished rows in planned_losses_runs.csv are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=results/titlesplit/planned_losses.log
for SEEDS in "0" "1 2"; do
  echo "=== seeds $SEEDS start $(date -Iseconds)" >> $LOG
  $PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs REF siglip asac_full >> results/titlesplit/w1.log 2>&1 &
  $PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs asac_no_tau asac_no_eps asac_no_weights >> results/titlesplit/w2.log 2>&1 &
  $PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs inter_intra_nce infonce_intra infonce_rdrop >> results/titlesplit/w3.log 2>&1 &
  wait
  echo "=== seeds $SEEDS done $(date -Iseconds)" >> $LOG
  $PY -m bgm.sweep.titlesplit table >> $LOG 2>&1
  echo "=== table written $(date -Iseconds)" >> $LOG
done
