#!/usr/bin/env bash
# Improvement round (exploratory): 6 new configs x 5 folds x seeds 0-2 on the title split, after the planned 3-seed run.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=results/titlesplit/planned_losses.log
while pgrep -f titlesplit_3seeds.sh >/dev/null; do sleep 20; done
echo "=== improve start $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs REF_ema REF_do05 >> results/titlesplit/w1.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs infonce_intra_ema infonce_intra_do05 >> results/titlesplit/w2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 --configs infonce_intra_rdrop siglip_intra >> results/titlesplit/w3.log 2>&1 &
wait
$PY -m bgm.sweep.titlesplit table --set improve >> $LOG 2>&1
echo "=== improve table written $(date -Iseconds)" >> $LOG
