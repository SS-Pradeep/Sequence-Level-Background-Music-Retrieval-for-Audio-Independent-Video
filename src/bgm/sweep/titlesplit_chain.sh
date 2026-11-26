#!/usr/bin/env bash
# Title-split planned losses: wait for seed 0, table, seeds 1-9 (3 parallel workers), 10-seed ensembles, final table.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=results/titlesplit/planned_losses.log
while pgrep -f "bgm.sweep.titlesplit run --seeds 0" >/dev/null; do sleep 20; done
echo "=== seed 0 done $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit table >> $LOG 2>&1
SEEDS="1 2 3 4 5 6 7 8 9"
$PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs REF siglip asac_full > results/titlesplit/w1.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs asac_no_tau asac_no_eps asac_no_weights > results/titlesplit/w2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $SEEDS --configs inter_intra_nce infonce_intra infonce_rdrop > results/titlesplit/w3.log 2>&1 &
wait
echo "=== seeds 1-9 done $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit ensemble >> $LOG 2>&1
echo "=== ensembles done $(date -Iseconds)" >> $LOG
