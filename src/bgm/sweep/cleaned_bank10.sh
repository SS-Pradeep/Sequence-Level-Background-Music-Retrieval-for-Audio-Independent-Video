#!/usr/bin/env bash
# N.1 (user 2026-11-21): the D144 tag-free loss bank (15 configs) on the CLEANED dataset, title split, seeds 0-9 x 5
# folds, 5 workers; multi-positive InfoNCE first. Resumable (finished (config, fold, seed) rows are skipped).
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BGM_SWEEP_DATA=cleaned
O=results/cleaned/loss_bank
mkdir -p $O
LOG=$O/loss_bank.log
S="0 1 2 3 4 5 6 7 8 9"
echo "=== cleaned bank10 start $(date -Iseconds)" >> $LOG
r() { $PY -m bgm.sweep.titlesplit run --seeds $S --configs "$@"; }
{ r multipos; r debiased infonce_fixtau; } >> $O/c1.log 2>&1 &
{ r REF; r siglip recipe_fixed; } >> $O/c2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs inter_intra_nce recipe_uncert vsepp >> $O/c3.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs infonce_intra all_uncert pq_tau >> $O/c4.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds $S --configs asac_full infonce_supcon_m infonce_vv >> $O/c5.log 2>&1 &
wait
echo "=== cleaned bank10 training done $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit table --set bank >> $LOG 2>&1
$PY -m bgm.sweep.train_vs_test > $O/train_vs_test.log 2>&1
echo "=== cleaned bank10 tables written $(date -Iseconds)" >> $LOG
