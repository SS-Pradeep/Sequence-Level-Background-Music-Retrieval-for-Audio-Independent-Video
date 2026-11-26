#!/usr/bin/env bash
# Per-encoder heads (CLAP-only, MERT-only) for similarity fusion: 2 configs x 5 folds x seeds 0-9, 4 workers.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=results/titlesplit/similarity.log
echo "=== enc start $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 3 4 --configs REF_clap >> results/titlesplit/e1.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 5 6 7 8 9 --configs REF_clap >> results/titlesplit/e2.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 0 1 2 3 4 --configs REF_mert >> results/titlesplit/e3.log 2>&1 &
$PY -m bgm.sweep.titlesplit run --seeds 5 6 7 8 9 --configs REF_mert >> results/titlesplit/e4.log 2>&1 &
wait
echo "=== enc done $(date -Iseconds)" >> $LOG
