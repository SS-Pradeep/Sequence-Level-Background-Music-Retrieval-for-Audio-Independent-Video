#!/usr/bin/env bash
# D145 chain: after the per-encoder heads finish, cache their score matrices, then write the similarity table.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
LOG=results/titlesplit/similarity.log
until grep -q "enc done" $LOG; do sleep 30; done
while pgrep -f "cmd_cache\(2, encs=\['both'\]\)" >/dev/null; do sleep 20; done
echo "=== cache clap/mert start $(date -Iseconds)" >> $LOG
$PY -c "
import logging; logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
from bgm.sweep import similarity as S
S.cmd_cache(4, encs=['both', 'clap', 'mert'])" >> results/titlesplit/sim_cache.log 2>&1 || { echo "=== cache FAILED $(date -Iseconds)" >> $LOG; exit 1; }
echo "=== table start $(date -Iseconds)" >> $LOG
$PY -m bgm.sweep.similarity table >> $LOG 2>&1 || { echo "=== table FAILED" >> $LOG; exit 1; }
echo "=== similarity done $(date -Iseconds)" >> $LOG
