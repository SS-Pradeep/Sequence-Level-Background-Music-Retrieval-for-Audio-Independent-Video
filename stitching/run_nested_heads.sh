#!/bin/bash
# Phase 1 heads: 10 nested splits x seeds 0-9, 5 processes at a time.
cd "$(dirname "$0")/.."
P=.venv/bin/python
L=stitching/results/logs
$P -m stitching.nested train --nested 0 --seeds 1-9 > $L/nested0.log 2>&1 &
for n in 1 2 3 4; do $P -m stitching.nested train --nested $n --seeds 0-9 > $L/nested$n.log 2>&1 & done
wait
for n in 5 6 7 8 9; do $P -m stitching.nested train --nested $n --seeds 0-9 > $L/nested$n.log 2>&1 & done
wait
echo "== NESTED HEADS DONE $(date)"
