#!/usr/bin/env bash
# Loss-function sweep, end to end (resume-safe: rerun after a stop or reboot; finished rows are skipped).
# Log: results/loss_sweep/sweep.log. A stop rule exits with code 3 and writes results/loss_sweep/STOPPED.txt.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
mkdir -p results/loss_sweep
rm -f results/loss_sweep/STOPPED.txt
for step in reference stage1 stage2 stage3 stage4 stage5 report; do
  echo "=== $step $(date -Iseconds)"
  $PY -m bgm.sweep.run "$step" 2> >(grep -v -i "warning" >&2)
  rc=$?
  if [ $rc -ne 0 ]; then echo "=== chain stopped at $step (exit $rc)"; exit $rc; fi
done
echo "=== done $(date -Iseconds)"
