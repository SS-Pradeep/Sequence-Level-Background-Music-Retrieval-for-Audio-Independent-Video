#!/bin/bash
# Add background music to a silent video with the best system (PROGRESS.md D75): the 10-seed ensemble of semantic
# heads trained on the 1,248-pair film split, semantic-only scoring, Viterbi stitching, beat-snapped rendering.
#   ./render_best.sh in.mp4 out.mp4 [overrides...]     e.g. render.audio_bitrate=320k
set -e
cd "$(dirname "$0")"
.venv/bin/bgm render --video "$1" --out "$2" --run runs/filmsplit/pipeline_ens10/fold0/seed0 \
  splits.folds_file=folds_film.json paths.runs=runs/filmsplit pipeline.dyn_variant=null \
  "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]" "${@:3}"
