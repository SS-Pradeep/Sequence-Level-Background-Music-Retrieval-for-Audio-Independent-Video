#!/bin/bash
# Add background music to a video with the final system.
#   ./add_music.sh input.mp4               -> writes input_bgm.mp4
#   ./add_music.sh input.mp4 output.mp4    -> writes output.mp4
# Same as: bgm render --config configs/best_semantic.yaml --video IN --out OUT
#          --run runs/cleaned/pipeline_ens10/fold0/seed0
#          pipeline.dyn_variant=null "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"
set -e
if [ -z "$1" ]; then
  echo "usage: ./add_music.sh input.mp4 [output.mp4]"
  exit 1
fi
if [ ! -f "$1" ]; then
  echo "error: video not found: $1"
  echo "copy your video here first, e.g.:  cp ~/Videos/my_clip.mp4 new_video.mp4"
  exit 1
fi
in="$(realpath "$1")"
out="${2:-${1%.*}_bgm.mp4}"
out="$(realpath -m "$out")"
cd "$(dirname "$0")"
HF_HUB_OFFLINE=1 .venv/bin/bgm render --config configs/best_semantic.yaml \
  --video "$in" --out "$out" \
  --run runs/cleaned/pipeline_ens10/fold0/seed0 \
  pipeline.dyn_variant=null "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"
