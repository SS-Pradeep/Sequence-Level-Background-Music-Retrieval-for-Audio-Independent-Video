#!/bin/bash
# Download everything the system needs besides the code.
#
#   ./download_assets.sh            for running (./add_music.sh), about 11.5 GB:
#       runs/cleaned/            trained semantic heads (5 folds x 30 seeds), fusion/stitch settings, cue predictor
#       cleaned_dataset/         music library: music + video features, 48 kHz music files, manifests
#       .checkpoints/            ImageBind weights (Meta's public release)
#       ~/.cache/huggingface/    SigLIP so400m, the video encoder
#
#   ./download_assets.sh --train    also the training data (TRAIN.md), about 4.4 GB more:
#       cleaned_dataset/video/, cleaned_dataset/audio/   the 1,011 cleaned video-music pairs
#
# Run from the repository root after installing (README section 6).
set -e
cd "$(dirname "$0")"
ASSETS_REPO="${ASSETS_REPO:-pradeepss007/bgm-render-assets}"    # Hugging Face dataset repo with runs/ and cleaned_dataset/
HF=.venv/bin/hf
[ -x "$HF" ] || HF=hf
TRAIN=0
[ "$1" = "--train" ] && TRAIN=1

echo "== 1/3  models and music library from huggingface.co/datasets/$ASSETS_REPO (3.3 GB)"
"$HF" download "$ASSETS_REPO" --repo-type dataset --local-dir . --include "runs/*" \
  --include "cleaned_dataset/manifests/*" --include "cleaned_dataset/features/*" --include "cleaned_dataset/interim/*"
if [ "$TRAIN" = 1 ]; then
  echo "== 1b    training data: cleaned videos and music (4.4 GB)"
  "$HF" download "$ASSETS_REPO" --repo-type dataset --local-dir . \
    --include "cleaned_dataset/video/*" --include "cleaned_dataset/audio/*"
fi

echo "== 2/3  SigLIP video encoder into the Hugging Face cache (3.5 GB; add_music.sh runs offline)"
"$HF" download google/siglip-so400m-patch14-384

echo "== 3/3  ImageBind weights (4.8 GB)"
mkdir -p .checkpoints
if [ -f .checkpoints/imagebind_huge.pth ] && [ "$(stat -c %s .checkpoints/imagebind_huge.pth)" = 4803584173 ]; then
  echo "already present"
else
  curl -L --fail -C - -o .checkpoints/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
fi

need="runs/cleaned/pipeline_ens10/fold0/seed0/fusion.json runs/cleaned/cue/eval/fold0/eval.json
      cleaned_dataset/manifests/pairs.csv cleaned_dataset/features/music cleaned_dataset/interim/music"
[ "$TRAIN" = 1 ] && need="$need cleaned_dataset/video cleaned_dataset/audio"
for p in $need; do
  [ -e "$p" ] || { echo "missing after download: $p"; exit 1; }
done
echo "done. Try:  ./add_music.sh Demo_Outputs/1/original_video.mp4 test_bgm.mp4"
