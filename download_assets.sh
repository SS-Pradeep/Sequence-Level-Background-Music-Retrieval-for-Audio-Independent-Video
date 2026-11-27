#!/bin/bash
# Download everything `bgm render` / `./add_music.sh` needs besides the code (about 11.5 GB):
#   runs/cleaned/            trained semantic heads (5 folds x 30 seeds), fusion/stitch settings, cue predictor
#   cleaned_dataset/         music library: music features, video features, 48 kHz music files, manifests
#   .checkpoints/            ImageBind weights (Meta's public release)
#   ~/.cache/huggingface/    SigLIP so400m, the video encoder
# Run from the repository root after installing (README section 6):  ./download_assets.sh
set -e
cd "$(dirname "$0")"
ASSETS_REPO="${ASSETS_REPO:-pradeepss007/bgm-render-assets}"    # Hugging Face dataset repo with runs/ and cleaned_dataset/
HF=.venv/bin/hf
[ -x "$HF" ] || HF=hf

echo "== 1/3  models and music library from huggingface.co/datasets/$ASSETS_REPO (3.3 GB)"
"$HF" download "$ASSETS_REPO" --repo-type dataset --local-dir . --include "runs/*" --include "cleaned_dataset/*"   # not its README.md

echo "== 2/3  SigLIP video encoder into the Hugging Face cache (3.5 GB; add_music.sh runs offline)"
"$HF" download google/siglip-so400m-patch14-384

echo "== 3/3  ImageBind weights (4.8 GB)"
mkdir -p .checkpoints
if [ -f .checkpoints/imagebind_huge.pth ] && [ "$(stat -c %s .checkpoints/imagebind_huge.pth)" = 4803584173 ]; then
  echo "already present"
else
  curl -L --fail -C - -o .checkpoints/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
fi

for p in runs/cleaned/pipeline_ens10/fold0/seed0/fusion.json runs/cleaned/cue/eval/fold0/eval.json \
         cleaned_dataset/manifests/pairs.csv cleaned_dataset/features/music cleaned_dataset/interim/music; do
  [ -e "$p" ] || { echo "missing after download: $p"; exit 1; }
done
echo "done. Try:  ./add_music.sh Demo_Outputs/1/original_video.mp4 test_bgm.mp4"
