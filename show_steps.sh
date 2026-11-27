#!/bin/bash
# Same run as add_music.sh, but shows every step while it runs (pictures + a live page in the browser).
#   ./show_steps.sh input.mp4               -> writes input_bgm.mp4
#   ./show_steps.sh input.mp4 output.mp4    -> writes output.mp4
# Pictures: renders/<output name>/steps/   Live page: renders/<output name>/steps/steps.html
set -e
if [ -z "$1" ] || [ ! -f "$1" ]; then
  echo "usage: ./show_steps.sh input.mp4 [output.mp4]"
  exit 1
fi
in="$(realpath "$1")"
out="$(realpath -m "${2:-${1%.*}_bgm.mp4}")"
cd "$(dirname "$0")"
.venv/bin/python tools/show_steps.py "$in" "$out" "${@:3}"
