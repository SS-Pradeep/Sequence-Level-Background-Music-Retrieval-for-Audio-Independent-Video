# src/bgm/render/

`bgm render`: from a silent input video to the same video with a background-music soundtrack.

```bash
bgm render --config configs/best_semantic.yaml --video in.mp4 --out out.mp4 \
           --run runs/cleaned/pipeline_ens10/fold0/seed0
```

## Used by the final system

| File | What it does |
|---|---|
| `run.py` | **Entry point.** Featurizes the video, loads the run (heads, `fusion.json`, `stitch.json`), retrieves candidates, runs the cue predictor and Viterbi, renders and muxes. Writes `renders/<name>.plan.json`. |
| `query.py` | Featurizes the input exactly like the dataset: silent copy, shots, SigLIP at 1 Hz, segments. Cached in `renders/<name>/`. |
| `render.py` | Builds the soundtrack: groups segments into pieces, snaps piece starts to beats, equal-power crossfades, loudness normalisation, limiter, mux with A/V sync correction. |

## Optional render-only branches (off by default)

| File | What it does |
|---|---|
| `branches.py` | Wraps the music index to add extra scoring branches. |
| `pace.py` | Pace match: fast-cut video → energetic music (D100). |
| `scene.py` | Scene brief: a vision-language model (Qwen2.5-VL-3B) describes each scene to steer the music (D101). |
| `replay.py` | Penalty for replaying the same passage of a music file again and again (D101). |
