# src/bgm/cue/

The learned **music-change (cue-switch) predictor**: for each boundary between two segments, the probability that the
music changes there. These probabilities set the switch cost in Viterbi stitching. Opt-in: nothing here runs unless
called through `bgm cue ...` or `bgm render` with `cue.enabled=true`. Plan: [`docs/cue_switch_plan.md`](../../../docs/cue_switch_plan.md).

## Used by the final system

| File | Command | What it does |
|---|---|---|
| `tokens.py` | | One vector per segment: mean SigLIP, log duration, motion mean and spread. |
| `sequences.py` | `bgm cue build` | Joins dataset clips into 60–180 s sequences and labels each boundary change / no change. |
| `model.py` | | `CueModel`: segment tokens → 2-layer Transformer → change probability per boundary. |
| `train.py` | `bgm cue train` | Trains the predictor per fold × seed × variant. |
| `stitch.py` | | Viterbi with per-boundary switch costs from the predicted probabilities. |
| `evaluate.py` | `bgm cue eval` | Tunes how strongly the probabilities are used, on validation; evaluates on test. |
| `baselines.py`, `scenes.py` | | Reference predictors (cosine rule, scene-cut heuristic) that `cue eval` compares against. |
| `hook.py` | | **Render step:** runs the trained predictors on a new video's segments and decodes the lattice. |

## Studies and reports

| File | What it does |
|---|---|
| `report.py` | `bgm cue report` → `results/cue_switch.md`. |
| `probes.py` | `bgm cue probe`: does the predictor detect music changes, or just clip joins? (D85) |
| `within_clip.py` | Join-free test: music changes inside test clips only (D98). |
| `audio_teacher.py` | Training targets read from the music itself (D98). |
| `real_eval.py` | `bgm cue real-eval`: metrics on hand-annotated real multi-cue excerpts. |
| `cuts.py`, `cut_model.py` | Music-change classifier at every shot cut inside a clip (D103). |
| `cut_hook.py` | That classifier inside the Viterbi stitcher (D104). |
