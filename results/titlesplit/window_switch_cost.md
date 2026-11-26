# Per-window key and tempo in the switch cost (D137)

Final title-split system (pipeline_ens10), 5 folds × 3 ensembles, test concatenations (50 per fold). Each mode is tuned on the val concatenations (part track accuracy + switch F1). *track* = the shipped per-file key / tempo; *wN* = key / tempo of the last N s of the outgoing window vs the first N s of the incoming one (file value where the window gives no estimate). *+mu*: μ grid extended to {0, 0.5, 1, 2, 4}. Values are mean ± std over the 15 runs; Δ vs track (same grid) with 95% CIs resampling concatenations. Seam distances use the 5 s window values for every mode (0 = same key / tempo; fifths distance and octave-folded tempo distance in [0, 1]); the wN modes optimise a similar quantity, so their seam rows are expected to drop.

| mode | switch F1 | Δ F1 | part track acc | Δ part acc | switches / min | seam key dist | Δ seam key | seam tempo dist | tuned μ (15 runs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| track | 0.731 ± 0.038 | — | 0.073 ± 0.027 | — | 3.415 ± 0.469 | 0.438 ± 0.043 | — | 0.352 ± 0.059 | 0 ×7, 0.5 ×5, 1 ×3 |
| track+mu | 0.730 ± 0.040 | — | 0.074 ± 0.026 | — | 3.468 ± 0.439 | 0.432 ± 0.052 | — | 0.345 ± 0.068 | 0 ×7, 0.5 ×5, 1 ×2, 4 ×1 |
| w3 | 0.730 ± 0.032 | -0.001 [-0.010, +0.009] | 0.074 ± 0.027 | +0.001 [-0.006, +0.007] | 3.480 ± 0.459 | 0.398 ± 0.055 | -0.040 [-0.054, -0.027] | 0.311 ± 0.077 | 0 ×4, 0.5 ×5, 1 ×6 |
| w3+mu | 0.730 ± 0.032 | -0.000 [-0.010, +0.010] | 0.074 ± 0.027 | +0.001 [-0.007, +0.007] | 3.470 ± 0.457 | 0.396 ± 0.057 | -0.036 [-0.052, -0.022] | 0.307 ± 0.082 | 0 ×4, 0.5 ×5, 1 ×5, 2 ×1 |
| w5 | 0.732 ± 0.023 | +0.001 [-0.009, +0.010] | 0.072 ± 0.022 | -0.002 [-0.007, +0.003] | 3.370 ± 0.429 | 0.292 ± 0.123 | -0.145 [-0.162, -0.129] | 0.293 ± 0.083 | 0 ×4, 0.5 ×5, 1 ×6 |
| w5+mu | 0.727 ± 0.026 | -0.003 [-0.013, +0.007] | 0.072 ± 0.022 | -0.002 [-0.008, +0.003] | 3.389 ± 0.403 | 0.293 ± 0.122 | -0.139 [-0.156, -0.123] | 0.292 ± 0.084 | 0 ×4, 0.5 ×5, 1 ×5, 2 ×1 |
| w8 | 0.723 ± 0.040 | -0.008 [-0.018, +0.002] | 0.069 ± 0.021 | -0.004 [-0.010, +0.001] | 3.498 ± 0.403 | 0.426 ± 0.066 | -0.012 [-0.026, +0.001] | 0.340 ± 0.080 | 0 ×7, 0.5 ×3, 1 ×5 |
| w8+mu | 0.726 ± 0.039 | -0.004 [-0.015, +0.007] | 0.070 ± 0.021 | -0.004 [-0.011, +0.001] | 3.456 ± 0.426 | 0.411 ± 0.070 | -0.021 [-0.036, -0.007] | 0.332 ± 0.077 | 0 ×6, 0.5 ×3, 1 ×5, 2 ×1 |
| greedy | 0.602 ± 0.033 | — | 0.051 ± 0.012 | — | 6.186 ± 0.494 | 0.473 ± 0.013 | — | 0.401 ± 0.025 | — |
| track (shipped penalties) | 0.731 ± 0.038 | — | 0.073 ± 0.027 | — | 3.399 ± 0.484 | 0.434 ± 0.041 | — | 0.346 ± 0.057 | 0 ×6, 0.5 ×6, 1 ×3 |

How often a window estimate exists and differs from the file value (test lattices, all candidates, both ends):

| window | key from window | key ≠ file key | tempo from window | tempo ≠ file tempo (±5%) |
| --- | --- | --- | --- | --- |
| w3 | 0.995 | 0.440 | 0.930 | 0.160 |
| w5 | 0.998 | 0.386 | 0.956 | 0.128 |
| w8 | 0.999 | 0.326 | 0.969 | 0.113 |

*track (shipped penalties)* = each run's stitch.json penalties, not re-tuned: they reproduce the saved val objective (asserted) and the saved test numbers (1 of 750 run-sequences differ, decoder near-ties). Re-tuning the track mode today picks the saved penalties in 14 of 15 runs; elsewhere a GPU-noise near-tie in the μ = 0 decode changes the val ranking. All Δs compare modes re-tuned on the same lattices.
