# Crossfade interpolation shapes

Every gain family of `crossfade_gains` (D50), each with the default 4 s seam gain ramps (D48); `equal_power_no_ramp` is the pre-D48 renderer for reference. Same held-out test bed and metrics as results/crossfade.md; Δ: paired against **equal_power (+ ramp, the shipped default)**, mean over seams, 95% cluster-bootstrap CI over videos. Theory: mid-fade level on uncorrelated material.

## Held-out test concatenations (750 renders per shape)

| shape | theory mid-fade dB | jump 3 s median | Δ jump 3 s | > 2 LU | jump 1 s median | Δ jump 1 s | dip median (signed) | |dip| Δ | dips < −3 LU | LUFS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_power | +0.0 | 1.20 | — | 37% | 1.91 | — | +0.71 | — | 6% | -16.78 |
| linear | -3.0 | 1.37 | +0.43 [+0.39, +0.47] | 40% | 2.15 | +0.42 [+0.39, +0.45] | -1.87 | +0.45 [+0.35, +0.54] | 32% | -17.02 |
| sqrt | +0.0 | 1.24 | -0.18 [-0.21, -0.15] | 38% | 1.83 | -0.26 [-0.29, -0.23] | +0.72 | -0.00 [-0.01, +0.00] | 7% | -16.78 |
| smoothstep | -3.0 | 1.30 | +0.52 [+0.46, +0.59] | 39% | 2.11 | +0.69 [+0.63, +0.74] | -1.78 | +0.41 [+0.32, +0.50] | 31% | -16.96 |
| smoothstep_power | +0.0 | 1.18 | +0.18 [+0.16, +0.21] | 37% | 1.96 | +0.30 [+0.27, +0.33] | +0.72 | -0.00 [-0.01, +0.00] | 6% | -16.76 |
| quadratic | -9.0 | 1.65 | +1.12 [+1.01, +1.23] | 45% | 2.85 | +1.52 [+1.42, +1.62] | -6.97 | +4.30 [+4.11, +4.48] | 83% | -17.18 |
| slow | +0.5 | 1.20 | -0.11 [-0.12, -0.09] | 37% | 1.86 | -0.13 [-0.14, -0.12] | +1.16 | +0.14 [+0.12, +0.16] | 5% | -16.73 |
| sinc | -3.0 | 1.19 | +0.71 [+0.61, +0.83] | 37% | 2.16 | +1.34 [+1.21, +1.47] | -0.80 | +0.03 [-0.03, +0.08] | 16% | -16.81 |
| smootherstep | -3.0 | 1.27 | +0.57 [+0.49, +0.65] | 38% | 2.09 | +0.84 [+0.77, +0.91] | -1.69 | +0.37 [+0.28, +0.46] | 30% | -16.92 |
| smootherstep_power | +0.0 | 1.18 | +0.28 [+0.23, +0.32] | 36% | 2.02 | +0.47 [+0.42, +0.51] | +0.72 | -0.01 [-0.02, +0.00] | 6% | -16.76 |
| equal_power_no_ramp | +0.0 | 2.19 | +0.78 [+0.72, +0.83] | 53% | 2.61 | +0.69 [+0.63, +0.75] | +1.18 | +0.41 [+0.36, +0.46] | 6% | -16.56 |

## 1.mp4 (same pieces as renders/1_bgm_final; videos in renders/shapes/, A/B clips in renders/shapes/ab/)

| shape | jump 3 s median | > 2 LU | jump 1 s median | dip median (signed) | dips < −3 LU | LUFS | file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_power | 0.28 | 1 | 1.06 | +0.00 | 1 of 30 | -16.04 | 1_bgm_equal_power.mp4 |
| linear | 0.40 | 1 | 1.57 | -2.27 | 11 of 30 | -16.22 | 1_bgm_linear.mp4 |
| sqrt | 0.42 | 1 | 1.41 | -0.00 | 1 of 30 | -16.04 | 1_bgm_sqrt.mp4 |
| smoothstep | 0.33 | 1 | 1.31 | -2.06 | 10 of 30 | -16.18 | 1_bgm_smoothstep.mp4 |
| smoothstep_power | 0.26 | 1 | 1.14 | +0.01 | 1 of 30 | -16.04 | 1_bgm_smoothstep_power.mp4 |
| quadratic | 0.54 | 1 | 1.97 | -7.72 | 25 of 30 | -16.36 | 1_bgm_quadratic.mp4 |
| slow | 0.30 | 1 | 1.06 | +0.37 | 1 of 30 | -16.00 | 1_bgm_slow.mp4 |
| sinc | 0.22 | 1 | 1.06 | -1.13 | 3 of 30 | -16.09 | 1_bgm_sinc.mp4 |
| smootherstep | 0.28 | 1 | 1.30 | -1.94 | 10 of 30 | -16.15 | 1_bgm_smootherstep.mp4 |
| smootherstep_power | 0.24 | 1 | 1.06 | +0.02 | 1 of 30 | -16.04 | 1_bgm_smootherstep_power.mp4 |
| equal_power_no_ramp | 0.48 | 3 | 1.11 | -0.01 | 1 of 30 | -16.02 | 1_bgm_equal_power_no_ramp.mp4 |
