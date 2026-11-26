# Experiment 1: hidden semi-Markov decoder (Table 3 protocol)

112 test sequences (Table 3 lattices: BaSSL scenes, same candidates and emissions), 5 title folds. Mean ± std over
folds (std for track acc. and switch F1, as Table 3). Δ = HSMM − final, per sequence, 95% cluster bootstrap over film groups
(sequences sharing a film group merged: 81 clusters), 10,000 resamples. * = CI excludes 0.

Protocol note: the Table 3 lattices were built by the stitching/ harness from data/manifests (1,248 pairs) with the
film-title folds that cleaned_dataset/manifests/folds_title.json carries over unchanged (0 fold/split mismatches);
121 of the 982 clips in these test sequences were later excluded by cleaning (77 of 112 sequences touched). Both rows
use these identical lattices; the final row is re-decoded and reproduces its saved Table 3 values exactly.

| Decoder | Track acc. | Switch F1 ±1 s | TBIoU | Switches/min | Jumps/min | Align. succ. | ms/video |
|---|---|---|---|---|---|---|---|
| HMM, prior from gold paths, α tuned (final) | 0.044 ± 0.009 | 0.694 ± 0.052 | 0.805 | 4.16 | 0.06 | 0.820 | 0.5 |
| Hidden semi-Markov (cue level) | 0.040 ± 0.006 | 0.688 ± 0.051 | 0.795 | 3.76 | 0.44 | 0.769 | 1.7 |

| Δ (HSMM − final) | track_acc | switch_f1 | TBIoU | switches_per_min | jumps_per_min | align_success |
|---|---|---|---|---|---|---|
| | −0.005 [−0.015, +0.005] | −0.007 [−0.021, +0.007] | −0.011 [−0.027, +0.004] | −0.41 [−0.55, −0.27]* | +0.39 [+0.27, +0.52]* | −0.052 [−0.075, −0.031]* |

## Tuned parameters (val)

| fold | α | λ_jump | μ | γ | m | val objective | cue prior median / mean (s) |
|---|---|---|---|---|---|---|---|
| 0 | 2.0 | 2.0 | 1.0 | 0.5 | 0.5 | 0.872 | 8.6 / 9.4 |
| 1 | 2.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.924 | 8.6 / 9.4 |
| 2 | 0.25 | 0.0 | 0.0 | 2.0 | 0.5 | 0.776 | 8.7 / 9.4 |
| 3 | 2.0 | 2.0 | 2.0 | 2.0 | 0.0 | 0.903 | 8.7 / 9.5 |
| 4 | 1.0 | 1.0 | 2.0 | 1.0 | 0.5 | 1.011 | 8.6 / 9.4 |
