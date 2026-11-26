# Re-ranked shortlists through the final decoder (Table 3 protocol, D148)

112 test sequences, 5 title folds, Table 3 lattices (stitching/cache; original 1,248-pair data with the title folds that cleaned_dataset carries over unchanged; no Table 3 lattices exist for the cleaned data). The final row is re-decoded with its saved parameters and reproduces its saved values exactly. The two re-rankers were chosen before this test, by the largest mean val gain of the segment-protocol ensemble in the retrieval study. Each scene's top-20 candidates are re-ordered (sorted sim values re-assigned in the new order); re-ranker parameters and λ chosen jointly on val sequences. Mean ± std over folds (std for track acc. and switch F1, as Table 3). Δ = re-ranked − final per sequence, 95% cluster bootstrap over film groups (81 clusters), 10,000 resamples; * = CI excludes 0.

| System | Track acc. | Switch F1 ±1 s | TBIoU | Switches/min | Jumps/min | Align. succ. | ms/video |
|---|---|---|---|---|---|---|---|
| HMM, gold prior, α tuned (final) | 0.044 ± 0.009 | 0.694 ± 0.052 | 0.805 | 4.16 | 0.06 | 0.820 | 0.5 |
| + k-reciprocal re-ranking (top-20) | 0.043 ± 0.010 | 0.694 ± 0.055 | 0.805 | 4.15 | 0.04 | 0.819 | 5.1 |
| + Sequence context re-ranking (top-20) | 0.043 ± 0.010 | 0.691 ± 0.055 | 0.805 | 4.16 | 0.07 | 0.817 | 4.3 |

| Δ vs final | track_acc | switch_f1 | TBIoU | switches_per_min | jumps_per_min | align_success |
|---|---|---|---|---|---|---|
| + k-reciprocal re-ranking (top-20) | −0.001 [−0.010, +0.008] | −0.001 [−0.005, +0.003] | −0.001 [−0.004, +0.003] | −0.01 [−0.05, +0.03] | −0.02 [−0.04, −0.00]* | −0.001 [−0.005, +0.002] |
| + Sequence context re-ranking (top-20) | −0.002 [−0.009, +0.006] | −0.003 [−0.007, +0.001] | −0.001 [−0.005, +0.004] | +0.00 [−0.03, +0.03] | +0.00 [−0.02, +0.03] | −0.003 [−0.007, +0.001] |

## Picks on val

| fold | k-reciprocal (params, λ) | Sequence context (params, λ) |
|---|---|---|
| 0 | {'k1': 10, 'k2': 1, 'lam': 0.5}, 0.707 | {'w': 0.25, 'n': 1}, 0.707 |
| 1 | {'k1': 10, 'k2': 3, 'lam': 0.5}, 0.354 | {'w': 0.25, 'n': 2}, 0.25 |
| 2 | {'k1': 20, 'k2': 3, 'lam': 0.1}, 1 | {'w': 0.25, 'n': 1}, 0.707 |
| 3 | {'k1': 10, 'k2': 3, 'lam': 0.1}, 0.5 | {'w': 0.5, 'n': 99}, 0.5 |
| 4 | {'k1': 5, 'k2': 1, 'lam': 0.1}, 0.707 | {'w': 0.25, 'n': 99}, 0.707 |
