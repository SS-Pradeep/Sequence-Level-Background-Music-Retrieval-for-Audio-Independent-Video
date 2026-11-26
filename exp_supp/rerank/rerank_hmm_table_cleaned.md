# Re-ranked shortlists through the final decoder (Table 3 protocol, D148b)

92 test sequences, 5 title folds, cleaned dataset (1,011 pairs): the Table 3 decoder chain rebuilt on cleaned_dataset by exp_supp.rerank.clean_t3 (C boundary head retrained on the cleaned train clips, theta on cleaned val, lattices from the cleaned 10-seed siglip-both emission, gold-path prior fitted on cleaned training sequences, λ on cleaned val). The final row is re-decoded with its saved parameters and reproduces its saved values exactly. The two re-rankers were chosen before this test, by the largest mean val gain of the segment-protocol ensemble in the retrieval study. Each scene's top-20 candidates are re-ordered (sorted sim values re-assigned in the new order); re-ranker parameters and λ chosen jointly on val sequences. Mean ± std over folds (std for track acc. and switch F1, as Table 3). Δ = re-ranked − final per sequence, 95% cluster bootstrap over film groups (72 clusters), 10,000 resamples; * = CI excludes 0.

| System | Track acc. | Switch F1 ±1 s | TBIoU | Switches/min | Jumps/min | Align. succ. | ms/video |
|---|---|---|---|---|---|---|---|
| HMM, gold prior, α tuned (final) | 0.048 ± 0.012 | 0.687 ± 0.049 | 0.782 | 4.07 | 0.11 | 0.765 | 0.5 |
| + k-reciprocal re-ranking (top-20) | 0.047 ± 0.012 | 0.687 ± 0.038 | 0.786 | 4.22 | 0.12 | 0.780 | 5.0 |
| + Sequence context re-ranking (top-20) | 0.050 ± 0.013 | 0.676 ± 0.043 | 0.774 | 4.04 | 0.13 | 0.748 | 4.6 |

| Δ vs final | track_acc | switch_f1 | TBIoU | switches_per_min | jumps_per_min | align_success |
|---|---|---|---|---|---|---|
| + k-reciprocal re-ranking (top-20) | −0.000 [−0.010, +0.009] | +0.001 [−0.009, +0.010] | +0.004 [−0.004, +0.013] | +0.16 [+0.07, +0.26]* | +0.01 [−0.01, +0.02] | +0.016 [+0.001, +0.033]* |
| + Sequence context re-ranking (top-20) | +0.002 [−0.006, +0.010] | −0.011 [−0.022, −0.000]* | −0.007 [−0.018, +0.002] | −0.03 [−0.11, +0.05] | +0.02 [−0.02, +0.05] | −0.017 [−0.032, −0.002]* |

## Picks on val

| fold | k-reciprocal (params, λ) | Sequence context (params, λ) |
|---|---|---|
| 0 | {'k1': 5, 'k2': 6, 'lam': 0.9}, 0.354 | {'w': 0.5, 'n': 1}, 0.707 |
| 1 | {'k1': 20, 'k2': 3, 'lam': 0.9}, 0.5 | {'w': 0.25, 'n': 99}, 0.707 |
| 2 | {'k1': 20, 'k2': 3, 'lam': 0.1}, 0.25 | {'w': 0.25, 'n': 1}, 1.41 |
| 3 | {'k1': 20, 'k2': 1, 'lam': 0.7}, 0.707 | {'w': 0.25, 'n': 2}, 0.707 |
| 4 | {'k1': 20, 'k2': 1, 'lam': 0.1}, 1 | {'w': 2, 'n': 99}, 0.5 |
