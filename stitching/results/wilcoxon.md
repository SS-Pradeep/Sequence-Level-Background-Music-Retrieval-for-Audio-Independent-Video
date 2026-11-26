# Paired Wilcoxon signed-rank tests over test sequences (5 folds pooled)

Row 10 (CRF, context + derived prior, mean over 3 seeds) minus each baseline, per test sequence; two-sided; zero differences dropped (Wilcoxon's method).

| vs | metric | n | mean Δ | median Δ | W | p |
|---|---|---|---|---|---|---|
| 1. Greedy top-1 | objective | 112 | -0.0458 | -0.0629 | 1969.0 | 0.0012 |
| 1. Greedy top-1 | track_acc | 112 | +0.0180 | +0.0000 | 410.0 | 0.068 |
| 1. Greedy top-1 | switch_f1 | 112 | -0.0638 | -0.0643 | 1533.5 | 5.9e-06 |
| 1. Greedy top-1 | TBIoU | 112 | -0.0955 | -0.0975 | 1113.0 | 4.4e-09 |
| 4. Prétet-style DTW (one track) | objective | 112 | +0.6690 | +0.6667 | 0.0 | 4.1e-20 |
| 4. Prétet-style DTW (one track) | track_acc | 112 | +0.0323 | +0.0000 | 234.0 | 0.00017 |
| 4. Prétet-style DTW (one track) | switch_f1 | 112 | +0.6367 | +0.6667 | 0.0 | 4.1e-20 |
| 4. Prétet-style DTW (one track) | TBIoU | 112 | +0.5728 | +0.5862 | 0.0 | 4.1e-20 |
| 5. Viterbi, hand-set costs (existing) | objective | 112 | -0.0272 | -0.0377 | 2345.0 | 0.025 |
| 5. Viterbi, hand-set costs (existing) | track_acc | 112 | +0.0201 | +0.0000 | 279.0 | 0.0071 |
| 5. Viterbi, hand-set costs (existing) | switch_f1 | 112 | -0.0473 | -0.0377 | 1860.0 | 0.00038 |
| 5. Viterbi, hand-set costs (existing) | TBIoU | 112 | -0.0801 | -0.0811 | 1402.0 | 5.2e-07 |
| 7. HMM, derived costs, λ tuned | objective | 112 | -0.0191 | -0.0258 | 2357.0 | 0.053 |
| 7. HMM, derived costs, λ tuned | track_acc | 112 | +0.0215 | +0.0000 | 262.0 | 0.011 |
| 7. HMM, derived costs, λ tuned | switch_f1 | 112 | -0.0406 | -0.0364 | 1869.0 | 0.0015 |
| 7. HMM, derived costs, λ tuned | TBIoU | 112 | -0.0538 | -0.0460 | 1746.0 | 9.8e-05 |
