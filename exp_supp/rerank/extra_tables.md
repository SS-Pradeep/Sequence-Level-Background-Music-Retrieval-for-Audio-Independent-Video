# Re-ranker study: extra tables (D148c)

Cleaned dataset (1,011 pairs), title split, 5 folds, hit rule track_id; same runs as exp_supp/rerank/rerank_table.md (T1–T3), the shortlist-size sweep exp_supp/rerank/ksweep.py (T4) and the cleaned decoder study exp_supp/rerank/rerank_hmm_table_cleaned.md (T5).

## T1. Headroom of re-ranking the top 20

Oracle = the hit inside the shortlist moved to rank 1 (an ideal re-ranker): oracle R@1 = oracle R@10 = base R@20. 'Best' = the re-ranker with the highest test R@10 in that column (picked after the fact, so optimistic). Captured = (best − base) / (oracle R@10 − base). Single head: means over 50 runs. R@k here is pooled over all test queries, so it can differ from the main table's mean over folds in the third decimal (ensemble 0.124 / 0.305 vs 0.125 / 0.306).

| protocol | base R@1 | base R@10 | base R@20 = oracle R@10 | hit outside top 20 | MedR | best re-ranker | best R@10 | captured |
|---|---|---|---|---|---|---|---|---|
| Segment, single head | 0.017 | 0.105 | 0.182 | 0.818 | 86.9 | Sequence context | 0.113 | 11% |
| Segment, 10-seed ens. | 0.018 | 0.124 | 0.214 | 0.786 | 71.0 | Sequence context | 0.137 | 14% |
| Track, single head | 0.043 | 0.253 | 0.405 | 0.595 | 28.7 | LambdaMART (learned, val-trained) | 0.267 | 9% |
| Track, 10-seed ens. | 0.053 | 0.305 | 0.462 | 0.538 | 23.0 | Song-group de-dup. | 0.313 | 5% |

## T2. Where the hit moves inside the shortlist (10-seed ensemble, test queries pooled over folds)

Only queries whose hit is in the base top 20 can move. ↑ / ↓ = rank improved / worsened; into / out of top 10 = crossing rank 10; to / from rank 1. Net top-10 = into − out.

### Segment (2117 test queries, 453 with the hit in the base top 20)

| re-ranker | ↑ | ↓ | = | into top 10 | out of top 10 | net top 10 | to rank 1 | from rank 1 |
|---|---|---|---|---|---|---|---|---|
| AQE | 173 | 216 | 64 | 49 | 50 | -1 | 15 | 28 |
| α-QE | 157 | 173 | 123 | 40 | 43 | -3 | 7 | 13 |
| DBA | 196 | 173 | 84 | 62 | 55 | +7 | 12 | 21 |
| k-reciprocal | 111 | 186 | 156 | 34 | 25 | +9 | 10 | 17 |
| Mutual kNN | 44 | 37 | 372 | 10 | 6 | +4 | 3 | 2 |
| Dual softmax | 150 | 151 | 152 | 29 | 18 | +11 | 11 | 7 |
| Track agg. (max) | 85 | 132 | 236 | 27 | 11 | +16 | 0 | 0 |
| Track agg. (mean) | 173 | 129 | 151 | 36 | 30 | +6 | 9 | 5 |
| Track agg. (top-3 mean) | 163 | 130 | 160 | 33 | 25 | +8 | 5 | 7 |
| Song-group de-dup. | 190 | 165 | 98 | 70 | 82 | -12 | 0 | 0 |
| Key/tempo re-scoring | 132 | 129 | 192 | 34 | 29 | +5 | 5 | 6 |
| Sequence context | 160 | 103 | 190 | 50 | 23 | +27 | 9 | 7 |
| LambdaMART (learned, val-trained) | 201 | 210 | 42 | 87 | 88 | -1 | 27 | 37 |

### Track (894 test queries, 413 with the hit in the base top 20)

| re-ranker | ↑ | ↓ | = | into top 10 | out of top 10 | net top 10 | to rank 1 | from rank 1 |
|---|---|---|---|---|---|---|---|---|
| AQE | 138 | 178 | 97 | 34 | 58 | -24 | 15 | 7 |
| α-QE | 128 | 162 | 123 | 27 | 53 | -26 | 8 | 4 |
| DBA | 160 | 202 | 51 | 47 | 61 | -14 | 18 | 36 |
| k-reciprocal | 122 | 199 | 92 | 42 | 61 | -19 | 7 | 14 |
| Mutual kNN | 20 | 20 | 373 | 0 | 0 | +0 | 4 | 7 |
| Dual softmax | 146 | 125 | 142 | 22 | 30 | -8 | 7 | 11 |
| Track agg. (max) | 107 | 162 | 144 | 27 | 22 | +5 | 8 | 11 |
| Track agg. (mean) | 134 | 95 | 184 | 10 | 11 | -1 | 8 | 5 |
| Track agg. (top-3 mean) | 119 | 113 | 181 | 14 | 12 | +2 | 6 | 6 |
| Song-group de-dup. | 100 | 0 | 313 | 7 | 0 | +7 | 0 | 0 |
| LambdaMART (learned, val-trained) | 173 | 179 | 61 | 51 | 64 | -13 | 20 | 27 |

## T3. R@10 per fold (10-seed ensemble)

Folds better / worse = folds where the method's R@10 is above / below no re-ranking.

### Segment

| re-ranker | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | folds better / worse |
|---|---|---|---|---|---|---|---|
| No re-ranking | 0.119 | 0.132 | 0.101 | 0.163 | 0.112 | 0.125 | — |
| AQE | 0.121 | 0.127 | 0.108 | 0.155 | 0.112 | 0.124 | 2 / 2 |
| α-QE | 0.121 | 0.127 | 0.099 | 0.157 | 0.114 | 0.124 | 2 / 3 |
| DBA | 0.129 | 0.134 | 0.108 | 0.147 | 0.123 | 0.128 | 4 / 1 |
| k-reciprocal | 0.116 | 0.132 | 0.108 | 0.165 | 0.125 | 0.129 | 3 / 1 |
| Mutual kNN | 0.119 | 0.136 | 0.101 | 0.160 | 0.118 | 0.127 | 2 / 1 |
| Dual softmax | 0.121 | 0.134 | 0.114 | 0.165 | 0.116 | 0.130 | 5 / 0 |
| Track agg. (max) | 0.129 | 0.141 | 0.108 | 0.168 | 0.118 | 0.133 | 5 / 0 |
| Track agg. (mean) | 0.124 | 0.141 | 0.097 | 0.157 | 0.121 | 0.128 | 3 / 2 |
| Track agg. (top-3 mean) | 0.126 | 0.139 | 0.106 | 0.160 | 0.114 | 0.129 | 4 / 1 |
| Song-group de-dup. | 0.119 | 0.127 | 0.099 | 0.144 | 0.107 | 0.119 | 0 / 4 |
| Key/tempo re-scoring | 0.116 | 0.136 | 0.106 | 0.160 | 0.118 | 0.127 | 3 / 2 |
| Sequence context | 0.124 | 0.141 | 0.116 | 0.176 | 0.132 | 0.138 | 5 / 0 |
| LambdaMART (learned, val-trained) | 0.093 | 0.136 | 0.099 | 0.141 | 0.147 | 0.124 | 2 / 3 |

### Track

| re-ranker | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | folds better / worse |
|---|---|---|---|---|---|---|---|
| No re-ranking | 0.327 | 0.337 | 0.274 | 0.322 | 0.269 | 0.306 | — |
| AQE | 0.269 | 0.314 | 0.246 | 0.306 | 0.258 | 0.279 | 0 / 5 |
| α-QE | 0.275 | 0.320 | 0.240 | 0.290 | 0.258 | 0.277 | 0 / 5 |
| DBA | 0.292 | 0.314 | 0.246 | 0.306 | 0.290 | 0.290 | 1 / 4 |
| k-reciprocal | 0.275 | 0.326 | 0.246 | 0.279 | 0.296 | 0.284 | 1 / 4 |
| Mutual kNN | 0.327 | 0.337 | 0.274 | 0.322 | 0.269 | 0.306 | 0 / 0 |
| Dual softmax | 0.333 | 0.326 | 0.268 | 0.284 | 0.274 | 0.297 | 2 / 3 |
| Track agg. (max) | 0.333 | 0.354 | 0.246 | 0.339 | 0.285 | 0.311 | 4 / 1 |
| Track agg. (mean) | 0.327 | 0.320 | 0.274 | 0.328 | 0.274 | 0.305 | 2 / 1 |
| Track agg. (top-3 mean) | 0.345 | 0.320 | 0.268 | 0.322 | 0.285 | 0.308 | 2 / 2 |
| Song-group de-dup. | 0.339 | 0.349 | 0.285 | 0.328 | 0.269 | 0.314 | 4 / 0 |
| LambdaMART (learned, val-trained) | 0.298 | 0.371 | 0.251 | 0.268 | 0.269 | 0.292 | 1 / 3 |

## T4. Shortlist size (10-seed ensemble; every parameter re-chosen on val for each K)

Δ R@10 vs no re-ranking (which does not depend on K), paired per test query, 95% film-group cluster bootstrap; * = CI excludes 0. Oracle R@10 = share of queries whose hit is in the base top K (for K ≥ 10). K = 20 = the main table (reproduced exactly). K = 10 cannot change R@10 by construction (re-ordering the top 10 keeps the same 10 items) and is listed as a check. LambdaMART was trained for K = 20 only and is not swept.

### Segment (no re-ranking R@10 = 0.124)

| re-ranker | K = 10 | K = 20 | K = 50 | K = 100 |
|---|---|---|---|---|
| oracle R@10 | 0.124 | 0.214 | 0.409 | 0.602 |
| AQE | +0.000 [+0.000, +0.000] | −0.000 [−0.011, +0.009] | −0.006 [−0.017, +0.005] | −0.006 [−0.017, +0.004] |
| α-QE | +0.000 [+0.000, +0.000] | −0.001 [−0.010, +0.007] | −0.001 [−0.010, +0.008] | −0.001 [−0.011, +0.008] |
| DBA | +0.000 [+0.000, +0.000] | +0.003 [−0.007, +0.014] | +0.004 [−0.007, +0.015] | −0.004 [−0.015, +0.008] |
| k-reciprocal | +0.000 [+0.000, +0.000] | +0.004 [−0.002, +0.011] | −0.004 [−0.013, +0.005] | −0.009 [−0.018, −0.001]* |
| Mutual kNN | +0.000 [+0.000, +0.000] | +0.002 [−0.002, +0.006] | +0.002 [−0.002, +0.006] | +0.002 [−0.002, +0.006] |
| Dual softmax | +0.000 [+0.000, +0.000] | +0.005 [−0.001, +0.012] | +0.004 [−0.003, +0.011] | +0.004 [−0.003, +0.011] |
| Track agg. (max) | +0.000 [+0.000, +0.000] | +0.008 [+0.003, +0.013]* | +0.010 [+0.004, +0.016]* | +0.010 [+0.004, +0.016]* |
| Track agg. (mean) | +0.000 [+0.000, +0.000] | +0.003 [−0.005, +0.011] | +0.003 [−0.005, +0.011] | +0.003 [−0.005, +0.011] |
| Track agg. (top-3 mean) | +0.000 [+0.000, +0.000] | +0.004 [−0.003, +0.010] | +0.003 [−0.003, +0.010] | +0.003 [−0.003, +0.010] |
| Song-group de-dup. | +0.000 [+0.000, +0.000] | −0.006 [−0.020, +0.008] | −0.003 [−0.018, +0.011] | −0.003 [−0.018, +0.011] |
| Key/tempo re-scoring | +0.000 [+0.000, +0.000] | +0.002 [−0.005, +0.009] | +0.001 [−0.007, +0.008] | +0.001 [−0.007, +0.008] |
| Sequence context | +0.000 [+0.000, +0.000] | +0.013 [+0.004, +0.022]* | +0.019 [+0.009, +0.030]* | +0.019 [+0.009, +0.031]* |

### Track (no re-ranking R@10 = 0.305)

| re-ranker | K = 10 | K = 20 | K = 50 | K = 100 |
|---|---|---|---|---|
| oracle R@10 | 0.305 | 0.462 | 0.728 | 0.894 |
| AQE | +0.000 [+0.000, +0.000] | −0.027 [−0.049, −0.005]* | −0.029 [−0.054, −0.005]* | −0.032 [−0.058, −0.008]* |
| α-QE | +0.000 [+0.000, +0.000] | −0.029 [−0.049, −0.010]* | −0.030 [−0.051, −0.011]* | −0.031 [−0.052, −0.012]* |
| DBA | +0.000 [+0.000, +0.000] | −0.016 [−0.040, +0.009] | −0.036 [−0.060, −0.012]* | −0.036 [−0.060, −0.011]* |
| k-reciprocal | +0.000 [+0.000, +0.000] | −0.021 [−0.042, +0.000] | −0.012 [−0.032, +0.006] | −0.013 [−0.033, +0.005] |
| Mutual kNN | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| Dual softmax | +0.000 [+0.000, +0.000] | −0.009 [−0.026, +0.007] | −0.008 [−0.024, +0.008] | −0.008 [−0.024, +0.008] |
| Track agg. (max) | +0.000 [+0.000, +0.000] | +0.006 [−0.009, +0.019] | +0.012 [−0.005, +0.029] | +0.012 [−0.005, +0.029] |
| Track agg. (mean) | +0.000 [+0.000, +0.000] | −0.001 [−0.012, +0.009] | +0.000 [−0.011, +0.011] | +0.000 [−0.011, +0.011] |
| Track agg. (top-3 mean) | +0.000 [+0.000, +0.000] | +0.002 [−0.009, +0.013] | +0.007 [−0.005, +0.018] | +0.007 [−0.005, +0.018] |
| Song-group de-dup. | +0.000 [+0.000, +0.000] | +0.008 [+0.003, +0.014]* | +0.008 [+0.003, +0.014]* | +0.008 [+0.003, +0.014]* |

## T5. Decoder per fold (cleaned 1,011 pairs; HMM, gold-path prior, α tuned; 92 test sequences)

| fold | seq. | system | track_acc | switch_f1 | TBIoU | switches_per_min | jumps_per_min | align_success | λ (val) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 18 | Final HMM | 0.036 | 0.747 | 0.831 | 4.41 | 0.21 | 0.883 | 0.354 |
| 0 | 18 | + k-reciprocal | 0.036 | 0.740 | 0.830 | 4.40 | 0.19 | 0.873 | 0.354 |
| 0 | 18 | + Sequence context | 0.037 | 0.733 | 0.810 | 4.20 | 0.21 | 0.841 | 0.707 |
| 1 | 18 | Final HMM | 0.048 | 0.711 | 0.795 | 4.34 | 0.21 | 0.794 | 0.5 |
| 1 | 18 | + k-reciprocal | 0.057 | 0.711 | 0.795 | 4.34 | 0.21 | 0.794 | 0.5 |
| 1 | 18 | + Sequence context | 0.064 | 0.704 | 0.801 | 4.27 | 0.15 | 0.785 | 0.707 |
| 2 | 19 | Final HMM | 0.070 | 0.639 | 0.752 | 3.64 | 0.00 | 0.683 | 1.41 |
| 2 | 19 | + k-reciprocal | 0.067 | 0.656 | 0.774 | 4.32 | 0.02 | 0.771 | 0.25 |
| 2 | 19 | + Sequence context | 0.067 | 0.631 | 0.744 | 3.60 | 0.02 | 0.671 | 1.41 |
| 3 | 18 | Final HMM | 0.036 | 0.719 | 0.788 | 4.16 | 0.12 | 0.788 | 0.5 |
| 3 | 18 | + k-reciprocal | 0.042 | 0.698 | 0.768 | 4.02 | 0.12 | 0.752 | 0.707 |
| 3 | 18 | + Sequence context | 0.037 | 0.687 | 0.767 | 4.02 | 0.12 | 0.739 | 0.707 |
| 4 | 19 | Final HMM | 0.048 | 0.620 | 0.742 | 3.81 | 0.02 | 0.677 | 1.41 |
| 4 | 19 | + k-reciprocal | 0.035 | 0.632 | 0.761 | 4.03 | 0.04 | 0.710 | 1 |
| 4 | 19 | + Sequence context | 0.045 | 0.622 | 0.749 | 4.09 | 0.13 | 0.702 | 0.5 |

