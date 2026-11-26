# Stitching ablation (title split, 5 folds; test = frozen blocked sequences, BaSSL C-head own cuts)

Mean ± std over the 5 test folds (each fold = mean over its test sequences; CRF rows also averaged over 3 training seeds). Every method decodes the same shortlist (10-head ensemble, top-20 windows + continuations); every hyperparameter tuned on the fold's val sequences (objective = track acc. + switch F1).

| # | Method | Track acc. (R@1 of path) | Switch F1 ±1 s | TBIoU | Switches / min | Jumps / min | Deviation T→P (s) | Switch precision ±1 s | Switch recall = alignment success ±1 s | |switches/min − 3.10| | Entropy | Runtime / video (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Greedy top-1 | 0.040 ± 0.009 | 0.703 ± 0.053 | 0.797 ± 0.029 | 3.81 ± 0.05 | 0.81 ± 0.09 | 0.61 ± 0.73 | 0.660 ± 0.048 | 0.789 ± 0.053 | 0.86 ± 0.06 | 0.804 ± 0.019 | 0.0 ± 0.0 |
| 2 | Greedy + smoothing | 0.039 ± 0.009 | 0.688 ± 0.052 | 0.773 ± 0.031 | 3.57 ± 0.14 | 0.89 ± 0.21 | 2.17 ± 1.45 | 0.679 ± 0.045 | 0.754 ± 0.053 | 0.86 ± 0.07 | 0.796 ± 0.024 | 0.1 ± 0.0 |
| 3 | Two-stage change-point + retrieve | 0.083 ± 0.035 | 0.727 ± 0.058 | 0.789 ± 0.027 | 3.26 ± 0.35 | 0.37 ± 0.08 | 1.05 ± 0.45 | 0.736 ± 0.077 | 0.756 ± 0.045 | 0.70 ± 0.14 | 0.754 ± 0.020 | 7.7 ± 2.9 |
| 4 | Prétet-style DTW (one track) | 0.025 ± 0.012 | 0.000 ± 0.000 | 0.127 ± 0.008 | 0.00 ± 0.00 | 4.62 ± 0.10 | nan ± nan | nan ± nan | 0.000 ± 0.000 | 3.10 ± 0.00 | 0.538 ± 0.019 | 1.3 ± 0.2 |
| 5 | Viterbi, hand-set costs (existing) | 0.038 ± 0.010 | 0.687 ± 0.069 | 0.783 ± 0.065 | 3.67 ± 0.51 | 0.58 ± 0.54 | 1.30 ± 1.44 | 0.662 ± 0.025 | 0.763 ± 0.113 | 0.89 ± 0.14 | 0.791 ± 0.021 | 1.1 ± 0.4 |
| 6 | HMM, derived costs, λ = 1 | 0.046 ± 0.006 | 0.655 ± 0.049 | 0.724 ± 0.029 | 2.97 ± 0.21 | 1.06 ± 0.13 | 1.54 ± 0.96 | 0.692 ± 0.036 | 0.655 ± 0.056 | 0.63 ± 0.07 | 0.783 ± 0.014 | 0.8 ± 0.3 |
| 7 | HMM, derived costs, λ tuned | 0.036 ± 0.009 | 0.679 ± 0.049 | 0.755 ± 0.040 | 3.29 ± 0.34 | 0.96 ± 0.18 | 0.84 ± 0.83 | 0.683 ± 0.036 | 0.709 ± 0.070 | 0.68 ± 0.08 | 0.791 ± 0.021 | 0.8 ± 0.3 |
| 8 | CRF, no context | 0.053 ± 0.011 | 0.631 ± 0.052 | 0.694 ± 0.058 | 2.87 ± 0.31 | 0.27 ± 0.07 | 1.95 ± 1.47 | 0.676 ± 0.044 | 0.622 ± 0.067 | 0.56 ± 0.17 | 0.753 ± 0.040 | 22.8 ± 2.2 |
| 9 | CRF, with context | 0.057 ± 0.011 | 0.652 ± 0.052 | 0.718 ± 0.040 | 3.01 ± 0.20 | 0.15 ± 0.10 | 1.64 ± 0.75 | 0.678 ± 0.046 | 0.657 ± 0.058 | 0.55 ± 0.05 | 0.761 ± 0.023 | 27.5 ± 10.4 |
| 10 | CRF, with context + derived prior | 0.058 ± 0.012 | 0.639 ± 0.052 | 0.702 ± 0.047 | 2.92 ± 0.25 | 0.23 ± 0.07 | 1.94 ± 1.10 | 0.678 ± 0.045 | 0.636 ± 0.062 | 0.57 ± 0.11 | 0.755 ± 0.032 | 18.8 ± 2.0 |
| 11 | HMM, π from gold paths, λ = 1 | 0.052 ± 0.005 | 0.692 ± 0.054 | 0.797 ± 0.037 | 3.88 ± 0.10 | 0.05 ± 0.04 | 0.51 ± 0.66 | 0.637 ± 0.048 | 0.787 ± 0.053 | 0.87 ± 0.08 | 0.799 ± 0.024 | 0.5 ± 0.0 |
| 12 | HMM, π from gold paths, λ tuned | 0.044 ± 0.009 | 0.694 ± 0.052 | 0.805 ± 0.033 | 4.16 ± 0.20 | 0.06 ± 0.03 | 0.21 ± 0.22 | 0.622 ± 0.046 | 0.820 ± 0.056 | 1.10 ± 0.19 | 0.813 ± 0.020 | 0.5 ± 0.0 |
| 13 | CRF, context + prior from gold paths | 0.059 ± 0.010 | 0.638 ± 0.050 | 0.707 ± 0.041 | 2.99 ± 0.22 | 0.15 ± 0.07 | 1.88 ± 0.82 | 0.667 ± 0.044 | 0.642 ± 0.057 | 0.54 ± 0.07 | 0.756 ± 0.029 | 19.5 ± 6.2 |
| 14 | Switch at every cut (calibration) | 0.041 ± 0.008 | 0.689 ± 0.052 | 0.808 ± 0.033 | 4.57 ± 0.07 | 0.05 ± 0.03 | 0.07 ± 0.06 | 0.597 ± 0.050 | 0.856 ± 0.051 | 1.48 ± 0.08 | 0.828 ± 0.018 | 0.1 ± 0.0 |
| 15 | Never switch (calibration) | 0.028 ± 0.007 | 0.000 ± 0.000 | 0.127 ± 0.008 | 0.00 ± 0.00 | 4.62 ± 0.09 | nan ± nan | nan ± nan | 0.000 ± 0.000 | 3.10 ± 0.00 | 0.508 ± 0.042 | 0.3 ± 0.0 |

- Shortlist (identical for every method): track R@10 0.281 ± 0.040, MRR 0.112 ± 0.009 (rank of the true track among the shortlisted files, per test scene). R@10 / MRR do not depend on the stitcher, so the table reports the decoded path's track accuracy (its R@1) instead.
- True changes: 3.10 / min. Alignment success = share of true changes with a predicted switch within ±1 s; deviation = median distance from a true change to the nearest predicted switch.
- Test sequences: 112; galleries ~196 files per test fold.
- Hand-set row reproduces D143's combo/own R0_tuned exactly in every fold: f0 True/0e+00, f1 True/0e+00, f2 True/0e+00, f3 True/0e+00, f4 True/0e+00

## Tuned hyperparameters (per fold)

| fold | τ | L (s) | ρ | smooth w | two-stage | hand-set λ_switch / λ_jump | HMM λ | CRF γ (prior, seeds) | CRF α / λ (ctx+prior, seed 0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0692 | 9.42 | 0.338 | 5 | threshold 0.221 | 1 / 2 | 1 | 0.01, 0.01, 0.01 | 13.84 / 1.10 |
| 1 | 0.0677 | 9.42 | 0.330 | 3 | kts 0.300 | 0.5 / 0.5 | 0.354 | 0.1, 0.1, 0.01 | 21.11 / 1.28 |
| 2 | 0.0673 | 9.46 | 0.359 | 3 | kts 0.200 | 0.5 / 0.5 | 0.5 | 0.01, 0.01, 0.01 | 21.46 / 1.16 |
| 3 | 0.0667 | 9.49 | 0.367 | 3 | threshold 0.323 | 0.5 / 2 | 0.5 | 0.01, 0.01, 0.01 | 21.61 / 1.13 |
| 4 | 0.0675 | 9.43 | 0.352 | 3 | threshold 0.119 | 0.5 / 0.5 | 0.25 | 0.01, 0.01, 0.01 | 21.29 / 1.12 |

## CRF checks

Gold window among the plain candidates (train sequences use the frozen heads' own training clips):

| fold | train | val | test | gold sim percentile train / val / test |
|---|---|---|---|---|
| 0 | 0.550 | 0.164 | 0.108 | 0.48 / 0.15 / 0.14 |
| 1 | 0.719 | 0.287 | 0.070 | 0.66 / 0.25 / 0.10 |
| 2 | 0.749 | 0.164 | 0.074 | 0.68 / 0.16 / 0.12 |
| 3 | 0.734 | 0.221 | 0.103 | 0.67 / 0.20 / 0.11 |
| 4 | 0.742 | 0.391 | 0.110 | 0.65 / 0.28 / 0.12 |

Shortcut guard (val, seed 0): predicted vs real switch rate.

| fold | variant | pred. switches / min | true / min | pred. switch / boundary | posterior P(switch) | true change / boundary | derived p_change |
|---|---|---|---|---|---|---|---|
| 0 | crf_noctx | 2.64 | 3.38 | 0.586 | 0.739 | 0.601 | 0.682 |
| 0 | crf_ctx | 2.98 | 3.38 | 0.660 | 0.734 | 0.601 | 0.682 |
| 0 | crf_ctx_prior | 2.49 | 3.38 | 0.551 | 0.694 | 0.601 | 0.682 |
| 1 | crf_noctx | 3.00 | 3.22 | 0.616 | 0.733 | 0.529 | 0.667 |
| 1 | crf_ctx | 2.96 | 3.22 | 0.611 | 0.713 | 0.529 | 0.667 |
| 1 | crf_ctx_prior | 2.94 | 3.22 | 0.604 | 0.726 | 0.529 | 0.667 |
| 2 | crf_noctx | 3.21 | 3.16 | 0.669 | 0.795 | 0.521 | 0.662 |
| 2 | crf_ctx | 3.36 | 3.16 | 0.707 | 0.792 | 0.521 | 0.662 |
| 2 | crf_ctx_prior | 3.34 | 3.16 | 0.702 | 0.795 | 0.521 | 0.662 |
| 3 | crf_noctx | 3.17 | 3.34 | 0.689 | 0.819 | 0.594 | 0.674 |
| 3 | crf_ctx | 3.14 | 3.34 | 0.685 | 0.832 | 0.594 | 0.674 |
| 3 | crf_ctx_prior | 3.18 | 3.34 | 0.689 | 0.832 | 0.594 | 0.674 |
| 4 | crf_noctx | 3.35 | 3.39 | 0.760 | 0.866 | 0.654 | 0.694 |
| 4 | crf_ctx | 3.39 | 3.39 | 0.768 | 0.869 | 0.654 | 0.694 |
| 4 | crf_ctx_prior | 3.37 | 3.39 | 0.764 | 0.864 | 0.654 | 0.694 |

## Prior counted from the training gold paths (rows 11-13)

| fold | L (s, MLE) | ρ | continue | jump | switch | π_c / π_j / π_s (raw frequencies) |
|---|---|---|---|---|---|---|
| 0 | 8.13 | 0.033 | 711 | 70 | 2042 | 0.252 / 0.025 / 0.723 |
| 1 | 8.68 | 0.024 | 786 | 51 | 2090 | 0.269 / 0.017 / 0.714 |
| 2 | 8.86 | 0.019 | 814 | 41 | 2071 | 0.278 / 0.014 / 0.708 |
| 3 | 9.76 | 0.033 | 911 | 67 | 1959 | 0.310 / 0.023 / 0.667 |
| 4 | 8.65 | 0.034 | 825 | 74 | 2080 | 0.277 / 0.025 / 0.698 |
