# Similarity functions on the title split (D145)

Title split (folds_title.json), 5 folds × 10 seeds. Every head is the REF trainer (InfoNCE, siglip-both settings, early stopping on val); only the test-time similarity changes. Hubness banks = the fold's training clips (1,500 random training segments for the segment protocol, all training clips for the track protocol); no test query is used to normalise anything. Every weight (CLAP vs MERT w, max-sim α) and every hubness parameter (CSLS k, inverted-softmax / QB-Norm β) is chosen per run on the fold's val films (criterion R@10 + 0.1·MRR), then test is scored once. Single = one head per run, mean ± std over 50 runs; Ensemble = mean of the 10 seeds' standardised scores per fold, mean ± std over 5 folds. CI: paired vs REF on per-query hit@10 (single: averaged over seeds), film-group cluster bootstrap, 10,000 resamples; * = excludes 0.

## Segment retrieval (segment_all, one hit, gallery ≈ 480 windows)

| similarity | single R@10 | single R@1 | single MRR | Δ R@10 vs REF [95% CI] | ensemble R@10 | ensemble MRR | Δ ens R@10 [95% CI] | typical val pick |
|---|---|---|---|---|---|---|---|---|
| REF (cosine, mean-pooled) | 0.095 ± 0.017 | 0.015 ± 0.004 | 0.048 ± 0.007 | — | 0.115 ± 0.009 | 0.060 ± 0.007 | — | α 0 |
| REF + CSLS | 0.094 ± 0.017 | 0.013 ± 0.005 | 0.046 ± 0.007 | -0.001 [-0.004, +0.003] | 0.116 ± 0.009 | 0.058 ± 0.003 | +0.000 [-0.009, +0.010] | α 0, csls 50 |
| REF + inverted softmax | 0.093 ± 0.018 | 0.013 ± 0.006 | 0.046 ± 0.008 | -0.002 [-0.007, +0.002] | 0.109 ± 0.019 | 0.056 ± 0.007 | -0.007 [-0.018, +0.003] | α 0, is 0.5 |
| REF + QB-Norm | 0.093 ± 0.018 | 0.013 ± 0.005 | 0.047 ± 0.008 | -0.001 [-0.005, +0.002] | 0.108 ± 0.016 | 0.055 ± 0.008 | -0.007 [-0.017, +0.002] | α 0, qb 0.5 |
| REF max-sim | 0.095 ± 0.018 | 0.014 ± 0.005 | 0.047 ± 0.007 | +0.000 [-0.003, +0.003] | 0.119 ± 0.013 | 0.056 ± 0.005 | +0.003 [-0.003, +0.009] | α 1 |
| REF mean + max-sim (α on val) | 0.096 ± 0.017 | 0.015 ± 0.005 | 0.048 ± 0.007 | +0.001 [-0.001, +0.002] | 0.116 ± 0.012 | 0.060 ± 0.006 | +0.001 [-0.003, +0.004] | α 0.25 |
| CLAP head only | 0.088 ± 0.016 | 0.013 ± 0.005 | 0.044 ± 0.007 | -0.007 [-0.013, -0.001]* | 0.110 ± 0.009 | 0.054 ± 0.006 | -0.006 [-0.018, +0.007] | α 0, w_CLAP 1 |
| MERT head only | 0.073 ± 0.014 | 0.009 ± 0.004 | 0.037 ± 0.007 | -0.023 [-0.033, -0.013]* | 0.088 ± 0.015 | 0.043 ± 0.006 | -0.029 [-0.046, -0.012]* | α 0 |
| CLAP + MERT fusion (w on val) | 0.099 ± 0.018 | 0.015 ± 0.005 | 0.049 ± 0.007 | +0.003 [-0.003, +0.010] | 0.114 ± 0.010 | 0.060 ± 0.005 | -0.002 [-0.015, +0.011] | α 0, w_CLAP 0.6 |
| CLAP + MERT fusion + max-sim | 0.099 ± 0.021 | 0.015 ± 0.005 | 0.048 ± 0.008 | +0.004 [-0.003, +0.011] | 0.118 ± 0.017 | 0.060 ± 0.005 | +0.002 [-0.011, +0.014] | α 0.5, w_CLAP 0.6 |
| Combination: fusion + max-sim + hubness | 0.098 ± 0.023 | 0.014 ± 0.005 | 0.048 ± 0.008 | +0.003 [-0.004, +0.010] | 0.117 ± 0.014 | 0.058 ± 0.005 | +0.002 [-0.012, +0.015] | α 0.5, w_CLAP 0.6 |
| All three heads + max-sim + hubness | 0.106 ± 0.014 | 0.016 ± 0.005 | 0.052 ± 0.007 | +0.011 [+0.006, +0.017]* | 0.113 ± 0.007 | 0.059 ± 0.006 | -0.002 [-0.015, +0.010] | α 0.5, w_CLAP 0.5 |

## Track retrieval (track_test_fold, gallery ≈ 196 files)

| similarity | single R@10 | single R@1 | single MRR | Δ R@10 vs REF [95% CI] | ensemble R@10 | ensemble MRR | Δ ens R@10 [95% CI] | typical val pick |
|---|---|---|---|---|---|---|---|---|
| REF (cosine, mean-pooled) | 0.242 ± 0.034 | 0.035 ± 0.011 | 0.107 ± 0.014 | — | 0.297 ± 0.045 | 0.128 ± 0.006 | — | α 0 |
| REF + CSLS | 0.243 ± 0.033 | 0.036 ± 0.012 | 0.108 ± 0.014 | +0.001 [-0.006, +0.007] | 0.298 ± 0.039 | 0.125 ± 0.012 | +0.001 [-0.017, +0.019] | α 0, csls 1 |
| REF + inverted softmax | 0.246 ± 0.035 | 0.038 ± 0.014 | 0.109 ± 0.016 | +0.004 [-0.005, +0.013] | 0.285 ± 0.018 | 0.131 ± 0.010 | -0.012 [-0.032, +0.006] | α 0, is 0.5 |
| REF + QB-Norm | 0.245 ± 0.034 | 0.038 ± 0.014 | 0.109 ± 0.016 | +0.003 [-0.005, +0.012] | 0.288 ± 0.018 | 0.131 ± 0.010 | -0.009 [-0.029, +0.009] | α 0, qb 0.5 |
| REF max-sim | 0.244 ± 0.030 | 0.033 ± 0.013 | 0.105 ± 0.014 | +0.002 [-0.005, +0.008] | 0.307 ± 0.051 | 0.120 ± 0.007 | +0.010 [-0.007, +0.029] | α 1 |
| REF mean + max-sim (α on val) | 0.244 ± 0.033 | 0.033 ± 0.013 | 0.105 ± 0.014 | +0.002 [-0.002, +0.006] | 0.304 ± 0.050 | 0.124 ± 0.008 | +0.007 [-0.003, +0.017] | α 0.25 |
| CLAP head only | 0.229 ± 0.034 | 0.030 ± 0.012 | 0.098 ± 0.012 | -0.013 [-0.025, -0.002]* | 0.280 ± 0.039 | 0.119 ± 0.012 | -0.016 [-0.039, +0.006] | α 0, w_CLAP 1 |
| MERT head only | 0.197 ± 0.031 | 0.025 ± 0.011 | 0.085 ± 0.012 | -0.046 [-0.065, -0.027]* | 0.223 ± 0.022 | 0.097 ± 0.006 | -0.074 [-0.110, -0.040]* | α 0 |
| CLAP + MERT fusion (w on val) | 0.248 ± 0.031 | 0.034 ± 0.012 | 0.108 ± 0.013 | +0.006 [-0.006, +0.018] | 0.286 ± 0.030 | 0.126 ± 0.013 | -0.011 [-0.033, +0.011] | α 0, w_CLAP 0.6 |
| CLAP + MERT fusion + max-sim | 0.249 ± 0.030 | 0.034 ± 0.013 | 0.108 ± 0.014 | +0.007 [-0.005, +0.019] | 0.282 ± 0.029 | 0.121 ± 0.013 | -0.015 [-0.042, +0.012] | α 0.25, w_CLAP 0.6 |
| Combination: fusion + max-sim + hubness | 0.253 ± 0.031 | 0.040 ± 0.015 | 0.113 ± 0.015 | +0.011 [-0.001, +0.023] | 0.298 ± 0.033 | 0.129 ± 0.016 | +0.001 [-0.025, +0.027] | α 0.25, w_CLAP 0.7 |
| All three heads + max-sim + hubness | 0.268 ± 0.029 | 0.042 ± 0.013 | 0.118 ± 0.012 | +0.026 [+0.015, +0.036]* | 0.304 ± 0.009 | 0.130 ± 0.008 | +0.007 [-0.019, +0.032] | α 0.25, w_CLAP 0.5, is 0.5 |

