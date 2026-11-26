# Improvement round on the title split (exploratory)

Title split (data/manifests/folds_title.json), 5 folds × seeds 0, 1, 2; mean ± std over folds × seeds. Shipped siglip-both trainer settings for every row (CLAP+MERT music, ≤ 30 clips per film per epoch, one segment per video per epoch, same-track masking, 25% frame drop, ±1 s jitter); early stopping on each fold's `val` (10% of its training films, v→m segment R@10, patience 20). Test = M5 protocol of `bgm train semantic`: segment = segment_all (gallery: every test segment's aligned window), track = track_test_fold. Defaults exactly, nothing tuned. Sorted by segment R@10. CI: per-query hit@10 averaged over seeds, paired against REF, film-group cluster bootstrap (10,000 resamples); * = excludes 0.

| config | runs | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF (paired) | Δ seg R@10 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| infonce_intra_do05 | 15 | 0.016 ± 0.005 | 0.061 ± 0.010 | 0.106 ± 0.012 | 92.4 ± 16.0 | 0.052 ± 0.006 | 0.261 ± 0.026 | +0.014 ± 0.019 | +0.013 [+0.004, +0.022]* |
| REF_ema | 15 | 0.017 ± 0.007 | 0.057 ± 0.017 | 0.106 ± 0.021 | 92.2 ± 15.8 | 0.052 ± 0.010 | 0.271 ± 0.047 | +0.013 ± 0.009 | +0.013 [+0.006, +0.020]* |
| REF_do05 | 15 | 0.018 ± 0.006 | 0.060 ± 0.012 | 0.104 ± 0.015 | 91.9 ± 16.3 | 0.053 ± 0.009 | 0.265 ± 0.028 | +0.012 ± 0.022 | +0.011 [+0.001, +0.020]* |
| infonce_intra_ema | 15 | 0.016 ± 0.005 | 0.060 ± 0.015 | 0.104 ± 0.017 | 91.6 ± 11.2 | 0.052 ± 0.008 | 0.260 ± 0.039 | +0.011 ± 0.014 | +0.011 [+0.003, +0.019]* |
| siglip_intra | 15 | 0.016 ± 0.007 | 0.056 ± 0.010 | 0.101 ± 0.018 | 101.0 ± 24.0 | 0.049 ± 0.009 | 0.247 ± 0.037 | +0.008 ± 0.023 | +0.008 [-0.003, +0.018] |
| infonce_intra | 15 | 0.015 ± 0.004 | 0.058 ± 0.010 | 0.100 ± 0.013 | 101.0 ± 13.9 | 0.049 ± 0.006 | 0.252 ± 0.025 | +0.008 ± 0.020 | +0.008 [+0.000, +0.015]* |
| infonce_intra_rdrop | 15 | 0.014 ± 0.004 | 0.057 ± 0.009 | 0.099 ± 0.013 | 102.6 ± 14.0 | 0.048 ± 0.006 | 0.249 ± 0.033 | +0.006 ± 0.019 | +0.006 [-0.003, +0.015] |
| REF | 15 | 0.015 ± 0.006 | 0.056 ± 0.018 | 0.092 ± 0.022 | 99.9 ± 18.6 | 0.048 ± 0.009 | 0.235 ± 0.046 | — | — |

Definitions: siglip = pairwise sigmoid loss (t = e^t', t' init log 10, b init −10); inter_intra_nce = InfoNCE(v, m) + 0.5·[InfoNCE(v′, v″) + InfoNCE(m′, m″)] with two 25% frame-drop views of the video and two ±1 s jitter crops of the music; infonce_intra = InfoNCE + 0.5 · intra (rows of the frozen vs projected similarity, both sides); infonce_rdrop = InfoNCE + 1.0 · R-Drop (symmetric KL of two dropout passes' similarity softmaxes).
