# Planned losses on the title split

Title split (data/manifests/folds_title.json), 5 folds × seeds 0, 1, 2; mean ± std over folds × seeds. Shipped siglip-both trainer settings for every row (CLAP+MERT music, ≤ 30 clips per film per epoch, one segment per video per epoch, same-track masking, 25% frame drop, ±1 s jitter); early stopping on each fold's `val` (10% of its training films, v→m segment R@10, patience 20). Test = M5 protocol of `bgm train semantic`: segment = segment_all (gallery: every test segment's aligned window), track = track_test_fold. Defaults exactly, nothing tuned. Sorted by segment R@10. CI: per-query hit@10 averaged over seeds, paired against REF, film-group cluster bootstrap (10,000 resamples); * = excludes 0.

| config | runs | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF (paired) | Δ seg R@10 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| infonce_rdrop | 15 | 0.017 ± 0.005 | 0.061 ± 0.012 | 0.102 ± 0.015 | 95.7 ± 11.2 | 0.052 ± 0.007 | 0.251 ± 0.036 | +0.009 ± 0.019 | +0.009 [+0.000, +0.019]* |
| asac_no_eps | 15 | 0.015 ± 0.005 | 0.059 ± 0.009 | 0.101 ± 0.010 | 100.0 ± 13.3 | 0.049 ± 0.005 | 0.252 ± 0.024 | +0.008 ± 0.017 | +0.008 [+0.001, +0.016]* |
| infonce_intra | 15 | 0.015 ± 0.004 | 0.058 ± 0.010 | 0.100 ± 0.013 | 101.0 ± 13.9 | 0.049 ± 0.006 | 0.252 ± 0.025 | +0.008 ± 0.020 | +0.008 [+0.000, +0.015]* |
| inter_intra_nce | 15 | 0.015 ± 0.006 | 0.054 ± 0.010 | 0.097 ± 0.017 | 103.4 ± 16.4 | 0.048 ± 0.008 | 0.242 ± 0.037 | +0.005 ± 0.020 | +0.005 [-0.004, +0.013] |
| asac_no_weights | 15 | 0.014 ± 0.004 | 0.057 ± 0.013 | 0.097 ± 0.013 | 102.4 ± 13.4 | 0.048 ± 0.007 | 0.238 ± 0.020 | +0.004 ± 0.018 | +0.005 [-0.004, +0.013] |
| asac_no_tau | 15 | 0.014 ± 0.005 | 0.056 ± 0.013 | 0.095 ± 0.017 | 103.9 ± 14.5 | 0.048 ± 0.007 | 0.238 ± 0.024 | +0.003 ± 0.016 | +0.003 [-0.006, +0.011] |
| asac_full | 15 | 0.015 ± 0.004 | 0.056 ± 0.013 | 0.094 ± 0.016 | 104.7 ± 15.9 | 0.048 ± 0.007 | 0.233 ± 0.021 | +0.002 ± 0.018 | +0.002 [-0.007, +0.010] |
| siglip | 15 | 0.014 ± 0.005 | 0.052 ± 0.015 | 0.094 ± 0.020 | 97.8 ± 22.5 | 0.048 ± 0.010 | 0.244 ± 0.039 | +0.002 ± 0.025 | +0.001 [-0.010, +0.012] |
| REF | 15 | 0.015 ± 0.006 | 0.056 ± 0.018 | 0.092 ± 0.022 | 99.9 ± 18.6 | 0.048 ± 0.009 | 0.235 ± 0.046 | — | — |

Definitions: siglip = pairwise sigmoid loss (t = e^t', t' init log 10, b init −10); inter_intra_nce = InfoNCE(v, m) + 0.5·[InfoNCE(v′, v″) + InfoNCE(m′, m″)] with two 25% frame-drop views of the video and two ±1 s jitter crops of the music; infonce_intra = InfoNCE + 0.5 · intra (rows of the frozen vs projected similarity, both sides); infonce_rdrop = InfoNCE + 1.0 · R-Drop (symmetric KL of two dropout passes' similarity softmaxes).
