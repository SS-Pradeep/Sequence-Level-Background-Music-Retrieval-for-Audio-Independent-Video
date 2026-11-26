# Tag-free loss bank on the title split (inter / intra / adaptive)

Title split (data/manifests/folds_title.json), 5 folds × seeds 0, 1, 2, 3, 4, 5, 6, 7, 8, 9; mean ± std over folds × seeds. Shipped siglip-both trainer settings for every row (CLAP+MERT music, ≤ 30 clips per film per epoch, one segment per video per epoch, same-track masking, 25% frame drop, ±1 s jitter); early stopping on each fold's `val` (10% of its training films, v→m segment R@10, patience 20). Test = M5 protocol of `bgm train semantic`: segment = segment_all (gallery: every test segment's aligned window), track = track_test_fold. Defaults exactly, nothing tuned. Sorted by segment R@10. CI: per-query hit@10 averaged over seeds, paired against REF, film-group cluster bootstrap (10,000 resamples); * = excludes 0.

| config | runs | seg R@1 | seg R@5 | seg R@10 | seg MedR ↓ | seg MRR | track R@10 | Δ seg R@10 vs REF (paired) | Δ seg R@10 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| siglip | 50 | 0.014 ± 0.006 | 0.054 ± 0.013 | 0.097 ± 0.018 | 99.7 ± 23.3 | 0.048 ± 0.008 | 0.245 ± 0.040 | +0.002 ± 0.020 | +0.002 [-0.006, +0.009] |
| infonce_vv | 50 | 0.016 ± 0.005 | 0.054 ± 0.012 | 0.096 ± 0.016 | 98.5 ± 15.3 | 0.049 ± 0.007 | 0.247 ± 0.032 | +0.001 ± 0.019 | +0.001 [-0.004, +0.006] |
| vsepp | 50 | 0.016 ± 0.006 | 0.056 ± 0.015 | 0.096 ± 0.019 | 99.7 ± 16.6 | 0.049 ± 0.009 | 0.243 ± 0.037 | +0.001 ± 0.013 | +0.001 [-0.003, +0.005] |
| infonce_fixtau | 50 | 0.015 ± 0.005 | 0.055 ± 0.014 | 0.096 ± 0.017 | 99.0 ± 15.9 | 0.048 ± 0.008 | 0.242 ± 0.034 | +0.001 ± 0.005 | +0.001 [-0.001, +0.003] |
| inter_intra_nce | 50 | 0.015 ± 0.005 | 0.052 ± 0.010 | 0.095 ± 0.013 | 102.5 ± 15.0 | 0.047 ± 0.006 | 0.242 ± 0.030 | +0.001 ± 0.015 | +0.001 [-0.004, +0.005] |
| multipos | 50 | 0.015 ± 0.005 | 0.054 ± 0.009 | 0.095 ± 0.013 | 102.6 ± 14.5 | 0.048 ± 0.006 | 0.243 ± 0.028 | +0.000 ± 0.016 | +0.000 [-0.005, +0.006] |
| infonce_intra | 50 | 0.015 ± 0.005 | 0.054 ± 0.010 | 0.095 ± 0.014 | 100.9 ± 15.4 | 0.048 ± 0.006 | 0.244 ± 0.032 | +0.000 ± 0.017 | +0.000 [-0.004, +0.004] |
| REF | 50 | 0.015 ± 0.004 | 0.055 ± 0.013 | 0.095 ± 0.017 | 99.7 ± 15.6 | 0.048 ± 0.007 | 0.242 ± 0.034 | — | — |
| infonce_supcon_m | 50 | 0.015 ± 0.004 | 0.053 ± 0.012 | 0.094 ± 0.016 | 101.8 ± 15.5 | 0.048 ± 0.007 | 0.236 ± 0.034 | -0.001 ± 0.015 | -0.001 [-0.006, +0.004] |
| asac_full | 50 | 0.015 ± 0.005 | 0.054 ± 0.012 | 0.093 ± 0.015 | 105.4 ± 16.9 | 0.047 ± 0.007 | 0.233 ± 0.031 | -0.001 ± 0.016 | -0.001 [-0.006, +0.003] |
| debiased | 50 | 0.014 ± 0.005 | 0.052 ± 0.013 | 0.093 ± 0.019 | 99.7 ± 19.2 | 0.047 ± 0.008 | 0.234 ± 0.037 | -0.002 ± 0.019 | -0.002 [-0.009, +0.004] |
| all_uncert | 50 | 0.014 ± 0.005 | 0.053 ± 0.010 | 0.093 ± 0.016 | 107.0 ± 16.6 | 0.046 ± 0.007 | 0.239 ± 0.032 | -0.002 ± 0.019 | -0.002 [-0.008, +0.003] |
| recipe_fixed | 50 | 0.014 ± 0.005 | 0.051 ± 0.009 | 0.091 ± 0.014 | 106.1 ± 15.5 | 0.046 ± 0.006 | 0.232 ± 0.033 | -0.004 ± 0.018 | -0.004 [-0.009, +0.002] |
| recipe_uncert | 50 | 0.014 ± 0.005 | 0.050 ± 0.010 | 0.091 ± 0.015 | 106.9 ± 16.0 | 0.045 ± 0.006 | 0.232 ± 0.033 | -0.004 ± 0.017 | -0.004 [-0.009, +0.002] |
| pq_tau | 50 | 0.015 ± 0.006 | 0.053 ± 0.013 | 0.091 ± 0.017 | 105.3 ± 19.3 | 0.047 ± 0.008 | 0.235 ± 0.034 | -0.004 ± 0.016 | -0.004 [-0.008, -0.001]* |

Definitions: siglip = pairwise sigmoid loss (t = e^t', t' init log 10, b init −10); inter_intra_nce = InfoNCE(v, m) + 0.5·[InfoNCE(v′, v″) + InfoNCE(m′, m″)] with two 25% frame-drop views of the video and two ±1 s jitter crops of the music; infonce_intra = InfoNCE + 0.5 · intra (rows of the frozen vs projected similarity, both sides); infonce_rdrop = InfoNCE + 1.0 · R-Drop (symmetric KL of two dropout passes' similarity softmaxes).

Loss-bank definitions (2026-11-20, all tag-free; positives come from track_id only): multipos = multi-positive InfoNCE, gallery [m; m′] (aligned window + a second ±1 s jitter crop), every same-track window a positive, both directions; debiased = Chuang et al. 2020, τ⁺ 0.1; vsepp = max-violation hinge, margin 0.2, InfoNCE warm-up 10 epochs; infonce_supcon_m = InfoNCE + 0.5 · music–music SupCon over two jitter crops (positives = same track_id, τ 0.1); infonce_vv = InfoNCE + 0.5 · InfoNCE(v′, v″) over two 25% frame-drop views; infonce_fixtau = REF with τ fixed at 0.07 (REF learns τ); pq_tau = per-query τ from an MLP on the frozen video feature; recipe_fixed = multipos + 0.5 · supcon_music + 0.5 · intra (distillation); recipe_uncert = the same with uncertainty weighting (Kendall et al. 2018) on the two aux terms; all_uncert = multipos + supcon_music + vv_nce + intra, uncertainty weighting; asac_full = soft targets + 0.5 · intra, uncertainty weighting. Similarity is cosine everywhere; only the training loss differs. Train vs test: results/titlesplit/train_vs_test.md.
