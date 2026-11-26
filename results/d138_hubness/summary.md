# Test-time hubness reduction (D138, Step 2)

894 test videos, cleaned title split, 5 folds × 3 disjoint 10-head ensembles; gallery = the fold's test music (179 files on average). No retraining. Parameters chosen per fold on validation clips only (track MRR); ± = std over the 5 fold means; [CI] = 95% film-group cluster bootstrap (10,000); Δ = paired, per query. Development numbers.

| system | param per fold | R@1 | R@10 [95% CI] | MedR | Δ R@10 [95% CI] | Δ MRR [95% CI] | N10 skew | max N10 |
|---|---|---|---|---|---|---|---|---|
| ensemble (current) | — | 0.052 ± 0.016 | 0.302 ± 0.026 [0.269, 0.338] | 24.1 ± 3.6 | — | — | 0.85 | 33 |
| inverted softmax (β tuned on val; bank: training clips) | 2, 3, 2, 5, 1 | 0.056 ± 0.018 | 0.304 ± 0.022 [0.271, 0.339] | 23.6 ± 4.7 | +0.003 [-0.007, +0.012] | +0.003 [-0.001, +0.006] | 0.69 | 30 |
| CSLS (k = 10; bank: training clips) | 10, 10, 10, 10, 10 | 0.047 ± 0.018 | 0.304 ± 0.030 [0.272, 0.338] | 23.3 ± 4.0 | +0.003 [-0.011, +0.016] | -0.003 [-0.009, +0.002] | 0.71 | 28 |
| CSLS (k tuned on val; bank: training clips) | 50, 10, 50, 50, 50 | 0.049 ± 0.016 | 0.309 ± 0.026 [0.277, 0.344] | 23.4 ± 3.7 | +0.007 [-0.004, +0.019] | -0.001 [-0.007, +0.004] | 0.62 | 27 |
| inverted softmax (β tuned on val; bank: test queries, transductive) | 7, 10, 1, 15, 7 | 0.051 ± 0.020 | 0.319 ± 0.026 [0.286, 0.353] | 22.1 ± 3.4 | +0.017 [-0.002, +0.035] | +0.001 [-0.007, +0.009] | 0.29 | 22 |
| CSLS (k = 10; bank: test queries, transductive) | 10, 10, 10, 10, 10 | 0.048 ± 0.022 | 0.317 ± 0.025 [0.284, 0.351] | 22.5 ± 3.3 | +0.015 [+0.001, +0.028] | +0.001 [-0.006, +0.006] | 0.48 | 23 |

N10 = number of test queries with a file in their top 10 (k-occurrence); skew > 0 = a few hub files fill many lists. Fold mean of the per-ensemble values.

## The 10 files most often in top-10 lists (summed over folds, ensembles averaged)

**ensemble (current)**

| rank | music file | track | top-10 lists | before (current) |
|---|---|---|---|---|
| 1 | combined_full_bgm_dataset_youtube__132__132_bgm_0002 | trk_0516 | 36.3 | 36.3 |
| 2 | combined_full_bgm_dataset_youtube__34__34_bgm_0001 | trk_0583 | 36.3 | 36.3 |
| 3 | sunnxt_326_326_bgm_0002 | trk_0900 | 31.3 | 31.3 |
| 4 | combined_full_bgm_dataset_youtube2__289__289_bgm_0002 | trk_0382 | 31.0 | 31.0 |
| 5 | sunnxt_567_567_bgm_0002 | trk_0744 | 30.7 | 30.7 |
| 6 | combined_full_bgm_dataset__Professor__Professor_bgm_0000 | trk_0269 | 30.0 | 30.0 |
| 7 | sunnxt_254_254_bgm_0001 | trk_0816 | 29.7 | 29.7 |
| 8 | combined_full_bgm_dataset_youtube2__10__10_bgm_0000 | trk_0333 | 29.7 | 29.7 |
| 9 | combined_full_bgm_dataset_youtube2__430__430_bgm_0001 | trk_0423 | 29.0 | 29.0 |
| 10 | combined_full_bgm_dataset__Lava kusa__Lava kusa_bgm_0000 | trk_0264 | 28.7 | 28.7 |

**inverted softmax (β tuned on val; bank: training clips)**

| rank | music file | track | top-10 lists | before (current) |
|---|---|---|---|---|
| 1 | combined_full_bgm_dataset_youtube__34__34_bgm_0001 | trk_0583 | 34.0 | 36.3 |
| 2 | combined_full_bgm_dataset_youtube__132__132_bgm_0002 | trk_0516 | 33.0 | 36.3 |
| 3 | sunnxt_567_567_bgm_0002 | trk_0744 | 29.7 | 30.7 |
| 4 | combined_full_bgm_dataset__Professor__Professor_bgm_0000 | trk_0269 | 29.0 | 30.0 |
| 5 | sunnxt_565_565_bgm_0000 | trk_1092 | 28.3 | 27.0 |
| 6 | combined_full_bgm_dataset__Lava kusa__Lava kusa_bgm_0001 | trk_0265 | 28.0 | 28.0 |
| 7 | sunnxt_326_326_bgm_0002 | trk_0900 | 27.7 | 31.3 |
| 8 | sunnxt_180_180_bgm_0003 | trk_0744 | 27.7 | 27.0 |
| 9 | sunnxt_113_113_bgm_0001 | trk_0666 | 27.7 | 27.7 |
| 10 | combined_full_bgm_dataset__Lava kusa__Lava kusa_bgm_0000 | trk_0264 | 27.3 | 28.7 |

**CSLS (k = 10; bank: training clips)**

| rank | music file | track | top-10 lists | before (current) |
|---|---|---|---|---|
| 1 | combined_full_bgm_dataset_youtube2__13__13_bgm_0001 | trk_0346 | 29.0 | 26.7 |
| 2 | combined_full_bgm_dataset_youtube2__283__283_bgm_0000 | trk_0376 | 26.7 | 21.0 |
| 3 | sunnxt_36_36_bgm_0001 | trk_0945 | 26.3 | 14.7 |
| 4 | sunnxt_567_567_bgm_0002 | trk_0744 | 25.7 | 30.7 |
| 5 | combined_full_bgm_dataset__Lava kusa__Lava kusa_bgm_0001 | trk_0265 | 25.3 | 28.0 |
| 6 | sunnxt_180_180_bgm_0004 | trk_0744 | 25.3 | 27.0 |
| 7 | sunnxt_254_254_bgm_0001 | trk_0816 | 25.3 | 29.7 |
| 8 | sunnxt_85_85_bgm_0002 | trk_1126 | 25.3 | 16.3 |
| 9 | combined_full_bgm_dataset_youtube2__104__104_bgm_0000 | trk_0326 | 25.0 | 26.7 |
| 10 | combined_full_bgm_dataset__Lava kusa__Lava kusa_bgm_0000 | trk_0264 | 24.7 | 28.7 |

