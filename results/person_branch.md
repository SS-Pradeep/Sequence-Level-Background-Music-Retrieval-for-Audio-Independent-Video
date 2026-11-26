# Person-focused branch (D87–)

1,248-pair film split (D36), 5 folds, test-fold galleries (195–199 files). Konjum salangai is train-only. Every weight and stopping epoch was chosen on the fold's val films. Per query, values are averaged over seeds (0–2) first; ensemble rows are one run per fold. Paired Δ with a 95% cluster bootstrap (10,000 resamples): film groups for videos and segments, concatenations for stitching. Plan: docs/person_branch_plan.md.

## Verdict (computed)

Does a person-branch system beat P0? A metric counts as better / worse only when its 95% CI excludes 0. P1–P3 are single models, so most of their deficit is the 10-seed ensemble that P0 has and they lack (P0s, the single frame heads, is behind P0 by about as much); their own effect is the Δ vs P0s table below. P4 is the like-for-like test against P0.

- **P1 + D77 facial branch, w on val**: better on nothing; worse on track R@1 -0.016 [-0.030, -0.004], track R@10 -0.038 [-0.058, -0.019], track R@10, all 1,248 -0.034 [-0.049, -0.019], segment R@10 -0.033 [-0.045, -0.022], pipeline track@1 -0.012 [-0.021, -0.002], pipeline track R@10 -0.037 [-0.056, -0.019], window@1 -0.008 [-0.013, -0.004], right track ±1 s -0.007 [-0.012, -0.003], switch F1 (Viterbi) -0.030 [-0.053, -0.008].
- **P2 + person-crop head (late fusion), w on val**: better on nothing; worse on track R@1 -0.014 [-0.027, -0.002], track R@10 -0.036 [-0.055, -0.016], track R@10, all 1,248 -0.035 [-0.050, -0.021], segment R@10 -0.032 [-0.044, -0.021], pipeline track@1 -0.012 [-0.023, -0.002], pipeline track R@10 -0.036 [-0.054, -0.019], window@1 -0.008 [-0.013, -0.004], window@J -0.010 [-0.017, -0.004], right track ±1 s -0.008 [-0.013, -0.004], part track accuracy -0.016 [-0.030, -0.002], natural videos, track after Viterbi -0.011 [-0.022, -0.001].
- **P3 frame ‖ person in one head (early fusion)**: better on nothing; worse on track R@1 -0.025 [-0.041, -0.011], track R@10 -0.063 [-0.087, -0.039], track R@10, all 1,248 -0.041 [-0.061, -0.024], segment R@10 -0.045 [-0.062, -0.028], pipeline track@1 -0.022 [-0.032, -0.011], pipeline track R@10 -0.063 [-0.087, -0.040], window@1 -0.011 [-0.016, -0.006], window@J -0.016 [-0.025, -0.008], right track ±1 s -0.011 [-0.016, -0.006], switch F1 (Viterbi) -0.030 [-0.053, -0.007], part track accuracy -0.020 [-0.036, -0.004], natural videos, track after Viterbi -0.019 [-0.033, -0.006].
- **P4 ensemble: D75 frames + 10 regularised person heads, w on val**: better on nothing; worse on part track accuracy -0.014 [-0.027, -0.002].
- **P4 distilled: D75 frames + one distilled person head per seed, w on val**: better on nothing; worse on nothing.

## Step 1: diagnosis

1,248-pair film split, 5 test folds, test-fold galleries (195–199 files). P0 = D75: 10-seed siglip-both ensemble, semantic-only fusion, coarse-to-fine pipeline with each fold's tuned fusion.json. Recomputed values reproduce the saved D75 per-segment hits and the saved seed-0 retrieval ranks, except one first-index window@1 hit on fold 4 where two windows tie exactly (the tie-shared hit, 0.5, is identical; GPU batch arithmetic flips the order). The recomputed value is used.

**Subsets** (docs/person_branch_plan.md; data/manifests/person_subsets_film.json). Pairs of test segments from different videos within a fold; setting = SigLIP span-mean cosine, music = mean of CLAP and MERT span-mean cosines; thresholds are the fold's 90th percentile (setting) and 30th percentile (music) of all cross-video pairs.

- **spec** (as specified): the segment is in at least one pair with setting ≥ q90 and music ≤ q30.
- **nn** (diagnostic): the segment's single nearest other-video neighbour by setting passes both thresholds.

| fold | segments | setting q90 | music q30 | pairs passing | spec | nn |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 493 | 0.828 | 0.672 | 3,161 of 120,621 | 392 | 62 |
| 1 | 498 | 0.827 | 0.657 | 3,110 of 123,072 | 392 | 56 |
| 2 | 546 | 0.831 | 0.664 | 3,478 of 147,910 | 448 | 70 |
| 3 | 482 | 0.835 | 0.653 | 2,802 of 115,208 | 356 | 39 |
| 4 | 533 | 0.825 | 0.649 | 2,499 of 140,912 | 407 | 45 |
| all | 2552 | | | | **1995** (78%) | **272** (11%) |

## Subset nn: 272 of 2552 segments; 181 of 982 videos have at least one

Clip-level track R@10 uses the videos with at least one subset segment. Δ = subset − rest, 95% film-group cluster bootstrap (10,000 resamples).

| metric | all | subset | rest | Δ subset − rest | chance (all / subset) |
| --- | --- | --- | --- | --- | --- |
| track@1 (pipeline) | 0.062 | 0.077 | 0.060 | +0.017 [-0.026, +0.070] | 0.009 / 0.007 |
| track R@10 (pipeline, per segment) | 0.313 | 0.279 | 0.317 | -0.038 [-0.112, +0.038] | 0.059 / 0.053 |
| window@1 | 0.018 | 0.015 | 0.019 | -0.004 [-0.018, +0.012] | 0.001 / 0.001 |
| window@J (J = 20) | 0.063 | 0.062 | 0.064 | -0.001 [-0.033, +0.035] | — |
| right track and within ±1 s | 0.021 | 0.020 | 0.021 | -0.001 [-0.017, +0.017] | 0.001 / 0.001 |
| right track and within ±5 s | 0.041 | 0.044 | 0.040 | +0.004 [-0.022, +0.033] | 0.003 / 0.003 |
| segment R@10 (all test windows) | 0.157 | 0.154 | 0.157 | -0.003 [-0.054, +0.049] | 0.020 / 0.020 |
| track R@10 (clip level, test-fold gallery) | 0.334 | 0.309 | 0.340 | -0.030 [-0.104, +0.045] | 0.058 / 0.053 |

## Subset spec: 1995 of 2552 segments; 829 of 982 videos have at least one

Clip-level track R@10 uses the videos with at least one subset segment. Δ = subset − rest, 95% film-group cluster bootstrap (10,000 resamples).

| metric | all | subset | rest | Δ subset − rest | chance (all / subset) |
| --- | --- | --- | --- | --- | --- |
| track@1 (pipeline) | 0.062 | 0.059 | 0.072 | -0.013 [-0.050, +0.022] | 0.009 / 0.009 |
| track R@10 (pipeline, per segment) | 0.313 | 0.296 | 0.375 | -0.079 [-0.143, -0.013] | 0.059 / 0.059 |
| window@1 | 0.018 | 0.018 | 0.020 | -0.002 [-0.016, +0.012] | 0.001 / 0.001 |
| window@J (J = 20) | 0.063 | 0.062 | 0.070 | -0.008 [-0.038, +0.021] | — |
| right track and within ±1 s | 0.021 | 0.021 | 0.023 | -0.002 [-0.017, +0.011] | 0.001 / 0.001 |
| right track and within ±5 s | 0.041 | 0.038 | 0.050 | -0.012 [-0.041, +0.014] | 0.003 / 0.003 |
| segment R@10 (all test windows) | 0.157 | 0.146 | 0.196 | -0.049 [-0.098, +0.000] | 0.020 / 0.020 |
| track R@10 (clip level, test-fold gallery) | 0.334 | 0.317 | 0.425 | -0.108 [-0.186, -0.019] | 0.058 / 0.058 |

All segments, window@1 with its film-group CI: 0.018 [0.013, 0.024].


## All test segments / videos: value (Δ vs P0 [95% CI])

| system | track R@1 | track R@10 | track R@10, all 1,248 | segment R@10 | pipeline track@1 | pipeline track R@10 | window@1 | window@J | right track ±1 s | switch F1 (Viterbi) | part track accuracy | switches / min | natural videos, track after Viterbi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 current system (D75, 10-seed ensemble) | 0.069 | 0.334 | 0.118 | 0.157 | 0.062 | 0.313 | 0.019 | 0.063 | 0.021 | 0.741 | 0.071 | 3.460 | 0.063 |
| P0s single siglip-both heads (reference) | 0.055 (-0.014 [-0.027, -0.003]) | 0.288 (-0.046 [-0.065, -0.027]) | 0.081 (-0.037 [-0.052, -0.023]) | 0.123 (-0.034 [-0.045, -0.024]) | 0.051 (-0.011 [-0.021, -0.002]) | 0.274 (-0.039 [-0.057, -0.022]) | 0.010 (-0.009 [-0.013, -0.005]) | 0.056 (-0.008 [-0.015, -0.001]) | 0.012 (-0.009 [-0.013, -0.005]) | 0.716 (-0.025 [-0.047, -0.002]) | 0.062 (-0.009 [-0.022, +0.004]) | 3.575 (+0.115 [+0.004, +0.231]) | 0.054 (-0.010 [-0.021, +0.001]) |
| P1 + D77 facial branch, w on val | 0.053 (-0.016 [-0.030, -0.004]) | 0.296 (-0.038 [-0.058, -0.019]) | 0.084 (-0.034 [-0.049, -0.019]) | 0.124 (-0.033 [-0.045, -0.022]) | 0.050 (-0.012 [-0.021, -0.002]) | 0.276 (-0.037 [-0.056, -0.019]) | 0.011 (-0.008 [-0.013, -0.004]) | 0.058 (-0.005 [-0.012, +0.002]) | 0.014 (-0.007 [-0.012, -0.003]) | 0.710 (-0.030 [-0.053, -0.008]) | 0.062 (-0.009 [-0.023, +0.005]) | 3.456 (-0.004 [-0.125, +0.117]) | 0.054 (-0.009 [-0.021, +0.002]) |
| P2 + person-crop head (late fusion), w on val | 0.055 (-0.014 [-0.027, -0.002]) | 0.298 (-0.036 [-0.055, -0.016]) | 0.083 (-0.035 [-0.050, -0.021]) | 0.125 (-0.032 [-0.044, -0.021]) | 0.050 (-0.012 [-0.023, -0.002]) | 0.277 (-0.036 [-0.054, -0.019]) | 0.011 (-0.008 [-0.013, -0.004]) | 0.053 (-0.010 [-0.017, -0.004]) | 0.013 (-0.008 [-0.013, -0.004]) | 0.724 (-0.017 [-0.040, +0.006]) | 0.055 (-0.016 [-0.030, -0.002]) | 3.534 (+0.074 [-0.048, +0.198]) | 0.052 (-0.011 [-0.022, -0.001]) |
| P3 frame ‖ person in one head (early fusion) | 0.044 (-0.025 [-0.041, -0.011]) | 0.271 (-0.063 [-0.087, -0.039]) | 0.077 (-0.041 [-0.061, -0.024]) | 0.113 (-0.045 [-0.062, -0.028]) | 0.040 (-0.022 [-0.032, -0.011]) | 0.250 (-0.063 [-0.087, -0.040]) | 0.008 (-0.011 [-0.016, -0.006]) | 0.047 (-0.016 [-0.025, -0.008]) | 0.010 (-0.011 [-0.016, -0.006]) | 0.710 (-0.030 [-0.053, -0.007]) | 0.051 (-0.020 [-0.036, -0.004]) | 3.342 (-0.118 [-0.242, +0.008]) | 0.044 (-0.019 [-0.033, -0.006]) |
| P4 ensemble: D75 frames + 10 regularised person heads, w on val | 0.072 (+0.003 [-0.008, +0.014]) | 0.333 (-0.001 [-0.017, +0.015]) | 0.112 (-0.006 [-0.016, +0.003]) | 0.154 (-0.003 [-0.011, +0.006]) | 0.060 (-0.002 [-0.010, +0.006]) | 0.319 (+0.006 [-0.006, +0.018]) | 0.016 (-0.003 [-0.007, +0.001]) | 0.065 (+0.002 [-0.004, +0.008]) | 0.018 (-0.003 [-0.007, +0.001]) | 0.726 (-0.014 [-0.036, +0.007]) | 0.056 (-0.014 [-0.027, -0.002]) | 3.542 (+0.082 [-0.023, +0.190]) | 0.059 (-0.004 [-0.012, +0.004]) |
| P4 distilled: D75 frames + one distilled person head per seed, w on val | 0.070 (+0.001 [-0.008, +0.009]) | 0.329 (-0.005 [-0.019, +0.010]) | 0.108 (-0.010 [-0.020, +0.000]) | 0.151 (-0.006 [-0.015, +0.003]) | 0.061 (-0.001 [-0.007, +0.005]) | 0.312 (-0.001 [-0.013, +0.011]) | 0.016 (-0.003 [-0.007, +0.000]) | 0.065 (+0.001 [-0.004, +0.007]) | 0.018 (-0.003 [-0.006, +0.001]) | 0.735 (-0.006 [-0.024, +0.011]) | 0.063 (-0.008 [-0.018, +0.001]) | 3.375 (-0.085 [-0.170, +0.001]) | 0.060 (-0.003 [-0.010, +0.005]) |

## Single-model comparison (choice of the P4 base): Δ vs P0s, same seeds

| system | track R@1 | track R@10 | track R@10, all 1,248 | segment R@10 | pipeline track@1 | pipeline track R@10 | window@1 | window@J | right track ±1 s | switch F1 (Viterbi) | part track accuracy | switches / min | natural videos, track after Viterbi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 + D77 facial branch, w on val | -0.002 [-0.007, +0.002] | +0.008 [-0.002, +0.018] | +0.003 [-0.003, +0.009] | +0.001 [-0.003, +0.005] | -0.001 [-0.004, +0.003] | +0.002 [-0.004, +0.009] | +0.001 [-0.001, +0.002] | +0.002 [-0.000, +0.006] | +0.001 [-0.001, +0.003] | -0.006 [-0.020, +0.008] | -0.000 [-0.007, +0.006] | -0.119 [-0.192, -0.050] | +0.000 [-0.005, +0.005] |
| P2 + person-crop head (late fusion), w on val | +0.000 [-0.006, +0.007] | +0.010 [-0.001, +0.021] | +0.002 [-0.005, +0.009] | +0.002 [-0.004, +0.009] | -0.001 [-0.006, +0.005] | +0.003 [-0.007, +0.013] | +0.001 [-0.001, +0.003] | -0.003 [-0.007, +0.001] | +0.000 [-0.002, +0.003] | +0.008 [-0.007, +0.023] | -0.007 [-0.015, +0.000] | -0.041 [-0.118, +0.036] | -0.002 [-0.008, +0.004] |
| P3 frame ‖ person in one head (early fusion) | -0.011 [-0.022, -0.000] | -0.017 [-0.037, +0.003] | -0.004 [-0.018, +0.008] | -0.010 [-0.022, +0.002] | -0.010 [-0.018, -0.003] | -0.024 [-0.044, -0.003] | -0.002 [-0.005, +0.001] | -0.009 [-0.016, -0.001] | -0.002 [-0.006, +0.001] | -0.005 [-0.025, +0.015] | -0.011 [-0.022, +0.000] | -0.233 [-0.330, -0.136] | -0.010 [-0.020, +0.000] |

## Step 1 nearest-neighbour subset (diagnostic): 272 segments; value (Δ vs P0 [95% CI])

| system | segment R@10 | pipeline track@1 | pipeline track R@10 | window@1 | right track ±1 s |
| --- | --- | --- | --- | --- | --- |
| P0 current system (D75, 10-seed ensemble) | 0.154 | 0.077 | 0.279 | 0.015 | 0.020 |
| P0s single siglip-both heads (reference) | 0.119 (-0.036 [-0.065, -0.009]) | 0.064 (-0.013 [-0.038, +0.010]) | 0.243 (-0.037 [-0.076, +0.001]) | 0.007 (-0.007 [-0.023, +0.007]) | 0.010 (-0.010 [-0.029, +0.008]) |
| P1 + D77 facial branch, w on val | 0.119 (-0.036 [-0.065, -0.008]) | 0.060 (-0.017 [-0.041, +0.005]) | 0.259 (-0.021 [-0.060, +0.017]) | 0.009 (-0.006 [-0.022, +0.008]) | 0.011 (-0.009 [-0.027, +0.008]) |
| P2 + person-crop head (late fusion), w on val | 0.124 (-0.031 [-0.060, -0.001]) | 0.067 (-0.010 [-0.035, +0.016]) | 0.254 (-0.026 [-0.064, +0.014]) | 0.010 (-0.005 [-0.020, +0.010]) | 0.012 (-0.008 [-0.027, +0.009]) |
| P3 frame ‖ person in one head (early fusion) | 0.108 (-0.047 [-0.087, -0.007]) | 0.062 (-0.015 [-0.039, +0.009]) | 0.227 (-0.053 [-0.103, -0.000]) | 0.004 (-0.011 [-0.025, +0.001]) | 0.007 (-0.013 [-0.030, +0.001]) |
| P4 ensemble: D75 frames + 10 regularised person heads, w on val | 0.151 (-0.004 [-0.029, +0.022]) | 0.077 (+0.000 [-0.023, +0.023]) | 0.298 (+0.018 [-0.007, +0.048]) | 0.026 (+0.011 [-0.004, +0.028]) | 0.028 (+0.007 [-0.007, +0.023]) |
| P4 distilled: D75 frames + one distilled person head per seed, w on val | 0.131 (-0.023 [-0.043, -0.005]) | 0.075 (-0.002 [-0.022, +0.019]) | 0.283 (+0.004 [-0.021, +0.031]) | 0.017 (+0.002 [-0.011, +0.018]) | 0.021 (+0.000 [-0.013, +0.015]) |

## Step 1 subset as specified: 1995 segments; value (Δ vs P0 [95% CI])

| system | segment R@10 | pipeline track@1 | pipeline track R@10 | window@1 | right track ±1 s |
| --- | --- | --- | --- | --- | --- |
| P0 current system (D75, 10-seed ensemble) | 0.146 | 0.059 | 0.296 | 0.019 | 0.021 |
| P0s single siglip-both heads (reference) | 0.114 (-0.032 [-0.044, -0.021]) | 0.050 (-0.009 [-0.019, +0.000]) | 0.259 (-0.037 [-0.055, -0.019]) | 0.009 (-0.010 [-0.015, -0.005]) | 0.012 (-0.009 [-0.014, -0.004]) |
| P1 + D77 facial branch, w on val | 0.116 (-0.031 [-0.043, -0.019]) | 0.050 (-0.009 [-0.019, -0.000]) | 0.262 (-0.033 [-0.054, -0.015]) | 0.010 (-0.009 [-0.014, -0.004]) | 0.013 (-0.008 [-0.013, -0.003]) |
| P2 + person-crop head (late fusion), w on val | 0.118 (-0.028 [-0.041, -0.015]) | 0.049 (-0.010 [-0.020, +0.001]) | 0.261 (-0.034 [-0.053, -0.016]) | 0.010 (-0.008 [-0.014, -0.003]) | 0.012 (-0.009 [-0.014, -0.003]) |
| P3 frame ‖ person in one head (early fusion) | 0.107 (-0.039 [-0.057, -0.023]) | 0.043 (-0.017 [-0.027, -0.006]) | 0.237 (-0.059 [-0.086, -0.032]) | 0.008 (-0.011 [-0.017, -0.006]) | 0.010 (-0.011 [-0.016, -0.005]) |
| P4 ensemble: D75 frames + 10 regularised person heads, w on val | 0.150 (+0.004 [-0.004, +0.013]) | 0.062 (+0.003 [-0.005, +0.010]) | 0.300 (+0.004 [-0.009, +0.018]) | 0.016 (-0.003 [-0.007, +0.002]) | 0.019 (-0.002 [-0.006, +0.002]) |
| P4 distilled: D75 frames + one distilled person head per seed, w on val | 0.145 (-0.001 [-0.010, +0.007]) | 0.061 (+0.002 [-0.004, +0.007]) | 0.291 (-0.005 [-0.018, +0.008]) | 0.016 (-0.003 [-0.006, +0.001]) | 0.018 (-0.002 [-0.006, +0.001]) |

## Fusion weights chosen on val (w of the branch; frame weight 1)

| system | w per run | mean |
| --- | --- | --- |
| P1 + D77 facial branch, w on val | 0, 0, 0.05, 0, 0.3, 0.05, 0, 0.3, 0.5, 0.5, 0.3, 0.75, 0.3, 0.2, 0.2 | 0.23 |
| P2 + person-crop head (late fusion), w on val | 0.05, 0.05, 0.75, 1, 0.3, 0.1, 0.75, 0.5, 0.75, 0.5, 0.1, 0.2, 0.2, 1, 0 | 0.42 |
| P4 ensemble: D75 frames + 10 regularised person heads, w on val | 0.3, 0.5, 0.2, 0, 1 | 0.40 |
| P4 distilled: D75 frames + one distilled person head per seed, w on val | 0.3, 1, 0.75, 0.3, 0, 0.5, 1, 0.2, 0.3, 0, 0.2, 0, 0.75, 0.5, 1 | 0.45 |

Person features (Step 2): 23,909 seconds of 1,248 clips; faces 63.8%, person boxes only 24.8%, no person 11.4%; segments without a person second 1.6%.

## Step 4: rendered outputs (outputs/person_branch/)

baseline/ = P0 (./render_best.sh); person/ = `runs/filmsplit/person/eval/P4-P2/fold0/seed0` with `person.enabled=true`. Both: fold 0, audio 320k; outputs under 1080 lines re-encoded to 1080 (Lanczos, CRF 16, source timestamps kept). Sync: music sample 0 vs the first frame, by cross-correlation with the render's soundtrack. Pieces per system: outputs/person_branch/index.csv.

| output | picture | segments | switches | music files | LUFS | music vs first frame |
| --- | --- | --- | --- | --- | --- | --- |
| baseline/1.mp4 | 1080x720 -> 1620x1080 | 54 | 27 | 20 | -16.2 | +0.00 ms |
| person/1.mp4 | 1080x720 -> 1620x1080 | 54 | 25 | 20 | -16.1 | +0.00 ms |
| baseline/12.mp4 | 1920x1080 kept | 52 | 19 | 12 | -16.9 | +0.00 ms |
| person/12.mp4 | 1920x1080 kept | 52 | 18 | 11 | -17.1 | +0.00 ms |
| baseline/123.mp4 | 1920x1080 kept | 37 | 16 | 15 | -16.1 | +0.00 ms |
| person/123.mp4 | 1920x1080 kept | 37 | 12 | 11 | -16.1 | +0.00 ms |
| baseline/1_BGM_Ori.mp4 | 1080x720 -> 1620x1080 | 54 | 27 | 20 | -16.2 | +0.00 ms |
| person/1_BGM_Ori.mp4 | 1080x720 -> 1620x1080 | 54 | 25 | 20 | -16.1 | +0.00 ms |
| baseline/with_audio.mp4 | 1080x720 -> 1620x1080 | 53 | 24 | 17 | -16.1 | +0.00 ms |
| person/with_audio.mp4 | 1080x720 -> 1620x1080 | 53 | 25 | 19 | -16.1 | +0.00 ms |

Failed: none.

