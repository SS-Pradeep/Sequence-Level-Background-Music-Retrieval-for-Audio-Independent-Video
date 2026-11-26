# Semantic heads on the full dataset (4,246 pairs)

Same method as the film split (PROGRESS.md D36): SigLIP + CLAP+MERT heads, 5 folds × 3 seeds, per-film cap of 30 clips per epoch, early stopping on validation segment R@10, Konjum salangai train-only. New in this run (D49): the 2,998 Demucs-separated clips of 32 films (data/manifests_full). The film split is **extended**, not redrawn: the 1,248 old pairs keep their folds except 15 moved by film groups that the new clips revealed (old SunNXT / YouTube clips of *Jilla* and *Pokkiri* under several source keys); each new film is tested in one fold.

## 1. Did the new data help? (paired, same queries and galleries)

Queries: the old film split's test clips that kept their fold (the moved ones left out). Both models are re-scored from their checkpoints by the same code (check: re-scoring an old run reproduces its saved ranks exactly).
Cells: mean ± std over 15 runs. Δ: full − old, per-query hits averaged over seeds, 95% cluster bootstrap over film groups.

### Track retrieval, test-fold gallery (967 queries, 396 film groups)

| trained on | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| old data (1,248 pairs) | 0.056 ± 0.020 | 0.185 ± 0.027 | 0.294 ± 0.024 | 27.133 ± 4.422 | 0.134 ± 0.017 |
| **full data (4,246 pairs)** | 0.051 ± 0.016 | 0.187 ± 0.029 | 0.301 ± 0.027 | 25.200 ± 4.157 | 0.134 ± 0.017 |

Δ full − old: R@1 **-0.005** [-0.017, +0.007]; R@5 **+0.001** [-0.018, +0.020]; R@10 **+0.006** [-0.018, +0.029]; MRR **-0.000** [-0.013, +0.012]

### Track retrieval, old library (1,248 files) (967 queries, 396 film groups)

| trained on | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| old data (1,248 pairs) | 0.012 ± 0.010 | 0.046 ± 0.016 | 0.082 ± 0.023 | 149.733 ± 22.957 | 0.042 ± 0.011 |
| **full data (4,246 pairs)** | 0.008 ± 0.006 | 0.045 ± 0.015 | 0.084 ± 0.020 | 141.600 ± 24.619 | 0.038 ± 0.009 |

Δ full − old: R@1 **-0.005** [-0.010, +0.000]; R@5 **-0.000** [-0.012, +0.011]; R@10 **+0.002** [-0.012, +0.016]; MRR **-0.004** [-0.011, +0.003]

### Segment retrieval, all windows (2503 queries, 396 film groups)

| trained on | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| old data (1,248 pairs) | 0.020 ± 0.007 | 0.076 ± 0.015 | 0.126 ± 0.018 | 87.867 ± 12.461 | 0.061 ± 0.009 |
| **full data (4,246 pairs)** | 0.019 ± 0.008 | 0.073 ± 0.009 | 0.125 ± 0.016 | 82.767 ± 7.853 | 0.059 ± 0.007 |

Δ full − old: R@1 **-0.001** [-0.006, +0.003]; R@5 **-0.003** [-0.013, +0.008]; R@10 **-0.000** [-0.014, +0.013]; MRR **-0.002** [-0.008, +0.004]

## 2. Whole dataset (extended film split, all test folds)

Every test clip against its test-fold gallery (old + new music files) or the whole library. Old = the 1,248-pair set; new = the 32 added films. Film-balanced: each film group weighs the same (the new films have ~95 clips each). Δ vs ImageBind: R@10, paired, 95% cluster bootstrap over film groups.

### Track retrieval, test-fold gallery

| queries | n | films | system | R@1 | R@5 | R@10 | MedR | MRR | film-balanced R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 3907 | 421 | **heads** | 0.018 ± 0.005 | 0.073 ± 0.015 | 0.126 ± 0.021 | 96.933 ± 16.588 | 0.057 ± 0.009 | 0.250 |
| all | | | ImageBind | 0.007 ± 0.002 | 0.021 ± 0.006 | 0.036 ± 0.008 | 272.800 ± 18.883 | 0.021 ± 0.003 | |
| all | | | random (exact) | 0.001 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | — | — | |
| old | 982 | 397 | **heads** | 0.045 ± 0.016 | 0.159 ± 0.019 | 0.259 ± 0.032 | 37.100 ± 7.656 | 0.114 ± 0.017 | 0.262 |
| old | | | ImageBind | 0.011 ± 0.008 | 0.033 ± 0.016 | 0.053 ± 0.019 | 225.800 ± 30.993 | 0.031 ± 0.012 | |
| old | | | random (exact) | 0.002 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.001 | — | — | |
| new | 2925 | 31 | **heads** | 0.009 ± 0.005 | 0.044 ± 0.016 | 0.081 ± 0.021 | 129.433 ± 22.749 | 0.038 ± 0.009 | 0.079 |
| new | | | ImageBind | 0.006 ± 0.005 | 0.017 ± 0.008 | 0.031 ± 0.013 | 290.200 ± 33.787 | 0.018 ± 0.006 | |
| new | | | random (exact) | 0.001 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | — | — | |

Δ R@10 heads − ImageBind: all **+0.089** [+0.071, +0.111]; old **+0.206** [+0.178, +0.234]; new **+0.050** [+0.032, +0.068]

### Track retrieval, all-library gallery

| queries | n | films | system | R@1 | R@5 | R@10 | MedR | MRR | film-balanced R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 3907 | 421 | **heads** | 0.003 ± 0.002 | 0.016 ± 0.006 | 0.030 ± 0.008 | 568.233 ± 220.557 | 0.015 ± 0.004 | 0.070 |
| all | | | ImageBind | 0.002 ± 0.001 | 0.007 ± 0.002 | 0.010 ± 0.005 | 1406.900 ± 142.802 | 0.006 ± 0.001 | |
| all | | | random (exact) | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | — | — | |
| old | 982 | 397 | **heads** | 0.007 ± 0.005 | 0.038 ± 0.013 | 0.071 ± 0.021 | 198.167 ± 35.687 | 0.033 ± 0.008 | 0.073 |
| old | | | ImageBind | 0.003 ± 0.003 | 0.011 ± 0.010 | 0.017 ± 0.015 | 1137.900 ± 151.877 | 0.010 ± 0.006 | |
| old | | | random (exact) | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | — | — | |
| new | 2925 | 31 | **heads** | 0.002 ± 0.002 | 0.008 ± 0.006 | 0.016 ± 0.009 | 760.767 ± 348.267 | 0.009 ± 0.005 | 0.016 |
| new | | | ImageBind | 0.001 ± 0.002 | 0.006 ± 0.005 | 0.008 ± 0.006 | 1497.500 ± 233.369 | 0.005 ± 0.003 | |
| new | | | random (exact) | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | — | — | |

Δ R@10 heads − ImageBind: all **+0.019** [+0.012, +0.028]; old **+0.054** [+0.038, +0.070]; new **+0.007** [-0.001, +0.015]

### Segment retrieval, all windows

| queries | n | films | system | R@1 | R@5 | R@10 | MedR | MRR | film-balanced R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 13309 | 421 | **heads** | 0.005 ± 0.002 | 0.020 ± 0.005 | 0.036 ± 0.008 | 465.700 ± 123.473 | 0.019 ± 0.004 | 0.110 |
| all | | | ImageBind | 0.002 ± 0.001 | 0.008 ± 0.003 | 0.011 ± 0.003 | 1022.400 ± 115.027 | 0.007 ± 0.002 | |
| all | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |
| old | 2552 | 397 | **heads** | 0.017 ± 0.008 | 0.059 ± 0.013 | 0.100 ± 0.021 | 150.233 ± 48.892 | 0.048 ± 0.010 | 0.115 |
| old | | | ImageBind | 0.002 ± 0.001 | 0.015 ± 0.009 | 0.019 ± 0.011 | 867.100 ± 125.003 | 0.011 ± 0.003 | |
| old | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |
| new | 10757 | 31 | **heads** | 0.003 ± 0.001 | 0.011 ± 0.005 | 0.021 ± 0.008 | 563.933 ± 139.617 | 0.012 ± 0.004 | 0.021 |
| new | | | ImageBind | 0.001 ± 0.002 | 0.006 ± 0.004 | 0.010 ± 0.004 | 1059.500 ± 133.254 | 0.006 ± 0.003 | |
| new | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |

Δ R@10 heads − ImageBind: all **+0.024** [+0.018, +0.032]; old **+0.080** [+0.065, +0.097]; new **+0.011** [+0.006, +0.016]

### Segment retrieval, other-video windows

| queries | n | films | system | R@1 | R@5 | R@10 | MedR | MRR | film-balanced R@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 13309 | 421 | **heads** | 0.008 ± 0.003 | 0.023 ± 0.005 | 0.038 ± 0.009 | 463.100 ± 123.061 | 0.021 ± 0.005 | 0.114 |
| all | | | ImageBind | 0.002 ± 0.001 | 0.008 ± 0.003 | 0.011 ± 0.003 | 1020.300 ± 114.755 | 0.008 ± 0.002 | |
| all | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |
| old | 2552 | 397 | **heads** | 0.025 ± 0.009 | 0.066 ± 0.012 | 0.107 ± 0.022 | 149.100 ± 48.584 | 0.055 ± 0.011 | 0.120 |
| old | | | ImageBind | 0.004 ± 0.002 | 0.015 ± 0.009 | 0.019 ± 0.011 | 865.800 ± 126.585 | 0.012 ± 0.004 | |
| old | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |
| new | 10757 | 31 | **heads** | 0.004 ± 0.002 | 0.012 ± 0.005 | 0.022 ± 0.008 | 561.333 ± 139.412 | 0.013 ± 0.004 | 0.022 |
| new | | | ImageBind | 0.002 ± 0.002 | 0.006 ± 0.004 | 0.010 ± 0.004 | 1058.100 ± 134.005 | 0.007 ± 0.003 | |
| new | | | random (exact) | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | — | — | |

Δ R@10 heads − ImageBind: all **+0.026** [+0.020, +0.035]; old **+0.087** [+0.070, +0.104]; new **+0.012** [+0.007, +0.017]

## Folds and training

| fold | train | val | test (old + new) | train films | test films | best epochs (seeds 0-2) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3392 | 111 | 743 (192 + 551) | 302 | 90 | 23, 13, 31 |
| 1 | 3362 | 73 | 811 (201 + 610) | 299 | 91 | 11, 41, 3 |
| 2 | 3230 | 241 | 775 (206 + 569) | 313 | 75 | 23, 21, 8 |
| 3 | 3288 | 192 | 766 (190 + 576) | 306 | 83 | 16, 11, 8 |
| 4 | 3347 | 87 | 812 (193 + 619) | 307 | 82 | 33, 27, 27 |
