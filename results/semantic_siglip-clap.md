# Semantic heads: siglip-clap

15 runs (5 folds x 3 seeds). Cells: mean ± std over runs; Δ vs ImageBind: per-query R@k averaged over seeds, paired, 95% cluster-bootstrap CI over track groups.

## Track retrieval, test-fold gallery

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (siglip-clap)** | 0.161 ± 0.030 | 0.380 ± 0.034 | 0.476 ± 0.035 | 12.167 ± 2.312 | 0.269 ± 0.029 |
| ImageBind zero-shot | 0.018 ± 0.010 | 0.056 ± 0.008 | 0.095 ± 0.013 | 75.000 ± 1.732 | 0.052 ± 0.008 |
| random (exact) | 0.006 ± 0.002 | 0.029 ± 0.010 | 0.056 ± 0.016 | — | — |

- Δ R@1 heads − ImageBind: **+0.143** [+0.119, +0.170] (1248 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.381** [+0.345, +0.417] (1248 queries, 1150 track groups)

## Track retrieval, all-library gallery

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (siglip-clap)** | 0.020 ± 0.012 | 0.148 ± 0.034 | 0.239 ± 0.043 | 55.400 ± 10.573 | 0.092 ± 0.018 |
| ImageBind zero-shot | 0.004 ± 0.003 | 0.018 ± 0.009 | 0.026 ± 0.009 | 367.800 ± 30.939 | 0.016 ± 0.005 |
| random (exact) | 0.001 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.004 | — | — |

- Δ R@1 heads − ImageBind: **+0.016** [+0.008, +0.025] (1248 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.213** [+0.185, +0.246] (1248 queries, 1150 track groups)

## Segment retrieval, all windows

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (siglip-clap)** | 0.059 ± 0.011 | 0.182 ± 0.026 | 0.270 ± 0.038 | 40.833 ± 9.639 | 0.129 ± 0.017 |
| ImageBind zero-shot | 0.005 ± 0.003 | 0.022 ± 0.005 | 0.043 ± 0.005 | 216.900 ± 11.160 | 0.023 ± 0.003 |
| random (exact) | 0.002 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | — | — |

- Δ R@1 heads − ImageBind: **+0.054** [+0.046, +0.062] (3282 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.227** [+0.202, +0.255] (3282 queries, 1150 track groups)

## Segment retrieval, other-video windows

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (siglip-clap)** | 0.098 ± 0.019 | 0.205 ± 0.031 | 0.290 ± 0.043 | 39.233 ± 9.386 | 0.161 ± 0.024 |
| ImageBind zero-shot | 0.009 ± 0.006 | 0.023 ± 0.005 | 0.046 ± 0.006 | 215.300 ± 11.223 | 0.026 ± 0.005 |
| random (exact) | 0.002 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | — | — |

- Δ R@1 heads − ImageBind: **+0.089** [+0.076, +0.102] (3282 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.244** [+0.217, +0.274] (3282 queries, 1150 track groups)

## H1

Trained heads beat ImageBind zero-shot on test-fold track R@10: **yes** (Δ +0.381, 95% CI [+0.345, +0.417]).

Validation (early stopping): segment R@10 0.443 ± 0.038; best epoch median 34.
