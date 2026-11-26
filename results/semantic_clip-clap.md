# Semantic heads: clip-clap

15 runs (5 folds x 3 seeds). Cells: mean ± std over runs; Δ vs ImageBind: per-query R@k averaged over seeds, paired, 95% cluster-bootstrap CI over track groups.

## Track retrieval, test-fold gallery

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (clip-clap)** | 0.145 ± 0.039 | 0.323 ± 0.026 | 0.439 ± 0.028 | 14.267 ± 2.086 | 0.243 ± 0.032 |
| ImageBind zero-shot | 0.018 ± 0.010 | 0.056 ± 0.008 | 0.095 ± 0.013 | 75.000 ± 1.732 | 0.052 ± 0.008 |
| random (exact) | 0.006 ± 0.002 | 0.029 ± 0.010 | 0.056 ± 0.016 | — | — |

- Δ R@1 heads − ImageBind: **+0.127** [+0.102, +0.155] (1248 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.343** [+0.308, +0.381] (1248 queries, 1150 track groups)

## Track retrieval, all-library gallery

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (clip-clap)** | 0.022 ± 0.009 | 0.130 ± 0.027 | 0.205 ± 0.036 | 69.800 ± 9.660 | 0.085 ± 0.015 |
| ImageBind zero-shot | 0.004 ± 0.003 | 0.018 ± 0.009 | 0.026 ± 0.009 | 367.800 ± 30.939 | 0.016 ± 0.005 |
| random (exact) | 0.001 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.004 | — | — |

- Δ R@1 heads − ImageBind: **+0.018** [+0.010, +0.027] (1248 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.179** [+0.151, +0.212] (1248 queries, 1150 track groups)

## Segment retrieval, all windows

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (clip-clap)** | 0.047 ± 0.012 | 0.152 ± 0.026 | 0.233 ± 0.034 | 51.367 ± 8.092 | 0.110 ± 0.016 |
| ImageBind zero-shot | 0.005 ± 0.003 | 0.022 ± 0.005 | 0.043 ± 0.005 | 216.900 ± 11.160 | 0.023 ± 0.003 |
| random (exact) | 0.002 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | — | — |

- Δ R@1 heads − ImageBind: **+0.042** [+0.035, +0.050] (3282 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.189** [+0.165, +0.217] (3282 queries, 1150 track groups)

## Segment retrieval, other-video windows

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| **heads (clip-clap)** | 0.077 ± 0.019 | 0.173 ± 0.030 | 0.248 ± 0.034 | 49.800 ± 8.095 | 0.135 ± 0.021 |
| ImageBind zero-shot | 0.009 ± 0.006 | 0.023 ± 0.005 | 0.046 ± 0.006 | 215.300 ± 11.223 | 0.026 ± 0.005 |
| random (exact) | 0.002 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | — | — |

- Δ R@1 heads − ImageBind: **+0.068** [+0.056, +0.081] (3282 queries, 1150 track groups)
- Δ R@10 heads − ImageBind: **+0.202** [+0.176, +0.230] (3282 queries, 1150 track groups)

## H1

Trained heads beat ImageBind zero-shot on test-fold track R@10: **yes** (Δ +0.343, 95% CI [+0.308, +0.381]).

Validation (early stopping): segment R@10 0.396 ± 0.031; best epoch median 26.
