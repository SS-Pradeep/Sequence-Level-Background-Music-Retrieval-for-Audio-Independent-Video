# One learned cross-attention layer before the semantic heads (D110, title split)

Single head, one layer, no feed-forward block, no positional encoding. Q = W_q LN(video 1 Hz tokens), K = W_k LN(music tokens), V = W_v LN(music tokens) (d = 64, value 512); α = softmax(q·k / √d) over the music seconds; the music vector is Σ_j (mean_t α_tj) V_j, i.e. the music pooled the way this video attends to it; score = cos(head_v(mean video token), head_m(music vector)). W_q, W_k start near zero (uniform attention). Baseline: the pooled heads, runs/titlesplit/semantic/siglip-both seeds 0–2. 5 folds × 3 seeds; paired per-query hits averaged over seeds, 95% cluster bootstrap over film (title) groups, 10,000 resamples.

## 1. Retrieval

| system | runs | track R@1 | track R@10 | track MRR | segment R@10 | Δ track R@10 [95% CI] | Δ track R@1 [95% CI] | Δ segment R@10 [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: pooled heads | 15 | 0.036 ± 0.013 | 0.235 ± 0.046 | 0.107 ± 0.018 | 0.092 ± 0.022 | — | — | — |
| **learned Q/K/V attention** | 15 | 0.040 ± 0.016 | 0.243 ± 0.020 | 0.111 ± 0.017 | 0.095 ± 0.013 | +0.008 [-0.008, +0.023] | +0.004 [-0.005, +0.014] | +0.002 [-0.008, +0.013] |

## 2. Is the attention doing anything? (same trained weights, test queries)

| scoring | track R@10 | segment R@10 |
| --- | --- | --- |
| learned attention | 0.243 ± 0.020 | 0.095 ± 0.013 |
| same weights, uniform attention | 0.246 ± 0.019 | 0.095 ± 0.016 |

Learned − uniform, track R@10 (paired): -0.002 [-0.013, +0.008].

## 3. Where does the attention go? (test clip vs its own music file)

| measure | value (mean ± std over 15 runs) | reading |
| --- | --- | --- |
| entropy of each video second's attention row | 0.758 ± 0.107 | 1 = spread evenly over the music seconds |
| entropy of the pooled music weights | 0.770 ± 0.098 | 1 = plain mean pooling |
| distance of the pooled weights from uniform (TV) | 0.422 ± 0.121 | 0 = mean pooling, 1 = all weight elsewhere |
| distance between two different videos' weights on the same music (TV) | 0.157 ± 0.090 | 0 = the video does not change where the music is read |

## 4. Segment hit@10 by segment length (attention − baseline)

| segment length | segments | baseline | attention | Δ [95% CI] |
| --- | --- | --- | --- | --- |
| 0–3 s | 607 | 0.081 | 0.092 | +0.011 [-0.009, +0.031] |
| 3–6 s | 916 | 0.085 | 0.078 | -0.008 [-0.022, +0.007] |
| 6–10 s | 416 | 0.091 | 0.100 | +0.009 [-0.010, +0.027] |
| ≥ 10 s | 613 | 0.115 | 0.119 | +0.004 [-0.017, +0.023] |

## 5. Fit vs generalization

Track R@10 of 300 training clips against the fold's training gallery (≈ 1,000 files) and on the test fold.

| model | train-clip R@10 (train gallery) | test R@10 (test gallery) |
| --- | --- | --- |
| baseline: pooled heads | 0.661 ± 0.171 | 0.235 ± 0.046 |
| learned Q/K/V attention | 0.636 ± 0.165 | 0.243 ± 0.020 |

Best epoch (early stopping on val): baseline 9.1, attention 11.8; val segment R@10 baseline 0.316, attention 0.285.
