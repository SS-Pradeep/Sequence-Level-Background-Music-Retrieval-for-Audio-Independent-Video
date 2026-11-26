# Learning curve: semantic heads on the film split

SigLIP + CLAP+MERT heads, seeds 0, 1, 2 (seeds per point: [3]), 5 folds pooled (982 test queries, test-fold gallery ≈ 196 files). Per fold and seed, nested random subsets of the training films other than Konjum salangai (25 / 50 / 75 / 100%; each seed draws its own subsets, so the spread includes which films are used); Konjum stays in, with the 30-clips-per-epoch cap; val and test are unchanged. 100% = runs/filmsplit/semantic/siglip-both. R@k: per-query hits averaged over seeds; MedR: mean over seeds of the median rank. 95% CIs: cluster bootstrap over film groups; ± is the std over seeds.

![learning curve](learning_curve.png)

| training films (mean over folds) | share | R@10 [95% CI] | R@1 [95% CI] | MedR [95% CI] |
| --- | --- | --- | --- | --- |
| 73 | 25% | 0.174 [0.154, 0.196] ± 0.007 | 0.022 [0.017, 0.028] ± 0.005 | 43.0 [38.7, 46.3] |
| 145 | 50% | 0.216 [0.194, 0.239] ± 0.006 | 0.036 [0.029, 0.044] ± 0.006 | 35.7 [32.0, 39.3] |
| 218 | 75% | 0.245 [0.219, 0.271] ± 0.014 | 0.045 [0.036, 0.055] ± 0.005 | 31.3 [28.0, 36.0] |
| 290 | 100% | 0.288 [0.258, 0.319] ± 0.013 | 0.055 [0.044, 0.067] ± 0.009 | 27.2 [23.0, 31.3] |

- Gain 25% → 100% of the films, R@10: **+0.114** [+0.087, +0.141] (seeds [0, 1, 2])
- Gain 25% → 100% of the films, R@1: **+0.033** [+0.021, +0.046] (seeds [0, 1, 2])
- Change 25% → 100%, MedR: -15.8 [-19.0, -12.0] (negative = better)
- Gain 50% → 100% of the films, R@10: **+0.072** [+0.050, +0.094] (seeds [0, 1, 2])
- Gain 50% → 100% of the films, R@1: **+0.019** [+0.009, +0.030] (seeds [0, 1, 2])
- Change 50% → 100%, MedR: -8.5 [-11.3, -5.3] (negative = better)
- Gain 75% → 100% of the films, R@10: **+0.043** [+0.026, +0.061] (seeds [0, 1, 2])
- Gain 75% → 100% of the films, R@1: **+0.010** [-0.001, +0.021] (seeds [0, 1, 2])
- Change 75% → 100%, MedR: -4.2 [-7.0, -2.0] (negative = better)
- Gain 25% → 50% of the films, R@10: **+0.042** [+0.024, +0.059] (seeds [0, 1, 2])
- Gain 25% → 50% of the films, R@1: **+0.014** [+0.005, +0.022] (seeds [0, 1, 2])
- Change 25% → 50%, MedR: -7.3 [-10.0, -4.2] (negative = better)

Least-squares slope of R@10 against log2(training films): +0.054 per doubling (descriptive; 4 points).
