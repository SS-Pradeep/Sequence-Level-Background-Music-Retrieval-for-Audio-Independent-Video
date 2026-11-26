# Unlabeled pretraining on the film split (H4)

Heads: SigLIP + CLAP+MERT. From scratch: runs/filmsplit/semantic/siglip-both; pretrained on the M8 pool, then fine-tuned at lr_finetune: runs/filmsplit/semantic/siglip-both-pt. 5 folds × 3 seeds each, 982 test queries. Δ: per-query hits averaged over seeds, paired, 95% cluster bootstrap over film groups.

| section | metric | from scratch | pretrained + fine-tuned | Δ [95% CI] |
| --- | --- | --- | --- | --- |
| Track retrieval, test-fold gallery | R@1 | 0.055 | 0.058 | **+0.003** [-0.007, +0.014] |
| Track retrieval, test-fold gallery | R@10 | 0.288 | 0.285 | **-0.003** [-0.024, +0.018] |
| Track retrieval, all-library gallery | R@1 | 0.012 | 0.013 | **+0.001** [-0.004, +0.006] |
| Track retrieval, all-library gallery | R@10 | 0.081 | 0.086 | **+0.004** [-0.009, +0.018] |
| Segment retrieval, all windows | R@1 | 0.020 | 0.022 | **+0.003** [-0.002, +0.007] |
| Segment retrieval, all windows | R@10 | 0.123 | 0.122 | **-0.001** [-0.012, +0.010] |
| Segment retrieval, other-video windows | R@1 | 0.031 | 0.030 | **-0.001** [-0.008, +0.005] |
| Segment retrieval, other-video windows | R@10 | 0.129 | 0.129 | **-0.000** [-0.012, +0.011] |

**H4** (pretraining beats training on the pairs alone, track R@10, test-fold gallery): **not shown** (Δ -0.003 [-0.024, +0.018]).
