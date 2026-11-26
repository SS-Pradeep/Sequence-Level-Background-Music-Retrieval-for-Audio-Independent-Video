# Learned change costs in the Viterbi stitcher (D104)

Title split, 5 folds. Held-out clips with >= 2 segments; true changes = the clip's own music changes at segment-boundary cuts (D103 labels). The soundtrack changes at a boundary when the chosen windows switch file or jump. Penalties, alpha and the scorers' thresholds-free costs tuned on each fold's VAL clips; retrieval = the title-split 10-head ensemble (seeds 0–9), semantic only. CIs: paired cluster bootstrap over film groups, 2,000 resamples.

| system | boundary F1 | precision | recall | soundtrack changes per boundary | segments on the clip's own track | Δ F1 vs fixed [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| fixed penalties (tuned on val clips) | 0.247 | 0.160 | 0.537 | 0.559 | 0.011 | — |
| learned: GBM video + music | 0.265 | 0.182 | 0.487 | 0.438 | 0.008 | +0.018 [-0.011, +0.048] |
| learned: MLP video + music | 0.263 | 0.178 | 0.508 | 0.459 | 0.007 | +0.016 [-0.020, +0.051] |
| learned: GBM video only | 0.276 | 0.184 | 0.550 | 0.486 | 0.008 | +0.029 [-0.006, +0.060] |

Test clips: 649; scored boundaries with a change: 240.
