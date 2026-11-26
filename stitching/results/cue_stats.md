# Cue statistics from the training films' BGM (per fold)

| fold | L (s) | ρ | cues | files | music (min) | cues ≥ 4 s | reprises | x-corr threshold (p99 of different-film pairs) |
|---|---|---|---|---|---|---|---|---|
| 0 | 9.42 | 0.338 | 2133 | 999 | 335 | 2133 | 722 | 0.950 |
| 1 | 9.42 | 0.330 | 2101 | 977 | 330 | 2101 | 693 | 0.953 |
| 2 | 9.46 | 0.359 | 2106 | 991 | 332 | 2106 | 756 | 0.949 |
| 3 | 9.49 | 0.367 | 2110 | 977 | 334 | 2110 | 774 | 0.948 |
| 4 | 9.43 | 0.352 | 2150 | 1007 | 338 | 2150 | 756 | 0.949 |

Mean over folds: L = 9.44 ± 0.02 s, ρ = 0.349 ± 0.013.

Caveat: every dataset file is one clip's BGM (~20 s median), and file ends count as cue ends, so L is bounded by the clip length; it is the cue length *within these clips*, not of whole film scores.
