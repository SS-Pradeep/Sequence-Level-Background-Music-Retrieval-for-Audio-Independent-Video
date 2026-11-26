# Final: best combined setting on test (D138, Step 6)

Chosen on validation only: heads = **d135-do05** (val track MRR {"siglip-both": 0.2996, "d135-do05": 0.3046}); candidates K = 10, C = 40, window step 2 s (best val stitching objective, Step 4); fixed penalties re-tuned on val per fold and run seed (Step 5). Cleaned title split, 50 test concatenations per fold, 5 folds × 3 run seeds (disjoint 10-head ensembles); ± = std over fold means; Δ = per sequence, run seeds averaged, 95% cluster bootstrap over sequences. Test evaluated once.

| system | switch F1 | switches / min | song acc. | natural-video track acc. | Δ switch F1 [95% CI] | Δ song acc. [95% CI] |
|---|---|---|---|---|---|---|
| shipped (K 10, C 20, 0.5 s) | 0.739 ± 0.069 | 3.84 | 0.054 ± 0.020 | 0.049 | — | — |
| best combined (val-chosen) | 0.770 ± 0.055 | 3.28 | 0.063 ± 0.020 | 0.053 | +0.032 [+0.012, +0.051] | +0.009 [-0.003, +0.022] |

## With the change predictor (R4, α and λ_jump re-tuned on the cue val sequences)

Blocked test sequences of `bgm cue eval` (emission run seed 0 = one 10-head ensemble; cue seeds 0–2 averaged).

| emission | system | α per fold | λ_jump per fold | μ per fold | switch F1 (all joins) | in-film F1 (paper headline) | song acc. |
|---|---|---|---|---|---|---|---|
| shipped | R0_tuned | — | 1; 0.25; 2; 2; 2 | 0; 0; 0; 0.5; 1 | 0.614 | 0.581 | 0.054 |
| shipped | R4 | 2/1/0.5; 0.5/0.5/0.125; 8/4/4; 2/1/2; 0.25/0.25/0.25 | 2/1/1; -1/-1/-1; -0.25/-0.25/-0.25; 1/1/1; 0.5/0.5/0.5 | 0/0/0; 0/0/0; 0/0/0; 0.5/0.5/0.5; 1/1/1 | 0.623 | 0.604 | 0.049 |
| best combined | R0_tuned | — | 1; 2; 1; 2; 1 | 0.5; 0.5; 0.5; 0; 0.5 | 0.618 | 0.596 | 0.048 |
| best combined | R4 | 8/8/4; 1/2/1; 1/1/1; 2/1/2; 0.125/0.125/0.5 | 2/1/1; 0.25/0.25/0.25; -0.25/0.25/-0.25; 1/0.5/0.5; 0/0/0.25 | 0.5/0.5/0.5; 0.5/0.5/0.5; 0.5/0.5/0.5; 0/0/0; 0.5/0.5/0.5 | 0.616 | 0.595 | 0.052 |
