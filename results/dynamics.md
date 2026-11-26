# Dynamics (M7, H2)

Grounding with chance levels (`bgm eval --exp grounding`): the saved predictions of the trained real-data models are re-scored (no real-data model retrained; M9 fused needs inference only). v1 numbers (before the motion-feature fix, D27) remain in results/final_v1.md.

Each query: a test segment at its exact duration against every 25 Hz offset of its own music file. Rules: exact = |error| ≤ 0.1 s; within 0.5 s = |error| ≤ 0.5 s (key `strict`); sync = exact, or the predicted shift still lines up (`lines_up`: a guard of `sample_negative_shifts` fires). Per query: hit = the rule at the predicted offset (when several offsets tie at the top score, the rule's mean over them: the expected hit under a random tie-break); chance = the share of offsets the rule accepts (a uniformly random offset's hit rate). Hits are averaged over seeds per query, then pooled over folds; means with 95% cluster-bootstrap CIs over track groups (10,000 resamples). "before" = the number reported before this change (mean over runs; old definitions: strict the same, sync = |error| ≤ 0.1 s or the sampler guards, judged on the beat grid alone near file edges); exact is new.

## Real test folds (3280 queries from 1150 track groups; learned and fused: 5 folds × 3 seeds, hand-crafted: 5 folds, deterministic)

| system | rule | before | accuracy | chance | accuracy − chance (95% CI) |
| --- | --- | --- | --- | --- | --- |
| learned dynamics, shifted negatives (M7) | exact (±0.1 s) | — | 0.138 [0.126, 0.151] | 0.115 [0.104, 0.127] | **+0.023 [+0.018, +0.029]** |
| learned dynamics, shifted negatives (M7) | within 0.5 s | 0.214 | 0.214 [0.199, 0.230] | 0.173 [0.160, 0.187] | **+0.041 [+0.034, +0.048]** |
| learned dynamics, shifted negatives (M7) | sync (exact, or the shift still lines up) | 0.443 | 0.444 [0.431, 0.458] | 0.435 [0.426, 0.444] | **+0.009 [-0.000, +0.019]** |
| learned dynamics, no shifted negatives (M7) | exact (±0.1 s) | — | 0.144 [0.132, 0.158] | 0.115 [0.104, 0.127] | **+0.029 [+0.024, +0.035]** |
| learned dynamics, no shifted negatives (M7) | within 0.5 s | 0.219 | 0.219 [0.204, 0.235] | 0.173 [0.160, 0.187] | **+0.046 [+0.039, +0.053]** |
| learned dynamics, no shifted negatives (M7) | sync (exact, or the shift still lines up) | 0.446 | 0.447 [0.434, 0.460] | 0.435 [0.426, 0.444] | **+0.012 [+0.003, +0.021]** |
| fused semantic + dynamics + mood (M9) | exact (±0.1 s) | — | 0.140 [0.129, 0.153] | 0.115 [0.104, 0.127] | **+0.025 [+0.021, +0.030]** |
| fused semantic + dynamics + mood (M9) | within 0.5 s | — | 0.232 [0.218, 0.248] | 0.173 [0.160, 0.187] | **+0.059 [+0.052, +0.066]** |
| fused semantic + dynamics + mood (M9) | sync (exact, or the shift still lines up) | — | 0.444 [0.432, 0.455] | 0.435 [0.426, 0.444] | **+0.009 [+0.002, +0.016]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | exact (±0.1 s) | — | 0.129 [0.116, 0.143] | 0.115 [0.104, 0.127] | **+0.014 [+0.007, +0.021]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | within 0.5 s | 0.199 | 0.199 [0.183, 0.216] | 0.173 [0.160, 0.187] | **+0.026 [+0.017, +0.035]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | sync (exact, or the shift still lines up) | 0.440 | 0.440 [0.423, 0.458] | 0.435 [0.426, 0.444] | **+0.006 [-0.009, +0.020]** |

Paired differences in (hit − chance), same queries (chance cancels, so this is also the difference in accuracy):

- shiftneg − hand: exact +0.009 [+0.001, +0.017]; within 0.5 s +0.015 [+0.005, +0.026]; sync +0.004 [-0.013, +0.022]
- noshift − hand: exact +0.015 [+0.006, +0.024]; within 0.5 s +0.020 [+0.010, +0.031]; sync +0.007 [-0.010, +0.023]
- shiftneg − noshift: exact -0.006 [-0.011, -0.001]; within 0.5 s -0.005 [-0.012, +0.001]; sync -0.003 [-0.014, +0.009]
- fused − hand: exact +0.011 [+0.003, +0.019]; within 0.5 s +0.033 [+0.023, +0.045]; sync +0.003 [-0.013, +0.020]

Median absolute error (s): shiftneg 3.88, noshift 3.96, fused 3.80, hand 3.92.
- Ties: 51% of fused queries have several offsets at the top score (in the runs whose tuned w_fine gives the dynamics branch weight 0, the score changes only at 1 Hz feature boundaries); learned dynamics and hand-crafted scores are continuous (0.2% tied, hand-crafted). Breaking fused ties by the first offset instead (offset 0 is the true start of every video's first segment) would report exact 0.168 (excess +0.053, vs +0.025); within 0.5 s 0.236 (excess +0.063, vs +0.059); sync 0.475 (excess +0.040, vs +0.009).
- 397 of 3280 queries leave ≤ 1 s of slack between segment and file, so every offset is within 0.5 s (chance 1, excess 0); they raise accuracy and chance alike.

## Synthetic beat-cut set (48 queries from 8 val clips; one learned run, seed 0)

| system | rule | before | accuracy | chance | accuracy − chance (95% CI) |
| --- | --- | --- | --- | --- | --- |
| learned dynamics, shifted negatives | exact (±0.1 s) | — | 0.458 [0.380, 0.565] | 0.010 [0.009, 0.011] | **+0.448 [+0.370, +0.555]** |
| learned dynamics, shifted negatives | within 0.5 s | — | 0.458 [0.380, 0.565] | 0.049 [0.045, 0.054] | **+0.410 [+0.333, +0.515]** |
| learned dynamics, shifted negatives | sync (exact, or the shift still lines up) | 0.979 | 0.979 [0.942, 1.000] | 0.390 [0.310, 0.463] | **+0.589 [+0.490, +0.687]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | exact (±0.1 s) | — | 0.229 [0.122, 0.356] | 0.010 [0.009, 0.011] | **+0.219 [+0.112, +0.345]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | within 0.5 s | — | 0.292 [0.200, 0.405] | 0.049 [0.045, 0.054] | **+0.243 [+0.153, +0.350]** |
| hand-crafted dynamics (cut-on-beat + motion/energy) | sync (exact, or the shift still lines up) | — | 0.646 [0.562, 0.733] | 0.390 [0.310, 0.463] | **+0.256 [+0.192, +0.316]** |

- learned − hand-crafted: exact +0.229 [+0.116, +0.327]; within 0.5 s +0.167 [+0.070, +0.255]; sync +0.333 [+0.250, +0.435]
- The synthetic model is the only one retrained (user-approved; the original synthetic checkpoint had not been saved): same generator, seed and settings as the M7 acceptance test (28 training clips, max_epochs 80), checkpoint now kept in runs/synthetic_grounding/. Val sync 0.979 (old definition, best epoch 5) vs the previously reported 0.979 (the acceptance test's printout). These val clips also choose the epoch, so this set is a sanity check, not a held-out result; the CIs rest on 8 clips.

## H2

- Learned dynamics (shifted negatives) above chance on sync: **not shown** (Δ +0.009, CI [-0.000, +0.019]).
- Learned dynamics beats hand-crafted on sync: **not shown** (Δ +0.004, CI [-0.013, +0.022]).
- Shifted negatives are necessary: **not shown** (Δ -0.003, CI [-0.014, +0.009]).

Training: 126 of 895 fold-0 training clips are too short for an 8 s window (> 10%), so 6 s / 4 s length buckets are on (SPEC §3.5); no clip is skipped.
