# Pipeline on the full dataset (4,246 pairs) with the 10-seed ensemble of heads (D73)

Full dataset, all 4,246 pairs of `dataset_merged`, extended film split, heads = the ensemble of seeds 0–9 (one run per fold), 5 folds × 1 seeds = 5 runs under `runs/full/pipeline_ens10`; heads `runs/full/semantic/siglip-both (seeds 0–9)`. Scoring is semantic only (`fusion.dyn_weight` 0, `fusion.attr_weight` 0; D55, D56) and no dynamics model is loaded. Test-fold gallery 781 (743–812) music files; 13,309 test segments. Tables: mean ± std over the runs. CIs: per-query values averaged over seeds, 95% cluster bootstrap over film groups (D39), 10,000 resamples, unless a section says otherwise. The single-head runs in `runs/full/pipeline` are shown for reference.

## 1. Pipeline accuracy (M9)

Per test segment after the coarse and fine stages, before Viterbi. Shortlist: top 10 files for the whole video and for each segment; windows every 0.5 s; J = 20. Track@1: the top window's file is in the right track group. Window@1 / @J: the query's own file and an offset within 0.5 s of the aligned one.

| system | track@1 | window@1 | window@J |
| --- | --- | --- | --- |
| **coarse-to-fine, semantic only** | 0.019 ± 0.005 | 0.004 ± 0.001 | 0.018 ± 0.005 |
| no coarse stage (every window of every file) | 0.019 ± 0.005 | 0.003 ± 0.001 | 0.018 ± 0.006 |
| attributes only (reference; weight 0 in the pipeline) | 0.016 ± 0.006 | 0.002 ± 0.002 | 0.013 ± 0.003 |
| _single heads, same data (D61; 15 runs)_ | 0.016 ± 0.004 | 0.003 ± 0.001 | 0.014 ± 0.004 |

- Chance (a random window of the gallery): track@1 0.003, window@1 0.0001. Coarse-to-fine minus chance: track@1 +0.016 [+0.012, +0.021]; window@1 +0.003 [+0.002, +0.005].
- Coarse-to-fine minus no coarse stage: track@1 +0.000 [-0.000, +0.001]; window@1 +0.000 [-0.000, +0.000]; window@J +0.000 [-0.001, +0.001].
- Tie check: neighbouring 0.5 s windows often pool the same seconds and tie exactly. Window@1 with ties sharing the hit is 0.003, against 0.003 for the first-index choice in the table.

## 2. Offset precision of the top-1 window

For each test segment: the top-1 (file, offset) after the coarse and fine stages. A hit needs the file in the right track group AND the offset within the tolerance of the true (aligned) offset; windows tied at the top share the hit. Chance is exact per query: the share of all gallery windows (every test-fold file, every 0.5 s) that the same rule accepts.

| rule | coarse-to-fine | chance | minus chance [95% CI] | verdict | no coarse stage |
| --- | --- | --- | --- | --- | --- |
| right track group, any offset | 0.019 ± 0.005 | 0.003 | +0.016 [+0.012, +0.021] | above chance | 0.019 ± 0.005 |
| right track group and within ±1 s | 0.004 ± 0.001 | 0.0002 | +0.004 [+0.003, +0.006] | above chance | 0.004 ± 0.001 |
| right track group and within ±3 s | 0.007 ± 0.001 | 0.0004 | +0.007 [+0.005, +0.009] | above chance | 0.007 ± 0.001 |
| right track group and within ±5 s | 0.008 ± 0.002 | 0.001 | +0.008 [+0.006, +0.011] | above chance | 0.008 ± 0.002 |

The joint numbers are mostly track retrieval: the chance levels are tiny because a random window is rarely in the right track group at all. The next table asks the offset question on its own. Among the segments whose top-1 file is in the right track group, how often is the offset within the tolerance, against a random window of that track group's files?

| tolerance | within tolerance, given the right track group | chance, given the right track group | minus chance [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| ±1 s | 0.232 | 0.186 | +0.047 [+0.010, +0.088] | above chance |
| ±3 s | 0.379 | 0.339 | +0.040 [-0.003, +0.086] | at chance |
| ±5 s | 0.451 | 0.469 | -0.017 [-0.066, +0.032] | at chance |

Once the right track group is found, the offset is within ±1 s 0.232 of the time against 0.186 for a random window of it. The three rows are nested and share the same segments, so they are one finding, not three independent tests: the offset is only slightly better than random.

Context for the chance levels: segments last 4.7 s (median) and the files are short, so a file offers few distinct windows and a wide tolerance covers most of them. 247 of 13,309 segments (seed-averaged) have their top-1 in the right track group.

## 3. Stitching (M10): Viterbi vs greedy

Test concatenations: 50 per fold, each 3–4 test clips with different track groups joined end to end (data/manifests_full/concat_sets_film.json). Penalties are tuned per run on the val concatenations (objective: part track accuracy + switch-boundary F1). Greedy takes each segment's top window. CIs resample concatenations.

| metric | Viterbi | greedy | Viterbi − greedy [95% CI] |
| --- | --- | --- | --- |
| part track accuracy (duration-weighted) | 0.038 ± 0.017 | 0.029 ± 0.012 | +0.009 [+0.000, +0.018] |
| switch-boundary F1 (±1 s) | 0.690 ± 0.057 | 0.497 ± 0.059 | +0.194 [+0.169, +0.218] |
| switches / min | 3.192 ± 0.668 | 6.724 ± 0.509 | -3.532 [-3.792, -3.282] |
| jumps / min | 1.433 ± 0.892 | 1.837 ± 0.322 | -0.404 [-0.606, -0.214] |
| seam loudness jump, median LU per video | 2.090 ± 0.690 | 2.016 ± 0.291 | +0.075 [-0.202, +0.362] |

For reference, the single heads on the same data (D61): part track accuracy 0.027 ± 0.016 vs 0.018 ± 0.009; switch F1 0.679 ± 0.043 vs 0.495 ± 0.051; switches / min 3.020 ± 0.495 vs 6.739 ± 0.448.

Natural test videos after decoding, duration-weighted per video (film-group CIs):

| measure | Viterbi | greedy | Viterbi − greedy [95% CI] |
| --- | --- | --- | --- |
| track accuracy | 0.023 ± 0.005 | 0.021 ± 0.004 | +0.002 [-0.000, +0.004] |
| window accuracy (own file, within 0.5 s) | 0.011 ± 0.005 | 0.008 ± 0.002 | +0.002 [+0.000, +0.005] |

Tuned penalties (val), most common of 5 runs: {"lam_jump": 2.0, "lam_switch": 2.0, "mu": 0.5} ×1, {"lam_jump": 0.25, "lam_switch": 0.5, "mu": 0.0} ×1, {"lam_jump": 1.0, "lam_switch": 2.0, "mu": 0.0} ×1.

## 4. Old clips and new clips

The same runs and galleries, with the test segments split by origin: the 1,248 pairs of the earlier dataset (old) and the Demucs-separated clips of the added films (new). Per-segment values averaged over seeds; film-group CIs.

| segments of | n | films | track@1 | chance | minus chance [95% CI] | right track and within ±1 s | chance | window@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all clips | 13,309 | 421 | 0.019 | 0.003 | +0.016 [+0.012, +0.021] | 0.004 | 0.0002 | 0.003 |
| old clips | 2,552 | 397 | 0.053 | 0.002 | +0.051 [+0.039, +0.064] | 0.015 | 0.0002 | 0.011 |
| new clips | 10,757 | 31 | 0.011 | 0.003 | +0.007 [+0.004, +0.011] | 0.002 | 0.0002 | 0.002 |

