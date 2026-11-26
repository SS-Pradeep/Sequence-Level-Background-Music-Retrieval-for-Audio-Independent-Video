# Segmentation methods: comparison study

Compiled from results/segmentation.md (D140), results/segmentation_improve.md (D142) and
results/titlesplit/cue_switch.md (D121). No new runs.

**Test set (shared by every table):** frozen title-split blocked test sequences (data/manifests/cue_sequences_title.json),
5 folds pooled: 112 sequences of 60–180 s, 279 min, 81 film clusters. Ground truth = the music changes at clip joins
(868 joins: 731 inside a film, 137 between films). Thresholds θ chosen per fold on val; learned heads average 3 seeds.
[95% CI] = cluster bootstrap over film groups, 10,000 resamples; Δ paired on the same sequences.

## Table 1. Main comparison (each method's own boundaries, no oracle joins)

| # | method | what decides a boundary | AP | precision ±1 s | recall ±1 s | F1 ±1 s | F1 ±1 s in film | Δ F1 ±1 s vs 1 | segments / min (mean length) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | Uniform 5 s windows | fixed clock | n/a | 0.122 | 0.457 | 0.193 [0.177, 0.209] | 0.169 | −0.251 [−0.279, −0.222] | 12.3 (4.9 s) |
| 1 | PySceneDetect + SigLIP merge (previous default) | pixel cut, merge similar shots | 0.316 | 0.303 | 0.829 | 0.443 [0.417, 0.470] | 0.397 | — | 9.2 (6.6 s) |
| 2 | TransNetV2 + SigLIP merge | learned shot cut, merge similar shots | 0.314 | 0.304 | 0.825 | 0.444 [0.417, 0.471] | 0.396 | +0.000 [−0.005, +0.005] | 9.1 (6.6 s) |
| 3 | TransNetV2 + BaSSL, original MovieNet head | scene boundary (pretrained) | 0.626 | 0.516 | 0.846 | 0.641 [0.607, 0.674] | 0.595 | +0.197 [+0.169, +0.223] | 5.2 (11.6 s) |
| 4 | TransNetV2 + BaSSL + trained head (D140) | scene boundary, head trained on our data | 0.729 | 0.561 | 0.861 | 0.679 [0.648, 0.710] | 0.636 | +0.236 [+0.209, +0.261] | 5.0 (11.9 s) |
| C | 4 + contrast features + BiGRU (D142, chosen on val) | as 4, plus left/right visual contrast and sequence context | 0.762 | — | — | 0.695 [0.665, 0.723] | 0.652 | +0.252 (via 4: +0.016 [+0.004, +0.031]) | — |
| 4b | 4 + log shot duration and motion (excluded) | as 4, plus shot length | 0.778 | 0.602 | 0.885 | 0.716 [0.688, 0.743] | 0.676 | +0.273 [+0.247, +0.297] | 4.9 (12.2 s) |

Why BaSSL wins: recall is about the same for every shot-based method (0.83–0.89), because nearly every music change
falls on a shot cut. The difference is **precision**: PySceneDetect / TransNetV2 keep too many shot cuts (0.30), BaSSL
keeps the ones that start a new scene (0.52 → 0.56 with our head). The shot detector itself does not matter (1 vs 2).

4b is excluded: its extra gain comes from shot length, which leaks the clip-join artifact (the last shot of a clip is
cut short where the clip was trimmed; D79).

## Table 2. Trained-head variants (D142), Δ vs head 4

Segment metrics: purity / coverage / mIoU higher is better; Pk / WindowDiff lower is better. ▲ = CI excludes 0, better.

| variant | val AP | AP | F1 ±1 s | coverage | mIoU | Pk ↓ | WindowDiff ↓ | Δ AP | Δ F1 ±1 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 BaSSL + head (D140) | 0.715 | 0.729 | 0.678 | 0.766 | 0.802 | 0.251 | 0.309 | — | — |
| L1 focal loss | 0.717 | 0.728 | 0.681 | 0.768 | 0.804 | 0.247 | 0.305 | −0.000 | +0.003 |
| L2 BCE + ranking loss | 0.715 | 0.732 | 0.682 | 0.767 | 0.803 | 0.249 | 0.305 | +0.003 | +0.004 |
| F1 + contrast features | 0.731 | 0.745 | 0.691 | 0.771 | 0.807 | 0.244 | 0.300 | +0.016 ▲ | +0.012 ▲ |
| A1 + BiGRU over boundaries | 0.759 | 0.759 | 0.684 | 0.771 | 0.805 | 0.247 | 0.303 | +0.031 ▲ | +0.005 |
| **C contrast + BiGRU (chosen)** | **0.763** | **0.762** | **0.695** | **0.772** | **0.809** | 0.245 | **0.300** | **+0.034 ▲** | **+0.016 ▲** |
| 4b + duration, motion (excluded) | 0.763 | 0.778 | 0.716 | 0.773 | 0.815 | 0.236 | 0.281 | +0.049 ▲ | +0.038 ▲ |

Losses change nothing; the extra visual inputs and the sequence layer give small, real gains. Purity is ~0.967 for all.

## Table 3. Cue-switch systems (Viterbi stitching, D121), same test sequences

**Not directly comparable with Tables 1–2.** Here the segments already contain every clip join (oracle joins), and the
system decides at which segment boundaries the *music* switches, using retrieval scores and a switch penalty. Metric:
boundary F1 ±1 s at music changes inside a film (comparable in spirit to "F1 ±1 s in film" above).

| system | F1 ±1 s in film | precision | recall |
| --- | --- | --- | --- |
| greedy (no switch penalty) | 0.518 [0.490, 0.543] | 0.368 | 0.873 |
| R0 fixed penalties, re-tuned | 0.567 [0.534, 0.596] | 0.462 | 0.732 |
| R1 scene cuts (PySceneDetect content score) | 0.569 [0.535, 0.599] | 0.466 | 0.729 |
| R2 cosine rule | 0.574 [0.544, 0.603] | 0.470 | 0.739 |
| R3 learned, pair-only | 0.580 [0.552, 0.606] | 0.456 | 0.796 |
| **R4 learned + context (shipped)** | **0.584 [0.561, 0.607]** | 0.477 | 0.754 |
| oracle p (true changes, upper bound) | 0.824 [0.804, 0.843] | 0.725 | 0.953 |

R4 − R0 re-tuned +0.018 [−0.003, +0.040]; R4 without log duration drops to 0.549 (the duration shortcut, D79).

## Table 4. Music-awareness probe: changes *inside* continuous clips

3,467 shot cuts inside test clips with the film's own music (no splices); 452 have a music change (D103 audio labels).
Random score: AP 0.130, AUROC 0.500.

| variant | AP [95% CI] | AUROC |
| --- | --- | --- |
| 3 BaSSL original head | 0.153 [0.126, 0.188] | 0.515 |
| 4 BaSSL + head | 0.151 [0.125, 0.186] | 0.520 |
| C contrast + BiGRU | 0.150 [0.123, 0.185] | 0.526 |
| 4b + duration, motion | 0.158 [0.130, 0.196] | 0.535 |
| R4 cue-switch (D85, length held fixed) | — | 0.553 [0.522, 0.591] |

Every method is at or near chance.

## Conclusions

1. BaSSL scene boundaries beat shot-cut segmentation by +0.24 F1 ±1 s (+0.25 with head C), significant in every
   metric. Shot cuts already have the right recall; the gain is precision.
2. The shot detector does not matter (PySceneDetect = TransNetV2). Uniform windows are far worse.
3. Training a head on our data adds +0.04 F1 over BaSSL's own head; contrast features + BiGRU add +0.016 more. New losses
   add nothing.
4. **Caveat:** every true boundary in the benchmark is a clip join, which is also a scene change, so these numbers
   show scene-change detection. The in-clip probe (Table 4) shows no method detects music changes inside continuous
   footage from vision alone.
