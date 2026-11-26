# Video-to-BGM Retrieval: Full Methodology

This document describes the method end to end: task, data, splits, segmentation, features, models, retrieval, sequence-level stitching, the music-change predictor, rendering, and the evaluation protocol, followed by the headline results and the caveats. Codes like "D75" are entries in the decision log (docs/PROGRESS.md). Full result tables, including every baseline, are in docs/PROJECT_BRIEFING.md.

**Status:** 2026-11-13, last decision D123. A loss-function ablation of the semantic heads (D124: sigmoid, hard-negative InfoNCE, DCL, VSE++) is coded but has no results yet.

---

## 1. Task and pipeline

- **Input:** a silent video, e.g. an Indian film scene.
- **Output:** the same video with a continuous background-music (BGM) soundtrack assembled from a fixed music library.

Pipeline:

1. **Segment** the video into pieces of 2–20 s (§4).
2. **Retrieve** candidate music windows per segment, coarse-to-fine (§7).
3. **Predict** at each boundary between segments the probability that the music should change (§9).
4. **Stitch:** choose one window per segment with a Viterbi pass that trades segment fit against musical continuity, with the change probabilities as switch costs (§8).
5. **Render:** beat-snapped piece starts, crossfades, loudness normalization, mux (§10).

**Contribution:** the sequence-level choice of music (when to keep a track and when to switch) on a leak-free benchmark. Per-clip video-to-music retrieval is an existing task.

Three branches compare video with music: **semantic** (what is on screen vs what the music sounds like), **dynamics** (motion and cuts over time vs beats and energy over time) and **attributes** (mood). Only the semantic branch is in the final system; the other two were built, evaluated and switched off (§13).

## 2. Data

| | Original | Cleaned (D115) |
|---|---|---|
| Video–music pairs | 1,248 | 1,011 |
| Music duration | 6.7 h | 4.65 h |
| Track groups (distinct songs) | 1,150 | 912 |
| Test clips over 5 folds | 982 | 894 |
| Gallery per test fold | ~196 files | ~179 files |
| Chance track R@10 | 0.058 | 0.069 |

- **Source:** Indian film clips (SunNXT, YouTube, compilations), original 1,248 pairs: 228 real films + 15 compilations (445 film keys, 219 film groups), clips 9.84–171 s, median 15.0 s; cleaned 1,011 pairs: 420 film keys, 211 film groups, clips 1.07–171 s, median 13.0 s. SunNXT supplies 499 of the 1,011 cleaned pairs. Each video's own soundtrack is its BGM, so music and picture are time-aligned. 1,255 delivered files minus 7 duplicates.
- **Konjum salangai subset (266 clips):** DC offset, 100 Hz mains hum, ~27 dB too quiet. The hum fakes duplicates in fingerprinting, so the audio is high-passed before matching. Used for training only, never for validation or test.
- **Cleaned version:** every pair hand-trimmed with a pair-trimmer tool to drop non-BGM audio; same folds.
- **How much is music (PANNs CNN14 per-second tags, 150 Hz high-pass):** 69.5% of seconds are clear music in the original data, 82.6% after cleaning; speech is under 1%.
- **Dropped extension:** 2,998 extra Demucs-separated clips from only 32 films gave no gain (recall grows with distinct films, not clips).

**Media preparation**

- Video: silent copy, constant 25 fps, short side ≤ 480 px. Every video feature is read from this copy, so no video feature hears the soundtrack.
- Music: mono 48 kHz FLAC, resampled in memory per encoder (CLAP 48 kHz, MERT 24 kHz, PANNs 32 kHz, librosa 22,050 Hz); the rate is asserted at every encoder boundary.
- **Timing rule:** music sample 0 is the instant of the first video frame; A/V start offsets reported by ffprobe are corrected.

**Grouping songs.** The same song must not be in train and test. Links come from Chromaprint fingerprints (after a 200 Hz high-pass, confirmed by log-mel correlation) and CLAP cosine ≥ 0.97; union-find over the links gives the track groups (`track_id`).

## 3. Splits

| Split | Grouped by | Problem |
|---|---|---|
| Track split | song | 65% of test clips had a same-film clip in training; the model memorised films (R@10 0.60 seen vs 0.28 unseen films) |
| Film split | source film ID | one movie uploaded under several IDs: 84 films in more than one test fold |
| **Title split (all headline numbers)** | real film title, 224 hand-checked links | films in more than one test fold: 84 → 1 |

- 5-fold stratified group k-fold over film groups, stratified by duration. Validation is 10% of the training films, disjoint from training.
- The test fold never touches training, early stopping, calibration, fusion weights, stitching penalties or the change predictor's tuning.
- Contrastive batches hold at most one segment per video; pairs sharing a `track_id` are masked out of each other's negatives.

## 4. Segmentation

1. **Shots:** PySceneDetect `AdaptiveDetector` on the silent 25 fps video.
2. **Merge** consecutive shots while their SigLIP embedding cosine is ≥ 0.85 and the segment stays ≤ 20 s.
3. **Fold** shots under 2 s into the more similar neighbour.
4. **Split** any single shot over 20 s into equal parts.
5. **Span rule:** the semantic and mood branches pool at least 5 s, widening the span symmetrically, on both video and music sides.

## 5. Features

All pretrained encoders are frozen (`eval()`, no gradients, fp16) and cached once per video / music file with a config hash.

| Branch | Video side (silent video) | Music side |
|---|---|---|
| Semantic | SigLIP so400m image embeddings, 1 Hz | CLAP `larger_clap_music_and_speech` ⊕ MERT-v1-330M (mean of 25 layers), 1 Hz |
| Dynamics (25 Hz curves) | optical-flow motion, visual onsets, cut impulses, luminance change, camera motion | onset strength, beat impulses, RMS loudness (dB), spectral-centroid change, novelty |
| Attributes | SigLIP zero-shot mood scores | CLAP zero-shot mood scores, calibrated; compared by Jensen–Shannon divergence |
| Stitching extras | — | tempo, key/mode |

`laion/larger_clap_music` (the spec's choice) has broken public weights (embeddings collapse), so the music_and_speech variant is used.

## 6. Models and training

**Semantic heads (the scoring model of the final system)**

- One MLP per side: LayerNorm → Linear(512) → GELU → Dropout → Linear(256), L2-normalized.
- Input: mean of the 1 Hz embeddings over the segment's span (≥ 5 s).
- Loss: symmetric InfoNCE with a learnable temperature. Optimizer AdamW.
- Augmentation: drop 25% of frames; jitter the music span by ±1 s.
- Early stopping on validation segment R@10.
- **Ensemble (D75):** 10 heads with different seeds, cosines averaged. +0.059 track R@10 over single heads [+0.041, +0.079]. Each of the 3 reported "seeds" of the final system is a disjoint set of 10 heads.

**Dynamics encoders (built, weight 0 in the final system)**

- Dilated 1-D residual CNNs, ~87K parameters, 2.4 s receptive field.
- Contrastive loss; negatives are other music in the batch plus time-shifted copies of the pair's own music, at shifts that never land on a whole number of beats (alignment is the label). Dynamics inputs are never time-shifted as augmentation.

**Self-supervised pretraining** on ~1,000 unlabeled music videos was tried (M8); no gain.

## 7. Retrieval

1. **Coarse:** score every library file against the whole video and against each segment; shortlist the union of the top 10 of each.
2. **Fine:** for each segment, slide windows every 0.5 s over each shortlisted file and score them with the heads.
3. **Fusion:** each branch's scores are z-scored per segment and summed with weights. **Final weights: semantic 1, dynamics 0, mood 0** (dynamics is at chance alone; val-tuned mood weights hurt test).

The z-scored fine score of a window is its **emission score**; Viterbi uses −score as the unary cost.

## 8. Viterbi stitching

- **Candidates per segment:** the top 20 windows, plus the contiguous continuation (same file, next stretch of audio) of each candidate of the previous segment.
- **Transition costs** between segments i and i + 1:

| Move | Fixed-penalty decoder (D75) | With change predictor (D121, default) |
|---|---|---|
| continue contiguously in the same file | 0 | −α · log(1 − p_i) |
| switch to another file | λ_switch + μ · (circle-of-fifths key distance + tempo mismatch) | −α · log(p_i) + μ · (key + tempo distance) |
| jump within the same file | λ_jump | −α · log(p_i) + λ_jump |

  With constant p the two decoders are identical (unit-tested), so the predictor only replaces one global switch penalty with a per-boundary one.

- **Tuning (validation only):**
  - Fixed penalties λ_switch, λ_jump, μ: per fold, on validation "concatenations" (3–4 clips of different songs joined), objective = part track accuracy + switch-boundary F1.
  - With the predictor: α ∈ {0.125 … 8} and λ_jump ∈ {−2 … 2} grid-tuned per fold on validation sequences; μ keeps the fold's D75 value.

## 9. Music-change predictor (D78, D121)

Decides *where* the soundtrack should change; the emission scores decide *what* plays.

- **Token per segment (1,155-d):** mean SigLIP embedding over the segment's own frames (no 5 s widening, so tokens don't leak into neighbours), L2-normalized; plus log duration and mean/std of the optical-flow motion curve, z-scored with training-fold statistics (frame 0 excluded, since it has no flow).
- **Architecture (R4, 1.68 M parameters):**
  1. token MLP: Linear(1155 → 256) → GELU → Dropout → Linear(256);
  2. a learned [CTX] token, then a 2-layer bidirectional Transformer encoder (d 256, 4 heads, FFN 512, dropout 0.4, pre-norm), no positional encoding;
  3. boundary head on each neighbouring pair: MLP([h_i, h_{i+1}, |h_i − h_{i+1}|, h_i ⊙ h_{i+1}]) → logit of p_i.
- **Training data:** 60–180 s sequences from the fold's training clips only, redrawn every epoch (blocks of 2–8 clips of one film, chained across films). Label 1 where the track group changes across a boundary.
- **Training:** BCE with pos_weight = negatives / positives; AdamW, lr 3e-4, weight decay 0.3, batch 32; early stopping on validation AP. 5 folds × 3 seeds; at render time the 3 seeds' probabilities are averaged.
- **Variant choice (validation only):** R4 `ctx` (val AP 0.714) over `ctx_splice` and `ctx_cross`.
- **Switch:** `cue.enabled: true` by default; `cue.enabled=false` reproduces the fixed-penalty decoder bit-identically. Recommended off for one long continuous film (§12).

## 10. Rendering

1. Consecutive segments using contiguous audio from one file form a **piece**.
2. **Beat snap:** each piece's start shifts to the beat within ±2 s that puts the most video cuts on beats (render time only).
3. Equal-power crossfades at seams with linear gain ramps meeting both sides' loudness (~10 other shapes tested; none better).
4. Each piece normalized to −16 LUFS; lookahead limiter at −1 dBFS.
5. Muxed onto the original video with a measured start-offset correction; sync verified to 0.0 ms by cross-correlation.

Opt-in demo extras, not part of the benchmarked system: pace match, a per-segment scene brief from Qwen2.5-VL-3B matched to music via CLAP text embeddings, replay/reprise limits, length fit, silent spans, 1080p upscaling, emotion showcases.

## 11. Evaluation protocol

**Retrieval**

- **Track R@k:** query = whole test video; gallery = the test fold's music; hit = any file of the right song. Also R@1/5, median rank, MRR, nDCG@10; chance computed exactly per query.
- **Exact-pair R@k** (only the video's own file counts) and **music → video** direction.
- **Segment R@k:** query = segment; target = its exact aligned window.
- **Pairwise accuracy** (MEMA protocol): P(own music scores above another song's music), ties ½.
- **Grounding:** given the right file, find the offset; tied top scores share the hit (avoids an offset-0 bias).

**Pipeline:** track@1 (right song) and window@1 (right song within 0.5 s of the right offset).

**Stitching**

- 50 test concatenations per fold (3–4 clips of different songs).
- Switch-boundary F1 (switch within ±1 s of a true join), TBIoU, duration-weighted part track accuracy, switches/min, seam loudness jump (LU).

**Change predictor**

- Frozen test sequences from each fold's test clips, each clip used once: blocked 1–3 min (112 sequences, 279 min), blocked 5–15 min (27 sequences), single-film 5–15 min (9 sequences).
- Headline: **in-film boundary F1 (±1 s)**, counting only song changes between clips of the same film; any switch inside a clip is a false positive.
- AP / AUROC, and probes with segment length held fixed (8 × 8 length bins).
- Comparisons: greedy; R0 fixed penalties (stored and re-tuned); R1 scene-cut rule; R2 SigLIP-cosine rule; R3 pair-only model; R4s splice-trained; R4 without duration; oracle.

**Content fit and other judges**

- ImageBind and LanguageBind (LB, tw-LB) cosine between video and chosen music; models the system never uses.
- KL of PANNs tags vs the original music; beat alignment (cut-on-beat excess over chance, ±70 ms); diversity (entropy, coverage, top-5 share).
- Robustness: features re-extracted from degraded media (270p/180p video, MP3 128/64/32 kbps, combined).

**Baselines (one shared evaluator, D120):** random; ImageBind and LanguageBind zero-shot; MMTrail captions → CLAP; VM-NET (CBVMR); EmoMV single- and multi-task; DecCM and its teacher. Each rebuilt from its own models and checked to reproduce its earlier numbers. Plus CFD-, UT-CMVMR- and MVPt-style models re-implemented on the same frozen features (D43).

**Statistics**

- 5 folds × 3 seeds; mean ± std.
- 95% CIs from a cluster bootstrap over film groups, 10,000 resamples. Paired Δ on the same test videos counts only if its CI excludes 0.
- Paired Wilcoxon with Holm correction; significant only if both per video and per film cluster.
- Every run saves config, metrics, git commit and seed; random generators seeded per item.

## 12. Headline results

| Claim | Result |
|---|---|
| Retrieval vs all baselines (cleaned title split, 894 videos) | track R@10 **0.301** [0.269, 0.338] vs 0.075–0.113 for every other method (chance 0.069); first in all 5 folds |
| Pairwise accuracy | **0.781** vs ≤ 0.601 |
| Trained heads vs ImageBind zero-shot (H1) | +0.126 [+0.092, +0.161] single heads; final ensemble 0.295 vs 0.110 |
| Viterbi vs greedy (H5) | switch F1 **0.730 vs 0.602**, Δ +0.128 [+0.109, +0.148] |
| Change predictor vs previous fixed penalties | in-film F1 0.584 vs 0.549, Δ **+0.035** [+0.005, +0.071]; vs re-tuned penalties +0.018 [−0.003, +0.040] (cleaned: +0.023 [+0.004, +0.043]) |
| Long sequences (5–15 min) | R4 0.588 (same as 1–3 min); on one long film fixed penalties best (0.639 vs 0.618, n.s.) |
| Content fit | ImageBind 0.224 for the top pick vs 0.223 for the original music |
| Robustness | −8% to −17% R@10 for one degradation, −29% for 180p + 32 kbps |
| Not shown | H2 learned dynamics sync (chance), H3 branch fusion, H4 unlabeled pretraining |

## 13. Tried and not adopted

All null or worse on the honest split: learned dynamics and mood fusion; unlabeled pretraining; V-JEPA 2 features; learned MERT layer weights; same-film hard negatives; temporal-token late interaction; deeper MLPs; attribute auxiliary loss; one-to-many / many-to-many targets; cross-attention around the heads; whole-video context (helps retrieval, hurts stitching); face-expression and person-crop branches; same-music splices and audio-teacher targets for the change predictor; ~10 crossfade shapes. Hand-cleaning the dataset gave no retrieval gain (Δ −0.007 [−0.024, +0.010]).

## 14. Interpretation and caveats

- The picture identifies which kind of song fits, not which moment lines up with which shot: beat-level sync and offset choice are at chance, and render-time beat snap aligns random music just as well.
- The limit is generalization across films (train R@10 ≈ 0.87, test ≈ 0.29; +0.054 R@10 per doubling of films, no plateau).
- The gap over published systems comes mostly from frozen foundation features plus ensembling: on equal features, CFD-style models tie the heads.
- The change predictor's gain rests mostly on segment length, which marks clip joins in concatenated data; with length held fixed its AUROC is ~0.55. The Transformer adds nothing measurable over the pair-only model. Oracle headroom is ~+0.24 F1.
- All stitching tests use joined dataset clips; no continuous film with annotated music changes exists on disk.
- ImageBind/LanguageBind judges are weak; no human listening study yet.
- Original-split (982 clips) and cleaned-split (894 clips) numbers come from different test sets; compare within a table.
