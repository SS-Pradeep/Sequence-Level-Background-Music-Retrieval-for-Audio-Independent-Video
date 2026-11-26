# Video-to-BGM Retrieval: Methodology and Results

**What this file is.** A standalone summary of a research project: the task, the data, the full method, how it is evaluated, every main result, the comparison against all other methods, and what was tried and dropped. It is meant as context for discussing the work (paper writing, viva preparation, reviewer questions) without access to the code. All numbers come from the project's result files; nothing here is estimated. Codes like "D75" are entries in the project's decision log (docs/PROGRESS.md); quote them when asking about a specific choice.

**Status:** 2026-11-12, last decision D123. The system is built and evaluated, all baselines have been scored under one shared evaluator, and the learned music-change predictor (a 2-layer Transformer that feeds the Viterbi pass) is now part of the final system (D121–D123).

**How to read the numbers:**
- "R@10" means Recall@10.
- "±" is the standard deviation over runs (5 folds × 3 seeds unless stated).
- "[a, b]" is a 95% confidence interval from a cluster bootstrap over **film groups** (10,000 resamples), so clips of the same film are never treated as independent.
- "Δ" is a paired difference on the same test videos. It counts as significant only if its CI excludes 0.

---

## 1. The project in brief

- **Task:** given a **silent video** (e.g. an Indian film scene), produce a continuous background-music (BGM) soundtrack from a fixed music library.
- **Pipeline:**
  1. Cut the video into segments (2–20 s).
  2. Retrieve candidate music windows for each segment (coarse: rank whole files; fine: slide 0.5 s windows over shortlisted files).
  3. Predict, at every boundary between two segments, the probability that the music should change (a small Transformer over the segment sequence).
  4. Choose one window per segment with a **Viterbi** pass that trades segment fit against musical continuity, using those change probabilities as the switch costs.
  5. Render (crossfades, loudness normalization, beat-snapped piece starts).
- **Claimed contribution:** the **sequence-level choice of music** (Viterbi stitching across segments, i.e. when to keep a track and when to switch), not per-clip retrieval alone. Per-clip video-to-music retrieval is an existing task; coherent soundtrack assembly is the new part.
- **Final system:**
  - Scoring (D75): a **10-model ensemble** of small MLP "semantic heads". The video side is SigLIP so400m (1 frame/s); the music side is CLAP (`larger_clap_music_and_speech`) concatenated with MERT-v1-330M. All encoders are frozen, and only the heads train. Retrieval is coarse-to-fine with semantic scores only.
  - Sequence choice (D121): a **2-layer Transformer change predictor** sets the switch cost at each segment boundary; Viterbi then picks the windows.
- **Three branches compare video with music:** semantic (what is on screen vs what the music sounds like), dynamics (motion and cuts over time vs beats and energy over time) and attributes (mood). Only the semantic branch is in the final system.
- **Built but switched off** because none helped significantly: the learned dynamics branch, the mood/attribute branch, whole-video context, a person-crop branch.

## 2. Data

| | Original dataset | Cleaned dataset (D115) |
|---|---|---|
| Pairs (video + its own BGM) | 1,248 | 1,011 |
| Music duration | 6.7 h | 4.65 h |
| Track groups (distinct songs) | 1,150 | 912 |
| Films | 228 real films + 15 compilations (445 film keys, 219 film groups) | 420 film keys, 211 film groups |
| Test clips (5 folds) | 982 | 894 |
| Gallery per test fold | ~196 music files | ~179 music files |
| Chance track R@10 | 0.058 | 0.069 |

- **Sources:** SunNXT, YouTube and compilations. Clip length: 9.84–171 s, median 15.0 s (original); 1.07–171 s, median 13.0 s (cleaned). The video's own soundtrack is its BGM, so music and picture are time-aligned. The 1,248 pairs are the 1,255 delivered files minus 7 duplicates.
- **"Konjum salangai" (266 clips, 117 after cleaning):** this subset has a DC offset, 100 Hz mains hum and audio about 27 dB too quiet. The hum made different clips look like duplicates to fingerprinting. It is used for **training only**, never for validation or test.
- **Media prep:**
  - Video: silent, constant 25 fps, short side at most 480 px. Every video feature is read from this silent copy, so no video feature ever hears the original soundtrack.
  - Music: mono 48 kHz, resampled in memory to each encoder's rate (CLAP 48 kHz, MERT 24 kHz, PANNs 32 kHz, librosa 22,050 Hz), asserted at every encoder boundary.
  - **Timing rule:** music sample 0 is the instant of the first video frame. A/V start offsets from ffprobe are corrected.
- **Grouping songs:** a song never appears in both train and test. Songs are linked by Chromaprint fingerprints (after a 200 Hz high-pass, plus a log-mel correlation check) and by CLAP cosine ≥ 0.97; union-find over these links gives the track groups.
- **Cleaned dataset:** the author hand-trimmed every pair with a pair-trimmer tool, dropping or trimming non-BGM audio. The folds were carried over unchanged.
- **How much is really music (PANNs CNN14 tags per second, after a 150 Hz high-pass; D114, D117):**
  - Original: 69.5% of seconds are clear music (79.4% on test clips, excluding Konjum salangai); speech is rare (0.9% of seconds). Files: 595 clean BGM, 322 mostly music, 331 with little music.
  - Cleaned: 82.6% (83.2% on test clips). Files: 629 clean BGM, 242 mostly BGM, 140 with little BGM. Konjum salangai still carries its DC offset and hum.
- **An extension was tried and dropped:** 2,998 extra Demucs-separated clips (4,246 pairs in total) gave no gain. Recall grows with the number of distinct *films*, and this batch added many clips from only 32 films. The author chose to stay on the 1,248 pairs.

## 3. Splits (why the numbers are honest)

| Split | Grouped by | Problem |
|---|---|---|
| Track split | song (`track_id`) | Leaked by film: 65% of test clips had a same-film clip in training. Neighbouring scenes use different songs but look alike, so the model memorised films (R@10 0.60 on seen films vs 0.28 on unseen) |
| Film split | source film ID | One movie uploaded under several IDs: 84 films in more than one test fold |
| **Title split (used for all headline numbers)** | real movie title (224 hand-checked links) | Films in more than one test fold: 84 → 1 |

- 5-fold stratified group k-fold over film groups, stratified by duration.
- Validation is 10% of the training films, disjoint from training.
- The test fold never touches training, early stopping, calibration, fusion weights, stitching penalties or the change predictor's tuning.
- Contrastive batches hold at most one segment per video, and pairs that share a song are masked out of each other's negatives.

---

## 4. Method

### 4.1 Segmentation

- **Shots:** PySceneDetect `AdaptiveDetector`.
- **Segments:**
  - Merge consecutive shots while their SigLIP image-embedding cosine is ≥ 0.85 and the segment stays ≤ 20 s.
  - Fold shots under 2 s into the more similar neighbour.
  - Split a single shot longer than 20 s into equal parts.
- **Span rule:** the semantic and mood branches always pool at least 5 s, widening the span symmetrically, on both the video and the music side.

### 4.2 Features

All pretrained encoders are frozen (eval mode, no gradients, fp16) and cached once with a config hash.

| Branch | Video side (from the silent video only) | Music side |
|---|---|---|
| Semantic | SigLIP so400m image embeddings, 1 per second | CLAP (`larger_clap_music_and_speech`) and MERT-v1-330M (25 layers averaged), concatenated |
| Dynamics, 25 Hz curves | optical-flow motion, visual onsets, cut impulses, luminance change, camera motion | onset strength, beat impulses, loudness (RMS dB), spectral-centroid change, novelty |
| Attributes | SigLIP zero-shot mood scores | CLAP zero-shot mood scores; calibrated, then compared with Jensen–Shannon divergence |
| Extras (stitching only) | — | tempo, key/mode |

The spec named `laion/larger_clap_music`, but its public weights are broken: all embeddings come out nearly identical. The music_and_speech variant is used instead.

### 4.3 Models and training

- **Semantic heads:**
  - One MLP per side: LayerNorm → Linear(512) → GELU → Dropout → Linear(256), L2-normalized.
  - Input is the mean of the 1 Hz embeddings over a span of at least 5 s.
  - Training: symmetric InfoNCE with a learnable temperature, AdamW, 25% frame dropout and ±1 s music-span jitter; early stopping on validation segment R@10.
- **Ensemble:** 10 heads trained with different seeds, cosines averaged. This was the first change that reliably helped (+0.059 track R@10 over single heads, CI [+0.041, +0.079]): averaging seeds cancels seed-specific overfitting. Each of the 3 "seeds" of the final system is a disjoint set of 10 heads.
- **Dynamics encoders (built, not used in the final system):**
  - Dilated 1-D residual CNNs (about 87K parameters, 2.4 s receptive field).
  - Contrastive loss; negatives are other music in the batch plus time-shifted copies of the pair's own music.
  - Shifts never land on a whole number of beats, because alignment is the label. Dynamics inputs are never time-shifted as augmentation.
- **Self-supervised pretraining** on about 1,000 unlabeled music videos was tried. It gave no gain.

### 4.4 Retrieval

1. **Coarse:** score every library file against the whole video and against each segment, and shortlist the union of the top 10 from each.
2. **Fine:** for each segment, slide windows every 0.5 s over each shortlisted file and score them with the heads. Beat-aligned window starts were planned, but beat tracking is unreliable on this music.
3. **Fusion:** each branch's scores are z-scored per segment, then combined as a weighted sum. **In the final system the weights are semantic 1, dynamics 0, mood 0**: dynamics scored at chance on its own, and mood weights tuned on validation hurt test results.

The z-scored fine score of each candidate window is its **emission score**; Viterbi uses −score as the unary cost.

### 4.5 Viterbi stitching

- **Candidates per segment:** the top 20 windows, plus the contiguous continuation (same file, next stretch of audio) of each candidate from the previous segment.
- **Costs** between segment i and i + 1:

  | Move | Fixed-penalty decoder (D75) | With the change predictor (D121) |
  |---|---|---|
  | continue contiguously in the same file | 0 | −α · log(1 − p_i) |
  | switch to another file | λ_switch + μ · (key distance on the circle of fifths + tempo mismatch) | −α · log(p_i) + μ · (key + tempo distance) |
  | jump within the same file | λ_jump | −α · log(p_i) + λ_jump |

  p_i is the predicted probability that the music changes at boundary i (section 4.6). With a constant p the two decoders are identical (unit-tested), so the predictor only replaces one fixed switch penalty with a per-boundary one.
- **Tuning:**
  - Fixed penalties (λ_switch, λ_jump, μ): per fold on validation "concatenations" (3–4 clips with different songs joined end to end), objective part track accuracy plus switch-boundary F1.
  - With the predictor: α ∈ {0.125 … 8} and λ_jump ∈ {−2 … 2} are grid-tuned per fold on validation sequences only; μ stays at the fold's D75 value.

### 4.6 Music-change predictor (Transformer before Viterbi; D78, D121)

**Purpose:** decide *where* the soundtrack should change. The emission scores say which music fits each segment; the predictor says, at each boundary, how likely a new cue is, so Viterbi switches where the picture suggests a change and holds the music elsewhere.

- **Input token per segment** (1,155-d):
  - mean SigLIP embedding of the segment's own 1 Hz frames, L2-normalized (the 5 s span rule is not applied, so a token never reaches into the neighbouring segment);
  - log segment duration, mean and std of the raw optical-flow motion curve, z-scored with training-fold statistics. Frame 0 of a video is left out of the motion statistics, because it has no flow and would mark every clip start.
- **Architecture (R4, `ctx`; 1.68 M parameters):**
  1. token MLP: Linear(1155 → 256) → GELU → Dropout → Linear(256);
  2. a learned [CTX] token prepended, then a **2-layer bidirectional Transformer encoder** (d 256, 4 heads, feed-forward 512, dropout 0.4, pre-norm, GELU), no positional encoding (the boundary head supplies the order, and any sequence length works);
  3. boundary head on each neighbouring pair: MLP([h_i, h_{i+1}, |h_i − h_{i+1}|, h_i ⊙ h_{i+1}]) → logit of p_i.
- **Training:**
  - Sequences of 60–180 s built from the fold's **training clips only**, redrawn every epoch: blocks of 2–8 clips of one film, chained across films. A boundary is labelled 1 when the song (track group) changes across it, else 0.
  - BCE with positive weight = negatives / positives; AdamW, lr 3e-4, weight decay 0.3, batch 32; early stopping on validation AP (patience 30 "epochs" of 8 steps).
  - 5 folds × 3 seeds. At render time the 3 seeds' probabilities are averaged.
- **Variant choice (on validation only):** mean val AP, title split: ctx 0.714, ctx_splice 0.713, ctx_cross 0.709 → R4 (`ctx`). The pair-only model (no Transformer) has higher val AP (0.777) and the same test F1; the Transformer was kept by the author's choice.
- **Comparison systems in the evaluation:**
  - R0: fixed penalties (stored D75 values, and re-tuned on the same val sequences as the learned systems).
  - R1: scene-cut rule (PySceneDetect ContentDetector on the joined sequence, p high at detected cuts).
  - R2: cosine rule (logistic on the SigLIP cosine between neighbouring segments).
  - R3 (`pair`): the same model without [CTX] and Transformer.
  - R4s (`ctx_splice`): R4 trained with same-music splices (a gap cut out of one clip and the ends joined: looks like a join, label 0).
  - Diagnostics: R4 without the log-duration feature; R5 trained on one-clip-per-film sequences; oracle (true change points).
- **Default:** on (`cue.enabled: true`). `cue.enabled=false` reproduces the fixed-penalty decoder bit-identically.

### 4.7 Rendering

1. Consecutive segments that use contiguous audio from one file form a "piece".
2. **Beat sync:** each piece's start shifts to the beat within ±2 s that puts the most video cuts on beats. This happens at render time only; the models are unchanged.
3. Equal-power crossfades at seams, plus linear gain ramps that meet the loudness of both sides. About 10 other fade shapes were tested and none did better.
4. Each piece is normalized to −16 LUFS, then a lookahead limiter caps peaks at −1 dBFS.
5. The audio is muxed onto the original video with a measured start-offset correction, and sync is verified to 0.0 ms by cross-correlation.

Opt-in extras exist for user-facing demo videos. They are not part of the benchmarked system:

- **Pace match:** fast-cut video gets energetic music.
- **Scene brief:** a local vision-language model (Qwen2.5-VL-3B) writes an editable scene/emotion/music description per segment, matched to the music with CLAP text embeddings.
- **Replay/reprise limits:** avoid replaying the same passage.
- **Length fit:** avoid looping files that are too short.
- **Silent spans:** parts marked as needing no music get faded out.
- **1080p upscaling.**
- **Emotion demos:** zero-shot emotion scores (SigLIP on frames + CLAP on the true music) pick the best held-out clip per emotion, and long test films are re-rendered once per emotion via the scene brief. These are showcase outputs only.

---

## 5. Evaluation protocol

**Retrieval:**
- **Track R@k:** the query is a whole test video and the gallery is the test fold's music. A hit is any file of the right song. Median rank and MRR are also reported, and chance is computed exactly per query.
- **Segment R@k:** the query is a segment and the target is the exact music window aligned with it.
- **Grounding:** given the right file, find the right offset. Ties at the top score share the hit, which avoids a bias toward offset 0.

**Pipeline:** track@1 (right song) and window@1 (right song, within 0.5 s of the right offset).

**Stitching:**
- Benchmark: 50 test concatenations per fold (3–4 clips of different songs).
- Switch-boundary F1: a switch within ±1 s of a true join counts as a hit.
- Part track accuracy, weighted by duration; seam loudness jump in LU.

**Music-change predictor:**
- Frozen test sequences built from each fold's test clips, each clip used once. "Blocked": each film's clips stay together and films are chained (1–3 min: 112 sequences, 279 min; 5–15 min: 27 sequences, 279 min). "Single-film": one film's clips per sequence.
- Headline metric: **in-film boundary F1 (±1 s)**: predicted music switches against the true song changes *between clips of the same film* (any switch inside a clip counts as a false positive). Changes between different films are reported separately, since they are easy.
- Predictor AP / AUROC on the same boundaries, and probes with segment length held fixed (8 × 8 length bins).

**Additional suite:**
- **Beat alignment** (named "LB" in the D111 suite): share of cuts with a beat within ±70 ms, minus chance. Not the LanguageBind score.
- **LanguageBind score (LB, tw-LB; D119):** cosine between the video and the rendered soundtrack in LanguageBind's video–audio space. LB scores whole clip vs whole soundtrack; tw-LB averages segment-level cosines weighted by segment length. LanguageBind was re-implemented module for module; the official checkpoints load with strict key matching.
- **ImageBind similarity (IB):** video–music similarity measured by a model the system never uses.
- **KL:** divergence of audio tags against the original music.
- **TBIoU and DTW:** how well the transitions match the true ones.
- **Diversity:** entropy, gallery coverage, and the top-5 share of picks.

**Statistics:**
- 5 folds × 3 seeds.
- 95% confidence intervals from a cluster bootstrap over film groups, 10,000 resamples.
- Paired Wilcoxon tests with Holm correction. A result counts only if it is significant both per video and per film cluster.

---

## 6. HEADLINE: all methods under one evaluator (D120, cleaned title-split test)

Every method's test-set score matrix was rebuilt from that method's own saved models or embeddings, then scored by one shared evaluator:
- Same 894 test videos, same galleries (~179 music files per fold, one file per test pair) and same hit rules.
- Each method's rebuilt scores reproduce its own earlier reported numbers.
- VM-NET was retrained with its own seeded code and reproduced 100% of its saved ranks.

### 6.1 How each other method was run on this data

| Method | Type | What was used |
|---|---|---|
| **ImageBind zero-shot** | untrained | `imagebind_huge`; cosine of whole-clip video and audio embeddings |
| **LanguageBind zero-shot** | untrained | LanguageBind_Video_FT / Audio_FT, re-implemented module for module; official checkpoints |
| **MMTrail captions → CLAP** | untrained | MMTrail pipeline's CoCa frame captions → CLAP text embeddings (length-weighted mean) vs CLAP audio embedding of each music file. MMTrail defines no retrieval model; this is the natural baseline built on its output |
| **VM-NET (CBVMR)** | trained | Original architecture and losses. Inception-v3 1 fps frames → PCA-whitened 1024-d (YouTube-8M recipe); music = 1140-d librosa features. 5 folds × 3 seeds; best-val checkpoint |
| **EmoMV (single-task, multi-task)** | trained | Original EmoMV retrieval models; SlowFast R50 video + VGGish music. Emotion labels predicted by a model trained on EmoMV-A (the EmoMV-B/C recipe). 1 run per fold |
| **DecCM (+ its teacher)** | trained | Original code; ResNet video, VGGish + OpenSmile music. "Uploader" = film; "genre" = k-means clusters. Grid configuration chosen by mean validation score (t1e-4_e500). Scored on the **full test gallery**: its own protocol ranks 1 positive against 100 sampled negatives, where it got R@10 0.108 vs chance 0.099 |

### 6.2 Video → music retrieval (track rule: any file of the right song counts)

| Method | Trained | R@1 | R@5 | R@10 [95% CI] | Median rank ↓ | MRR | Pairwise acc. [95% CI] |
|---|---|---|---|---|---|---|---|
| **Final system (10-head ensemble)** | yes | **0.052** | **0.190** | **0.301 [0.269, 0.338]** | **24.1** | **0.135** | **0.781 [0.763, 0.801]** |
| LanguageBind zero-shot | no | 0.014 | 0.066 | 0.113 [0.091, 0.137] | 62.4 | 0.058 | 0.592 [0.567, 0.615] |
| ImageBind zero-shot | no | 0.012 | 0.054 | 0.110 [0.082, 0.140] | 62.5 | 0.053 | 0.601 [0.578, 0.625] |
| EmoMV, single-task | yes | 0.007 | 0.053 | 0.103 [0.083, 0.125] | 65.9 | 0.048 | 0.577 [0.550, 0.604] |
| EmoMV, multi-task | yes | 0.014 | 0.056 | 0.093 [0.073, 0.114] | 67.2 | 0.053 | 0.574 [0.550, 0.598] |
| VM-NET (CBVMR) | yes | 0.011 | 0.047 | 0.092 [0.078, 0.108] | 71.7 | 0.047 | 0.558 [0.541, 0.576] |
| MMTrail captions → CLAP | no | 0.009 | 0.049 | 0.092 [0.068, 0.116] | 71.5 | 0.045 | 0.547 [0.522, 0.572] |
| DecCM teacher (no deconfounding) | yes | 0.010 | 0.050 | 0.088 [0.069, 0.109] | 68.0 | 0.046 | 0.564 [0.543, 0.586] |
| DecCM | yes | 0.010 | 0.040 | 0.075 [0.057, 0.093] | 78.6 | 0.042 | 0.527 [0.506, 0.547] |
| chance | – | 0.007 | 0.035 | 0.069 | 90.0 | – | 0.500 |

Pairwise accuracy is the probability that a video's own music scores above a music file of another song (all mismatches, ties ½; the MEMA protocol).

### 6.3 Both directions and the exact-pair rule

Exact pair: only the video's own music file counts (or, for music → video, the music's own video).

| Method | v2m track R@10 | m2v track R@10 | v2m exact R@10 | m2v exact R@10 | m2v median rank |
|---|---|---|---|---|---|
| **Final system** | **0.302** | **0.313** | **0.281** | **0.284** | **23.2** |
| LanguageBind zero-shot | 0.113 | 0.125 | 0.096 | 0.116 | 61.3 |
| ImageBind zero-shot | 0.110 | 0.184 | 0.101 | 0.170 | 46.5 |
| EmoMV, single-task | 0.103 | 0.117 | 0.093 | 0.106 | 63.4 |
| EmoMV, multi-task | 0.094 | 0.112 | 0.084 | 0.101 | 67.1 |
| VM-NET | 0.092 | 0.087 | 0.082 | 0.080 | 70.1 |
| MMTrail captions → CLAP | 0.092 | 0.108 | 0.082 | 0.096 | 70.6 |
| DecCM teacher | 0.088 | 0.096 | 0.079 | 0.087 | 67.2 |
| DecCM | 0.075 | 0.060 | 0.062 | 0.056 | 84.6 |
| chance | 0.069 | 0.069 | 0.056 | 0.056 | 90.0 |

### 6.4 Paired differences: final system − each method (same 894 videos)

| vs method | Δ track R@10 | Δ pairwise acc. | Δ ImageBind score | Δ LanguageBind score |
|---|---|---|---|---|
| LanguageBind zero-shot | +0.188 [+0.150, +0.227] | +0.189 [+0.159, +0.221] | −0.022 (judge favours itself) | −0.101 (judge favours itself) |
| ImageBind zero-shot | +0.192 [+0.154, +0.232] | +0.180 [+0.153, +0.207] | −0.178 (judge favours itself) | −0.019 |
| EmoMV, single-task | +0.198 [+0.163, +0.238] | +0.204 [+0.177, +0.232] | +0.008 [−0.003, +0.018] n.s. | −0.001 n.s. |
| EmoMV, multi-task | +0.208 [+0.172, +0.248] | +0.207 [+0.178, +0.237] | +0.021 [+0.010, +0.032] | +0.005 n.s. |
| VM-NET | +0.209 [+0.174, +0.247] | +0.223 [+0.199, +0.247] | +0.032 [+0.023, +0.041] | +0.009 [+0.004, +0.013] |
| MMTrail captions → CLAP | +0.210 [+0.164, +0.258] | +0.234 [+0.198, +0.272] | +0.095 [+0.076, +0.112] | +0.058 [+0.050, +0.066] |
| DecCM teacher | +0.213 [+0.175, +0.253] | +0.218 [+0.190, +0.246] | +0.039 [+0.026, +0.051] | +0.010 [+0.005, +0.015] |
| DecCM | +0.226 [+0.189, +0.267] | +0.254 [+0.226, +0.285] | +0.049 [+0.030, +0.068] | +0.017 [+0.008, +0.025] |

### 6.5 How well the top-1 pick fits the video (one shared judge)

The judge is the cosine between the query video and the music file each method ranks first, using whole-clip ImageBind and LanguageBind vectors. ImageBind and LanguageBind zero-shot pick the file their own judge scores highest, so in their own column they are an upper bound, not a competitor.

| Music used | ImageBind score | LanguageBind score |
|---|---|---|
| original music | 0.223 | 0.151 |
| **Final system top-1** | **0.224** | 0.145 |
| EmoMV single-task top-1 | 0.216 | 0.146 |
| EmoMV multi-task top-1 | 0.203 | 0.140 |
| VM-NET top-1 | 0.192 | 0.136 |
| DecCM teacher top-1 | 0.185 | 0.136 |
| DecCM top-1 | 0.175 | 0.129 |
| MMTrail top-1 | 0.129 | 0.087 |
| random gallery file | 0.188 | 0.136 |
| (ImageBind zero-shot top-1) | (0.402, own judge) | 0.164 |
| (LanguageBind zero-shot top-1) | 0.245 | (0.246, own judge) |

Each method minus a random file:
- Final system: ImageBind +0.036 [+0.027, +0.044], LanguageBind +0.009 [+0.005, +0.013].
- EmoMV single-task: +0.028 and +0.010.
- VM-NET: +0.004 and +0.000 (n.s., i.e. random level).
- DecCM: −0.013 and −0.008.
- MMTrail: −0.059 and −0.049 (below random).

### 6.6 Per fold, video → music track R@10

| Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| **Final system** | 0.302 | 0.345 | 0.276 | 0.297 | 0.289 |
| LanguageBind zero-shot | 0.099 | 0.109 | 0.151 | 0.082 | 0.124 |
| ImageBind zero-shot | 0.117 | 0.109 | 0.106 | 0.115 | 0.102 |
| EmoMV, single-task | 0.117 | 0.131 | 0.101 | 0.109 | 0.059 |
| EmoMV, multi-task | 0.123 | 0.120 | 0.084 | 0.071 | 0.070 |
| VM-NET | 0.105 | 0.086 | 0.080 | 0.091 | 0.099 |
| MMTrail captions → CLAP | 0.099 | 0.097 | 0.128 | 0.055 | 0.081 |
| DecCM teacher | 0.088 | 0.097 | 0.078 | 0.066 | 0.113 |
| DecCM | 0.076 | 0.080 | 0.101 | 0.066 | 0.054 |
| chance | 0.068 | 0.076 | 0.071 | 0.063 | 0.066 |

### 6.7 What the comparison shows

1. **Your system is about 3× better than every other method on retrieval.** Its track R@10 is 0.30, while all others sit at 0.075–0.113 (chance 0.069). The gain is +0.19 to +0.23 R@10, with every CI clear of 0, and the system is first in all 5 folds. The lead holds in both directions and under the exact-pair rule.
2. **No published trained baseline beats ImageBind or LanguageBind used out of the box.** VM-NET, EmoMV and DecCM sit near or below ImageBind's level. With ~780 training pairs, their end-to-end models overfit (VM-NET's training loss is ~0 within 20–200 steps).
3. **DecCM is barely above chance** on the full gallery. Its own 1 + 100 protocol inflated its apparent result.
4. **The ImageBind/LanguageBind "fit" judges are weak.** They barely separate methods:
   - Your picks score the same as the original music on ImageBind (0.224 vs 0.223).
   - EmoMV single-task scores the same as yours despite a third of the recall.
   - Treat these scores as supporting evidence only. Retrieval and pairwise accuracy are the main evidence.
5. **Important caveat: why the gap is so large.** Much of the advantage comes from strong frozen encoders (SigLIP, CLAP, MERT) plus ensembling, not from the head architecture itself. When three published architectures (CFD, UT-CMVMR, MVPt) were re-implemented **on the same frozen features** (section 7), they matched the heads. So the defensible claim is that "frozen foundation features + light heads + ensembling" beats these published systems as published, while the architecture alone is not better than CFD-style models.

---

## 7. Published architectures re-implemented on the same features (D43, original title split, 982 test videos)

CFD-, UT-CMVMR- and MVPt-style models were trained on the same frozen SigLIP/CLAP+MERT features and folds.

| System | Runs | Track R@1 | Track R@10 | Median rank | Segment R@10 | Δ track R@10, heads − this |
|---|---|---|---|---|---|---|
| Heads (ours), single | 15 | 0.036 | 0.235 | 30.7 | 0.092 | — |
| CFD-style, single | 15 | 0.038 | 0.242 | 31.0 | 0.092 | −0.007 [−0.024, +0.013] n.s. |
| UT-CMVMR-style, single | 15 | 0.027 | 0.215 | 35.9 | 0.081 | +0.021 [−0.001, +0.044] n.s. |
| MVPt-style, single | 15 | 0.031 | 0.215 | 35.0 | 0.089 | +0.021 [+0.005, +0.038] |
| Heads, ensemble of 3 | 5 | 0.042 | 0.271 | 24.9 | 0.108 | — |
| CFD-style, ensemble of 3 | 5 | 0.041 | 0.280 | 25.9 | 0.106 | −0.008 n.s. |
| UT-CMVMR-style, ensemble of 3 | 5 | 0.042 | 0.262 | 30.8 | 0.102 | +0.010 n.s. |
| MVPt-style, ensemble of 3 | 5 | 0.042 | 0.260 | 27.8 | 0.110 | +0.011 n.s. |

**Reading:** on equal features, the heads tie the re-implemented baselines. Retrieval-architecture novelty is therefore **not** the claim. The claims are the sequence-level stitching plus an honest, leak-free benchmark.

---

## 8. Final system on the original title split (982 test videos; results/titlesplit/final.md)

### 8.1 Hypotheses

| # | Hypothesis | Measure | Δ [95% CI] | Verdict |
|---|---|---|---|---|
| H1 | Trained heads beat ImageBind zero-shot | track R@10, single heads | +0.126 [+0.092, +0.161] | **holds** |
| H2 | Learned dynamics beat hand-crafted features on sync | grounding sync excess | −0.001 [−0.023, +0.021] | not shown |
| H3 | Fusing branches beats the semantic branch alone | window@1, val-tuned fusion − semantic | −0.002 [−0.004, +0.000] | not shown |
| H4 | Unlabeled pretraining helps | track R@10 | +0.007 [−0.014, +0.027] | not shown |
| H5 | Viterbi beats greedy | switch-boundary F1 (concatenations only) | **+0.128 [+0.109, +0.148]** | **holds on concatenations; ties on blocked in-film sequences (D150)** |

The listening study (human preference) has not been run.

### 8.2 Retrieval

| System | Runs | Track R@1 | Track R@5 | Track R@10 | Median rank | MRR | Segment R@10 |
|---|---|---|---|---|---|---|---|
| random (exact chance) | — | 0.006 | 0.029 | 0.058 | ~92 | 0.033 | 0.020 |
| ImageBind zero-shot | 5 | 0.015 | 0.054 | 0.110 | 65.5 | 0.054 | 0.042 |
| text bridge, zero-shot | 5 | 0.014 | 0.059 | 0.105 | 67.8 | 0.052 | 0.033 |
| mood attributes, zero-shot | 5 | 0.008 | 0.049 | 0.095 | 72.9 | 0.045 | 0.032 |
| heads SigLIP → CLAP+MERT | 15 | 0.036 | 0.141 | 0.235 | 30.7 | 0.107 | 0.092 |
| heads SigLIP → CLAP | 15 | 0.028 | 0.132 | 0.238 | 31.4 | 0.099 | 0.089 |
| heads SigLIP → MERT | 15 | 0.024 | 0.101 | 0.189 | 40.9 | 0.082 | 0.069 |
| heads CLIP → CLAP | 15 | 0.031 | 0.118 | 0.202 | 36.7 | 0.092 | 0.079 |
| **final: 10-head ensemble** | 15 | **0.049** | **0.182** | **0.295** [0.263, 0.332] | **24.8** | **0.130** | **0.117** |

- Final − ImageBind: +0.185 [+0.146, +0.225].
- Ensemble − single heads: +0.059 [+0.041, +0.079].
- nDCG@10: final 0.147 vs ImageBind 0.051 (Δ +0.097 [0.075, 0.118]).

### 8.3 Stitching (H5): Viterbi vs greedy on test concatenations

50 concatenations per fold, each 3–4 clips of different songs joined end to end; fixed-penalty decoder.

| Metric | Viterbi | Greedy | Δ [95% CI] |
|---|---|---|---|
| switch-boundary F1 (±1 s) | **0.730** | 0.602 | **+0.128 [+0.109, +0.148]** |
| TBIoU | 0.848 | 0.749 | +0.099 [+0.085, +0.114] |
| part track accuracy | 0.073 | 0.051 | +0.022 [+0.014, +0.030] |
| switches / min (truth 2.6) | 3.40 | 6.19 | −2.79 |
| jumps / min | 1.60 | 1.84 | −0.25 |
| seam loudness jump (LU) | 1.49 | 1.37 | +0.13 (slightly worse) |

Other baselines for switch F1: a random window per segment scores 0.510, and random boundaries (Viterbi's switch count) score 0.395. On the cleaned dataset, Viterbi − greedy switch F1 is +0.089 [+0.069, +0.109]. A paired Wilcoxon test with Holm correction: switch F1 197 wins / 46 losses, p = 3e-24.

### 8.4 Evaluation suite (D111): alignment, content fit, diversity

| Group | Metric | Final | Reference |
|---|---|---|---|
| Content fit | ImageBind segment–window similarity | 0.211 | original 0.210; random 0.171 (Δ +0.040 [0.033, 0.047]) |
| Content fit | KL of PANNs tags vs original ↓ | 1.257 | random 1.498 (Δ −0.242 [−0.289, −0.193]) |
| Beat alignment | excess cut-on-beat, all cuts | +0.000 | original +0.010; random −0.009 (all ≈ chance) |
| Beat alignment | within-piece, with render beat sync | +0.105 | random music + same sync +0.115 (sync explains all of it) |
| Diversity | segment picks: entropy / coverage / top-5 share | 0.803 / 0.544 / 0.209 | original 0.961 / 1.000 / 0.090 |
| Diversity | track top-1: entropy / coverage / top-5 share | 0.794 / 0.441 / 0.201 | ImageBind 0.444 / 0.143 / 0.735 (hub-dominated) |

### 8.5 LanguageBind score of the rendered soundtrack (D119)

| Soundtrack | LB (whole) | tw-LB (segments) |
|---|---|---|
| original music | 0.140 | 0.151 |
| **final (Viterbi)** | 0.139 | 0.148 |
| greedy | 0.142 | 0.148 |
| random | 0.134 | 0.136 |

- Final − random: LB +0.006 [0.002, 0.009], tw-LB +0.011 [0.008, 0.014].
- The final soundtrack is level with the original music on LB.
- The scale is tiny (the original beats random by only +0.006), and LanguageBind's own zero-shot retrieval is R@10 0.110, the same as ImageBind, so this judge is weak.

### 8.6 Temporal grounding (H2): given the right file, find the offset

| System | Rule | Accuracy | Chance | Excess |
|---|---|---|---|---|
| learned dynamics | exact (±0.1 s) | 0.149 | 0.120 | +0.029 |
| learned dynamics | sync (shift still on the beat grid) | 0.439 | 0.442 | −0.003 (chance) |
| hand-crafted | sync | 0.440 | 0.442 | −0.001 (chance) |

Cuts in this data do not follow beats. Even the original music's alignment is at chance.

---

## 9. External evaluation protocols (D118, MEMA, Zhang et al., Sensors 2026; cleaned test, 894 clips)

### 9.1 Pairwise matching accuracy (chance 0.5)

| System | Track | Segment |
|---|---|---|
| **Final ensemble** | **0.782** | **0.753** |
| ImageBind zero-shot | 0.602 | 0.589 |

Ensemble − ImageBind, track: +0.180 [+0.153, +0.207].

### 9.2 Retrieval in MEMA's Table-4 format

| System | Track R@1 / R@5 / R@10 | Track median rank | Segment R@10 | Segment median rank |
|---|---|---|---|---|
| **Final** | **0.052 / 0.190 / 0.302** | **24.1** | **0.126** | **72.5** |
| ImageBind | 0.012 / 0.054 / 0.110 | 62.5 | 0.048 | 159.9 |

### 9.3 Robustness: same models on degraded test media (features re-extracted)

| Condition | Final track R@10 | Drop |
|---|---|---|
| clean (360p, lossless) | 0.302 | — |
| video 270p | 0.277 | −8.2% |
| video 180p | 0.262 | −13.3% |
| MP3 128 kbps | 0.282 | −6.4% |
| MP3 64 kbps | 0.270 | −10.5% |
| MP3 32 kbps | 0.249 | −17.4% |
| mild (270p + 64k) | 0.243 | −19.6% |
| severe (180p + 32k) | 0.214 | −29.1% (still ~2× ImageBind on clean media) |

Pairwise accuracy stays ≥ 0.70 in every condition. ImageBind barely moves because it is near chance to begin with.

### 9.4 Speed (RTX 4080 SUPER)

- Ensemble track query against ~171 files: 0.79 ms warm, 5.32 ms cold.
- Feature extraction: SigLIP 0.28 s per video clip; CLAP + MERT 0.22 s per music file.
- Change predictor: negligible; a 5.8-minute video (54 segments) renders end to end in 93 s with it on.

### 9.5 Limits

MEMA's NEC/TI/TIO aesthetic scores and its correlation metrics were not computed: its model is not public and there are no aesthetic ratings. The numbers are not comparable with MEMA's SymMV results, because the data, galleries and hit rules differ.

---

## 10. Music-change predictor results (D121–D123)

Metric: in-film boundary F1 (±1 s) on blocked test sequences; 5 folds × 3 seeds; the emission (which music fits each segment) is identical for every system, so only the switch decisions differ. Full reports: results/titlesplit/cue_switch.md, results/cleaned/cue_switch.md, results/titlesplit/cue_switch_long.md.

### 10.1 Main results, 1–3 min test sequences

| System | Title split, 1,248 pairs (112 seq., 279 min) | Cleaned, 1,011 pairs (92 seq., 230 min) |
|---|---|---|
| greedy (no Viterbi) | 0.518 | 0.538 |
| R0 fixed penalties, stored (D75) | 0.549 | 0.575 |
| R0 fixed penalties, re-tuned | 0.567 [0.534, 0.596] | 0.581 [0.547, 0.611] |
| R1 scene-cut rule | 0.569 | 0.593 |
| R2 cosine rule | 0.574 | 0.600 |
| R3 learned, no Transformer | 0.580 | 0.602 |
| **R4 learned + Transformer (final system)** | **0.584 [0.561, 0.607]** | **0.604 [0.572, 0.633]** |
| R4s = R4 + same-music splices | 0.575 | 0.605 |
| R4 without segment length (diagnostic) | 0.549 | 0.570 |
| oracle (true change points) | 0.824 | 0.837 |

| Paired Δ in-film F1 | Title split | Cleaned |
|---|---|---|
| R4 − R0 stored (the previous final system) | **+0.035 [+0.005, +0.071]** | +0.029 [−0.007, +0.069] |
| R4 − R0 re-tuned | +0.018 [−0.003, +0.040] | **+0.023 [+0.004, +0.043]** |
| R4 − R3 (what the Transformer adds) | +0.004 [−0.015, +0.024] | +0.001 [−0.017, +0.020] |
| R4 without segment length − R4 | −0.035 [−0.055, −0.018] | −0.033 [−0.061, −0.010] |
| R4s − R4 | −0.010 [−0.017, −0.003] | +0.001 [−0.010, +0.011] |

- Predictor quality: AP 0.582 / AUROC 0.743 (title), AP 0.589 / AUROC 0.738 (cleaned).
- **Segment-length probe:** with segment length held fixed, R4's AUROC is 0.553 [0.522, 0.591] (title) and 0.549 [0.506, 0.590] (cleaned), near chance (0.5). A logistic on segment length alone reaches the same plain AUROC as R4 (0.745 vs 0.744).
- On the old film split (which leaked 84 films) the gain over re-tuned penalties was +0.035 and significant; on the honest title split it shrinks to +0.018.

### 10.2 Long sequences, 5–15 min (title split, same trained models)

Sequences rebuilt to 5–15 min with penalties re-tuned on 5–15 min validation sequences. These are still joined dataset clips; no continuous film footage exists on disk.

| System | 1–3 min (112 seq.) | 5–15 min, several films (27 seq., median 10.7 min) | one film, 5–15 min (9 seq.) |
|---|---|---|---|
| R0 fixed penalties, stored | 0.549 | 0.542 | **0.639** |
| R0 fixed penalties, re-tuned | 0.567 | 0.567 | 0.609 |
| R3 learned, no Transformer | 0.580 | **0.596** | 0.631 |
| **R4 learned + Transformer** | **0.584** | 0.588 | 0.618 |
| R4 − R0 stored | **+0.035 [+0.005, +0.071]** | **+0.047 [+0.008, +0.087]** | −0.021 [−0.068, +0.041] |
| R4 − R0 re-tuned | +0.018 [−0.003, +0.040] | +0.021 [−0.007, +0.044] | +0.009 [−0.027, +0.057] |
| oracle | 0.824 | 0.830 | 0.833 |

### 10.3 Rendered videos

- 1.mp4 (347 s, 54 segments): 38 music switches with the predictor vs 27 with the fixed penalties, consistent with R4's +1.05 switches/min over the stored penalties in the test sequences.
- Earlier listening renders of all 189 held-out film-split test sequences (D98, splice variant): F1 0.354 vs 0.350 for the fixed penalties; 18% more true changes found but 33% more false switches.

### 10.4 What the change-predictor results mean

1. **The learned change costs help a little.** R4 has the best F1 of the learned systems and is significantly better than the previous final system on the title split (+0.035) and than re-tuned penalties on the cleaned data (+0.023).
2. **The Transformer itself adds nothing measurable** over the same model without it (+0.001 to +0.004).
3. **Most of the signal is segment length.** In concatenated data the music changes at clip joins, and single-segment clips (one 10–20 s segment) mark joins; without the length feature the gain disappears, and with length held fixed the predictor is near chance.
4. **Length of the video does not hurt it:** 0.588 on 5–15 min vs 0.584 on 1–3 min. On a single long film (9 sequences), though, the stored fixed penalties scored highest (0.639 vs 0.618, n.s.).
5. **Large headroom:** the oracle reaches 0.82–0.84, so a predictor that really knew where the music changes would gain about +0.24 F1.

**Use by video type:**

| Video | Setting |
|---|---|
| under ~20 s (one segment) | no boundaries; output identical either way |
| 1–15 min, several scenes or films | predictor on (default) |
| one long continuous film | `cue.enabled=false`; the fixed penalties scored at least as well |

---

## 11. Cleaned dataset vs original (D116)

- Retrained on the cleaned data, the final system scores track R@10 0.302 and segment R@10 0.126 on the cleaned test (chance 0.069). Viterbi still beats greedy on concatenations (switch F1 +0.089), but ties it on the 92 blocked in-film sequences (greedy − HMM −0.007 [−0.022, +0.008], D150).
- **Cleaning gave no measurable retrieval gain.** On the same cleaned test clips, cleaned-trained − original-trained ensemble: −0.007 [−0.024, +0.010].
- On the 121 test clips the author excluded, the cleaned-trained model is **worse** (−0.088 [−0.162, −0.027]). Those clips were not noise to the model.
- The cleaned data is measurably more musical (82.6% vs 69.5% clear-music seconds), yet retrieval did not improve.
- The change predictor beats re-tuned penalties on the cleaned data (section 10.1: +0.023 [+0.004, +0.043]), but this gain depends on duration: without log duration it ties them (−0.010 [−0.035, +0.010]), and the full predictor ties a duration-only prior (+0.007 [−0.015, +0.029]).
- Conclusion: the limit is the number of distinct films, not clip quality.

## 12. Learning curve (film split): more films help

| Training films | R@10 |
|---|---|
| 73 (25%) | 0.174 |
| 145 (50%) | 0.216 |
| 218 (75%) | 0.245 |
| 290 (100%) | 0.288 |

- About +0.054 R@10 per doubling of films, with no plateau.
- Adding 2,998 clips from only 32 extra films gave Δ ≈ 0: more clips of the same films don't help.
- Train R@10 is about 0.87 vs test about 0.29, so the gap is generalization across films.

## 13. Tried and not adopted (all null or worse on the honest split)

- Learned dynamics fusion (H3), mood fusion, val-tuned fusion weights
- Unlabeled pretraining on ~1,000 music videos (H4)
- V-JEPA 2 video features, for either the semantic or the dynamics branch
- Learned MERT layer weights
- Same-film hard negatives
- Temporal-token models with late interaction (a 1-layer Transformer per side: worse, −0.036 track R@10)
- Deeper MLPs
- Attribute auxiliary loss
- One-to-many and many-to-many targets (near-duplicate cues as extra positives)
- Cross-attention before or after the heads (null; code since removed)
- Whole-video context in the score (helps retrieval, hurts stitching)
- Face-expression and person-crop branches (person crops 0.217 R@10 alone, Δ −0.001 when fused)
- Same-music splices and audio-teacher targets for the change predictor (no gain over R4)
- ~10 crossfade shapes (equal-power is as good as any)

**Partly promising:** a classifier for whether the music changes at a shot cut *inside* a clip beats simple rules offline (AP 0.275 vs 0.167). Inside Viterbi the gain is small and not significant (+0.016 to +0.029 boundary F1); a 2-layer Transformer over a clip's cuts did worse than a small MLP there (AP 0.215 vs 0.275, about 430 positive examples).

## 14. What it all means

- **The visual signal identifies which kind of song fits a clip, not which moment of it lines up with which shot.** Fine alignment, beat sync and in-clip music changes are not predictable from the picture in this data.
- **The limit is generalization across films, not model capacity.** Train R@10 is about 0.87 while test is about 0.29. More distinct films help; more clips from the same films don't.
- **The sequence-level stitching result depends on the test protocol:** Viterbi beats greedy on concatenations of different songs but ties it on blocked in-film sequences, where two-stage and segmental Viterbi lead (D150). The learned change costs add no gain beyond a duration prior.

## 15. Safe claims and caveats for the paper

**Safe claims:**
- On a leak-free title split, the final system retrieves the right song about 3× as often as every published baseline and zero-shot model tested: track R@10 0.30 vs ≤ 0.113, chance 0.069, all CIs clear.
- Pairwise accuracy is 0.78 vs ≤ 0.60.
- On concatenations of different songs, Viterbi beats greedy on music-switch placement: F1 +0.128 [+0.109, +0.148] (original data), +0.089 (cleaned). Do not generalise this to blocked in-film sequences, where they tie (D150).
- Learned per-boundary change costs beat the fixed penalties of the previous system: in-film boundary F1 +0.035 [+0.005, +0.071] (title split), and beat re-tuned penalties on the cleaned data (+0.023 [+0.004, +0.043]).
- The chosen music fits the picture as well as the original soundtrack by ImageBind similarity, and sounds closer to the original than random music (KL).
- The system is robust to moderate degradation (−8% to −17% for a single degradation).

**Caveats to state:**
- The gain over published systems comes mainly from frozen foundation features plus ensembling. On equal features, CFD-style models tie the heads.
- The change predictor's gain rests mostly on segment length, which marks clip joins in concatenated data; it is not shown to learn scene-driven music changes. On the title split it is not significant against penalties re-tuned on the same validation data (+0.018 [−0.003, +0.040]), and the Transformer adds nothing over a pair-only model.
- All stitching tests use joined dataset clips; no continuous film with annotated music changes was available.
- The ImageBind and LanguageBind judges are weak and barely separate methods.
- Beat-level sync and fine offset choice are at chance. Render-time beat sync aligns random music just as well.
- No human listening study yet.
- The original-split (982 clips, chance 0.058) and cleaned-split (894 clips, chance 0.069) numbers come from different test sets. Compare within a table, not across.
- All baselines except VM-NET were run with their published settings on this data with one run per fold. VM-NET has 3 seeds per fold, and DecCM's configuration was chosen by validation score.

## 16. Project rules worth knowing

- Raw data is read-only.
- Pretrained encoders stay frozen. Only the small heads, the dynamics encoders and the change predictor train.
- Test data never touches training, early stopping, calibration, fusion weights or stitching penalties.
- Every run saves its config, metrics, git commit and seed; random generators are seeded per item.
- Final numbers are mean ± std over 5 folds × 3 seeds with cluster-bootstrap CIs. Single-fold numbers are labelled as development numbers.
- Stack: Python, PyTorch, Hugging Face transformers, Typer CLI (`bgm ...` commands), OmegaConf configs.

## 17. Glossary

- **Track R@k:** the query is a whole test video, and the gallery is the test fold's music files. A hit means any file of the right song (track group) is in the top k.
- **Exact pair R@k:** only the video's own music file counts.
- **v2m / m2v:** video → music (video query, music gallery) and music → video (music query, video gallery).
- **Segment R@k:** the query is a segment, and the target is its exact aligned music window among all test windows.
- **Median rank / MRR:** the median rank of the first hit, and the mean of 1/rank.
- **Pairwise accuracy:** P(own music scores above another song's music), averaged over all mismatches.
- **Switch F1:** predicted music switches matched to true clip joins within ±1 s.
- **In-film boundary F1:** switch F1 counted only at song changes between clips of the same film (the hard case); any switch inside a clip is a false positive.
- **Change probability p_i:** the predictor's probability that the music changes between segment i and i + 1.
- **Oracle:** the decoder given the true change points (p = 1 at true changes, 0 elsewhere); an upper bound for any change predictor.
- **TBIoU:** each true piece's best temporal IoU with a predicted piece, averaged.
- **IB / LB score:** ImageBind / LanguageBind cosine between video and music. These are independent judges the system never uses.
- **Chance:** the exact expected value of random ranking per query.
