# Video-to-BGM Retrieval: Methodology and Results

A self-contained brief for building a presentation. Every number below was measured in this project. Section 11 suggests a slide outline.

## 1. The problem

**Input:** a silent video. **Output:** the same video with a continuous background-music (BGM) soundtrack assembled from a music library.

The system cuts the video into scenes, retrieves music for each scene, chooses one coherent sequence of music pieces across all scenes, and renders it onto the video.

**Main idea.** Choose the soundtrack as a *sequence*, not one song per clip. Each scene gets a shortlist of candidates from a learned video-to-music model; a Viterbi pass then picks the sequence that balances how well each piece fits its scene against musical continuity.

## 2. Dataset

| item | value |
| --- | --- |
| Video–music pairs | 1,248 Indian film clips, each with its own background music |
| Total video | about 6.7 hours |
| Films | 405 film groups |
| Song groups (identical audio linked) | 1,150 |
| Scenes (segments) | 3,282, median about 5 s, range 2–20 s |
| Music file length | median about 14 s |

**Splits.** 5-fold cross-validation grouped by film, so no film appears in both training and test. Each fold has about 196 test clips (982 in total); one large film (266 clips) is used for training only; 10% of training films are held out for validation.

**Why film-level splits matter.** With the usual clip-level split (grouped by song only), the same film appears in training and test, and recall looks about 1.8× higher (R@10 0.502 against 0.279) because the model recognises films it has seen. All headline numbers use the stricter film-level split.

## 3. Pipeline overview

1. **Prepare.** Convert each video to a silent 25 fps file and each music file to 48 kHz mono. Music sample 0 is aligned to the first video frame.
2. **Segment.** Detect shots (PySceneDetect), then merge neighbouring shots into scenes of 2–20 s when their image embeddings are similar (cosine ≥ 0.85).
3. **Embed.** Frozen pretrained encoders produce one embedding per second of video and of music.
4. **Match.** Two small trained networks map video and music into a shared space. Ten such models are averaged (ensemble).
5. **Retrieve, coarse to fine.** Shortlist music files, then score windows inside them.
6. **Stitch.** A Viterbi pass picks one music window per scene.
7. **Render.** Crossfade, match loudness, snap entries to beats, and mux onto the video.

## 4. Models

### 4.1 Frozen encoders (never trained)
| role | model | output |
| --- | --- | --- |
| Video frames | SigLIP `so400m-patch14-384` | 1,152-d per second (one frame per second) |
| Music | CLAP `larger_clap_music_and_speech` | 512-d per second (10 s window) |
| Music | MERT `v1-330M` | 1,024-d per second (mean of its 25 layers) |

The two music embeddings are concatenated (1,536-d). A scene or a music window is represented by the mean of its per-second embeddings over at least 5 s.

### 4.2 Trained matching heads (the only learned part)
One head for video and one for music, same shape:

LayerNorm → Linear (512) → GELU → Dropout 0.2 → Linear (256) → L2-normalise

- The match score is the cosine between the video output and the music output.
- **Loss:** symmetric InfoNCE with a learnable temperature (starts at 0.07). Each batch holds one scene per video; pairs from the same song are not used as negatives.
- **Training:** AdamW, learning rate 1e-3, cosine schedule, batch size 128, at most 30 clips per film per epoch.
- **Augmentation:** drop 25% of video frames before averaging; shift the music span by up to ±1 s.
- **Early stopping** on validation recall, patience 20. Training runs a median of 33 epochs and keeps the model from about epoch 12. One model trains in about one minute.

### 4.3 Ensemble (the final system)
Ten copies of the heads are trained from different random seeds. At retrieval time their ten cosine scores are averaged. This is the single change that improved recall the most.

## 5. Retrieval and stitching

**Coarse stage.** Score every music file against the whole video and against each scene; keep the top 10 files for each.

**Fine stage.** Slide a window every 0.5 s across each shortlisted file, score each window against the scene, and keep the top 20 windows per scene.

**Viterbi stitching.** Choose one window per scene by minimising a total cost:
- the negative match score of each chosen window;
- zero cost for continuing the same music exactly where it left off;
- a penalty for jumping to another place in the same file;
- a larger penalty for switching to another file, plus a term for key and tempo clashes.

The penalties are tuned on validation videos built by joining 3–4 clips, so the true switch points are known.

**Rendering.** Equal-power crossfades at each seam (about 4 beats, 1–2 s), loudness ramps so both sides meet, normalisation to −16 LUFS, a limiter at −1 dBFS, and each piece shifted so it starts on a beat.

## 6. Evaluation protocol

- **Runs.** 5 folds × 3 seeds per experiment (10 seeds for the main model).
- **Recall@K.** For each test video, rank the music in the test gallery (about 196 files). R@10 is the share of videos whose correct music is in the top 10; R@1 is the share ranked first.
- **Whole-clip and segment level.** Whole-clip retrieval ranks music files; segment retrieval ranks one music window per test scene (about 511 windows).
- **Pipeline accuracy.** After coarse and fine retrieval: is the top window from the right track (track@1), and is it also at the right position (window@1, within 0.5 s).
- **Offset precision.** Right track *and* within ±1, ±3 or ±5 s of the true position.
- **Stitching.** On joined test videos: how well music switches line up with the true scene joins (switch F1, within ±1 s), switches per minute, and the share of the video on the right track.
- **Chance.** Every metric is reported with its exact chance level.
- **Significance.** Per-query paired differences with 95% cluster-bootstrap intervals that resample whole films. A change counts only if the whole interval is above zero.

## 7. Main results (1,248 pairs, film-level split)

### 7.1 Retrieval
| system | whole-clip R@10 | segment R@10 |
| --- | --- | --- |
| Chance | 0.058 | 0.020 |
| ImageBind zero-shot | 0.108 | 0.045 |
| Ours, single model | 0.279 | 0.121 |
| **Ours, 10-model ensemble** | **0.334** | **0.157** |

The ensemble gain over a single model is +0.055 [+0.040, +0.071] at whole-clip level and +0.036 [+0.027, +0.046] at segment level.

### 7.2 Pipeline accuracy per scene (10-model ensemble)
| measure | single model | ensemble | chance |
| --- | --- | --- | --- |
| track@1 | 0.051 | 0.062 | 0.009 |
| window@1 | 0.010 | 0.019 | 0.001 |
| right track and within ±1 s | 0.012 | 0.021 | 0.001 |

Once the right track is found, its position is within ±1 s 34% of the time against 22% by chance.

### 7.3 Stitching: Viterbi against greedy per-scene choice (ensemble)
| measure | Viterbi | greedy |
| --- | --- | --- |
| Switch placement F1 (±1 s) | 0.741 | 0.614 |
| Switches per minute | 3.5 | 6.2 |
| Share of video on the right track | 0.071 | 0.068 |

Viterbi places switches at true scene joins far better (+0.127 [+0.104, +0.150]) with about half as many switches. The share on the right track is not distinguishable between the two.

### 7.4 Comparison with published methods (reimplemented on the same features)
| system | single model R@10 | 3-model ensemble R@10 |
| --- | --- | --- |
| Ours | 0.288 | 0.314 |
| CFD-style | 0.261 | 0.300 |
| UT-CMVMR-style | 0.261 | 0.299 |
| MVPt-style | 0.252 | 0.312 |

As single models, ours leads by about 3 points (significant). When every method is ensembled the same way, the gaps at whole-clip level are no longer significant. The fair claim is: **on par with stronger temporal baselines using a much simpler model.**

(The single-model figure is 0.288 averaged over the first 3 seeds and 0.279 over all 10 seeds; both are the same model.)

## 8. What did and did not help

| change | outcome |
| --- | --- |
| Ensemble of 10 models | **Helped**: whole-clip R@10 0.279 → 0.334 |
| Distilling the ensemble into one model | Helped: single model 0.279 → 0.302 |
| Stronger regularisation (weight decay 0.3, dropout 0.4) | Helped: single model 0.279 → 0.297 |
| CLAP + MERT together instead of either alone | Helped |
| Dynamics branch (motion and cuts vs beats and loudness) | No gain; near chance |
| Mood branch (zero-shot mood words) | No gain |
| Deeper heads (2 and 3 hidden layers) | No gain, then worse |
| Temporal transformer with late interaction | Worse |
| Motion-aware video encoder (V-JEPA 2) | Worse alone; no gain combined |
| One-to-many training on near-duplicate cues (three variants) | No gain |
| Second-stage reranker | Worse |
| Hubness correction (CSLS, inverted softmax) | No gain |
| Self-supervised pretraining on unlabeled music videos | No gain |
| Adding 2,998 more clips from 32 films | No gain |
| Whole-video context in the scene score | Helps single clips; hurts stitching on multi-scene videos, so not used |

**The three-branch design.** The system was designed with three branches: semantic (what is on screen vs what the music is), dynamics (movement over time vs musical energy over time) and attributes (mood). On this data only the semantic branch carries signal. Tested alone, per-scene track@1 is 0.051 for semantic, 0.033 for dynamics and 0.035 for mood, and no weighted combination beats semantic alone. The final system therefore scores with the semantic branch only.

## 9. Key findings

1. **The model memorises films.** Recall is 87% on training clips but about 29% on unseen films. The limit is generalisation, not model capacity, which is why larger models did not help and averaging models did.
2. **More films help; more clips per film do not.** Recall rose steadily with the number of training films (73 films: 0.174; 145: 0.216; 218: 0.245; 290: 0.288) with no plateau. Adding 2,998 clips from only 32 films changed nothing.
3. **There is no beat-level alignment in this data.** Cuts do not follow the music's beats (no difference from shifted music, p = 0.76), so a synchronisation branch has little to learn.
4. **Exact-match recall understates quality.** A scene can suit many pieces, but only the original counts as correct. 23% of wrong top picks come from the same film, and 133 are near-identical to the true music.
5. **Stitching results depend on the test protocol.** Viterbi beats greedy on concatenations of different songs, but ties it on blocked in-film sequences, where two-stage and segmental Viterbi lead (D150).

## 10. Robustness check, limitations, next steps

**Full 4,246-pair dataset.** On an extended set that adds 2,998 source-separated clips, the same method gives whole-clip R@10 0.123 for a single model and 0.160 for the ensemble (chance 0.015; galleries four times larger). The original clips score 0.334 there, unchanged; the added clips score 0.102, likely because vocals bleed through the separation.

**Limitations.**
- The top-ranked piece is the original music only about 6% of the time; the system is a strong shortlister, not an exact matcher.
- Position within a track is only moderately better than chance, and there is no learned beat sync (beat-snapping at render time handles entries).
- Recall measures recovery of the original music, not whether the chosen music suits the scene. A listening study has not been run.
- With equal ensembling, published baselines match the retrieval numbers.

**Next steps.** Collect clips from many more distinct films with clean original audio; run a listening study; add human-marked alternative positives; label Indian-aesthetic mood (rasa) for a mood model trained on this domain.

## 11. Suggested slide outline

1. Title and one-line problem statement
2. Problem: from a silent video to a stitched soundtrack (input and output)
3. Main idea: choose a sequence, not one song per clip
4. Dataset and the film-level split (include the 0.502 vs 0.279 leak comparison)
5. Pipeline overview (the seven steps as a diagram)
6. Encoders and matching heads
7. Ensemble of ten models
8. Coarse-to-fine retrieval
9. Viterbi stitching and rendering
10. Evaluation protocol (recall, chance, film-level confidence intervals)
11. Retrieval results (table 7.1)
12. Pipeline and offset results (table 7.2)
13. Stitching results (table 7.3)
14. Comparison with published methods (table 7.4, with the fair-claim sentence)
15. What helped and what did not (table 8)
16. The three-branch design and why only semantic is used
17. Key findings (section 9)
18. Limitations and next steps
