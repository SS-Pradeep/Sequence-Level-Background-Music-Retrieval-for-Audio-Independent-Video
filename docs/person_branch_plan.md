# Person-focused branch: plan (Step 0)

Status: **done 2026-10-20** (results/person_branch.md, PROGRESS.md D87–D92); approved 2026-10-20 with the answers in "Decisions on the open questions" at the end. Decision numbers start at **D87**, the next free
number (D78–D86 are already used by the cue-switch work and the HD renders).

## Goal

SigLIP encodes the whole frame, so the setting dominates the embedding. Two temple scenes, one grieving and one
celebrating, embed close together and retrieve similar music. A person-focused branch embeds only the people on screen
(faces, or person boxes when no face is found) with the same SigLIP encoder. It learns from the video–music pairs with
InfoNCE: no text, no emotion labels.

This task is separate from the cue-switch predictor (docs/cue_switch_plan.md). It does not import or change `bgm/cue/`, and
`cue.enabled` stays false in every run.

## What exists (Step 0 findings)

### D77 facial-expression branch

| part | what it is |
| --- | --- |
| detector | OpenCV YuNet (`data/models/face_detection_yunet_2023mar.onnx`), score ≥ 0.6, box side ≥ 24 px, one frame per second (the frame at k + 0.5 s of the silent 25 fps interim video) |
| features | `trpakov/vit-face-expression` (FER-2013 classifier, frozen) on each face crop (box + 20%). `face_expr_1hz` [T1, 17]: soft and hard counts of 7 expressions, number of faces, face area share, no-face flag. Stored in data/features/video_face/ (5.3 MB) |
| code | src/bgm/features/face.py (extraction); src/bgm/analysis/arch_three_branch.py (evaluation) |
| heads | `face-both`: the semantic-head trainer with `train.semantic.video_feature=face_expr_1hz` against CLAP+MERT, seeds 0–2 → runs/filmsplit/semantic/face-both (and -m2m) |
| fusion | analysis only, not in the pipeline: z-score each branch per query, weighted sum, single models, 3 seeds |
| coverage | 64% of seconds have a face |

D77 results (track R@10, test-fold gallery 196; segment R@10; paired Δ vs one-to-one semantic heads at 0.288 / 0.123):

| system | track R@10 | segment R@10 |
| --- | --- | --- |
| facial alone | 0.121 (chance 0.058) | 0.042 |
| semantic 0.9 + facial 0.1 | 0.296, +0.008 [+0.000, +0.016] | 0.128, +0.005 [+0.002, +0.008] |
| semantic 0.8 + facial 0.2 | 0.301, **+0.013 [+0.001, +0.025]** | 0.129, +0.006 [+0.002, +0.011] |
| weights tuned on val (3 branches) | 0.275, −0.013 [−0.030, +0.003] | 0.118 |

Caveats recorded in D77: the 0.8 / 0.2 weight was **not chosen on val**, and the result was never tested on top of the
10-seed ensemble. The facial branch is a 17-number summary of expressions. It does not encode *who* or *how* the people
look, which is what this task adds.

The facial branch is a usable base for this task: YuNet and its frame reader can be reused as they are. The FER classifier
is not reused, because the new branch encodes crops with SigLIP.

### Where things live

| piece | location |
| --- | --- |
| SigLIP extraction (dataset) | src/bgm/features/extract.py `Extractor._sem_video` → `sem_siglip_1hz` [T1, 1152] in data/features/video/<pair>.npz; encoder `SiglipEncoder` in src/bgm/features/hf.py (`google/siglip-so400m-patch14-384`, fp16) |
| SigLIP extraction (render inputs) | src/bgm/render/query.py `featurize` → renders/<name>/query.npz |
| matcher heads | src/bgm/models/heads.py `SemanticHead` (LayerNorm → 512 → GELU → dropout → 256, unit norm); trainer src/bgm/train/semantic.py (`SemanticInputs`, `train_one`, InfoNCE with same-track masking, film cap 30, early stopping on val segment R@10) |
| ensemble | src/bgm/retrieval/index.py `EnsembleSemantic` (mean cosine of the members), `FoldModels.load` with `pipeline.sem_seeds` |
| fusion weights | src/bgm/retrieval/fuse.py (`tune_coarse`, `tune_fine`, per-run fusion.json); `fusion.dyn_weight: 0.0` and `fusion.attr_weight: 0.0` pin D75 to semantic-only (D55, D56) |
| pipeline and stitching | src/bgm/retrieval/pipeline.py (coarse shortlist, fine windows, per-segment z-scored branch fusion), stitch_eval.py and tune_stitch.py (Viterbi, penalties tuned on val concatenations) |
| D75 runs | runs/filmsplit/pipeline_ens10/fold{0..4}/seed0 (one run per fold; the 10-seed ensemble ignores the run seed). Single-head pipeline: runs/filmsplit/pipeline/fold*/seed{0,1,2} (D60) |
| ./render_best.sh | `bgm render --run runs/filmsplit/pipeline_ens10/fold0/seed0` with the film split, `pipeline.dyn_variant=null`, `pipeline.sem_seeds=[0..9]` → src/bgm/render/run.py (`plan_for`, `render_video`), src/bgm/render/render.py (`mux`) |

### Videos in media/inputs/

| file | duration | video | own audio | note |
| --- | --- | --- | --- | --- |
| 123.mp4 | 207.3 s | 1920×1080, 25 fps, starts 0.000 | yes | |
| 12.mp4 | 287.5 s | 1920×1080, 25 fps, **starts 0.040** | yes | the "known issue" input |
| 1.mp4 | 347.3 s | 1080×720, 24 fps, starts 0.041 | none (silent) | |
| 1_BGM_Ori.mp4 | 347.3 s | 1080×720, 24 fps, starts 0.000 | yes (original BGM) | same picture as 1.mp4 |
| with_audio.mp4 | 347.4 s | 1080×720, 24 fps, starts 0.042 | yes | same picture as 1.mp4 |

Render strips every input's audio first (silent 25 fps copy), so 1_BGM_Ori and with_audio will render almost exactly like 1.mp4.

**The 0.04 s start issue is fixed already.** D86 (2026-10-19) changed `mux` to pad the gap with silence, and
renders/hd_final/12_bgm.mp4 rendered and checked at 0.0 ms offset. I will not touch the render code. If any input still
fails, I will skip it and list it.

## Hard constraints and how they are kept

1. **New code only in `src/bgm/person/`.** Changes to existing files are opt-in hooks only:
   - a `person:` section in configs/default.yaml (`person.enabled: false`);
   - in `render/run.py::plan_for`, one `if cfg.person.enabled:` branch that builds the index and query through `bgm.person`, the same pattern as the cue hook;
   - a `bgm person ...` command group in cli.py;
   - `outputs/person_branch/_work/` added to .gitignore.

   semantic.py, index.py, pipeline.py, fuse.py and tune_stitch.py are not edited. The person module subclasses or wraps
   them: a `PersonMusicIndex(MusicIndex)` adds a `person` branch to `window_scores` and `file_scores`. The existing `fuse`
   and z-score code already iterate over whatever branch keys are present.
2. **Bit-identical with the flag off.** Before any code is written (D78 procedure):
   - run `bgm fuse` + `bgm tune-stitch` for fold 0 × seed 0 with the D75 settings into runs/person_check/before;
   - run `./render_best.sh media/inputs/123.mp4` with `paths.renders` pointed at a scratch folder.

   After the code is written, rerun both with the flag off into runs/person_check/after. Every array and json must be
   identical, except wall-clock seconds and output paths. The decoded PCM sha256 of the audio must also match. pytest
   must pass.
3. **Splits.** The film-level split (folds_film.json, D36) is used unchanged. Konjum is train-only through that file.
   Every weight, threshold and stopping epoch is chosen on each fold's val films only. Test films are touched only for
   the final numbers.
4. **Training signal.** Only video–music InfoNCE. No text prompts, mood words or emotion labels anywhere in the branch.
   The D77 FER classifier is used only by P1, which is D77's own branch, and P1 is a comparison row.
5. **New caches only.**
   - Person features go to data/features/video_person/<pair_id>.npz, with a config hash in `meta`.
   - Render-input person features go to <work>/person.npz.
   - Nothing in data/features/video, video_face or renders/<name>/query.npz is written.

## Step 1: diagnosis first (report, then continue)

**Subset definition, as specified.** Take every pair of test segments from different videos in the same test fold:
- setting similarity: cosine of their span-mean SigLIP vectors;
- music similarity: cosine of their paired music spans' CLAP+MERT vectors. Each part is unit-normalised, and the two are concatenated and divided by √2, so this equals the mean of the CLAP and MERT cosines.

A segment is in the subset if it belongs to at least one pair with setting similarity in the top 10% and music similarity
in the bottom 30%. Both percentiles are taken over that fold's cross-video pairs.

**Sizing check (already run, read-only):** 1,995 of 2,552 test segments (78%) fall in the subset. 1,903 of them qualify
through a cross-film pair. With about 500 segments per fold, almost every segment has *some* same-looking partner with
different music, so this subset is nearly "all segments" and cannot show a difference. **Proposed tighter companion
subset (Q1):** a segment qualifies only if its single nearest other-video neighbour by SigLIP is in the top 10% and their
music is in the bottom 30%. That gives **272 segments (11%)**. Both subsets would be reported, with the tight one used as
the diagnostic.

**D75 numbers on each subset vs all segments**, taken from the saved D75 runs where possible. A recomputed number must
reproduce the saved per-segment value before it is used.
- per-segment pipeline: track@1, window@1, window@J, offset within ±1 / ±3 / ±5 s;
- per-segment **track R@10**: the right track group among the 10 best-scoring distinct files, ranked by each file's best fine window. This is the R@10 version of the existing track@1;
- segment R@10 (M5 protocol, every window of the test-fold gallery);
- clip-level track R@10 over videos with at least one subset segment.

Each number is reported against its chance level. The difference between subset and complement gets a film-group
cluster-bootstrap CI.

## Step 2: person crops (new features)

Per second k, the frame at k + 0.5 s of the silent interim video, read with D77's `read_second_frames`.

- **Faces:** YuNet with D77's settings (score ≥ 0.6, side ≥ 24 px). Keep at most 5 faces per frame, the largest first. Crop = box widened by 30% on each side, then padded to a square inside the frame, because SigLIP resizes to 384×384 without keeping aspect.
- **Fallback, only on seconds with no face:** person boxes from torchvision `keypointrcnn_resnet50_fpn` (COCO person class, score ≥ 0.7, box height ≥ 48 px, at most 5). Its weights are already in ~/.cache/torch, so nothing is downloaded. Crop = box padded to a square.
- **Encoding:** each crop goes through the same SigLIP so400m image encoder (frozen, eval, no_grad, fp16; existing `EncoderBank`). Crop embeddings are unit vectors. The second's vector is their mean.
- **Stored:**
  - `person_siglip_1hz` [T1, 1152] f16, zero where the second has no person;
  - `person_kind_1hz` [T1] int8 (0 none, 1 face, 2 person box);
  - `person_n_1hz` [T1] int8;
  - the meta config hash.

  T1 matches `sem_siglip_1hz`.
- **Segment input:** the mean of the person seconds inside the span (span rule as for SigLIP), unit-normalised, plus a mask flag. A span with no person second gets mask 0, and the head replaces the input with a **learned "no person" vector** (an `nn.Parameter`, 1152-d). Frame-drop augmentation applies to person seconds the same way it does to frames.
- **Settings are fixed in advance** (margins, thresholds, cap of 5); they are not tuned. Coverage gets reported: the share of seconds with a face, a person box, or neither.
- **Render inputs** are 480 px on the short side, while the dataset is 360 px (D12), so faces there are slightly larger. This is noted as a domain difference and not corrected.

Expected cost: about 20–30 min of GPU time for 1,248 clips plus the 5 inputs.

## Step 3: runs

All runs use the film split, 5 folds, and test-fold galleries. Heads use the semantic-head protocol unchanged: CLAP+MERT
music side, segment positives, one segment per video per batch, same-track masking, film cap 30, early stopping on val
segment R@10, and the same lr, batch and τ.

| run | video side | score | new training |
| --- | --- | --- | --- |
| **P0** | D75: 10-seed siglip-both ensemble | semantic only | none (runs/filmsplit/pipeline_ens10) |
| P0s (reference) | single siglip-both heads, seeds 0–2 | semantic only | none (D60 runs) |
| **P1** | frame score + D77 face-both head (seeds 0–2) | z(s_frame) + w · z(s_face), w from val | none |
| **P2** | frame score + new person head (InfoNCE, person crops vs CLAP+MERT, seeds 0–2) | z(s_frame) + w · z(s_person), w from val | 15 runs, `person-both` |
| **P3** | one head on [SigLIP frame mean ‖ person mean ‖ mask] (no-person vector learned) | its own cosine | 15 runs, `siglipperson-both` |
| **P4** | the best of P1–P3, in ensemble form (see Q3) | as that run | up to 50 + 50 runs |

- **Late fusion inside the pipeline:** the person (or face) branch becomes a fourth key in `raw`. It is z-scored per segment over the window set, like the other branches. The coarse stage uses the same w (file-level scores). Viterbi decodes the fused lattice.
- **w grid:** {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}, with the frame weight fixed at 1. Ties keep the smaller w. The val objective is the mean of val track R@10 and val segment R@10, because one metric on 61–92 clips overfits more (D55). See Q2 for how w is chosen.
- **Metrics, every run:**
  - retrieval: track R@1 / R@10 / MedR (test fold and all 1,248), segment R@10;
  - pipeline: track@1, track R@10 (per segment), window@1, window@J, offset ±1 / ±3 / ±5 s;
  - stitching: switch F1, part track accuracy, switches / min (penalties tuned per fold × seed on val concatenations, as in D60);
  - each per-segment metric also on both Step 1 subsets.
- **CIs and protocol:** paired bootstrap of each run against P0, and also against P0s, which isolates the branch from the ensemble effect (see Q3). 95% film-group cluster bootstrap (D39), 10,000 resamples. Concatenation metrics resample concatenations (D60). Values per query are averaged over seeds first. The cue hook stays off.
- **Outputs:** runs in runs/filmsplit/person/...; report `python -m bgm.person.report` → results/person_branch.md. results/final.md is not edited.

Expected cost: about 1–2 min per head run, about 2–3 min per pipeline + stitch run, a few GPU hours in total.

## Step 4: render

- `outputs/person_branch/baseline/<name>.mp4`: `./render_best.sh` (P0), unchanged.
- `outputs/person_branch/person/<name>.mp4`: the same command plus `person.enabled=true` and the best run's settings, fold 0, the same as render_best.
- Work folders and plan files go to `outputs/person_branch/_work/{baseline,person}/` through `paths.renders=...`. Both systems' files share stems (123.mp4 → "123"), and the default would write into renders/, so this override is required. Nothing existing in renders/ is touched.
- All 5 inputs get rendered, with audio at 320k (user: high-quality outputs). Inputs that fail are skipped and listed.
- **Upscale to 1080p (user request).** The render stream-copies the source picture, so each output starts at its input's
  resolution. 123 and 12 are already 1920×1080 and are kept as rendered. The three 1080×720 outputs (1, 1_BGM_Ori,
  with_audio) are re-encoded to 1080 lines: Lanczos scaling to 1620×1080, which keeps the 3:2 aspect with no stretching
  or bars, libx264 CRF 16, preset slow. Their audio is stream-copied from the render. The final files in baseline/ and
  person/ are the 1080p ones; the native-resolution renders stay in _work/. Sync is checked after the re-encode: the
  video and audio stream start times, plus a cross-correlation of the output audio against bgm.wav, as in D86.
- `outputs/person_branch/index.csv` has one row per piece: `input, duration_s, system, piece, video_start_s, video_end_s, music_id, track_id, music_offset_s`, taken from each plan.json.

## Logging

- D87 onward in docs/PROGRESS.md, in the existing style.
- results/person_branch.md.
- A milestone row in PROGRESS.
- One commit when done.

## Decisions on the open questions (user, 2026-10-20)

- **Q1 Step 1 subset:** both subsets are reported, as specified (78% of segments) and the nearest-neighbour subset (272 segments, the diagnostic). (Default; no objection.)
- **Q2 w:** chosen per fold on that fold's val films, coarse grid {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1}, objective = mean of val track R@10 and val segment R@10, ties keep the smaller w. The D55 overfitting risk is stated in the report.
- **Q3 P4:** 10 seeds of the winning head with weight decay 0.3 + dropout 0.4, ensembled (D65), plus a separate distillation row (one student per seed, 10 teachers). The best of P1–P3 is picked against P0s (single models, same seeds); every row is also compared with P0.
- **Q4 renders:** all five inputs; audio 320k; every output at 1080 lines (above).
- **Results:** "upscaled" means higher-quality outputs only. The evaluation stays on the 1,248-pair film split.
