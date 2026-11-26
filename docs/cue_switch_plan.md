# Cue-switch predictor: Step 0 findings and plan

Status: **approved 2026-10-19 and implemented** (PROGRESS.md D78–D82; results/cue_switch.md). User answers to §3:
Q1 film-blocked main sets with inside-film boundaries as the headline, plus single-film same_film / timeline sets
≥ 40 s; Q2 ContentDetector cached in a new file; Q3 negative λ_jump allowed, μ kept identically in all runs; Q4 both
stored and re-tuned R0, re-tuned is the main comparison; Q5 optional film_key, local files only, 5-fold average for
films outside the dataset; Q6 R6 is the only emission change. Changes during implementation (D79): motion statistics
skip frame 0; a no-duration diagnostic (R4nodur) was added; training settings were chosen on val; no positional
encoding. The text below is the Step 0 plan as approved.

## 1. Findings

### 1.1 Where things live

| What | Where |
| --- | --- |
| Segments | `src/bgm/data/segments.py` (`make_segments`: shots merged at SigLIP cosine ≥ 0.85, min shot 2 s, max 20 s, long shots split, D13) → `data/manifests/segments.csv` (3,282 segments) |
| Shots | `src/bgm/data/shots.py` (PySceneDetect **AdaptiveDetector**, threshold 3.0) → `shots.csv`, `cut_times_s` in each video npz |
| Viterbi | `src/bgm/retrieval/stitch.py` (`transition_cost`, `viterbi`, the tested SPEC reference) and the vectorised lattice version actually used: `stitch_eval.build_lattice` / `transition_matrix` / `decode_lattice` (beam = 5 × J = 100) |
| Transition costs | continue (same file, contiguous ±0.04 s) = 0; jump (same file, not contiguous) = `lam_jump`; switch (other file) = `lam_switch + mu·(fifths + tempo distance)`. Only between segments; nothing inside a segment |
| Penalty tuning | `src/bgm/retrieval/tune_stitch.py::tune_one`: grid `lam_switch` {0.5, 1, 2, 4, 8, 1000} × `lam_jump` {0.25, 0.5, 1, 2} × `mu` {0, 0.5, 1}; objective = mean over val concatenations of part track accuracy + switch F1; result in `stitch.json` |
| Validation / test concatenations | `stitch_eval.make_concat_sets` → `data/manifests/concat_sets_film.json` (film split) |
| Switch F1 | `stitch_eval.switch_f1` (greedy one-to-one matching, ±1 s). Predicted = segment boundaries where the chosen **music file** changes; truth = every join between parts |
| Render | `src/bgm/render/run.py::plan_for` calls `decode_lattice` with the run's `stitch.json` penalties; `./render_best.sh` = `runs/filmsplit/pipeline_ens10/fold0/seed0`, 10-seed ensemble |
| D72 context score | `retrieval.context_weight` → `VideoQuery.set_context` (index.py / pipeline.py), default 0 |

D75 tuned penalties (ensemble runs, one per fold): fold 0 switch 1.0 / jump 0.25 / mu 0; fold 1 1.0 / 0.5 / 0;
fold 2 2.0 / 2.0 / 1.0; fold 3 1.0 / 1.0 / 1.0; fold 4 1.0 / 1.0 / 0.5.

### 1.2 Current concatenations

Per fold: 50 val + 50 test. Each is 3–4 random clips of that split with distinct track_ids, joined end to end
(`concat_query`: each clip keeps its own segments, so every join is a segment boundary). They are not film-aware:
83% (val) and 92% (test) of them have all parts from different films.
**Total length: median 57 s** (val 30–138 s, IQR 45–70; test 31–160 s, IQR 48–69). That is shorter than the target
of 60–180 s, and nearly every boundary is a change between films.

### 1.3 Training clips (1,248)

- Duration: **min 9.8 s, median 15.0 s, max 171 s** (10th / 90th percentile 10.5 / 33.4 s; mean 19.5 s).
- Segments per clip: median 2, max 17. 397 clips have a single segment. 2,034 within-clip boundaries in total.
- Per fold: train 961–991 clips (322–333 min, 266 of them Konjum = 125 min), val 61–92 (16–26 min), test 195–199 (55–57 min).

### 1.4 Metadata

| Field | Available? |
| --- | --- |
| film id | yes, all clips: `film_group` in folds_film.json (405 groups; D36) and `meta_source:meta_film` |
| YouTube id | **no**: the youtube sources are numbered folders ("youtube 158"), with no video ids. The full-dataset manifest has `video_id` only for the 2,998 Demucs clips you dropped on 2026-10-06 |
| timestamps within the film | **sunnxt only, 539 clips**: `meta_movie_start_s` / `_end_s`. The source "movies" are excerpts of about 5 min (median 295 s, 36–986 s), not whole films |
| track_id | yes, all |
| track group (D17) | `track_id` in pairs.csv already is the D17 group (1,150 groups) |

### 1.5 Timeline structure (sunnxt)

- 261 sunnxt films (246 film groups), median 2 clips each (max 12). The `bgm_NNNN` index follows film time in 261 of 261 films.
- 278 adjacent clip pairs. Gap between one clip's end and the next one's start: never overlapping; median 30 s; 10th / 90th percentile 3.4 / 128 s.
  **< 5 s: 45, < 10 s: 76, < 30 s: 137, < 60 s: 188.** Counting every ordered pair (not only neighbours) with a gap < 30 s gives 147.
- The gaps are what the source detector labelled speech (11,731 s), music shorter than its 10 s minimum (1,814 s) or silence (596 s).
  6 adjacent pairs share a track group.
- Is the music really a different cue across a short gap? Cached CLAP, last 3 s of clip i vs first 3 s of clip i+1 (median cosine):

  | pair | CLAP | MERT |
  | --- | --- | --- |
  | within one clip (two halves) | 0.960 | 0.962 |
  | neighbours, gap < 5 s | 0.791 | 0.937 |
  | neighbours, gap 5–30 s | 0.670 | 0.902 |
  | neighbours, gap > 30 s | 0.612 | 0.906 |
  | same film, not neighbours | 0.598 | 0.890 |
  | different films | 0.540 | 0.884 |

  Short-gap neighbours are closer than random same-film pairs but far from within-clip continuity. Most are real
  changes, but some may be one cue that continues under dialogue. These labels have not been checked by ear.
- Chains of clips with gaps < 30 s: 402 chains, median 17.5 s of clip time, 20 of them ≥ 60 s.

### 1.6 Can sequences of 60–180 s come from one film?

| per fold (non-Konjum) | train | val | test |
| --- | --- | --- | --- |
| films | 286–298 | 32–34 | 72–86 |
| films with ≥ 2 clips | 157–166 | 16–20 | 41–47 |
| films with ≥ 60 s of clips | 47–55 | **3–8** | **12–18** |
| sunnxt films (timeline) with ≥ 60 s | — | **1–4** | **6–12** |

Over the whole dataset, a film has a median of 27 s of clips (IQR 14–45 s); 71 of 404 films reach 60 s.
**Sequences of 60–180 s from a single film are possible for only a few films per fold**, and val is especially thin.

### 1.7 Input videos (media/inputs/)

| file | duration | note |
| --- | --- | --- |
| 123.mp4 | 207.3 s | |
| 12.mp4 | 287.5 s | video starts at 0.04 s: render's A/V assertion fails (known, D75) |
| 1.mp4 | 347.3 s | silent |
| 1_BGM_Ori.mp4 | 347.3 s | 1.mp4 with its original soundtrack |
| with_audio.mp4 | 347.4 s | |

All are 3.5–5.8 min, longer than both the brief's 1–3 min and the 180 s training sequences.

### 1.8 Baseline captured before any change (constraint 2)

`runs/cue/baseline/run_baseline.sh` (git 6b9b8fd): `bgm fuse` + `bgm tune-stitch` for fold 0 × seed 0 with the D75
settings into a **new** exp dir, `runs/filmsplit/cue_baseline_ens10/`, then `./render_best.sh media/inputs/123.mp4`
→ `renders/cue_baseline/123_best.mp4`. The fuse outputs are byte-identical to the existing
`runs/filmsplit/pipeline_ens10/fold0/seed0` (fusion.json, perquery.npz; metrics.json differs only in wall-clock
seconds). stitch.json, stitch_metrics.json and all 24 arrays of stitch_perquery.npz are also identical to the D65
run, so the pipeline reproduces bit for bit today. Fold 0 test: Viterbi switch F1 0.742, part track accuracy 0.054.
Render of 123.mp4: 37 segments → 32 pieces, penalties 1.0 / 0.25 / 0. The fingerprints (decoded PCM sha256, AAC
stream sha256, plan.json sha256) are in `runs/cue/baseline/123_best.audio_pcm.sha256`, and the plan is copied there.
After the change, the same script will be rerun with `cue.enabled=false` and diffed against these.

## 2. Plan

### 2.1 Code layout (new module; existing files touched only for opt-in hooks)

```
src/bgm/cue/
  sequences.py   build_sequences(cfg, store, fold, split, mode, n, seed) -> list[Sequence]; label_boundaries()
  tokens.py      segment_tokens(store, seq) -> [N, D] (span-mean SigLIP, log duration, motion mean/std); train-set z-stats
  model.py       CueModel (MLP tokens, [CTX], 2-layer bidirectional Transformer, boundary head); PairOnly variant
  train.py       train_cue(cfg, fold, seed, variant, mode) -> runs/filmsplit/cue/<variant>/fold<k>/seed<s>/
  baselines.py   cosine rule (R2), scene-cut rule (R1)
  stitch.py      decode_lattice_cue(layers, p, alpha, lam_jump, mu, beam, tol): own beam + Viterbi loop reusing
                 stitch_eval.transition_matrix's pieces. decode_lattice itself is NOT edited
  tune.py        alpha x lam_jump grid on val sequences, same objective as tune_stitch
  evaluate.py    R0-R6 on test sequences; metrics; cluster-bootstrap report -> results/cue_switch.md
  real_eval.py   Step 5: annotation CSV -> boundary metrics
tests/test_cue.py
data/manifests/cue_sequences_film.json      frozen sequence sets (fold, split, mode, seed)
data/annotations/cue_changes_template.csv   empty template (header only)
```

Hooks in existing files, all default OFF:
- `configs/default.yaml`: a new `cue:` section (`enabled: false`, `run: null`, model and training settings). Adding
  keys changes no feature hash; runs' config.yaml files gain the section.
- `src/bgm/render/run.py::plan_for`: `if cfg.cue.enabled:` predict p for the video's segments and call
  `decode_lattice_cue` with the cue run's tuned α / λ_jump; otherwise the existing line runs unchanged.
- `src/bgm/cli.py`: a new `bgm cue build|train|tune|eval|real-eval` command group.

`tune_stitch.py`, `stitch_eval.py`, `stitch.py`, `pipeline.py` and `index.py` are not edited.

### 2.2 Step 1: sequences and labels

- A sequence is a list of clips joined with the existing `concat_query`, so its segments are exactly the
  `segments.csv` segments of each clip and the emission lattice is the D75 pipeline's.
- Total length: draw a target uniformly from 60–180 s and add clips until it is reached. Stop before going over 180 s
  unless the sequence is still under 60 s.
- Label of the boundary between segments i and i+1: **0** if both are in the same clip (shot cuts inside a clip
  included); else **1** if their track groups differ, **0** if they are the same.
- Modes: `timeline`, `same_film`, `cross_film` as in the brief (blocked fallback: see Q1).
- Folds: train sequences only from the fold's train clips (Konjum allowed, at most 1 Konjum block per sequence, so
  its 266 clips do not dominate); val from val; test from test. Seeds: `[seed, fold, split_code, mode_code]` per
  sequence. Training sequences are redrawn every epoch (per-item seeds); val and test sets are drawn once and frozen in
  `cue_sequences_film.json`.
- Test sets **partition** each fold's test film groups: a film group appears in only one test sequence of a given
  set. So resampling sequences resamples disjoint film-group clusters, as D39 asks. Val sets may reuse clips (val is too
  small otherwise); they only tune and early-stop.

### 2.3 Step 2: model

- Token = MLP([span-mean SigLIP over the segment's own 1 Hz frames (not the 5 s expanded music span, which would
  reach into the neighbour), log duration, mean and std of `log_motion` over the segment]).
  `log_motion` is the cached log1p flow magnitude with cuts masked. It is read **raw** from `dyn_25hz` and z-scored with
  **train-fold statistics**, not with `store.dyn`'s per-clip normalisation: per-clip z-scores would mark clip joins (a
  leak). Render computes the same raw curve on the input video, so training and inference match.
- A learned [CTX] token is prepended. Then a 2-layer bidirectional Transformer encoder (d 256, 4 heads, dropout 0.4)
  with **no positional encoding**: the pair head supplies the order, and the model is not tied to the training lengths
  (inputs are up to 5.8 min).
- Boundary head: MLP([h_i, h_{i+1}, |h_i − h_{i+1}|, h_i ⊙ h_{i+1}]) → logit. Pair-only variant: the same head on the
  MLP tokens, without CTX or Transformer.
- AdamW, weight decay 0.3; BCE with pos_weight = neg/pos counted on the training sequences. Early stopping on **val
  average precision** of the boundary logits (threshold-free, robust to the class ratio), patience 10.
- 3 seeds × 5 folds per variant. Each run writes config.yaml, metrics.json, git commit and seed.

### 2.4 Step 3: Viterbi hook

At boundary i, as additive costs (the negated scores of the brief):
continue = −α·log(1 − p_i); switch = −α·log p_i + μ·(fifths + tempo); jump = −α·log p_i + λ_jump.
Every path crosses each boundary exactly once, so subtracting the continue cost at each boundary leaves the path and
the beam ranking unchanged. With a constant p this reduces to lam_switch = α·log((1 − p)/p) and
lam_jump = lam_switch + λ_jump.

Unit tests:
1. With constant p, α, λ_jump = lam_jump − lam_switch and the same μ, `decode_lattice_cue` returns the same picks as
   `decode_lattice`. Checked on random lattices and on the fold-0 val lattices, for all five folds' D75 penalties.
2. Labels: within-clip boundaries are always 0.
3. No fold leakage: no train or val sequence contains a test film group, no train sequence contains a val clip, and
   Konjum never appears in val or test.

Tuning: on the val sequences, grid α {0.25, 0.5, 1, 2, 4, 8} × λ_jump {−1, −0.5, −0.25, 0, 0.5, 1} × μ {0, 0.5, 1};
objective = part track accuracy + switch F1, as in `tune_one`. α scales p, so R1 and R2 get the same grid.

### 2.5 Step 4: runs

- Emission: the D75 ensemble pipeline, one per fold (`fusion.json` of `pipeline_ens10`). The lattices are built once
  per sequence and shared by R0–R5, so track R@k and the per-segment hits are identical by construction. I will
  assert array equality, not just report it.
- "3 seeds" are the cue model's training seeds. R0–R2 do not depend on a seed; that is reported, not repeated.
- R0: current penalties (see Q4). R1: scene cuts. R2: σ(a·(1 − cos) + b), logistic fit on train-sequence boundaries.
  R3: pair-only. R4: full model. R5: R4 trained on cross_film. R6: R4 + `retrieval.context_weight = 1` (emission
  changes on purpose, so R6 is excluded from the R@k identity check).
- Metrics:
  - all existing stitching metrics (part track accuracy, switch F1 ±1 s, switches/min, jumps/min, seam loudness jump);
  - boundary F1 at ±0.5 s and ±1 s against track-group changes;
  - the true number of changes per minute, for reference;
  - the predictor's own AP and AUROC per boundary.
- CIs: cluster bootstrap over sequences (disjoint film groups), 10,000 resamples; deltas paired per sequence.
  Results are reported separately for each test set (timeline, same_film, the mixed set) and pooled.
- Expected on concatenations: ±0.5 s and ±1 s F1 will be almost equal. Switches happen only at segment boundaries,
  every true change is a segment boundary, and segments are ≥ 2 s long. The tolerances will differ only on real videos (Step 5).

### 2.6 Step 5: real multi-cue tool

`bgm cue real-eval --csv data/annotations/cue_changes.csv`; columns `youtube_id,start,end,cue_change_times`.
I create the template with the header only. The tool:
- reads a **local** file `media/annotated/<youtube_id>.mp4` and downloads nothing;
- cuts [start, end] to a silent 25 fps interim video (rule 8);
- runs the render's query feature path (SigLIP 1 Hz, shots, flow curves, segments; cached);
- runs R0–R4 with the fold-0 deployment run;
- scores boundary F1 ±0.5 / ±1 s and switches/min against the annotated times.

There is no track truth (the music is outside the library), so the tool reports boundary metrics only.

### 2.7 Logging

New decisions go in PROGRESS.md as D78 onward. Results go in `results/cue_switch.md`; `results/final.md` is not touched.

### 2.8 Risks seen so far

1. **The labels mostly encode "is this a clip join".** Every 1 is a join between dataset clips and every 0 is a shot
   cut inside a clip. In timeline mode, each join also skips a stretch of film (median 30 s of dialogue or short
   music) that a real continuous video would contain. Only Step 5's real annotations test the real task.
2. Some short-gap timeline 1s may be one cue continuing under dialogue (§1.5).
3. A real cue change can fall inside a segment. Segment merging (cosine 0.85) can swallow it, and "inside a segment,
   keep current behaviour" means Viterbi cannot switch there.
4. Clips differ by source in frame size, letterboxing and grading. Cross-film joins are trivially visible, which is
   why R5 matters.
5. Val is small (16–26 min per fold; 1–8 eligible single-film val films), so early stopping and α tuning will be noisy.
6. Inputs run to 5.8 min, beyond the 180 s training sequences.

## 3. Questions before I start

**Q1. Sequence design, given §1.6.** Single-film sequences of 60–180 s exist for only 12–18 test and 3–8 val films
per fold (timeline: 6–12 / 1–4). My proposal:
- (a) The main training and test sets are **film-blocked**: a sequence is a chain of blocks, each block 2+ clips of
  one film (timeline-ordered when the film has timestamps, random order otherwise), with blocks from different films
  until 60–180 s. Boundaries are then within-clip (0), same-film join (1) and cross-film join (1); metrics are
  reported per boundary type.
- (b) Plus pure single-film `same_film` and `timeline` test sets for the films that reach a relaxed **≥ 40 s**,
  reported separately.
- (c) `cross_film` as in the brief (R5 only).

Or would you rather keep the brief's literal modes and accept tiny single-film sets?

**Q2. R1 needs new detection.** Running PySceneDetect ContentDetector (threshold 30, minimum scene 8 s) is a new pass
over the silent interim videos (no encoder; about 2 min for 1,248 clips). I would cache per-frame content scores per
clip and replay the detector on each concatenated sequence, including the frame pair across each join, so min-scene
8 s behaves as on one continuous video. Scene boundaries snap to the nearest segment boundary within ±1 s;
p = 1 − ε there and ε elsewhere. OK under constraint 4?

**Q3. Jump and μ.** To reproduce the tuned penalties (fold 0: switch 1.0, jump 0.25), λ_jump must be allowed to be
negative (λ_jump = lam_jump − lam_switch = −0.75). In your formula a jump is otherwise always at least as costly as a
switch. Folds 2–4 also use μ > 0, so I keep the μ·(key + tempo) term on switches. Agree with both?

**Q4. R0 penalties.** Either (i) the stored D75 penalties, tuned on the old 3–4-clip concatenations, or (ii) the same
grid re-tuned on the new val sequences. I propose reporting both, with (ii) as the main comparison, because R1–R4 are
tuned on the new val sequences.

**Q5. Step 5 inputs.** The metadata has no YouTube ids, and downloading needs your approval. The tool expects local
files `media/annotated/<youtube_id>.mp4`. May I add an optional `film_key` column? If an excerpt comes from a dataset
film, it lets the tool use a fold whose test set holds that film; without it, an excerpt from a training film would leak.

**Q6. R6** changes the emission by design (the D72 context score), as an exception to constraint 3. Confirming that is intended.
