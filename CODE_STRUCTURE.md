# Code structure

This document explains how the code is organised and what each file the system needs does. Of the 248 Python
files in `src/bgm/`, **104 are needed to run the system** through the `bgm` command. The rest are analysis, sweep
and paper scripts (see section 5). The 104 were found by tracing every `import` from `src/bgm/cli.py`, including
imports that only run behind a flag.

The "Needed for" column in the tables means:

- **[render]**: needed to add music to a video (`bgm render`). These 74 files are the minimum to use the system.
- **[train]**: needed only to rebuild the data, train or evaluate.
- **[opt-in]**: only runs when a config flag turns it on (e.g. `person.enabled=true`). It is still imported, so the
  file must exist.

---

## 1. How the system works

```
                    ┌──────────── offline: build the library and train (bgm audit … tune-stitch) ────────────┐
 dataset pairs ──►  data/        manifests, folds, silent 25 fps video, 48 kHz music, shots, segments
 (video + its BGM)  features/    frozen encoders: SigLIP (video), CLAP + MERT (music), motion curves, key
                    train/       semantic heads (small MLPs) trained with contrastive InfoNCE
                    retrieval/   fusion weights (fuse.py), Viterbi penalties (tune_stitch.py)
                    cue/         music-change predictor (2-layer Transformer)
                    └────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │  runs/…  (trained heads, fusion.json, stitch.json, cue models)
                                                        ▼
 silent video ──►  render/query.py   silent copy → shots → SigLIP 1 Hz → segments
                   retrieval/        rank music files (coarse) → slide 0.5 s windows (fine) → lattice of candidates
                   cue/hook.py       change probability at each segment boundary
                   retrieval/stitch  Viterbi: one window per segment, trading fit against continuity
                   render/render.py  crossfades, loudness, beat sync, limiter → mux → output.mp4
```

The code uses three conventions throughout:

- **One CLI.** Every stage is a `bgm <command>` in `cli.py`. Each command takes `--config <yaml>` plus dotted
  overrides (`fold=2 train.semantic.lr=3e-4`) and imports its stage module only when it runs.
- **Config, not hard-coded paths.** Functions receive the OmegaConf `cfg`, and every path comes from `cfg.paths.*`.
- **One writer per artefact.** Only `data/manifest.py` reads and writes the manifests. Features are cached once
  by `features/extract.py` and read through `eval/store.py`. Every training run writes `config.yaml`,
  `metrics.json` and the git commit (`runs.py`).

---

## 2. Folder structure

```
.
├── pyproject.toml            package definition; installs the `bgm` command          [render]
├── environment.yml           conda environment (all dependencies)                    [render]
├── configs/
│   ├── default.yaml          base config: paths, media, features, training, retrieval, stitching, render   [render]
│   └── best_semantic.yaml    current system: cleaned 1,011-pair dataset, title split [render]
├── render_best.sh            one-command render with the older film-split ensemble
├── tools/setup_external.sh   downloads ImageBind weights; clones TransNetV2 / BaSSL  [render]
├── tests/                    pytest suite + synthetic fixtures (optional)
└── src/bgm/
    ├── (core)       cli, config, log, runs, tools, doctor, smoke
    ├── data/        stage 1: dataset → manifests, folds, prepared media, shots, segments
    ├── features/    stage 2: frozen encoders → cached features
    ├── models/      trainable network modules
    ├── train/       stage 3: training the heads
    ├── retrieval/   stage 4: index, coarse-to-fine retrieval, fusion, Viterbi stitching
    ├── cue/         stage 5: learned music-change predictor
    ├── render/      stage 6: external video → soundtrack → output video
    ├── eval/        metrics, protocols, feature store, reports
    ├── person/      opt-in person-focused branch
    ├── pretrain/    self-supervised pretraining (stopped, still wired into the CLI)
    └── analysis/    3 of 80 files are needed (imported by render / CLI)
```

---

## 3. What each file does

### 3.1 Core (`src/bgm/`)

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `cli.py` | render | The `bgm` command (Typer). Defines every command: `doctor, smoke, audit, folds, prep, shots, peek, extract, check-features, segment, train, pretrain, fuse, tune-stitch, eval, render, brief, check, report` and the `bgm cue …` group. Each command loads the config and calls one stage module. |
| `config.py` | render | `load_config`: reads the YAML and applies dotted `key=value` overrides. |
| `log.py` | render | `setup_logging`: shared logging format. |
| `runs.py` | render | `run_dir`, `write_run`: creates a run folder and saves its config, metrics, git commit and seed. |
| `tools.py` | render | `find_tool`: finds `ffmpeg` / `ffprobe` / `fpcalc` on PATH or in `.checkpoints/bin/`. |
| `doctor.py` | train | `bgm doctor`: prints GPU, ffmpeg, fpcalc and package versions. |
| `smoke.py` | train | `bgm smoke --fake`: runs synthetic clips through the pipeline with fake encoders on CPU. |

### 3.2 `data/`: dataset → manifests and prepared media

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `manifest.py` | render | **The only reader and writer of the manifests**: `pairs.csv`, `tracks.csv`, `folds*.json`, `shots.csv`, `segments.csv`. Fixed column schemas and a manifest hash. |
| `media.py` | render | `ffprobe`: duration, fps, size and start time of a media file. |
| `prep.py` | render | `bgm prep`: makes silent 25 fps H.264 videos and A/V-aligned 48 kHz mono FLAC music (`interim/`). Render uses its path helpers. |
| `shots.py` | render | `bgm shots` / `bgm peek`: shot-cut detection (PySceneDetect) and contact sheets. |
| `segments.py` | render | `bgm segment`: merges shots into 2–20 s segments using SigLIP similarity. Render uses `make_segments` on the input video. |
| `films.py` | render | Film keys and film groups, so that clips of one film never span train and test. |
| `tracks.py` | render | Chromaprint fingerprinting: groups reused songs into track groups. |
| `themes.py` | render | Groups near-identical music cues of one film (MERT + CLAP similarity). |
| `audit.py` | train | `bgm audit`: scans the raw dataset and writes `pairs.csv`, `tracks.csv` and `reports/audit.md`. |
| `folds.py` | train | `bgm folds`: 5-fold split grouped by track or by film, with checks for leakage. |
| `near_dups.py` | train | Lists near-duplicate music files (CLAP cosine) for manual confirmation. |
| `alignment.py` | train | Alignment audit: checks that each pair's music is time-aligned with its video (cuts on beats above chance). |

### 3.3 `features/`: frozen encoders and cached features

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `encoders.py` | render | Interfaces `FrameEncoder`, `AudioEncoder`, `TextEncoder`, plus deterministic fakes for tests. |
| `hf.py` | render | The real frozen encoders: `SiglipEncoder`, `ClipEncoder`, `ClapEncoder`, `MertEncoder`, `ImageBindEncoder` (fp16, eval mode, no gradients). |
| `extract.py` | render | `bgm extract`: runs every encoder once per video and music file and caches the result in `features/`. `EncoderBank` and `frames_1hz` are reused by render. |
| `dynamics.py` | render | 25 Hz curves: video (motion, cuts, luma, camera motion) and music (onsets, beats, loudness, novelty). |
| `keys.py` | render | Key and tempo estimation, and the key distance used as a stitching cost. |
| `timing.py` | render | Timeline helpers (1 Hz and 25 Hz sample counts, upsampling). |
| `face.py` | render | 1 Hz facial-expression features (used by the person branch). |
| `video_temporal.py` | render | Optional temporal video encoder (V-JEPA) at 1 Hz. |
| `check.py` | train | `bgm check-features`: validates the cached features against the spec. |

### 3.4 `models/` and `train/`: the trained parts

| File | Needed for | What it does |
|---|---|---|
| `models/__init__.py` | render | Package marker. |
| `models/heads.py` | render | Trainable modules: `SemanticHead` (MLP that maps frozen features to a shared space), `AttnPool`, `DynamicsEncoder`, `AttributeMap`. |
| `train/__init__.py` | render | Package marker. |
| `train/semantic.py` | render | `bgm train semantic`: trains the semantic heads (video ↔ music) with symmetric InfoNCE. Render loads trained heads through its `SemanticModel` / `Scorer`. |
| `train/semantic_tokens.py` | render | Variant with temporal tokens and late interaction (off by default). |
| `train/losses.py` | render | Contrastive losses: InfoNCE variants, sigmoid loss, hard-negative, debiased, distillation, dynamics loss. |
| `train/dynamics.py` | render | `bgm train dynamics`: per-step dynamics encoders for video–music sync (not used in the final score). |
| `train/shifts.py` | render | Time-shifted negatives for dynamics training. |
| `train/attr_distill.py` | render | Optional auxiliary loss: the head also predicts the music's mood and energy profile. |
| `train/baselines.py` | train | `bgm train baseline --variant cfd\|utcmvmr\|mvpt`: published methods reimplemented on the same features, for comparison. |

### 3.5 `retrieval/`: finding and stitching music

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `index.py` | render | `MusicIndex` (per-file embeddings and prefix sums for O(1) window scores), `VideoQuery` / `SegmentQuery`, `EnsembleSemantic` (the 10-head ensemble as one mean-cosine scorer), `FoldModels` (loads a run). |
| `pipeline.py` | render | Coarse-to-fine retrieval: ranks whole files, slides windows over the shortlisted files, standardises and fuses the branch scores, and builds candidates per segment. |
| `stitch.py` | render | Core stitching: candidates, continuations, `transition_cost`, `viterbi`, `greedy`. |
| `stitch_eval.py` | render | Builds the candidate lattice and decodes it (`build_lattice`, `decode_lattice`); also the concatenated test videos and switch F1. |
| `chunked.py` | render | Long videos (≥ 60 s): split into chunks, score tracks per chunk, ensemble with z-scores. |
| `fuse.py` | train | `bgm fuse`: tunes branch fusion weights on val and measures pipeline accuracy on test. Writes `fusion.json`. |
| `tune_stitch.py` | train | `bgm tune-stitch`: tunes the Viterbi switch / jump / key penalties on val and compares Viterbi with greedy on test. Writes `stitch.json`. |

### 3.6 `cue/`: learned music-change predictor

At each boundary between two segments, this predicts how likely the music is to change. That probability replaces
Viterbi's fixed switch cost.

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `model.py` | render | `CueModel`: segment tokens → 2-layer Transformer → change logit per boundary. |
| `tokens.py` | render | Segment tokens: mean SigLIP, log duration, motion mean and std. |
| `hook.py` | render | **Render hook**: runs the trained cue models on the input video's segments and decodes with their change probabilities (`cue.enabled=true`, on by default). |
| `stitch.py` | render | Viterbi with per-boundary costs from the predicted probability. |
| `train.py` | render | `bgm cue train`: trains the predictor per fold × seed × variant; `load_cue` is used by render. |
| `sequences.py` | render | Builds sequences of joined clips with change / no-change labels (training and test sets). |
| `audio_teacher.py` | render | Training targets read from the music itself (MERT distance across a boundary). |
| `cuts.py` | render | Features and labels for music change at shot cuts inside a clip. |
| `cut_model.py` | render | Trains and evaluates the change-at-cut classifier. |
| `cut_hook.py` | opt-in | The change-at-cut classifier inside Viterbi (`cue.cuts_render.enabled`). |
| `baselines.py` | train | R2 baseline: logistic rule on the SigLIP cosine between neighbouring segments. |
| `scenes.py` | train | R1 baseline: PySceneDetect scene cuts as switch points. |
| `evaluate.py` | train | `bgm cue eval`: all predictors (R0–R6) on frozen val / test sequences. |
| `report.py` | train | `bgm cue report`: aggregates into `results/cue_switch.md` with bootstrap CIs. |
| `probes.py` | train | `bgm cue probe`: checks whether the predictor finds music changes or only clip joins. |
| `within_clip.py` | train | Test on music changes inside single clips (no joins). |
| `real_eval.py` | train | `bgm cue real-eval`: boundary metrics on hand-annotated real excerpts. |

### 3.7 `render/`: from an input video to an output video

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `run.py` | render | **Entry point of `bgm render`.** `render_video`: featurize → `plan_for` (retrieve, cue, Viterbi) → beat sync → render soundtrack → mux. Writes `renders/<name>.plan.json`. |
| `query.py` | render | `featurize`: silent copy of the input, then shots, SigLIP 1 Hz, mood logits, ImageBind, motion curves and segments. Cached in `renders/<name>/query.npz`. |
| `render.py` | render | Audio rendering: groups segments into runs, beat-syncs and loops pieces, equal-power crossfades, loudness normalisation, limiter, and `mux` (with A/V sync correction). |
| `branches.py` | render | Extra render-only scoring layers wrapped around a `MusicIndex`. |
| `pace.py` | opt-in | Pace match: fast-cut video → energetic music. |
| `scene.py` | opt-in | Scene brief: a vision-language model (Qwen2.5-VL) describes each segment, and the text guides the music choice (`bgm brief`). |
| `replay.py` | render | Penalty against replaying the same passage of a file again and again. |

### 3.8 `eval/`: feature access, metrics and reports

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker. |
| `store.py` | render | `FeatureStore`: in-memory access to the cached features with O(1) span means (the span rule). Used by training, retrieval and render. |
| `metrics.py` | render | Recall@K, MRR, chance levels, cut-on-beat rate, cluster bootstrap CIs. |
| `protocol.py` | render | Evaluation protocols: track retrieval, segment retrieval, grounding. |
| `attributes.py` | render | Mood-attribute calibration and score. |
| `report_filmsplit.py` | render | Film-split report tables. |
| `run.py` | train | `bgm eval --exp`: evaluates an experiment over 5 folds and aggregates it. |
| `zeroshot.py` | train | Zero-shot references: ImageBind, text bridge, hand-crafted dynamics. |
| `check.py` | train | `bgm check`: reference-free quality check of a render against random beat-snapped music. |
| `final.py` | train | `bgm report --final --split trackid`. |
| `final_film.py` | train | `bgm report --final`: the paper tables on the film split. |
| `report_pipeline_filmsplit.py` | train | Pipeline and stitching tables on the film split. |
| `report_dynamics.py` | train | Dynamics-branch tables. |
| `rescore_grounding.py` | train | `bgm eval --exp grounding`: re-scores saved predictions with chance levels. |

### 3.9 `person/`: person-focused branch (opt-in, `person.enabled=true`)

| File | Needed for | What it does |
|---|---|---|
| `__init__.py` | render | Package marker and description. |
| `render.py` | opt-in | Render hook for a person-branch run. |
| `pipeline.py` | opt-in | `PersonVideoQuery` / `PersonMusicIndex`: add the person score to retrieval and stitching. |
| `features.py` | opt-in | Face / person crops at 1 Hz, encoded with SigLIP. |
| `train.py` | opt-in | Trains the person-branch heads. |
| `evaluate.py` | opt-in | Shared per-query metrics for every person-branch system. |
| `diagnose.py` | opt-in | Baseline system on "same setting, different music" subsets. |
| `subset.py` | opt-in | Builds those subsets. |

### 3.10 `pretrain/` and `analysis/` (needed only because the CLI and render import them)

| File | Needed for | What it does |
|---|---|---|
| `pretrain/__init__.py`, `train.py`, `unlabeled.py` | train | `bgm pretrain`: self-supervised pretraining on unlabeled clips (stopped, D28). |
| `analysis/__init__.py` | render | Package marker. |
| `analysis/long_video.py` | render | Long-video helpers; render imports its model loading for the chunked shortlist. |
| `analysis/full_dataset.py` | render | `load_scorer` / `CachedScorer`, imported by the person branch and the pipeline report. |
| `analysis/arch_phase0.py` | train | `film_cfg`, imported by `eval/report_pipeline_filmsplit.py`. |

---

## 4. Which files run for each command

| Command | Main files |
|---|---|
| `bgm render` | `cli` → `render/run` → `render/query` (+ `features/extract`, `hf`, `dynamics`, `data/shots`, `data/segments`) → `retrieval/index`, `pipeline`, `stitch_eval`, `stitch`, `chunked` → `cue/hook`, `cue/model`, `cue/stitch` → `render/render` |
| `bgm audit / folds / prep / shots / segment` | `data/audit`, `folds`, `prep`, `shots`, `segments`, `manifest` |
| `bgm extract` | `features/extract`, `hf`, `encoders`, `dynamics`, `keys` |
| `bgm train semantic` | `train/semantic`, `models/heads`, `train/losses`, `eval/store` |
| `bgm fuse` / `tune-stitch` | `retrieval/fuse`, `retrieval/tune_stitch`, `retrieval/pipeline`, `stitch_eval` |
| `bgm cue …` | `cue/train`, `evaluate`, `report`, `sequences`, `tokens` |
| `bgm eval` / `report` | `eval/run`, `final_film`, `metrics`, `protocol` |

---

## 5. Not needed to run the system

Removing any of these does not break `bgm`:

| Path | What it is |
|---|---|
| `src/bgm/analysis/` (77 other files) | one-off scripts for paper tables, figures and diagnostics (`python -m bgm.analysis.<name>`) |
| `src/bgm/sweep/` | loss and similarity sweeps on the title split |
| `src/bgm/tagging/` | separate LLM music / scene tagging study (D141). **The system is trained and run without tags**: no file in training, retrieval, the cue predictor or rendering reads them. |
| `src/bgm/rerank/` | re-ranking and hubness study |
| `src/bgm/segmentation/` | scene-segmentation study (TransNetV2, BaSSL) |
| `src/bgm/e2e/` | end-to-end CRF study |
| `src/bgm/listening/` | listening-test kits |
| `src/bgm/losses/` | loss registry for the sweeps (the training losses are in `train/losses.py`) |
| `exp_supp/`, `stitching/` | supplementary paper experiments and the decoder study |
| `results/`, `docs/` | result tables and documentation |
| `tools/` (except `setup_external.sh`) | dataset building and download scripts |

Beyond the code, rendering also needs the music library (`cleaned_dataset/`: features, manifests and interim
music), the trained runs (`runs/cleaned/`) and the ImageBind weights (`.checkpoints/imagebind_huge.pth`). None of
these are in the repository; see the README.
