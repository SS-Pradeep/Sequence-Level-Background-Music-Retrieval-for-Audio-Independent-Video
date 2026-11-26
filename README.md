# Code structure: the final system

This document covers only the files that make up the **final version** of the system. The repository also holds
earlier experiments, baselines and paper scripts. They are not needed and are not described here.

**Final system:**

- Data: the cleaned dataset of 1,011 video–music pairs, with a 5-fold split by film title.
- Scoring: a 10-head ensemble of semantic heads. Video frames are encoded with SigLIP and music with CLAP + MERT;
  all encoders are frozen and only the heads are trained. Only the semantic score is used.
- Retrieval: coarse-to-fine. Whole music files are ranked first, then 0.5 s windows are slid over the top files.
- Music changes: a 2-layer Transformer predicts how likely the music is to change at each segment boundary.
- Stitching and rendering: Viterbi picks one window per segment, then the soundtrack is rendered with beat sync,
  crossfades and loudness normalisation.

**60 Python files run for the final system.** This was checked by tracing the imports of every command the system
uses, and the set is self-contained: it imports nothing outside these files.

---

## 1. Pipeline

```
 TRAINING (once)                                              RENDERING (per video)

 cleaned dataset (1,011 pairs)                                input.mp4 (silent)
        │                                                          │
        ▼                                                          ▼
 data/       prepare media, shots, segments, folds           render/query.py   silent copy → shots → SigLIP → segments
        │                                                          │
        ▼                                                          ▼
 features/   SigLIP (video), CLAP + MERT (music)              retrieval/        rank files → slide windows → candidates
        │                                                          │
        ▼                                                          ▼
 train/      10 semantic heads per fold (InfoNCE)             cue/hook.py       change probability at each boundary
        │                                                          │
        ▼                                                          ▼
 retrieval/  fuse.py (fusion.json)                            retrieval/stitch  Viterbi: one window per segment
             tune_stitch.py (Viterbi penalties)                    │
        │                                                          ▼
        ▼                                                     render/render.py  beat sync, crossfades, loudness → mux
 cue/        change predictor (train + eval)                       │
        │                                                          ▼
        └──────────► runs/cleaned/ ───────────────────────────► output.mp4
```

---

## 2. Files in the final system

```
.
├── pyproject.toml                 installs the package and the `bgm` command
├── environment.yml                all dependencies (conda)
├── configs/
│   ├── default.yaml               base settings for every stage
│   └── best_semantic.yaml         final system: cleaned dataset paths, title split, runs/cleaned
├── tools/
│   ├── setup_external.sh          downloads the ImageBind weights
│   └── build_cleaned.py           builds cleaned_dataset/ and its manifests
└── src/bgm/
    ├── __init__.py  cli.py  config.py  log.py  runs.py  tools.py
    ├── data/        __init__  manifest  media  prep  shots  segments  folds  films  audit  tracks  near_dups
    ├── features/    __init__  extract  encoders  hf  dynamics  keys  timing
    ├── models/      __init__  heads
    ├── train/       __init__  semantic  losses  dynamics  shifts
    ├── eval/        __init__  store  metrics  protocol  attributes
    ├── retrieval/   __init__  index  pipeline  stitch  stitch_eval  chunked  fuse  tune_stitch
    ├── cue/         __init__  tokens  sequences  model  train  stitch  hook  evaluate  baselines  scenes
    ├── render/      __init__  run  query  render
    └── analysis/    __init__  long_video
```

Every subfolder's `__init__.py` is an empty package marker and must be present.

---

## 3. What each file does

### 3.1 Root and configs

| File | What it does |
|---|---|
| `pyproject.toml` | Package definition. `pip install -e . --no-deps` installs the `bgm` command. |
| `environment.yml` | Conda environment: Python 3.11, PyTorch, transformers, librosa, ffmpeg and the other dependencies. |
| `configs/default.yaml` | Base settings for every stage: paths, media format, encoders, training, retrieval, stitching, render, cue predictor. |
| `configs/best_semantic.yaml` | Final-system config: points at `cleaned_dataset/` and `runs/cleaned/` and uses the title split. |
| `tools/setup_external.sh` | Downloads the ImageBind weights to `.checkpoints/`. Rendering needs them. |
| `tools/build_cleaned.py` | Builds `cleaned_dataset/` (1,011 pairs) from the hand-trimmed pairs, and writes its manifests and title-split folds. |

### 3.2 Core (`src/bgm/`)

| File | What it does |
|---|---|
| `cli.py` | The `bgm` command. Each subcommand (`prep`, `extract`, `train`, `fuse`, `tune-stitch`, `cue …`, `render`) loads the config and calls one stage. |
| `config.py` | Loads a YAML config and applies `key=value` overrides from the command line. |
| `log.py` | Shared logging setup. |
| `runs.py` | Creates run folders and saves each run's config, metrics, git commit and seed. |
| `tools.py` | Finds `ffmpeg`, `ffprobe` and `fpcalc`, and defines the repository root. |

### 3.3 `data/`: preparing the dataset

| File | What it does |
|---|---|
| `manifest.py` | The only code that reads and writes the manifests (`pairs.csv`, `tracks.csv`, folds, `shots.csv`, `segments.csv`). |
| `media.py` | `ffprobe`: duration, frame rate, size and start time of a video or audio file. |
| `prep.py` | `bgm prep`: silent 25 fps videos and time-aligned 48 kHz mono music files. |
| `shots.py` | `bgm shots`: detects shot cuts (PySceneDetect). Also used on the input video at render time. |
| `segments.py` | `bgm segment`: merges shots into 2–20 s segments by SigLIP similarity. Also used at render time. |
| `folds.py` | Builds and checks the 5-fold split, so that no film appears in both train and test. |
| `films.py` | Groups clips by film; this is the unit of the split. |
| `audit.py` | `bgm audit`: scans the raw dataset and writes the first manifests. |
| `tracks.py` | Audio fingerprinting (Chromaprint): finds songs reused across clips so they stay in one fold. |
| `near_dups.py` | Reads the manually confirmed near-duplicate music links that the audit merges into track groups. |

### 3.4 `features/`: frozen encoders

| File | What it does |
|---|---|
| `extract.py` | `bgm extract`: runs each encoder once per video and music file and caches the features. At render time it samples the input video's frames (1 per second) and loads the encoders. |
| `hf.py` | The frozen pretrained encoders: SigLIP (video), CLAP and MERT (music), ImageBind. |
| `encoders.py` | Common encoder interfaces, plus fake encoders for tests. |
| `dynamics.py` | Motion and cut curves of a video (25 per second); render stores them with the input's features. |
| `keys.py` | Musical key and tempo of a music window; used in the stitching cost between two pieces. |
| `timing.py` | Timeline helpers: sample counts at 1 and 25 per second. |

### 3.5 `models/`, `train/`, `eval/`: training the semantic heads

| File | What it does |
|---|---|
| `models/heads.py` | `SemanticHead`: the small MLP (LayerNorm → 512 → 256) that maps frozen video or music features into a shared space. One head per side. |
| `train/semantic.py` | `bgm train semantic`: trains the video and music heads with symmetric InfoNCE on matching segments; saves a checkpoint per fold and seed. The 10-head ensemble is seeds 0–9. |
| `train/losses.py` | The contrastive loss (`info_nce`) and its variants. |
| `train/dynamics.py`, `train/shifts.py` | The dynamics model class. The retrieval index imports it, but the final system doesn't use it (`pipeline.dyn_variant=null`). |
| `eval/store.py` | `FeatureStore`: loads cached features and gives fast averages over any time span. Used by training, retrieval and render. |
| `eval/metrics.py` | Recall@K, ranks and confidence intervals; used to pick the best epoch and in tuning. |
| `eval/protocol.py` | Track and segment retrieval evaluation, run on validation during training. |
| `eval/attributes.py` | Mood-attribute calibration. The retrieval index imports it; its weight is zero in the final system. |

### 3.6 `retrieval/`: finding and stitching music

| File | What it does |
|---|---|
| `index.py` | `MusicIndex`: every library file's head embeddings, with prefix sums so any window's score is computed in O(1). `EnsembleSemantic` combines the 10 heads into one score (mean cosine). `FoldModels` loads a trained run. |
| `pipeline.py` | Coarse-to-fine retrieval: shortlist the top K files for the whole video, then score 0.5 s sliding windows of those files for each segment. |
| `stitch_eval.py` | Builds the lattice of candidate windows per segment and decodes it with Viterbi. |
| `stitch.py` | The Viterbi and greedy decoders, and the transition cost (switch, jump, key change). |
| `chunked.py` | Videos of 60 s or more: scores music per chunk of the video and adds the best files to the shortlist. |
| `fuse.py` | `bgm fuse`: on validation, sets the branch weights (semantic only in the final system) and saves them with the shortlist size K, candidates per segment J and window hop in `fusion.json`. Render reads it. |
| `tune_stitch.py` | `bgm tune-stitch`: tunes the Viterbi penalties on validation and writes `stitch.json`. Render reads it. |

### 3.7 `cue/`: music-change predictor

| File | What it does |
|---|---|
| `tokens.py` | One feature vector per segment: mean SigLIP, log duration, motion mean and spread. |
| `sequences.py` | Joins dataset clips into 60–180 s training and test sequences, labelling each boundary as change or no change. |
| `model.py` | `CueModel`: segment tokens → 2-layer Transformer → change probability for each boundary. |
| `train.py` | `bgm cue train`: trains the predictor (variant `ctx`, seeds 0–2 per fold) and loads it at render time. |
| `stitch.py` | Viterbi in which each boundary's switch cost comes from the predicted change probability. |
| `evaluate.py` | `bgm cue eval`: tunes how strongly the probabilities are used on validation and saves the values that render reads. |
| `baselines.py`, `scenes.py` | Reference predictors that `cue eval` compares against. `bgm cue build` also caches the scene scores they need. |
| `hook.py` | **Render step**: runs the trained predictors on the input video's segments and decodes the lattice with their probabilities. |

### 3.8 `render/`: from input video to output video

| File | What it does |
|---|---|
| `run.py` | **Entry point of `bgm render`.** Featurizes the video, loads the run (heads, `fusion.json`, `stitch.json`), retrieves candidates, runs the cue predictor and Viterbi, renders the audio and muxes it. Writes `renders/<name>.plan.json`. |
| `query.py` | Featurizes the input video exactly like the dataset: silent copy, shots, SigLIP once per second, segments. Cached in `renders/<name>/`. |
| `render.py` | Builds the soundtrack: groups segments into pieces, moves piece starts onto beats, equal-power crossfades, loudness normalisation, a limiter, and `mux` onto the video with A/V sync correction. |
| `analysis/long_video.py` | Loads every fold's heads for the long-video shortlist (`chunked.py`). |

---

## 4. Commands and the files they use

| Step | Command | Files |
|---|---|---|
| Build the cleaned dataset | `python tools/build_cleaned.py` | `data/manifest`, `data/folds` |
| Prepare media | `bgm prep` / `bgm shots` | `data/prep`, `data/shots`, `data/media` |
| Extract features | `bgm extract` | `features/*` |
| Make segments | `bgm segment` | `data/segments` |
| Train heads | `bgm train semantic --grid` | `train/semantic`, `models/heads`, `train/losses`, `eval/*` |
| Tune retrieval | `bgm fuse --grid` | `retrieval/fuse`, `index`, `pipeline` |
| Tune Viterbi | `bgm tune-stitch --grid` | `retrieval/tune_stitch`, `stitch_eval`, `stitch` |
| Change predictor | `bgm cue build`, `cue train`, `cue eval` | `cue/*` |
| **Add music to a video** | `bgm render --config configs/best_semantic.yaml --video in.mp4 --out out.mp4 --run runs/cleaned/pipeline_ens10/fold0/seed0 …` | `render/*`, `retrieval/*`, `cue/hook`, `cue/model`, `cue/stitch` |

Pass `--config configs/best_semantic.yaml` to every command.

---

## 5. Needed besides the code (not in the repository)

| What | Path |
|---|---|
| Music library: features, manifests, prepared music | `cleaned_dataset/` |
| Trained runs: heads, `fusion.json`, `stitch.json`, cue models | `runs/cleaned/semantic/`, `runs/cleaned/pipeline_ens10/`, `runs/cleaned/cue/` |
| ImageBind weights | `.checkpoints/imagebind_huge.pth` (from `tools/setup_external.sh`) |
