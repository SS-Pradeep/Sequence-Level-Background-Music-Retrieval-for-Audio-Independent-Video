# Video-to-BGM Retrieval

Give it a silent video and it adds a continuous background-music (BGM) soundtrack, picked from a fixed music library.

How it works:

1. Cut the video into segments of 2–20 s.
2. Rank the library's music files for each segment, then slide 0.5 s windows over the top-ranked files to find the best part of each.
3. Predict, at each segment boundary, how likely the music is to change. This uses a small Transformer.
4. Choose one music window per segment with a Viterbi pass. It balances how well the music fits each segment against keeping the same track going.
5. Render the soundtrack (crossfades, loudness normalisation, piece starts moved onto beats) and mux it onto the video.

Final system: a 10-head ensemble of semantic heads. Video frames go through SigLIP and music through CLAP + MERT; all
encoders are frozen and only the heads are trained. It uses the cleaned 1,011-pair dataset, the title-level split,
Viterbi stitching and the learned change predictor. The method and results are written up in
[docs/PROJECT_BRIEFING.md](docs/PROJECT_BRIEFING.md) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and every design choice is logged in
[docs/PROGRESS.md](docs/PROGRESS.md) (entries D1–D154).

---

## 1. Requirements

- Linux with an NVIDIA GPU (the measurements in this repo come from an RTX 4080 SUPER, 16 GB). Feature extraction and
  training expect CUDA. Tests and `bgm smoke --fake` also run on CPU.
- Conda (or mamba).
- `ffmpeg` / `ffprobe` (installed by the conda environment).
- `fpcalc` (Chromaprint), used only for audio fingerprinting during the data audit: `sudo apt install libchromaprint-tools`.
- Disk: the full working set (datasets, features, runs) is about 270 GB. Only the code is stored in git.

## 2. Install

```bash
git clone https://github.com/SS-Pradeep/<repo-name>.git && cd <repo-name>
conda env create -f environment.yml        # creates the `bgm` env (Python 3.11, torch, transformers, librosa, ...)
conda activate bgm
pip install -e . --no-deps                 # installs the `bgm` command; dependencies come from environment.yml
bgm doctor                                 # prints GPU, ffmpeg, ffprobe, fpcalc and package versions
```

The scripts in this repo call `.venv/bin/bgm`, so if your environment lives in `.venv/`, that path works as is.
Otherwise, use `bgm` from the activated conda env.

Hugging Face models download on first use: SigLIP so400m, CLIP ViT-L/14, CLAP `larger_clap_music_and_speech` and
MERT-v1-330M. Some parts need files that are not in git:

| Path | Needed for |
|---|---|
| `.checkpoints/` | ImageBind weights (`features.imagebind`, zero-shot baseline) |
| `third_party/` | TransNetV2 and BaSSL clones (`shots.detector: transnetv2`, cue / segmentation studies) |

## 3. Check the install (no data needed)

```bash
pytest -q                     # unit tests on synthetic fixtures; add `-m "not slow"` to skip the training checks
bgm smoke --fake              # runs synthetic clips end to end with fake encoders on CPU
```

## 4. Data layout (not in git)

Every path comes from `paths:` in the config. Two configs are provided:

| Config | Dataset | Manifests / features | Runs |
|---|---|---|---|
| [configs/default.yaml](configs/default.yaml) | original 1,248 pairs (`data/raw` → `dataset_merged/`) | `data/manifests`, `data/features` | `runs/` |
| [configs/best_semantic.yaml](configs/best_semantic.yaml) | **cleaned 1,011 pairs (current system)** | `cleaned_dataset/manifests`, `cleaned_dataset/features` | `runs/cleaned/` |

`cleaned_dataset/` is built from the pair trimmer's output with `python tools/build_cleaned.py`, which also writes its
manifests and `folds_title.json`.

Generated outputs go to `results/` (tables), `reports/`, `renders/` and `runs/`.

## 5. Add music to a video

### With the current system (cleaned data, title split)

You need the trained runs under `runs/cleaned/` (`semantic/`, `pipeline_ens10/`, `cue/`) and the cleaned features.

```bash
bgm render --config configs/best_semantic.yaml \
  --video input.mp4 --out output.mp4 \
  --run runs/cleaned/pipeline_ens10/fold0/seed0 \
  pipeline.dyn_variant=null "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"
```

### With the older film-split system (one command)

```bash
./render_best.sh input.mp4 output.mp4 [overrides...]     # e.g. render.audio_bitrate=320k
```

This loads `runs/filmsplit/pipeline_ens10/fold0/seed0`, a 10-head ensemble trained on the original 1,248 pairs.

### What a render produces

- `output.mp4`: the input picture with the new soundtrack. Any audio already in the input is replaced.
- `renders/<name>.plan.json`: the segments, the music file and offset chosen for each segment, the penalties and the runs used.
- `renders/<name>/`: the work folder (silent 25 fps copy, features, `bgm.wav`).

### Related commands

```bash
bgm brief --video input.mp4 --out output.mp4     # draft an editable scene brief without rendering
bgm check --plan renders/output.plan.json        # check a render without a reference soundtrack
```

### Useful render overrides

| Override | Effect |
|---|---|
| `cue.enabled=false` | fixed Viterbi switch penalties instead of the learned change predictor |
| `render.audio_bitrate=320k` | higher-bitrate AAC (default 192k) |
| `render.beat_sync=false` | turn off moving piece starts onto beats |
| `person.enabled=true` | person-focused branch (needs its own run, see [docs/person_branch_plan.md](docs/person_branch_plan.md)) |

Videos of 60 s or more add a chunked shortlist from every fold's heads, so they take longer.

## 6. Train the full pipeline

Every command reads `--config` (default `configs/default.yaml`) followed by dotted `key=value` overrides. For the
current system, pass `--config configs/best_semantic.yaml` to each step. `--grid` runs all 5 folds × `train.seeds`.
Without it, a step runs only the config's `fold` and `seed`.

```bash
CFG="--config configs/best_semantic.yaml"

# Data (writes only to the manifests folder)
bgm audit $CFG                       # pairs.csv, tracks.csv, reports/audit.md
bgm folds $CFG --film --out folds_title.json   # film/title-level 5-fold split (cleaned data: made by build_cleaned.py)
bgm prep $CFG                        # silent 25 fps videos + aligned 48 kHz music
bgm shots $CFG                       # shot detection -> shots.csv
bgm peek $CFG <pair_id>              # optional: contact sheet of one pair's shots

# Features (GPU)
bgm extract $CFG                     # SigLIP, CLIP, CLAP, MERT, motion, attributes, ImageBind, key
bgm check-features $CFG              # validate the cache
bgm segment $CFG                     # shots -> segments.csv, then the alignment audit

# Models
bgm train semantic $CFG --grid       # semantic heads (seeds 0-9 are needed for the 10-head ensemble)
bgm fuse $CFG --grid                 # fusion weights on val, pipeline accuracy on test
bgm tune-stitch $CFG --grid          # Viterbi penalties on val, Viterbi vs greedy on test

# Learned music-change predictor (used by render while cue.enabled=true)
bgm cue build $CFG
bgm cue train $CFG
bgm cue eval $CFG --fold 0
bgm cue report $CFG

# Evaluation and tables
bgm eval $CFG --exp zeroshot
bgm report $CFG --final
```

Notes:

- The ensemble pipeline is the `fuse` / `tune-stitch` step run with
  `pipeline.exp=pipeline_ens10 pipeline.dyn_variant=null "pipeline.sem_seeds=[0,...,9]"`.
- `bgm train … --all` (deployment models trained on every pair) is not built. `bgm render` uses one fold's run, chosen
  with `--run`.
- Training cost of the final system: about 34 min for the 5-fold ensemble on one GPU, plus about 27 min of feature
  extraction. Inference takes about 7.7 s per video, roughly 2× real time
  ([results/cleaned/compute_cost.md](results/cleaned/compute_cost.md)).

## 7. Repository layout

```
.
├── configs/              YAML configs
│   ├── default.yaml          original 1,248-pair dataset
│   └── best_semantic.yaml    cleaned 1,011-pair dataset (current system)
├── docs/                 spec, methodology, decision log (PROGRESS.md), plans, inventory/
├── exp_supp/             supplementary paper experiments (re-rankers, HSMM, shot-cut probes) and their tables
├── results/              result tables (Markdown / CSV / LaTeX); results/cleaned/ = current system
├── src/bgm/              the Python package; one subpackage per pipeline stage
│   ├── cli.py                the `bgm` command (Typer)
│   ├── config.py, log.py, runs.py, doctor.py, smoke.py
│   ├── data/                 manifests, audit, folds, prep, shots, segments (the only code that writes manifests)
│   ├── features/             frozen encoders and feature extraction
│   ├── segmentation/         scene segmentation heads and evaluation
│   ├── train/ losses/ models/    semantic heads, dynamics encoders, baselines, loss registry
│   ├── retrieval/            index, coarse-to-fine retrieval, fusion, Viterbi stitching
│   ├── cue/                  learned music-change predictor
│   ├── render/               query featurization, plan, beat sync, mixing and muxing
│   ├── eval/ sweep/ analysis/    metrics, protocols, sweeps, paper tables and figures
│   └── rerank/ person/ tagging/ e2e/ pretrain/ listening/   studies and opt-in branches
├── stitching/            decoder study (CRF, two-stage, semi-Markov, segmental Viterbi), with its own tests/
├── tests/                pytest suite (synthetic fixtures in tests/fixtures/)
├── tools/                one-off scripts (cleaned-dataset builder, pair trimmer, downloads)
├── environment.yml       conda environment (all dependencies)
├── pyproject.toml        package definition; installs the `bgm` command
├── render_best.sh        one-command render with the film-split ensemble
└── README.md
```

Folders the code expects but that are not in this repository (create them, or point `paths:` in the config at
them): `data/`, `cleaned_dataset/`, `runs/`, `reports/`, `renders/`, `.checkpoints/`, `third_party/`.

## 8. Further reading

- [docs/SPEC.md](docs/SPEC.md): the build spec. CLI docstrings point to its sections.
- [docs/PROGRESS.md](docs/PROGRESS.md): decision log with every result. Look up a `Dxxx` number there.
- [docs/inventory/CURRENT_VERSION.md](docs/inventory/CURRENT_VERSION.md): which data and runs make up the current version.
- [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md): open items.
