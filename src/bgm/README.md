# src/bgm/

The `bgm` package: video-to-background-music retrieval. The build specification is
[`docs/SPEC.md`](../../docs/SPEC.md); `D<number>` in comments refers to [`docs/PROGRESS.md`](../../docs/PROGRESS.md).

## Top-level modules

| File | What it does |
|---|---|
| `cli.py` | The `bgm` command. Every subcommand loads a config and calls one stage. |
| `config.py` | Loads a YAML config and applies dotted `key=value` overrides. |
| `runs.py` | Creates run folders and saves each run's config, metrics, git commit and seed. |
| `log.py` | Shared logging setup. |
| `tools.py` | Finds `ffmpeg`, `ffprobe` and `fpcalc`; defines the repository root. |
| `doctor.py` | `bgm doctor`: prints GPU, tool and package versions. |
| `smoke.py` | `bgm smoke --fake`: runs synthetic clips through the pipeline on CPU with fake encoders. |

## Subpackages

**Used by the final system**

| Folder | Stage |
|---|---|
| [`data/`](data/) | Dataset audit, manifests, media preparation, shots, segments, folds. |
| [`features/`](features/) | Frozen encoders and cached features (SigLIP, CLAP, MERT, ...). |
| [`models/`](models/) | The small trainable heads. |
| [`train/`](train/) | Training the semantic heads. |
| [`eval/`](eval/) | Feature store, metrics, evaluation protocols, reports. |
| [`retrieval/`](retrieval/) | Coarse-to-fine retrieval, fusion and Viterbi stitching. |
| [`cue/`](cue/) | Learned music-change predictor. |
| [`render/`](render/) | Silent video in → video with background music out. |
| [`analysis/`](analysis/) | Analysis scripts, figures and tables (only `long_video.py` is used by the final system). |

**Experiments (not in the final system)**

| Folder | Topic |
|---|---|
| [`losses/`](losses/) | Alternative loss functions for the loss sweep. |
| [`sweep/`](sweep/) | Loss-function sweep runner and configs. |
| [`segmentation/`](segmentation/) | Segmentation study (TransNetV2 + BaSSL + boundary head). |
| [`rerank/`](rerank/) | Bidirectional retrieval and hubness re-scoring. |
| [`e2e/`](e2e/) | End-to-end CRF training of heads and transitions. |
| [`person/`](person/) | Person/face-focused scoring branch. |
| [`pretrain/`](pretrain/) | Self-supervised pretraining on unlabeled clips (M8, stopped). |
| [`listening/`](listening/) | Listening-study kit and analysis (M12). |

`a.txt` is a stray placeholder file and is not used by any code.
