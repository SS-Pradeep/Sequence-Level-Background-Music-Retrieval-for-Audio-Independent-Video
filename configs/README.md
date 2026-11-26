# configs/

YAML configuration files read by every `bgm` command (`--config <file>`). Any value can be overridden on the command
line with a dotted key, e.g. `bgm train semantic fold=2 seed=1 train.semantic.lr_finetune=3e-4`.

| File | What it is |
|---|---|
| `default.yaml` | Base settings for every stage: paths, media format, encoders, segmentation, training, retrieval, stitching, rendering and the cue (music-change) predictor. |
| `best_semantic.yaml` | **Final system.** `default.yaml` plus the cleaned 1,011-pair dataset (`cleaned_dataset/`), the title-level split and run folder `runs/cleaned/`. |

Pass `--config configs/best_semantic.yaml` to every command when working with the final system.
Loss-sweep configs live separately in [`src/bgm/sweep/configs/`](../src/bgm/sweep/configs/).
