# results/

Result tables written by the analysis and report scripts. Each file is generated; to change one, re-run the script
named at its top (most are `python -m bgm.analysis.<name>` or `bgm report ...`). `D<number>` refers to the decision
log entry in [`docs/PROGRESS.md`](../docs/PROGRESS.md).

Final-system results on the cleaned dataset are under `results/cleaned/` (written by `configs/best_semantic.yaml`, not
stored in git); the title-split results are in [`titlesplit/`](titlesplit/).

## Dataset

| File | Contents |
|---|---|
| `dataset_stats.md` | Statistics of the cleaned 1,011-pair dataset (D115). |
| `bgm_presence.md` | How much of each music track is real background music (PANNs audit, D114). |
| `full_dataset.md` | Semantic heads on the full 4,246-pair dataset. |

## Retrieval and heads

| File | Contents |
|---|---|
| `final_v1.md` | First full results report. |
| `semantic_siglip-clap.md`, `semantic_clip-clap.md` | Semantic heads with SigLIP or CLIP on the video side. |
| `learning_curve.md` | Recall vs amount of training data (film split). |
| `pretraining_filmsplit.md` | Unlabeled pretraining on the film split (H4). |
| `arch_ablation.md` | Architecture ablation ledger. |
| `qkv.md` | One learned cross-attention layer before the heads (D110). |
| `dynamics.md` | Learned vs hand-crafted dynamics, with/without shifted negatives (M7, H2). |
| `tagging.md` | LLM tags and tag-overlap retrieval (D141). |
| `person_branch.md` | Person/face-focused branch (D87+). |
| `query_1_grounding.md` | Grounding with chance levels on one external video. |

## Pipeline, stitching and rendering

| File | Contents |
|---|---|
| `pipeline_full.md`, `pipeline_full_ens10.md` | Coarse-to-fine retrieval and stitching on the full dataset (single model / 10-seed ensemble). |
| `cue_switch_audio.md`, `cue_switch_splice.md`, `cue_switch_full_splice.md` | Learned music-change probabilities in Viterbi stitching. |
| `cue_demo.md` | Cue-switch demo renders, grouped by visual mood. |
| `crossfade_shapes.md` | Crossfade interpolation shapes compared. |
| `segmentation_comparison.md`, `segmentation.tex` | Segmentation methods compared. |
| `segmentation_improve.md` | Boundary-head losses and layers (D142). |
| `hyperparameters.tex` | Hyperparameter table for the paper. |

## Subfolders

| Folder | Contents |
|---|---|
| [`titlesplit/`](titlesplit/) | Results on the title-grouped film split (final protocol). |
| [`d138_hubness/`](d138_hubness/) | D138 Step 2: test-time hubness reduction. |
| [`d138_candidates/`](d138_candidates/) | D138 Steps 4–5: candidate ablation and re-tuned penalties. |
| [`d138_final/`](d138_final/) | D138 Step 6: best combined setting on test. |
