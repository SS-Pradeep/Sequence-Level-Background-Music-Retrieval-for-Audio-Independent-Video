# Current version: what is kept and what is deleted (2026-11-26)

The current system is the cleaned 1,011 pairs on the title split, with the 10-head ensemble and Viterbi
stitching (PROGRESS.md D150–D154). Code snapshot: commit `5f8778c` on branch `oracle-p-table`.

Full file lists (path, bytes), `.git/` and `.venv/` not listed:
- `keep_files.tsv.gz`: 138,971 files, 268 GB
- `deleted_files.tsv.gz`: 111,293 files, 294 GB, deleted on 2026-11-26 (no backup)

After cleanup the disk went from 91% to 75% full (444 GB free); `pytest -q`: 498 passed.

Read them with `zcat docs/inventory/keep_files.tsv.gz | less`.

## Kept

| Path | Why |
|---|---|
| `src/ configs/ tests/ tools/ docs/ results/ exp_supp/ stitching/` (minus cache) | code, configs, result tables, decoder study |
| `cleaned_dataset/` | the 1,011-pair dataset; its manifests point at its own video/ and audio/ |
| `data/` (minus `unlabeled/`) | features, silent interim media, manifests (title folds) |
| `dataset_merged/` | `data/raw` links to it (read-only, CLAUDE.md rule 1) |
| `dataset_1248/` | source of the cleaned set; `trimmed/edits.json` feeds the dataset stats table |
| `full_bgm_dataset_sunnxt/` | `render.sunnxt_manifests` in configs/default.yaml |
| `runs/cleaned runs/titlesplit runs/exp_supp runs/compute_cost` | cleaned heads, ensembles, D148–D154 runs |
| `runs/filmsplit` | `render_best.sh` loads this run |
| `.checkpoints/ third_party/ media/ reports/` | ImageBind weights, TransNetV2/BaSSL, query media, reports |
| `.venv/` | the environment (`ven/` duplicated it) |
| `outputs/` figure and brief folders | paper figures (qualitative_cleaned, dataset_figure, method_figures, …) |
| `outputs/{cue_splice_demo, transition_showcase, person_branch, emotion_variants, cue_splice, emotion_sections, pipeline_train_test, silence_demo}` | demo videos, kept at the user's request (59 GB) |
| `godfather.mp4 KGF.mp4 sarpatta.mp4 website/ demo_outputs/ Viva_Voce_BGM_Retrieval.pptx` | demo inputs and presentation material |

## Deleted

| Path | GB | Why |
|---|---|---|
| `dataset_himv dataset_himv_stream dataset_himv_test` | 107.5 | YouTube streaming downloads; features already extracted |
| `yt_dataset` + `yt_dataset_*.log` | 89.3 | YouTube 5k download / separation, not used by the current system |
| `dataset_combined` | 6.5 | early combined set, no code reference |
| `Data/` and `data/unlabeled/` | 58.5 | M8 pretraining pool; M8 was stopped |
| `legacy/` | 2.2 | old artefacts |
| `renders/` | 15.1 | old render outputs (D75 era) |
| `ven/` | 6.5 | duplicate virtualenv (every package also in `.venv`) |
| `stitching/cache/` | 5.2 | lattices of the original data; cleaned lattices live in runs/exp_supp/rerank/clean_t3/cache |
| `runs/{full, titlesplit_nested, semantic, person_check, synthetic_grounding, segmentation, loss_sweep, mema, _v1, demo_cache, dynamics, pipeline, verify_noattn, zeroshot, cue}` + 3 logs | 3.3 | runs of the 1,248-pair / track split, superseded |
