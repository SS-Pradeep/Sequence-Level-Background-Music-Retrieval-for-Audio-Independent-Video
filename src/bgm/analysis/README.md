# src/bgm/analysis/

Analysis scripts: ablations, diagnostics, paper figures, demo renders and extra tables. Each is run on its own,
`python -m bgm.analysis.<name>`, and its docstring names its output (usually `results/...`, `reports/...` or
`outputs/...`). Only **`long_video.py`** is imported by the final system (it loads every fold's heads for the
long-video shortlist); everything else is analysis. `D<number>` refers to [`docs/PROGRESS.md`](../../../docs/PROGRESS.md).

## Final system: evaluation and ablations (cleaned data, title split)

| File | What it does |
|---|---|
| `eval_suite.py`, `eval_suite_report.py` | Evaluation suite: retrieval, alignment, transitions, diversity (D111). |
| `eval_lb.py` | LanguageBind score of the final system (D117). |
| `mema_protocols.py` | MEMA's evaluation protocols applied to the final system (D118). |
| `baselines_compare.py` | All baselines and the final system under one evaluator (D120). |
| `cleaned_compare.py` | Cleaned dataset vs the full 1,248 pairs (D116). |
| `ablation.py` | Leave-one-out ablation of the final system (D127). |
| `loss_compare.py`, `loss_why.py` | Loss-function ablation and why alternatives do not beat InfoNCE (D124, D127). |
| `cv_schemes.py` | Cross-validation schemes (D133). |
| `underfit_diag.py`, `overfit_fix.py`, `reg_sweep.py` | Under/overfitting diagnosis and regularisation sweep (D134, D135, D138). |
| `topk_sweep.py` | Shortlist-size sweep (D135). |
| `zs_blend.py` | Blending zero-shot scores into the heads' scores (D136). |
| `window_switch_cost.py` | Per-window key and tempo in the switch cost (D137). |
| `error_sources.py`, `hubness.py`, `cand_ablation.py`, `final_d138.py`, `d138_tables.py` | D138 study: error sources, hubness, candidate ablation, final combined setting, LaTeX tables. |
| `cue_hazard_table.py` | Duration-prior vs fixed penalties for the change predictor (D145). |
| `segment_retrieval.py` | Segment-level retrieval for the paper (D126). |
| `wilcoxon.py` | Paired Wilcoxon signed-rank tests between systems (D112). |
| `train_vs_test_rank.py` | How often training vs test segments find their own music. |
| `compute_cost.py`, `inference_time.py` | Compute cost and inference time. |
| `long_video.py` | Long-video retrieval: whole-clip pooling vs chunked querying. **Used by the final system.** |

## Dataset

| File | What it does |
|---|---|
| `dataset_stats.py`, `dataset_figure.py` | Dataset statistics table and example-pairs figure (D115). |
| `bgm_presence.py` | How much of each music track is real background music (D114). |
| `song_labels.py` | Song/dance vs background-score label per music file. |
| `section_audit.py` | Do scene boundaries coincide with musical section changes? |
| `film_screen.py`, `full_film_screen.py` | Screening for the same film under two names (D39, D49). |
| `full_dataset.py` | Semantic heads on the full 4,246-pair dataset (D49). |

## Earlier architecture experiments (film split)

`arch_phase0.py`, `arch_1a.py`, `arch_phase2.py`, `arch_mood_fusion.py`, `arch_inference.py`, `arch_ensemble.py`,
`arch_ensemble_video.py`, `arch_theme.py`, `arch_video_encoder.py`, `arch_branches.py`, `arch_vjdyn.py`,
`arch_context.py`, `arch_one_to_many.py`, `arch_three_branch.py`, `arch_full_best.py`, `arch_paired.py`:
one file per step of the architecture plan (D52–D77): MERT layer mixing, mood fusion, ensembles, hubness, theme
groups, V-JEPA 2 video features, dynamics branches, whole-video context, one-to-many targets, face branch.
Results are summarised in `results/arch_ablation.md`.

## Figures (paper, poster, slides)

| File | Figure |
|---|---|
| `worked_example.py`, `qualitative_cleaned_timeline.py` | Worked example of the whole system on one test case. |
| `qualitative_cleaned.py`, `qualitative_cleaned_fig.py` | Failure table and failure-case figure. |
| `retrieval_figure.py` | Video → music and music → video examples, distance histograms. |
| `method_figures.py` | Viterbi trellis and crossfade gains. |
| `viterbi_real_figure.py`, `penalty_sweep.py` | Viterbi on a real test video; switch-penalty sensitivity. |
| `viterbi_explainer.py`, `viterbi_explainer.html` | Data and page for the interactive Viterbi explainer. |
| `music_change_figure.py` | "Where does the music change?" |
| `poster_pipeline.py` | One image per pipeline stage for the poster. |
| `slide_steps34.py` | Slide images for steps 3–4 (change probability, Viterbi route). |
| `godfather_sections.py` | Section figures for one rendered video. |
| `crossfade_shapes.py`, `crossfade_eval.py` | Crossfade shapes and seam-rendering variants (D50/D51). |

## Demo renders

| File | What it does |
|---|---|
| `test_inputs.py` | Builds held-out demo inputs from one film's test clips (D106). |
| `cue_demo.py`, `cue_splice_demo.py`, `test_results.py` | Change-predictor demo renders and test outputs (D84, D94–D97). |
| `emotion_variants.py`, `emotion_showcase.py`, `emotion_sections.py`, `emotion_transitions.py` | Renders per emotion of the scene brief (D106–D108). |
| `pipeline_train_test.py` | Same clips rendered by heads that did / did not train on them. |

## Grounding and out-of-domain checks

| File | What it does |
|---|---|
| `query_grounding.py` | Grounding with chance levels on one external video. |
| `synthetic_grounding.py` | Synthetic beat-cut grounding check. |
| `youtube_eval.py` | Out-of-domain test on YouTube music videos. |

`a.txt` is a stray placeholder file and is not used by any code.
