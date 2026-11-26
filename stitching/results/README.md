# stitching/results/

Outputs of the stitching study. Written by `python -m stitching.report` (tables, plots) and by the per-fold scripts
(`.json` files).

| File | Contents |
|---|---|
| `main_table.md`, `main_table.csv` | **Main table**: every stitching method, mean ± std over the 5 test folds (track accuracy, switch F1, TBIoU, switches/min, runtime, ...). |
| `wilcoxon.md` | Paired Wilcoxon signed-rank tests over test sequences. |
| `decision.md` | Phase 3 decision and paired tests. |
| `cue_stats.md`, `cue_stats.json` | Cue length and reprise rate from training films, per fold. |
| `fold<k>.json` | Per-episode results of every method on fold k. |
| `fold<k>_phase0.json` | Phase 0 calibration rows. |
| `fold<k>_phase2.json` | Phase 2 semi-Markov rows. |
| `fold<k>_gold.json` | Gold-path transition prior rows. |
| `fold<k>_semi.json` | Extra semi-Markov rows. |
| `crf_switch_vs_siglip.png` | CRF switch behaviour plot. |
| `timelines_fold0.png` | Example decoded timelines on fold 0. |
