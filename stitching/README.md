# stitching/

The **stitching ablation study**: different ways to choose one music window per video segment, all compared on the
same candidate lattices (title split, 5 folds, 10-head ensemble shortlist, BaSSL segment cuts). This package imports
the main pipeline but does not change it. Results: [`results/main_table.md`](results/main_table.md).

## Run order

```bash
.venv/bin/python -m stitching.data --fold k        # build the shared episodes of fold k
.venv/bin/python -m stitching.cue_stats            # cue length / reprise statistics
.venv/bin/python -m stitching.evaluate --fold k    # fit every method on train/val, decode test
.venv/bin/python -m stitching.extra_rows --fold k  # semi-Markov rows
.venv/bin/python -m stitching.report               # -> results/main_table.{csv,md}, wilcoxon.md, plots
```

`run_nested_heads.sh` and `run_phase1_rest.sh` run the nested cross-fitting phase (100 heads, then cross-fitted
episodes and CRF retraining).

## Methods (rows of the main table)

| File | Method |
|---|---|
| `base.py` | Common interface every method implements. |
| `greedy.py` | 1: greedy top-1 per segment. |
| `smooth.py` | 2: greedy + sliding-window majority vote. |
| `two_stage.py` | 3: change-point grouping, then one track per group. |
| `dtw_single.py` | 4: one track with monotonic DTW alignment (Prétet-style). |
| `hmm_derived.py` | 5: HMM with costs derived from cue statistics, and the existing hand-set Viterbi. |
| `learnable_viterbi.py`, `crf.py` | 6: learnable Viterbi / linear-chain CRF over the lattice. |
| `semi_markov.py`, `semi_viterbi.py` | Segmental (semi-Markov) Viterbi. |
| `gold_prior.py` | Transition prior counted from training gold paths. |
| `phase0.py` | Calibration rows (e.g. change at every cut). |
| `phase2.py` | Semi-Markov rows 2a–2c and equal-budget re-tune of two-stage. |
| `cue_bassl.py` | Change predictor retrained on BaSSL segments (row 2c). |

## Data, training and reporting

| File | What it does |
|---|---|
| `data.py` | Builds the shared episodes of one fold. |
| `win10.py` | Fixed 10 s music-window embeddings for the semi-Markov rows. |
| `cue_stats.py` | Mean cue length and reprise rate from the training films' music. |
| `nested.py`, `crossfit.py`, `crf_crossfit.py` | Phase 1: nested cross-fitted emissions and CRF retraining. |
| `evaluate.py` | Fits every method and decodes the test episodes once. |
| `metrics.py` | Per-episode metrics (reuses the main evaluation functions). |
| `decision.py` | Phase 3 decision rule and Wilcoxon tests vs two-stage and greedy. |
| `report.py` | Aggregates everything into the tables and figures in `results/`. |
| [`tests/`](tests/) | Synthetic sanity tests. |
