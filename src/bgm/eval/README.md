# src/bgm/eval/

Feature access, metrics, evaluation protocols and result reports.

## Used by the final system

| File | What it does |
|---|---|
| `store.py` | `FeatureStore`: loads cached features; fast span averages via cumulative sums. Used by training, retrieval and render. |
| `metrics.py` | Recall@K, ranks, confidence intervals, cut-on-beat rate. |
| `protocol.py` | Track retrieval, segment retrieval and grounding protocols. |
| `attributes.py` | Mood-attribute calibration (its weight is zero in the final system). |

## Evaluation commands and reports

| File | Command / output |
|---|---|
| `run.py` | `bgm eval --exp <name>`: evaluate an experiment on all 5 folds and aggregate. |
| `zeroshot.py` | Zero-shot references: ImageBind, text bridge, hand-crafted dynamics, random. |
| `rescore_grounding.py` | `bgm eval --exp grounding`: re-score grounding predictions with chance levels. |
| `ensemble.py` | Seed ensembles evaluated like single runs (D99). |
| `check.py` | `bgm check --plan ...`: reference-free report for one render. |
| `final_film.py` | `bgm report --final`: the paper tables (`results/final.md`). |
| `final.py` | `bgm report --final --split trackid`: the older track_id-split report. |
| `report_semantic.py` | `results/semantic.md`: heads vs ImageBind zero-shot. |
| `report_filmsplit.py` | Film-split semantic results and the learning curve. |
| `report_pipeline_filmsplit.py` | Pipeline accuracy and Viterbi vs greedy on the film splits. |
| `report_dynamics.py` | `results/dynamics.md`. |
