# src/bgm/sweep/

Runner for the **loss-function sweep**: the semantic heads trained exactly as in the final system, with only the loss
swapped. Losses come from [`../losses/`](../losses/); each run is described by a YAML file in [`configs/`](configs/).
Results: `results/titlesplit/{planned_losses,improve_losses,loss_bank,similarity,train_vs_test}.md`.

## Python modules

| File | What it does |
|---|---|
| `run.py` | Sweep driver: `python -m bgm.sweep.run <command>` (split, pairs, smoke, stages, ...). |
| `train.py` | One sweep run: the normal trainer with the loss swapped. |
| `eval.py` | Segment retrieval in both directions (video → music and music → video). |
| `split.py` | The loss-sweep split (`folds_losssweep.json`). |
| `titlesplit.py` | Planned losses on the title split: `python -m bgm.sweep.titlesplit run --seeds 0`. |
| `similarity.py` | Similarity functions beyond mean-pooled cosine, at test time (D145). |
| `train_vs_test.py` | Re-scores saved checkpoints on train and test clips. |
| `report.py` | Final test stage and report of the sweep. |

## Shell scripts (resume-safe; rerun after a stop)

| Script | What it runs |
|---|---|
| `chain.sh` | The original loss sweep, end to end. |
| `titlesplit_chain.sh` | Planned losses: seed 0, table, seeds 1–9, 10-seed ensembles, final table. |
| `titlesplit_3seeds.sh` | Planned losses, seeds 0–2. |
| `titlesplit_improve.sh` | Exploratory improvement round. |
| `titlesplit_bank.sh`, `titlesplit_bank10.sh` | Tag-free loss bank, 3 and then 10 seeds. |
| `cleaned_bank10.sh` | The loss bank on the cleaned dataset, 10 seeds. |
| `titlesplit_enc.sh` | Per-encoder heads (CLAP-only, MERT-only). |
| `similarity_chain.sh` | Caches score matrices, then writes the similarity table. |
