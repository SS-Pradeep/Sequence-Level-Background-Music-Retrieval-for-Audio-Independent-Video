# exp_supp/

Supplementary experiments for the paper. These scripts import the final system's code **read-only**; they only write
inside `exp_supp/` and `runs/exp_supp/`. Every run is logged to `results.csv`.

Run any experiment from the repository root, e.g. `.venv/bin/python -m exp_supp.run_hsmm`.

| File | What it does |
|---|---|
| `common.py` | Shared helpers: run logging, clusters, paired cluster bootstrap, LaTeX table output. |
| `hsmm.py` | Hidden semi-Markov (cue-level) decoder on the Table 3 lattices. |
| `run_hsmm.py` | Experiment 1: the HSMM decoder vs the final system's Viterbi decoder → `hsmm_table.{md,tex}`. |
| `test_hsmm.py` | Checks the HSMM dynamic program against brute force on small random lattices. |
| `stitch_clean.py` | Re-runs the stitching study ([`stitching/`](../stitching/)) on the cleaned 1,011-pair dataset (D150) → `stitch_clean_table.{md,tex}`. |
| `cut_probe_cleaned.py` | Supplementary C: music changes at shot cuts inside continuous test clips, cleaned data (D147) → `cut_probe_cleaned.{md,tex}`. |
| `cut_probe_retrained.py` | Same question, fully retrained on the cleaned dataset (D152) → `cut_probe_retrained.{md,tex}`. |
| `results.csv` | Log of every supplementary run. |
| [`rerank/`](rerank/) | Re-ranker study on the top-20 shortlist (D148). |

Each `.md` table has a matching `.tex` (booktabs) version for the paper.
