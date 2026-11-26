# exp_supp/rerank/

Re-ranker study (supplementary "Re-Ranker Algorithms", PROGRESS.md D148). The final system's semantic heads are kept
fixed; only the order of the top-20 music shortlist is changed, and the re-ranked lists are then fed through the
final Viterbi decoder. Cleaned dataset, title split.

## Run order

```bash
.venv/bin/python -m exp_supp.rerank.cache        # cache embeddings / score matrices
.venv/bin/python -m exp_supp.rerank.run          # evaluate the re-rankers
.venv/bin/python -m exp_supp.rerank.table        # -> rerank_table.{md,tex}
.venv/bin/python -m exp_supp.rerank.lattice      # per-head scores of the Table 3 lattices
.venv/bin/python -m exp_supp.rerank.hmm          # best re-rankers through the decoder -> rerank_hmm_table*.{md,tex}
.venv/bin/python -m exp_supp.rerank.extra_tables # -> extra_tables.{md,tex}
```

## Files

| File | What it does |
|---|---|
| `data.py` | Split, galleries and hit masks (a hit = same `track_id`). |
| `cache.py` | Embedding and score cache. |
| `methods.py` | The re-rankers themselves, in score space (pure numpy). |
| `ltr.py` | LambdaMART learning-to-rank re-ranker on the top-20 shortlist. |
| `ksweep.py` | Shortlist-size sweep, K ∈ {10, 20, 50, 100}. |
| `run.py` | Main driver: evaluates every re-ranker. |
| `lattice.py` | Per-head scores of the Table 3 lattice candidates. |
| `hmm.py` | Feeds the best two re-rankers into the final decoder. |
| `clean_t3.py` | Rebuilds Table 3's final decoder on the cleaned dataset (D148b). |
| `table.py`, `extra_tables.py` | Write the result tables. |
| `test_methods.py` | Unit tests on synthetic matrices: `pytest -q exp_supp/rerank/test_methods.py`. |

## Result tables

| File | Contents |
|---|---|
| `rerank_table.{md,tex}` | Re-rankers on the top-20 shortlist. |
| `rerank_hmm_table.{md,tex}` | Re-ranked shortlists through the final decoder (D148). |
| `rerank_hmm_table_cleaned.{md,tex}` | Same, cleaned-dataset decoder (D148b). |
| `extra_tables.{md,tex}` | Extra tables (D148c). |
