# src/bgm/rerank/

**Bidirectional retrieval and hubness re-scoring** (D140): segment retrieval in both directions (video → music and
music → video), and training-free re-scoring of the score matrix so that a few "hub" music windows stop dominating
every query. Uses the shipped 10-head ensembles; nothing is retrained. Not part of the final system.

```bash
python -m bgm.rerank.run_all
```

| File | What it does |
|---|---|
| `run_all.py` | Driver for the whole study. |
| `scores.py` | Segment score matrices of the ensembles for both directions. |
| `normalize.py` | Re-scoring methods (pure functions). |
| `tune.py` | Grid search of re-scoring parameters on validation only. |
| `hubness.py` | k-occurrence hubness diagnostic. |
| `evaluate.py` | Ranks and recall in both directions. |

A later re-ranker study on the shortlist is in [`exp_supp/rerank/`](../../../exp_supp/rerank/).
