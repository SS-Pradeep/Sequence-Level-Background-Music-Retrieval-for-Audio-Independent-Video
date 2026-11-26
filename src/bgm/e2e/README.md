# src/bgm/e2e/

**End-to-end soundtrack retrieval** (D141): the semantic heads, the emission scale and cross-modal transition costs
trained together through a linear-chain CRF over the stitching lattice ("retrieval trained for the soundtrack, not
for single clips"). Compared against the shipped decoder; not part of the final system.

| File | What it does |
|---|---|
| `data.py` | Builds sequence lattices from the shipped pipeline's candidates. |
| `bounds.py` | Boundary features (beat distance, etc.) for the cross-modal switch cost. |
| `crf.py` | Linear-chain CRF over the lattice (same energy as the Viterbi objective). |
| `model.py` | The model: 10 heads, emission scale, transition parameters. |
| `train.py` | Training and evaluation per fold and ensemble. |
| `report.py` | End-to-end model vs shipped decoder, with bootstrap CIs. |
