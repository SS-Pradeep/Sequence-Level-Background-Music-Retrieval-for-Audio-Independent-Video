# src/bgm/train/

Training code. The final system uses only `semantic.py` and `losses.py`: 10 semantic heads per fold (seeds 0–9),
trained with symmetric InfoNCE.

| File | Command | What it does |
|---|---|---|
| `semantic.py` | `bgm train semantic` | Trains the video and music heads on matching segments; saves a checkpoint per fold and seed. |
| `losses.py` | | Contrastive losses (`info_nce` and variants). |
| `dynamics.py` | `bgm train dynamics` | Dynamics encoders for video–music synchronisation (M7; not used by the final system). |
| `shifts.py` | | Shifted negatives for dynamics training and the "in sync" test. |
| `semantic_tokens.py` | | Temporal tokens with late interaction (Phase 1b, D59; experimental). |
| `attr_distill.py` | | Attribute distillation as an auxiliary loss (D62; experimental). |
| `baselines.py` | `bgm train baseline` | Published video–music retrieval methods (CFD, UT-CMVMR, MVPT) reimplemented on the same frozen features (D43). |
