# src/bgm/pretrain/

**Unlabeled pretraining** (SPEC M8): self-supervised pretraining of the heads on unlabeled film clips before
fine-tuning on the labeled pairs. This milestone was stopped and its data deleted (see
[`docs/inventory/CURRENT_VERSION.md`](../../../docs/inventory/CURRENT_VERSION.md)); not part of the final system.
Results: `results/pretraining_filmsplit.md`.

Run through the CLI: `bgm pretrain --stage data | extract | train`.

| File | What it does |
|---|---|
| `unlabeled.py` | Builds the unlabeled set: speech filtering with PANNs, then clip extraction. |
| `train.py` | Stage 1: symmetric InfoNCE on random 5–10 s aligned video/music spans. |
