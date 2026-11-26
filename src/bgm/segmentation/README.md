# src/bgm/segmentation/

**Segmentation study** (D140, D142, D143): can a learned segmenter find where the music should change better than the
current rule (PySceneDetect shots merged by SigLIP similarity)? It uses TransNetV2 shots, frozen BaSSL shot embeddings
and a new boundary head trained to predict music changes. Not part of the final system.
Results: `results/segmentation_comparison.md`, `results/segmentation_improve.md`.

Needs `third_party/transnetv2` and `third_party/bassl` (not in git).

## Run

```bash
python -m bgm.segmentation.run_all <stage>      # shots -> features -> train -> report (see run_all.py)
python -m bgm.segmentation.improve train|eval   # D142 loss/layer variants
python -m bgm.segmentation.retrieval_eval segments  # D143: segmenters inside the retrieval pipeline
```

## Files

| File | What it does |
|---|---|
| `run_all.py` | Study driver, stage by stage. |
| `shots_transnet.py` | TransNetV2 shot detection on the silent videos. |
| `bassl_features.py` | Frozen BaSSL embeddings over shot sequences. |
| `sequences.py` | Chained shot sequences with music-change labels. |
| `head.py` | The boundary head on BaSSL embeddings. |
| `train_head.py` | Trains the boundary head per fold and seed. |
| `improve.py` | Loss and layer variants for the head (D142). |
| `segment.py` | Boundary scores → cuts → segments. |
| `evaluate.py` | Boundary metrics (AP, F1) against true music changes, with cluster bootstrap. |
| `report.py` | Evaluates all variants and writes the tables. |
| `retrieval_eval.py`, `seg_recall.py` | Each segmenter's effect on retrieval accuracy (D143). |
