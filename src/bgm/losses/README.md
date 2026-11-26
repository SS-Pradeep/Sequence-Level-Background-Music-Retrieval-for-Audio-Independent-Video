# src/bgm/losses/

Alternative loss functions for the **loss-function sweep** (run by [`../sweep/`](../sweep/)). The final system keeps
plain InfoNCE (`train/losses.py`); these were compared against it.

| File | What it does |
|---|---|
| `registry.py` | Loss registry: every loss term maps (video embeddings, music embeddings, batch) → scalar, selected by name in a sweep config. |
| `score_aware_loss.py` | Score-Aware contrastive loss: soft targets from music–music similarity, plus intra-video and transition terms. |
| `tag_aware_loss.py` | Tag-Aware version: adds LLM tag probabilities (Qwen2-Audio for music, Qwen2.5-VL for scenes) as a teacher and/or extra term. |
| `tag_data.py` | Loads and pools the tag probabilities (D141). |
