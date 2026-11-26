# src/bgm/models/

The trainable modules. The pretrained encoders stay frozen; only these learn.

| File | What it does |
|---|---|
| `heads.py` | `SemanticHead`: a small MLP (LayerNorm → 512 → 256) that maps frozen video or music features into a shared embedding space; one head per side. **This is what the final system trains.** The file also holds `DynamicsEncoder` (M7), `AttnPool` and `AttributeMap`, used only by experiments. |
