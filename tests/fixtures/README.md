# tests/fixtures/

Synthetic data for the tests and for `bgm smoke --fake`.

| File | What it does |
|---|---|
| `synth.py` | Generates synthetic video–music pairs with known ground truth (SPEC M0): 12–30 s of 25 fps 320×240 video with music on a beat grid, so cuts, beats and alignment are known exactly. |
| `pipeline.py` | Runs the synthetic pairs through the built pipeline stages in a temporary folder. |
