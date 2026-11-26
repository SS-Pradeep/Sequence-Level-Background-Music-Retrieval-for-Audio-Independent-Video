# src/bgm/person/

**Person-focused branch** (D87+; plan: [`docs/person_branch_plan.md`](../../../docs/person_branch_plan.md)). SigLIP
encodes the whole frame, so the setting dominates the score. This branch also embeds only the people on screen (face
crops, or person boxes) with the same SigLIP encoder. Experimental, not part of the final system.
Results: `results/person_branch.md`.

## Steps

```bash
python -m bgm.person.diagnose                      # 1: current system on "same setting, different music" subsets
python -m bgm.person.features                      # 2: person crops at 1 Hz -> SigLIP
python -m bgm.person.train --mode person --variant NAME   # 3: train person-branch heads
python -m bgm.person.run --system P2               # 3: evaluate one system
python -m bgm.person.report                        # 3: results/person_branch.md
python -m bgm.person.outputs --person-run <run>    # 4: render demo videos
```

| File | What it does |
|---|---|
| `subset.py` | Builds the "same setting, different music" test subsets. |
| `diagnose.py` | Step 1 diagnosis. |
| `features.py` | Person crops and their SigLIP embeddings. |
| `train.py` | Person-branch heads, trained like the semantic heads. |
| `pipeline.py` | The person branch inside retrieval, stitching and render. |
| `stitch.py` | Stitching tuned for person-branch systems. |
| `evaluate.py` | Per-query metrics for any system. |
| `run.py` | Evaluates one system per fold × seed. |
| `report.py` | Writes the results table. |
| `render.py` | Render hook (`person.enabled=true`). |
| `outputs.py` | Renders every input video with the baseline and the best person system. |
