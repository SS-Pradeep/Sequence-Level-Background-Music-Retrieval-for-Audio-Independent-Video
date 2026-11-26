# src/bgm/retrieval/

Finding music for a video and stitching it into one soundtrack.

**Coarse-to-fine retrieval:** rank whole music files for the whole video (shortlist of K files), then slide 0.5 s
windows over the shortlisted files and score them for each segment. **Stitching:** Viterbi picks one window per
segment, trading off fit against switching or jumping inside the music.

| File | Command | What it does |
|---|---|---|
| `index.py` | | `MusicIndex`: every library file's head embeddings with prefix sums (O(1) window scores). `EnsembleSemantic` averages the 10 heads; `FoldModels` loads a trained run. |
| `pipeline.py` | | Coarse-to-fine retrieval and pipeline accuracy. |
| `chunked.py` | | Long videos (≥ 60 s): scores music per chunk of the video and adds the best files to the shortlist. |
| `stitch.py` | | Viterbi and greedy decoders and the transition cost (switch, jump, key change). |
| `stitch_eval.py` | | Builds the candidate lattice per segment and decodes it; evaluation on concatenated test videos. |
| `fuse.py` | `bgm fuse` | Sets branch weights, shortlist size K, candidates per segment and window hop on validation → `fusion.json`. |
| `tune_stitch.py` | `bgm tune-stitch` | Tunes the Viterbi penalties on validation → `stitch.json`. |

`bgm render` reads `fusion.json` and `stitch.json` from the trained run.
