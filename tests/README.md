# tests/

Unit and integration tests for the `bgm` package (pytest). They use small synthetic data and fake encoders, so they
run on CPU without the dataset.

```bash
pytest -q                       # everything
pytest -q tests/test_m6.py      # one file
```

## Layout

| File(s) | What they test |
|---|---|
| `conftest.py` | Shared pytest fixtures. |
| [`fixtures/`](fixtures/) | Synthetic video–music pairs and a helper that runs them through the pipeline. |
| `test_m0.py` … `test_m12.py` | One file per SPEC milestone: M0 fixtures, M2 prep/shots, M3/M8 extraction, M4 segments, M5 metrics and evaluation, M6 semantic heads, M7 dynamics, M9–M10 retrieval and stitching, M12 listening kit. |
| `test_cue*.py`, `test_cut_changes.py` | Music-change (cue-switch) predictor, probes, timelines, hazard table, cut classifier. |
| `test_seg_*.py` | Segmentation study: sequences, segments, retrieval. |
| `test_loss_*.py`, `test_losses_extra.py`, `test_tag_*.py` | Loss registry, loss bank, tag-based losses. |
| `test_film_split.py`, `test_splits.py`, `test_cv_schemes.py`, `test_film_batches.py` | Fold construction: no film or track in both train and test. |
| `test_fp_index.py`, `test_themes.py`, `test_build_cleaned.py`, `test_pair_trimmer.py` | Dataset building: fingerprints, theme groups, cleaned dataset, trimmer tool. |
| `test_render_*.py`, `test_pace.py`, `test_scene.py`, `test_chunked.py` | Rendering and long-video retrieval. |
| other `test_<module>.py` | One file per analysis or feature module of the same name. |

`a.txt` is a stray placeholder file and is not used by any test.
