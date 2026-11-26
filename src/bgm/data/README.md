# src/bgm/data/

Preparing the dataset: from raw clips to manifests, silent videos, aligned music, shots, segments and folds.
`manifest.py` is the only code that reads or writes the manifest files.

| File | Command | What it does |
|---|---|---|
| `audit.py` | `bgm audit` | Scans `data/raw/` and writes the first manifests (`pairs.csv`, `tracks.csv`) and an audit report. |
| `manifest.py` | | Reads and writes `pairs.csv`, `tracks.csv`, folds, `shots.csv`, `segments.csv`. |
| `media.py` | | `ffprobe` helpers: duration, frame rate, size, start time. |
| `prep.py` | `bgm prep` | Silent 25 fps videos and A/V-aligned 48 kHz mono music. |
| `shots.py` | `bgm shots`, `bgm peek` | Shot-cut detection (PySceneDetect) and contact sheets. Also used at render time. |
| `segments.py` | `bgm segment` | Merges shots into 2–20 s segments by SigLIP similarity. Also used at render time. |
| `alignment.py` | | Checks that each pair's music is time-aligned with its video (cut-on-beat rate). |
| `tracks.py` | | Chromaprint fingerprinting: finds songs reused across clips so they stay in one fold. |
| `fp_index.py` | | Vectorised version of the fingerprint grouping for ~10⁵ files (D139). |
| `near_dups.py` | | Near-duplicate music candidates by CLAP similarity, for manual confirmation. |
| `films.py` | | Groups clips by film, the unit of the film-level split. |
| `themes.py` | | Theme groups: near-identical music cues of one film (D66). |
| `folds.py` | `bgm folds` | Builds and checks the 5-fold split (no film or track in both train and test). |
