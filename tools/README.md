# tools/

Stand-alone scripts for building and extending the dataset. Run them from the repository root.

| File | What it does |
|---|---|
| `build_cleaned.py` | Builds `cleaned_dataset/` (the final 1,011 pairs) from the pair trimmer's output, with its manifests and title-split folds. `python tools/build_cleaned.py` |
| `pair_trimmer.py` | Local web page for cutting unwanted parts out of dataset pairs, video and music together. Run it, then open the printed `http://127.0.0.1:...` address. |
| `pair_trimmer.html` | The page served by `pair_trimmer.py`. |
| `download_yt_dataset.py` | Resumable, parallel downloader for a YouTube-URL video–music dataset (e.g. HIMV-200K style). |
| `separate_bgm.py` | Keeps only the background music of downloaded audio, using Demucs source separation. |
| `stream_youtube.py` | Streams a downloaded folder into cached features and deletes the media as it goes (for lists too big for the disk). |

The YouTube tools were used for an out-of-domain test (`bgm.analysis.youtube_eval`); the final system does not need them.
