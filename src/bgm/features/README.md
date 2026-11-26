# src/bgm/features/

Frozen pretrained encoders and the feature cache. Every encoder is frozen (eval mode, no gradients, fp16 on GPU);
features are computed once per file and cached in `data/features/` (or `cleaned_dataset/features/`).

| File | What it does |
|---|---|
| `extract.py` | `bgm extract`: runs each encoder once per video and music file and caches the features. |
| `check.py` | `bgm check-features`: validates the cached features (shapes, dtypes, keys). |
| `encoders.py` | Common encoder interfaces (frame / audio / text) and deterministic fakes for tests. |
| `hf.py` | The real encoders: SigLIP and CLIP (video), CLAP and MERT (music), ImageBind. |
| `timing.py` | Timeline helpers: 1 Hz sample k is centred at k + 0.5 s; 25 Hz sample t is at t/25 s. |
| `dynamics.py` | 25 Hz motion / cut curves for video and onset / beat / loudness curves for music. |
| `keys.py` | Musical key and mode estimation, used in the stitching switch cost. |
| `face.py` | Facial-expression features at 1 Hz (D77, experimental). |
| `video_temporal.py` | Temporal video features with V-JEPA 2 (D67, experimental). |
| `languagebind.py` | LanguageBind video/audio encoders, used only for the LanguageBind evaluation score (D117). |
| `panns_tags.py` | PANNs CNN14 AudioSet tags per second of music, used by the evaluation suite (D111). |

The final system uses SigLIP for video and CLAP + MERT for music.
