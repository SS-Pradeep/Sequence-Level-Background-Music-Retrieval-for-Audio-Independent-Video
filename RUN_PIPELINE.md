# How the system runs: file by file, in order

This document follows one run of the final system, from a silent input video to the output video with background
music. It describes **every file that runs**, in the order it is used, and how each one works. To train the models
yourself, see [TRAIN.md](TRAIN.md).

## First run after cloning

The repository holds only the code. The trained models and the music library are too large for git, so they are
downloaded once with `download_assets.sh`. You need Linux, an NVIDIA GPU, Python 3.11 or newer, `ffmpeg`, and about
20 GB of free disk.

```bash
# 1. Get the code
git clone https://github.com/SS-Pradeep/Sequence-Level-Background-Music-Retrieval-for-Audio-Independent-Video.git
cd Sequence-Level-Background-Music-Retrieval-for-Audio-Independent-Video

# 2. Install the Python packages and the `bgm` command (once)
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e . --no-deps

# 3. Download the trained models, music library, ImageBind and SigLIP (once, about 11.5 GB)
./download_assets.sh

# 4. Add music to a video
./add_music.sh my_video.mp4              # writes my_video_bgm.mp4
```

Steps 1–3 are done once. After that, only step 4 is needed for each new video. To check the install, run
`.venv/bin/bgm doctor`, or try a demo clip: `./add_music.sh Demo_Outputs/1/original_video.mp4 test_bgm.mp4`.

## The command

`add_music.sh` runs this command:

```bash
bgm render --config configs/best_semantic.yaml \
           --video input.mp4 --out output.mp4 \
           --run runs/cleaned/pipeline_ens10/fold0/seed0 \
           pipeline.dyn_variant=null "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"
```

The last line is required. `pipeline.sem_seeds=[0,...,9]` turns on the **10-head ensemble**; without it only one head
(the run's own seed) is used. `pipeline.dyn_variant=null` stops the index from loading a dynamics model, which the
final system does not use.

**Short form.** [`add_music.sh`](add_music.sh) runs exactly this command, with the fold 0 run and the repository's
`.venv`:

```bash
./add_music.sh input.mp4                 # writes input_bgm.mp4
./add_music.sh input.mp4 output.mp4      # or choose the output name
```

It sets `HF_HUB_OFFLINE=1`, so the encoders load from the local Hugging Face cache without contacting the hub. A video
under 60 s takes about 45 s on an RTX 4080 SUPER. A video of 60 s or longer takes about 1.5–2 min, because it also
uses the long-video path (step 4).

All paths below are relative to `src/bgm/`.

---

## Overview

```
 input.mp4
    │
    ▼
 0. Start the command         cli.py → config.py, log.py, tools.py, runs.py
    │
    ▼
 1. Read and segment          render/query.py → data/media.py, data/prep.py, data/shots.py, data/segments.py
    │                         (data/manifest.py is used by the music library)
    ▼
 2. Encode the video          features/extract.py → features/hf.py, encoders.py, dynamics.py, keys.py, timing.py
    │
    ▼
 3. Load models and library   retrieval/index.py → models/heads.py, train/semantic.py, train/losses.py,
    │                         train/dynamics.py, train/shifts.py, eval/store.py, eval/attributes.py,
    │                         eval/metrics.py, eval/protocol.py
    ▼
 4. Find candidate music      retrieval/pipeline.py (+ retrieval/chunked.py, analysis/long_video.py for ≥ 60 s)
    │
    ▼
 5. Predict music changes     cue/hook.py → cue/model.py, cue/tokens.py, cue/train.py, cue/sequences.py
    │
    ▼
 6. Viterbi stitching         retrieval/stitch_eval.py → retrieval/stitch.py, cue/stitch.py
    │
    ▼
 7. Build and mux             render/render.py
    │
    ▼
 output.mp4  +  renders/output.plan.json
```

`render/run.py` coordinates steps 1–7. Its two main functions are:

- **`render_video()`**: the whole run: featurize the video → plan the soundtrack → render the audio → mux.
- **`plan_for()`**: the decisions: load the models → retrieve → predict changes → Viterbi → list of pieces.

---

## Step 0: Start the command

| File | How it works |
|---|---|
| `cli.py` | Defines the `bgm` command with Typer. `bgm render` reads `--video`, `--out`, `--run` and `--config`, then calls `render.run.render_video()`. Imports for each command happen inside its function, so only the code needed for rendering is loaded. |
| `config.py` | `load_config()` reads `configs/best_semantic.yaml` with OmegaConf and applies any `key.sub=value` overrides from the command line. Every later step reads its settings from this one config object. |
| `log.py` | `setup_logging()` sets one log format for all modules, so each step's progress messages appear in the terminal. |
| `tools.py` | `find_tool()` locates `ffmpeg` and `ffprobe` (first on `PATH`, then in `.checkpoints/bin/`) and defines the repository root. All video and audio work later goes through these tools. |
| `runs.py` | Helpers for run folders (config, metrics, git commit, seed). During rendering it is only imported by the model-loading code. |

---

## Step 1: Read and segment the input video

`render_video()` creates a work folder `renders/<name>/` and calls `render/query.py: featurize()`. The results are
cached in `renders/<name>/query.npz`, so running the same video again skips steps 1–2.

| File | How it works |
|---|---|
| `render/query.py` | **`featurize()`** runs steps 1 and 2 for the input video: (1) `silent_copy()` writes `silent25.mp4`, a 25 fps copy **with no audio**, so the system never hears the original soundtrack; (2) detects shot cuts; (3) samples one frame per second; (4) encodes the frames (step 2); (5) builds segments; (6) saves everything to `query.npz`. **`as_video_query()`** then wraps these arrays in a `VideoQuery` object that retrieval can score. |
| `data/media.py` | `ffprobe()` returns duration, frame rate, size and stream start time of a file. It is used to measure the silent copy and to read the original video's start time for A/V sync at the end. |
| `data/prep.py` | `scale_filter()` builds the ffmpeg filter that shrinks the short side to 480 px for the silent copy, the same format as the dataset videos. `interim_music_path()` gives the path of each library music file; step 7 uses it to read the audio. |
| `data/shots.py` | `detect_cuts()` runs **PySceneDetect's AdaptiveDetector** on the silent copy and returns the time of every shot cut in seconds. |
| `data/segments.py` | `make_segments()` turns shots into segments. It walks through the shots and merges the next shot into the current segment while the **SigLIP cosine** between them is ≥ 0.85 and the segment stays ≤ 20 s. Shots under 2 s are folded into the more similar neighbour; a single shot over 20 s is split into equal parts. The segments tile the whole video exactly. |
| `data/manifest.py` | The only reader of the manifest files (`pairs.csv`, `tracks.csv`, folds). It is used when the music library is loaded in step 3, to know which music files exist and which song (track group) each belongs to. |

---

## Step 2: Encode the video (frozen models)

Called from `featurize()`. All encoders are frozen: eval mode, no gradients, fp16 on the GPU.

| File | How it works |
|---|---|
| `features/extract.py` | `frames_1hz()` reads frame 25k + 12 of the silent video (the middle of each second k). `EncoderBank` loads each encoder once and runs it on the frames. `word_embeddings()` builds text embeddings for the mood words, giving per-second mood scores. |
| `features/hf.py` | The real encoders. **`SiglipEncoder`** (`google/siglip-so400m-patch14-384`) turns each frame into a 1,152-d vector; this is the main video feature. **`ImageBindEncoder`** also encodes the frames; it is stored with the features but not used for scoring in the final system. |
| `features/encoders.py` | The common interfaces (`FrameEncoder`, `AudioEncoder`, `TextEncoder`) that every encoder implements, and `l2_normalize()`. |
| `features/dynamics.py` | `video_curves()` computes five motion curves at 25 Hz from the silent video (optical-flow motion, visual onsets, cut impulses, luminance change, camera motion). It refuses a file that still has audio. The change predictor uses the motion curve in step 5. |
| `features/keys.py` | Musical key handling: `stitch_key()` and `fifths_distance()` measure how far apart two keys are on the circle of fifths. Used in step 6's switch cost. |
| `features/timing.py` | Timeline rules shared everywhere: `n_1hz()` (one sample per second, sample k centred at k + 0.5 s) and `n_25hz()`. |

---

## Step 3: Load the trained models and the music library

`plan_for()` reads `fusion.json` and `stitch.json` from the run folder, then loads the models and builds the index.

| File | How it works |
|---|---|
| `retrieval/index.py` | **`FoldModels.load()`** loads the trained heads of the run's fold from `runs/cleaned/semantic/siglip-both/fold<k>/seed<s>/ckpt.pt`: all **10 heads (seeds 0–9)**, because of the `pipeline.sem_seeds` override. **`EnsembleSemantic`** combines them: every library file and video span is embedded by all 10 heads, and the score is the mean cosine. **`MusicIndex`** holds every library music file's embeddings with **prefix sums**, so the score of any window of any file is computed in O(1). `Prefix` provides those fast span means. `VideoQuery.prepare()` embeds the input video's segments. |
| `models/heads.py` | **`SemanticHead`**: LayerNorm → Linear 512 → GELU → Dropout → Linear 256, L2-normalised. One head maps video features and one maps music features into the same 256-d space. |
| `train/semantic.py` | Defines **`SemanticModel`** (the video + music head pair) and **`SemanticInputs`** (how span means are pooled: SigLIP for video, CLAP + MERT concatenated for music). During rendering only these classes are used, to rebuild the heads from their checkpoints; no training happens. |
| `train/losses.py` | The contrastive losses used in training. It is loaded here because `train/semantic.py` imports it. |
| `train/dynamics.py`, `train/shifts.py` | The dynamics-model classes. The index imports them, but the final system gives dynamics zero weight, so they do no work at render time. |
| `eval/store.py` | **`FeatureStore`** lazily loads the library's cached music features from `cleaned_dataset/features/` (CLAP, MERT, beat times, duration). `Series1Hz` and `apply_span_rule()` give fast span means, widening any span to at least 5 s, the same rule used in training. |
| `eval/attributes.py` | The mood calibration (`Calibration`, `s_attr`). Imported by the index; its weight is 0 in the final system. |
| `eval/metrics.py`, `eval/protocol.py` | Ranking and evaluation helpers. Loaded through imports of the modules above; they do no work during a render. |

---

## Step 4: Find candidate music

| File | How it works |
|---|---|
| `retrieval/pipeline.py` | **Coarse:** `coarse_shortlist()` scores every library file against the whole video and against each segment, and keeps the union of the **top K files** of each (K comes from `fusion.json`). **Fine:** `fine()` slides windows every **0.5 s** over each shortlisted file and scores each window against each segment. Scores are z-scored per segment (`fuse()`), so they are comparable across segments. |
| `retrieval/chunked.py` | For videos **≥ 60 s**: `equal_chunks()` splits the video into equal chunks, `chunked_track_scores()` gives each file the mean over chunks of its best window, and `ensemble()` averages standardised scores across models. |
| `analysis/long_video.py` | Used by the long-video path above: `load_models()` loads every fold's heads and `Pooler` pools the inputs, so the extra files added to the shortlist come from a larger ensemble. Called from `render/run.py: chunked_ensemble_top()`. |

`stitch_eval.build_lattice()` (step 6) then keeps each segment's **top J windows** (J from `fusion.json`) as its
candidates.

---

## Step 5: Predict where the music should change

Turned on by `cue.enabled: true` in the config. Called from `plan_for()` through `cue/hook.py: cue_decoder()`.

| File | How it works |
|---|---|
| `cue/hook.py` | **`cue_decoder()`** builds one token per segment of the input video (`video_tokens()`), runs the trained change predictors, **averages the change probability over the fold's 3 seeds** (`cue_logits()`), and averages their tuned settings (α, λ_jump). It returns a decode function that step 6 uses. |
| `cue/tokens.py` | `segment_features()` builds each segment's 1,155-d token: the mean SigLIP vector of the segment's own frames (L2-normalised), log segment duration, and the mean and std of motion. `Scaler` z-scores the three scalar features with training-fold statistics. |
| `cue/model.py` | **`CueModel`**: token MLP → a learned [CTX] token → **2-layer bidirectional Transformer** (d 256, 4 heads) → a boundary head on each pair of neighbouring segments → one logit per boundary. `predict()` runs it in eval mode. |
| `cue/train.py` | `load_cue()` loads the saved predictor checkpoints from `runs/cleaned/cue/`. No training happens at render time. |
| `cue/sequences.py` | Sequence and clip helpers that `cue/train.py` imports. They do no work during a render. |

The output is $p_i$: the probability that the music changes at the boundary between segment $i$ and segment $i+1$.

---

## Step 6: Viterbi stitching (choose one window per segment)

| File | How it works |
|---|---|
| `retrieval/stitch_eval.py` | **`build_lattice()`** makes one `Layer` per segment: its top J windows plus the **contiguous continuation** of each previous candidate (the same file, the next stretch of audio), so "keep playing" is always an option. `transition_matrix()` adds the key (`fifths_matrix`) and tempo (`tempo_matrix`) distances. **`chosen_candidates()`** turns the chosen indices back into music files and offsets. |
| `cue/stitch.py` | **`decode_lattice_cue()`** runs Viterbi with a **different switch cost at every boundary**, taken from step 5's $p_i$: continue costs $-\alpha\log(1-p_i)$; switching to another file costs $-\alpha\log p_i + \mu\cdot$(key + tempo distance); jumping inside the same file costs $-\alpha\log p_i + \lambda_{jump}$. |
| `retrieval/stitch.py` | The core algorithm: `Candidate`, `transition_cost()`, `forward_costs()` and `viterbi()`. Viterbi finds the path through the lattice with the lowest total cost, where each window's own cost is its negated retrieval score, and each move between segments adds a transition cost. A beam keeps only the best paths at each step for speed. |

After decoding, `render/render.py: runs_from_path()` groups consecutive segments that play contiguous audio from one
file into **pieces**. `plan_for()` returns the full plan: segments, chosen files, offsets, scores and pieces.

---

## Step 7: Build the soundtrack and mux it onto the video

Back in `render_video()`:

| File | How it works |
|---|---|
| `render/render.py` | 1. **`beat_sync_runs()`** moves each piece's start in its music file to a nearby beat (within ±2 s) that puts the most video cuts on beats. 2. **`fill_short_runs()`** loops a piece at a beat if it is longer than the audio left in its file, instead of leaving silence. 3. **`render_soundtrack()`** reads each piece's audio, joins pieces with **equal-power crossfades** (`crossfade_gains`) plus gain ramps that match loudness at each seam (`seam_gain_ramps`), normalises each piece to **−16 LUFS**, and applies **`lookahead_limiter()`** to cap peaks at −1 dBFS. 4. **`mux()`** puts the soundtrack onto the **original** video with ffmpeg, correcting the video's start offset, and checks that the audio and video start times differ by less than 1 ms. 5. **`seam_loudness_jumps()`** measures the loudness jump at each seam for the report. |
| `render/run.py` | Writes the soundtrack as `renders/<name>/bgm.wav`, the final video to `--out`, and the full decision record to `renders/<name>.plan.json` (segments, chosen music, offsets, scores, change probabilities, pieces, loudness). It also writes plans from the fold's other seeds (`<name>.seed<k>.plan.json`) for an agreement check. |

---

## Outputs

| File | Contents |
|---|---|
| `output.mp4` | The input video with the new background music. |
| `renders/<name>/bgm.wav` | The soundtrack alone. |
| `renders/<name>.plan.json` | Every decision: segments, chosen file and offset per segment, scores, change probabilities, pieces, seam loudness, output loudness. |
| `renders/<name>/query.npz` | Cached video features, reused if the same video is rendered again. |

## What the run needs besides the code

| Path | Used in | Where it comes from |
|---|---|---|
| `configs/best_semantic.yaml`, `configs/default.yaml` | Every step | In git |
| `.venv/` with the `bgm` command | Every step | Created in step 2 of [First run after cloning](#first-run-after-cloning) |
| `cleaned_dataset/manifests/` (`pairs.csv`, `segments.csv`, `tracks.csv`, `folds_title.json`) | Step 3 | `download_assets.sh` |
| `cleaned_dataset/features/music/`, `cleaned_dataset/features/video/` | Steps 3 and 4 | `download_assets.sh` |
| `cleaned_dataset/interim/music/` (the 1,011 music files) | Step 7 | `download_assets.sh` |
| `runs/cleaned/semantic/siglip-both/fold<k>/seed<s>/ckpt.pt` (all 5 folds are loaded for videos ≥ 60 s) | Steps 3 and 4 | `download_assets.sh` |
| `runs/cleaned/pipeline_ens10/fold<k>/seed<s>/` (`fusion.json`, `stitch.json`) | Steps 3, 4 and 6 | `download_assets.sh` |
| `runs/cleaned/cue/ctx/`, `runs/cleaned/cue/eval/` (change predictor) | Step 5 | `download_assets.sh` |
| `.checkpoints/imagebind_huge.pth` | Step 2 | `download_assets.sh` (Meta's public file) |
| SigLIP `google/siglip-so400m-patch14-384` (Hugging Face cache) | Step 2 | `download_assets.sh` |

The downloaded folders sit inside the repository but are listed in `.gitignore`, so they are never committed. This
list was checked by tracing which files a real run opens.

**Common errors**

| Error | Cause | Fix |
|---|---|---|
| `error: video not found` from `add_music.sh` | The input path is wrong | Check the path with `ls`; copy the video into the repo or give its full path |
| `No such file ... runs/cleaned/...` or `cleaned_dataset/...` | Assets not downloaded | Run `./download_assets.sh` |
| `.venv/bin/bgm: No such file` | Package not installed | Run step 2 of [First run after cloning](#first-run-after-cloning) |
| SigLIP fails to load in offline mode | SigLIP not in the Hugging Face cache | Run `./download_assets.sh` (it fetches SigLIP) |

To see what the run decided, print one line per segment from the plan:

```bash
.venv/bin/python -c "import json; p=json.load(open('renders/output.plan.json'))
for s in p['segments']: print(f\"{s['start_s']:6.2f}-{s['end_s']:6.2f} s   {s['music_id']}   from {s['offset_s']:.1f} s\")
print('change probability at each boundary:', [round(x, 2) for x in p['cue']['p']])"
```

## Optional features (off in the final config)

`render/run.py` can also call `render/pace.py`, `render/scene.py`, `render/replay.py`, `render/branches.py`, the person
branch and `cue/cut_hook.py`. They are switched off in `best_semantic.yaml` and do not run.
