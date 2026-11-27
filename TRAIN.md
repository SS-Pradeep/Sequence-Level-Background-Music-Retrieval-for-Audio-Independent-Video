# How to train the system

This guide trains every model the final system uses, from the cleaned dataset of 1,011 video–music pairs. When it
is done, you have your own version of what `download_assets.sh` downloads into `runs/cleaned/`, and you can add music
to videos with it.

For running the already trained system, see [RUN_PIPELINE.md](RUN_PIPELINE.md).

```
 cleaned_dataset/ (1,011 pairs, 5-fold title split)
        │
        ▼
 A. Data stage (optional)  bgm prep → bgm shots → bgm extract → bgm segment → bgm cue build
        │                  silent 25 fps videos, shot cuts, frozen-encoder features, segments, cue tokens
        ▼
 1. bgm train semantic     150 semantic heads: 5 folds × 30 seeds (InfoNCE)
        ▼
 2. bgm fuse               retrieval settings per fold → fusion.json      (10-head ensemble, seeds 0–9)
        ▼
 3. bgm tune-stitch        Viterbi penalties per fold → stitch.json
        ▼
 4. bgm cue train          music-change predictor: 5 folds × 3 seeds
        ▼
 5. bgm cue fit-cosine     baseline rule, then
    bgm cue eval           tunes how strongly the change probabilities are used → eval.json
        ▼
 6. bgm render             add music to a video with your models
```

---

## 0. Before you start

1. Install the code and the `.venv`, as in [README section 6](README.md#6-install-and-run).
2. Download the assets:

   ```bash
   ./download_assets.sh            # manifests, cached features, music library, ImageBind, SigLIP
   ./download_assets.sh --train    # add this only for the full data stage (A); +4.4 GB of videos and music
   ```

3. Set these shell variables once per terminal. Every command below uses them.

   ```bash
   cd Sequence-Level-Background-Music-Retrieval-for-Audio-Independent-Video
   source .venv/bin/activate
   CFG="--config configs/best_semantic.yaml"
   RUNS=runs/retrained       # your models go here; the downloaded runs/cleaned/ stays untouched
   PIPE="pipeline.exp=pipeline_ens10 pipeline.dyn_variant=null pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9] eval.concat_sets_file=concat_sets_title.json train.semantic.music_feature=sem_clap_1hz train.semantic.max_clips_per_film_per_epoch=null"
   CUE="cue.sequences_file=cue_sequences_title.json"
   ```

   `PIPE` and `CUE` are the exact settings the final models were trained with. They come from each saved run's
   `config.yaml`.

Requirements: an NVIDIA GPU (all times below are on an RTX 4080 SUPER, 16 GB) and about 25 GB of free disk.

---

## A. Data stage (optional)

**You can skip this stage.** `download_assets.sh` already gives you its outputs: the cached features
(`cleaned_dataset/features/`), the music files (`cleaned_dataset/interim/music/`), `shots.csv`, `segments.csv`, the
cue tokens and the frozen sequence sets. Run it only if you want to rebuild everything from the videos, for example
after changing an encoder. It needs `./download_assets.sh --train`, and the encoders (SigLIP, CLAP, MERT) download from
Hugging Face on first use.

```bash
bgm prep    $CFG          # silent 25 fps videos + 48 kHz mono music       → cleaned_dataset/interim/
bgm shots   $CFG          # shot cuts (PySceneDetect)                     → cleaned_dataset/manifests/shots.csv
bgm extract $CFG          # SigLIP, CLAP + MERT, ImageBind, motion, keys  → cleaned_dataset/features/
bgm segment $CFG          # shots merged into 2–20 s segments             → cleaned_dataset/manifests/segments.csv
bgm cue build $CFG $CUE   # segment tokens, scene cache, sequence sets    → cleaned_dataset/features/cue_tokens.npz
```

| Command | Files that run |
|---|---|
| `bgm prep` | `data/prep.py`, `data/media.py`, `data/manifest.py` |
| `bgm shots` | `data/shots.py` |
| `bgm extract` | `features/extract.py`, `features/hf.py`, `features/dynamics.py`, `features/keys.py` |
| `bgm segment` | `data/segments.py`, `data/alignment.py` |
| `bgm cue build` | `cue/tokens.py`, `cue/scenes.py`, `cue/sequences.py` |

---

## 1. Train the semantic heads

```bash
bgm train semantic --grid $CFG paths.runs=$RUNS "train.seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]"
```

- **What it does:** for each of the 5 folds and each of the 30 seeds, trains one video head and one music head
  (`models/heads.py`) with symmetric InfoNCE (`train/losses.py`), so that matching video and music segments land close
  together. Early stopping picks the epoch with the best validation segment Recall@10.
- **Files that run:** `train/semantic.py`, `models/heads.py`, `train/losses.py`, `eval/store.py`, `eval/metrics.py`,
  `eval/protocol.py`
- **Output:** `$RUNS/semantic/siglip-both/fold<k>/seed<s>/` with `ckpt.pt`, `metrics.json` (val and test recall) and
  `train_log.csv`
- **Why 30 seeds:** the 10-head ensemble uses seeds 0–9. Videos of 60 s or longer also use seeds 10–29 to widen
  the shortlist.
- **Time:** a 2-epoch test took 14 s. A full run stops after about 25 epochs, so expect roughly 1–2 min per run and
  a few hours for all 150 (estimated).

To train one fold and one seed only (a quick check), leave out `--grid` and the seed list: it uses `fold` and `seed`
from the config (fold 0, seed 0).

---

## 2. Tune retrieval: `bgm fuse`

```bash
bgm fuse --grid $CFG paths.runs=$RUNS $PIPE
```

- **What it does:** for each fold, loads the 10 heads (seeds 0–9) as one ensemble and, on validation, sets the
  shortlist size K, the candidates per segment J and the window hop. The final system uses the semantic score only.
- **Files that run:** `retrieval/fuse.py`, `retrieval/index.py`, `retrieval/pipeline.py`
- **Output:** `$RUNS/pipeline_ens10/fold<k>/seed<s>/fusion.json` and `metrics.json` (seeds 0–2 per fold)
- **Time:** about 1 min per fold and seed, so about 15 min.

---

## 3. Tune the Viterbi penalties: `bgm tune-stitch`

```bash
bgm tune-stitch --grid $CFG paths.runs=$RUNS $PIPE
```

- **What it does:** on validation concatenations of clips, grid-searches the switch, jump and key penalties of
  Viterbi, then compares Viterbi with greedy decoding on test.
- **Files that run:** `retrieval/tune_stitch.py`, `retrieval/stitch_eval.py`, `retrieval/stitch.py`
- **Output:** `$RUNS/pipeline_ens10/fold<k>/seed<s>/stitch.json` and `stitch_metrics.json`
- **Time:** about 1 min per fold and seed, so about 15 min.

Steps 2 and 3 were checked on fold 0, seed 0: rerunning them gives the **same** `fusion.json` settings and the same
`stitch.json` penalties as the released models.

---

## 4. Train the music-change predictor: `bgm cue train`

```bash
bgm cue train --variants ctx --folds 0,1,2,3,4 --seeds 0,1,2 $CFG paths.runs=$RUNS $CUE $PIPE
```

- **What it does:** trains the 2-layer Transformer (`cue/model.py`) that reads one token per segment and predicts,
  at each boundary, the probability that the music changes. Training sequences are 60–180 s chains of training clips,
  redrawn every epoch.
- **Files that run:** `cue/train.py`, `cue/model.py`, `cue/sequences.py`, `cue/tokens.py`
- **Output:** `$RUNS/cue/ctx/fold<k>/seed<s>/ckpt.pt` and `metrics.json` (val AP and AUROC)
- **Time:** a few seconds per run, so about 1–2 min for all 15.

---

## 5. Tune the change predictor: `bgm cue eval`

```bash
bgm cue fit-cosine --folds 0,1,2,3,4 $CFG paths.runs=$RUNS $CUE $PIPE      # baseline that cue eval compares against
for k in 0 1 2 3 4; do
  bgm cue eval --fold $k --variants ctx --no-render $CFG paths.runs=$RUNS $CUE $PIPE
done
```

- **`fit-cosine` first:** `cue eval` also scores a simple rule (switch when the SigLIP cosine between neighbouring
  segments is low) and needs that rule's fitted parameters, `$RUNS/cue/cosine/fold<k>/params.json`. Without them it
  stops with `FileNotFoundError: .../cosine/fold0/params.json`.
- **What it does:** on validation sequences, tunes how strongly the change probabilities set Viterbi's switch cost
  (α) and the jump cost (λ_jump), then reports the test results.
- **Files that run:** `cue/evaluate.py`, `cue/stitch.py`, `cue/baselines.py`, `retrieval/stitch_eval.py`
- **Output:** `$RUNS/cue/eval/fold<k>/eval.json`, which render reads
- **Needs:** steps 2–4 finished for that fold. It builds its candidates with the step 2–3 models.
- **Time:** `fit-cosine` takes a few seconds; `cue eval` about 2.5 min per fold, so about 12 min.

Step 5 was checked on fold 0: rerunning it gives the **same** tuned α and λ_jump for all 3 seeds as the released
`eval.json`, and the same validation scores for every system.

---

## 6. Use your models

Add music to a video with your own models instead of the downloaded ones:

```bash
HF_HUB_OFFLINE=1 bgm render $CFG paths.runs=$RUNS \
  --video my_video.mp4 --out my_video_bgm.mp4 \
  --run $RUNS/pipeline_ens10/fold0/seed0 \
  pipeline.dyn_variant=null "pipeline.sem_seeds=[0,1,2,3,4,5,6,7,8,9]"
```

The only differences from `./add_music.sh` are `paths.runs=$RUNS` and the `--run` folder.

---

## Where to see the results

| What | File |
|---|---|
| Semantic heads: val and test recall per fold and seed | `$RUNS/semantic/siglip-both/fold<k>/seed<s>/metrics.json` |
| Training curve of one head | `$RUNS/semantic/siglip-both/fold<k>/seed<s>/train_log.csv` |
| Retrieval accuracy of the pipeline | `$RUNS/pipeline_ens10/fold<k>/seed<s>/metrics.json` |
| Viterbi vs greedy (switch F1, part track accuracy) | `$RUNS/pipeline_ens10/fold<k>/seed<s>/stitch_metrics.json` |
| Change predictor: val AP and AUROC | `$RUNS/cue/ctx/fold<k>/seed<s>/metrics.json` |
| Change predictor: tuned α, λ_jump and test results | `$RUNS/cue/eval/fold<k>/eval.json` |

## Notes

- **Fold split:** `cleaned_dataset/manifests/folds_title.json` splits by film title, so no film is in both train and
  test of a fold. Every step uses this file through `configs/best_semantic.yaml`.
- **Building `cleaned_dataset/` itself:** `tools/build_cleaned.py` made it from the hand-trimmed pairs of an earlier
  1,248-pair dataset. That earlier data and its trimming tool are not in this repository, so training starts from
  `cleaned_dataset/`.
- **Writing into `runs/cleaned/`:** setting `RUNS=runs/cleaned` would overwrite the downloaded models. Keep a
  separate folder unless you mean to replace them.
