# Next steps: film-level split, M8 unblock, learning curve

Instructions for Claude Code (written 2026-09-29). Read CLAUDE.md, docs/SPEC.md and docs/PROGRESS.md first.

## Context

A per-query analysis of `runs/semantic/siglip-both` shows that the current folds leak by film:

- Folds are grouped by `track_id`, and fingerprints only link identical audio. So neighbouring clips of the same scene end up in train and test.
- 809 of 1,248 test clips (65%) have an adjacent clip of the same film in train/val.
- Track R@10 is 0.603 for those clips, 0.316 for the rest, and 0.278 on films never seen in training (273 clips).
- ImageBind is no better on the neighbour clips (0.084 vs 0.116), so this is memorisation.
- Konjum salangai alone is 266 of the 1,248 clips.

## Approvals (my sign-off per CLAUDE.md)

- Change the split logic as in step 1. Write NEW folds, and leave `data/manifests/folds.json` and all existing runs/results untouched.
- Fix the M8 extraction bug and resume M8.

Still ask me before deleting anything, downloading data, anything over 5 GB, unfreezing encoders, or collecting new data.

While working:

- Log every decision in docs/PROGRESS.md as D35, D36, ...
- Run pytest after each code change.
- Run long jobs in the background with logs so they survive a reboot.

## Step 0: small fixes

- In `src/bgm/eval/final.py`, compute H1 by averaging per-seed hits, like `report_semantic.py` does, not by thresholding seed-averaged ranks. Today final.py gives +0.356 and the semantic report gives +0.407.
- Update the out-of-date fold table in PROGRESS.md.

## Step 1: film-level split

**Film groups**

- Film key = `meta_source + ":" + meta_film`.
- Merge film keys that share a `track_id`. 47 tracks span more than one key, which gives 405 film groups.
- Check whether `full_bgm_dataset_youtube` and `full_bgm_dataset_youtube2` use the same numbers for the same films, by comparing SigLIP/CLAP between same-number films. If they do, merge those too.

**Folds**

- Konjum salangai is training-only: it goes in every fold's train set and never in val or test.
- Test folds: `StratifiedGroupKFold(5, shuffle=True, random_state=0)` over the other 982 clips. Groups are the film groups; stratify by duration quartile.
- Val: `GroupShuffleSplit` 10%, grouped by film.
- Save the folds as `data/manifests/folds_film.json`.
- Extend `check_folds` so that a film group never crosses train/val/test.

**Training**

- Add a config option `max_clips_per_film_per_epoch`: off by default, 30 for the film split, resampled every epoch.
- Rerun the zero-shot baselines and the semantic heads (5 folds x 3 seeds) into new run directories (`runs/filmsplit/...`). Do siglip-both first, then the other three variants.

**Report**

- Write `results/semantic_filmsplit.md` with the same tables as `results/semantic_siglip-both.md`. State the gallery sizes and show the old split side by side.
- Do NOT rerun dynamics, fusion, stitching or rendering yet.

## Step 2: unblock M8 (can run while step 1 trains)

**The bug:** `bgm pretrain --stage extract` stops at `u01126_00`, where MERT gives 20 frames and CLAP gives 19. Pool clips are cut to whole seconds, and the 48 → 24 kHz resample most likely rounds the length up by one sample.

**Fix:** in `_sem_music` (`src/bgm/features/extract.py`), add this just before the assert:

```python
if len(arr["sem_mert_1hz"]) == T1 + 1:
    arr["sem_mert_1hz"] = arr["sem_mert_1hz"][:T1]
    if "sem_mert_layers_1hz" in arr:
        arr["sem_mert_layers_1hz"] = arr["sem_mert_layers_1hz"][:, :T1]
```

Add a test for this case, then resume the extraction from its cache.

**Screen the pool for labeled films.** With a film split, every labeled film is a test film in some fold, so pretraining must not see any of them.

1. Calibrate a SigLIP similarity threshold on the labeled set, using same-film vs different-film clips.
2. Flag pool source videos above that threshold.
3. **STOP** and show me the count and about 20 example pairs before dropping anything.

**After I approve:**

1. Pretrain, then fine-tune on the film-split folds, into new run directories.
2. Write `results/pretraining_filmsplit.md`: pretrained vs scratch, track R@10 with cluster-bootstrap CIs (H4).

## Step 3: learning curve (after step 1)

- Per fold, take nested random subsets of the non-Konjum training films: 25%, 50%, 75% and 100%. Keep Konjum in, with the cap. Val and test stay unchanged.
- Run seed 0 first (20 runs). Add seeds 1 and 2 only if the trend is unclear.
- Write `results/learning_curve.md` with:
  - a plot of test R@10, R@1 and MedR against the number of training films, with bootstrap CIs;
  - the gains from 50% to 100% and from 75% to 100%.

## Then STOP

Report the new headline numbers, the learning-curve verdict, and H4. Don't start data collection or model changes; I'll decide.
