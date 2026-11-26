# Demo videos on the cleaned dataset (final system)

Final system: cleaned 1,011 pairs, title split, 10-seed ensemble (`runs/cleaned/pipeline_ens10`), Viterbi stitching.
Each clip is run with the models of the fold that holds it in its TEST split (film and music unseen in training).
Library searched: all 1,011 cleaned music files. Made by `08_experimental_results.py videos ...` (log: `videos.log`).

Each folder is named by the clip's unique pair ID (`cleaned_dataset/manifests/pairs.csv`) and holds:

- `original_video.mp4`: the cleaned clip with its **original film soundtrack**
- `original_music.wav`: the original background music alone (ground truth)
- `output_with_bgm.mp4`: the same clip with the music our system picked
- `soundtrack.wav`: our soundtrack alone; `plan.json`: scenes, chosen files and windows
- `summary.txt`: the full report (scenes, chosen music, ranks of the true music); `timeline.png`, `contact_sheet.png`

| Unique pair ID | Scene | Test fold | Length (s) | Hand edit |
|---|---|---|---|---|
| `combined_full_bgm_dataset_youtube__14__14_bgm_0002` | step-by-step example (steps 01-07 of the old demo) | 2 | 44.0 | kept whole |
| `sunnxt_255_255_bgm_0000` | playful / light romance (cafe, waitress) | 4 | 33.0 | kept whole |
| `sunnxt_437_437_bgm_0001` | joyful / comedy (friends on a street, bike ride) | 1 | 18.2 | trimmed to 13.4-31.6 s of `sunnxt_437_437_bgm_0001` |
| `sunnxt_460_460_bgm_0001_p1` | romance (couple, ring, fairy lights), part 1 | 0 | 37.0 | trimmed to 2.6-39.7 s of `sunnxt_460_460_bgm_0001` |
| `sunnxt_460_460_bgm_0001_p2` | romance (couple, ring, fairy lights), part 2 | 0 | 16.1 | trimmed to 42.0-58.1 s of `sunnxt_460_460_bgm_0001` |
| `sunnxt_526_526_bgm_0000` | emotional / tense (mother and child at night) | 4 | 33.5 | kept whole |
| `sunnxt_545_545_bgm_0002` | thriller (dark house, torch light) | 0 | 53.0 | kept whole |
| `sunnxt_553_553_bgm_0002` | village drama (rural meal, serious faces) | 1 | 40.0 | kept whole |
| `sunnxt_85_85_bgm_0002` | suspense / outdoor (car sinking in a lake) | 4 | 49.0 | kept whole |
| `combined_full_bgm_dataset_youtube2__104__104_bgm_0001` | war / action (soldiers in desert dunes, tank) | 4 | 46.4 | kept whole |
| `combined_full_bgm_dataset_youtube__91__91_bgm_0002` | grief / tragedy (mourning at a temple) | 0 | 66.4 | kept whole |
