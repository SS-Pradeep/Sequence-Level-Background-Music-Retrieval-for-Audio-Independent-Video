# Supplementary C: music changes at shot cuts inside continuous test clips (cleaned data)

Candidates: every TransNetV2 boundary inside a test clip of the 112 frozen title-split test sequences, scored by the
frozen segmentation variants (no retraining). Labels: D103 rule (music change within ±1 s: calibrated MERT jump or a
drop to silence), rebuilt on cleaned_dataset with the calibration refit on each fold's cleaned training clips; cuts in
excluded clips or trimmed-away spans drop out. 95% CI: cluster bootstrap over film groups, 10,000 resamples.

Cleaned: 2,791 cuts, 417 with a music change (base rate 0.149 = AP of a random score), 195 film groups. Original data (paper): 3,467 cuts, 452 changes (reproduced exactly).

| Segmenter | AP [95% CI] | AP − chance [95% CI] | AUROC [95% CI] | AUROC, original data |
|---|---|---|---|---|
| Random score | 0.149 | 0 | 0.500 | 0.500 |
| TransNetV2 + BaSSL, original head | 0.154 [0.131, 0.185] | +0.005 [-0.010, 0.027] | 0.504 [0.466, 0.543] | 0.515 |
| TransNetV2 + BaSSL, trained head (4) | 0.161 [0.134, 0.195] | +0.011 [-0.004, 0.036] | 0.503 [0.469, 0.535] | 0.520 |
| TransNetV2 + BaSSL, trained head (ours, C) | 0.163 [0.135, 0.201] | +0.013 [-0.002, 0.037] | 0.522 [0.481, 0.560] | 0.526 |
| 4b + duration, motion | 0.170 [0.143, 0.205] | +0.020 [0.003, 0.047] | 0.517 [0.482, 0.550] | 0.535 |
