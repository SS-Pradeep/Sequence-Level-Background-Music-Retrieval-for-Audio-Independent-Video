# Music change at shot cuts (D103)

Title-grouped film split (`folds_title.json`), 5 folds. Every measurable shot cut **inside** a clip; label 1 if the clip's own music changes within ±1 s (MERT jump, calibrated per fold on train clips only, or a drop to silence). Test cuts pooled over folds. Thresholds for F1 chosen on val (small: about 150 cuts per fold). Boundary F1: ±1 cut, one-to-one. CIs: paired cluster bootstrap over film groups, 2,000 resamples.

Test cuts 3821, changes 487 (12.7%); chance AP = 0.127.

| system | AP | AUROC | boundary F1 | precision | recall | Δ AP vs visual rule [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| every cut | 0.127 | 0.500 | 0.226 | 0.127 | 1.000 | — |
| time since start (rule) | 0.130 | 0.496 | 0.231 | 0.131 | 0.969 | -0.036 [-0.074, +0.001] |
| visual distance (rule) | 0.167 | 0.536 | 0.240 | 0.153 | 0.554 | — |
| LR video | 0.195 | 0.577 | 0.248 | 0.195 | 0.339 | +0.028 [+0.003, +0.057] |
| LR time | 0.123 | 0.492 | 0.239 | 0.145 | 0.688 | -0.043 [-0.069, -0.020] |
| LR music | 0.214 | 0.606 | 0.254 | 0.178 | 0.444 | +0.048 [+0.010, +0.087] |
| LR video + time | 0.194 | 0.576 | 0.247 | 0.196 | 0.333 | +0.027 [+0.003, +0.056] |
| LR video + music | 0.238 | 0.621 | 0.279 | 0.215 | 0.400 | +0.072 [+0.041, +0.106] |
| LR all | 0.239 | 0.622 | 0.284 | 0.214 | 0.421 | +0.073 [+0.042, +0.108] |
| MLP all | 0.275 | 0.641 | 0.338 | 0.265 | 0.466 | +0.108 [+0.077, +0.141] |
| MLP video | 0.177 | 0.552 | 0.264 | 0.185 | 0.466 | +0.011 [-0.012, +0.037] |
| GBM all | 0.229 | 0.599 | 0.312 | 0.378 | 0.265 | +0.062 [+0.032, +0.096] |
| deep MLP all | 0.251 | 0.626 | 0.293 | 0.235 | 0.388 | +0.084 [+0.052, +0.120] |
| transformer (clip sequence) | 0.215 | 0.599 | 0.258 | 0.190 | 0.405 | +0.049 [+0.021, +0.082] |
| ensemble (GBM + deep MLP + transformer) | 0.266 | 0.648 | 0.302 | 0.232 | 0.433 | +0.099 [+0.071, +0.131] |

Ablations (paired Δ AP [95% CI]):

- LR video + music − LR video: +0.044 [+0.017, +0.071]
- LR video + time − LR video: -0.001 [-0.002, +0.000]
- LR all − LR video: +0.045 [+0.018, +0.072]
- LR all − LR music: +0.025 [-0.007, +0.058]
- MLP all − LR all: +0.035 [+0.004, +0.067]
- LR video − time since start (rule): +0.064 [+0.024, +0.109]
- GBM all − MLP all: -0.046 [-0.076, -0.013]
- deep MLP all − MLP all: -0.024 [-0.049, +0.001]
- transformer (clip sequence) − MLP all: -0.059 [-0.091, -0.029]
- ensemble (GBM + deep MLP + transformer) − MLP all: -0.009 [-0.034, +0.017]

Per fold (labels): fold 0: 75/722 test, 18/156 val, d at p=0.5 0.092; fold 1: 122/777 test, 37/266 val, d at p=0.5 0.090; fold 2: 130/891 test, 31/325 val, d at p=0.5 0.089; fold 3: 85/601 test, 37/356 val, d at p=0.5 0.091; fold 4: 75/830 test, 22/159 val, d at p=0.5 0.092
