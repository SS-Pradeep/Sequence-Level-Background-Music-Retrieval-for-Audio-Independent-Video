# zeroshot

Mean ± std over 5 folds; R@10 and grounding rates add a 95% cluster-bootstrap CI (resampling track_id groups over the pooled test queries).

## Track retrieval, gallery = the test fold's music files

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| imagebind | 0.015 ± 0.006 | 0.054 ± 0.013 | 0.110 ± 0.016 [0.088, 0.133] | 65.500 ± 5.723 | 0.054 ± 0.007 |
| random | 0.006 ± 0.000 | 0.029 ± 0.001 | 0.058 ± 0.003 | — | 0.033 ± 0.001 |
| text_bridge | 0.014 ± 0.013 | 0.059 ± 0.023 | 0.105 ± 0.022 [0.085, 0.126] | 67.800 ± 6.301 | 0.052 ± 0.012 |
| attribute | 0.008 ± 0.009 | 0.049 ± 0.018 | 0.095 ± 0.025 [0.075, 0.115] | 72.900 ± 6.485 | 0.045 ± 0.009 |

## Track retrieval, gallery = all library music files

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| imagebind | 0.005 ± 0.004 | 0.018 ± 0.009 | 0.028 ± 0.013 [0.017, 0.041] | 334.000 ± 47.509 | 0.018 ± 0.006 |
| random | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | — | 0.007 ± 0.000 |
| text_bridge | 0.000 ± 0.000 | 0.008 ± 0.011 | 0.018 ± 0.017 [0.010, 0.027] | 411.000 ± 29.283 | 0.010 ± 0.004 |
| attribute | 0.002 ± 0.005 | 0.008 ± 0.006 | 0.017 ± 0.015 [0.009, 0.027] | 450.100 ± 42.205 | 0.011 ± 0.004 |

## Segment retrieval, all windows

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| imagebind | 0.008 ± 0.004 | 0.023 ± 0.004 | 0.042 ± 0.007 [0.032, 0.054] | 183.700 ± 16.498 | 0.025 ± 0.004 |
| random | 0.002 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.002 | — | 0.013 ± 0.001 |
| text_bridge | 0.002 ± 0.002 | 0.014 ± 0.006 | 0.033 ± 0.010 [0.024, 0.042] | 200.200 ± 29.295 | 0.018 ± 0.003 |
| attribute | 0.004 ± 0.004 | 0.016 ± 0.004 | 0.032 ± 0.006 [0.023, 0.041] | 212.800 ± 26.395 | 0.019 ± 0.005 |

## Segment retrieval, other-video windows only

| system | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- |
| imagebind | 0.011 ± 0.004 | 0.025 ± 0.004 | 0.044 ± 0.006 [0.034, 0.056] | 182.500 ± 16.101 | 0.028 ± 0.004 |
| random | 0.002 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.002 | — | 0.014 ± 0.001 |
| text_bridge | 0.004 ± 0.003 | 0.016 ± 0.007 | 0.035 ± 0.009 [0.026, 0.044] | 198.400 ± 28.902 | 0.020 ± 0.004 |
| attribute | 0.007 ± 0.006 | 0.017 ± 0.004 | 0.033 ± 0.007 [0.024, 0.042] | 210.600 ± 25.851 | 0.022 ± 0.006 |

## Grounding (own music file given, every 25 Hz offset; rules and chance levels as in results/dynamics.md)

| system | rule | accuracy | chance | accuracy − chance |
| --- | --- | --- | --- | --- |
| hand_dynamics | exact (±0.1 s) | 0.135 ± 0.031 [0.117, 0.149] | 0.122 ± 0.028 [0.107, 0.134] | 0.013 ± 0.003 [0.004, 0.021] |
| hand_dynamics | within 0.5 s | 0.215 ± 0.042 [0.194, 0.232] | 0.184 ± 0.033 [0.167, 0.198] | 0.031 ± 0.013 [0.020, 0.042] |
| hand_dynamics | sync (exact, or the shift still lines up) | 0.443 ± 0.035 [0.421, 0.461] | 0.443 ± 0.016 [0.431, 0.452] | 0.000 ± 0.022 [-0.018, 0.015] |
| hand_dynamics | median abs error s | 3.388 ± 0.388 | | |

Calibration temperatures per fold: [{"attr": {"video": 0.1, "music": 0.1}, "bridge": {"video": 0.2, "music": 0.1}}, {"attr": {"video": 0.1, "music": 0.1}, "bridge": {"video": 0.2, "music": 0.1}}, {"attr": {"video": 0.1, "music": 0.1}, "bridge": {"video": 0.2, "music": 0.2}}, {"attr": {"video": 0.1, "music": 0.1}, "bridge": {"video": 0.2, "music": 0.1}}, {"attr": {"video": 0.1, "music": 0.1}, "bridge": {"video": 0.2, "music": 0.1}}]
