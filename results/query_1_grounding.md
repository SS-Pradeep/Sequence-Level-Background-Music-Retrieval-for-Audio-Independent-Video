# 1.mp4: grounding with chance levels

Silent query media/inputs/1.mp4 (347.3 s); its own soundtrack from media/inputs/1_BGM_Ori.mp4 (verified aligned: identical frames, audio lag 0). 54 segments (cut-based, the pipeline's segmentation), each against every 25 Hz offset of the soundtrack. No training; the soundtrack is in no library file, so all 15 models per system are leak-free. Hits averaged over the 15 models per segment; 95% CIs resample segments (one video, so they treat neighbouring segments as independent and are descriptive).

| system | rule | accuracy | chance | accuracy − chance (95% CI) | per-model accuracy (min–max) |
| --- | --- | --- | --- | --- | --- |
| learned dynamics, shifted negatives (M7) | exact (±0.1 s) | 0.001 [0.000, 0.004] | 0.001 | **+0.001 [-0.001, +0.003]** | 0.000–0.019 |
| learned dynamics, shifted negatives (M7) | within 0.5 s | 0.002 [0.000, 0.006] | 0.003 | **-0.000 [-0.003, +0.003]** | 0.000–0.019 |
| learned dynamics, shifted negatives (M7) | sync (exact, or the shift still lines up) | 0.267 [0.235, 0.299] | 0.314 | **-0.047 [-0.079, -0.015]** | 0.185–0.370 |
| learned dynamics, no shifted negatives (M7) | exact (±0.1 s) | 0.002 [0.000, 0.007] | 0.001 | **+0.002 [-0.001, +0.007]** | 0.000–0.019 |
| learned dynamics, no shifted negatives (M7) | within 0.5 s | 0.005 [0.000, 0.011] | 0.003 | **+0.002 [-0.003, +0.008]** | 0.000–0.019 |
| learned dynamics, no shifted negatives (M7) | sync (exact, or the shift still lines up) | 0.275 [0.243, 0.309] | 0.314 | **-0.038 [-0.070, -0.006]** | 0.185–0.389 |
| fused semantic + dynamics + mood (M9) | exact (±0.1 s) | 0.003 [0.000, 0.009] | 0.001 | **+0.003 [-0.001, +0.008]** | 0.000–0.019 |
| fused semantic + dynamics + mood (M9) | within 0.5 s | 0.005 [0.000, 0.011] | 0.003 | **+0.002 [-0.003, +0.008]** | 0.000–0.019 |
| fused semantic + dynamics + mood (M9) | sync (exact, or the shift still lines up) | 0.282 [0.252, 0.312] | 0.314 | **-0.032 [-0.059, -0.005]** | 0.185–0.407 |
| hand-crafted dynamics (cut-on-beat + motion/energy) | exact (±0.1 s) | 0.000 [0.000, 0.000] | 0.001 | **-0.001 [-0.001, -0.001]** | 0.000–0.000 |
| hand-crafted dynamics (cut-on-beat + motion/energy) | within 0.5 s | 0.000 [0.000, 0.000] | 0.003 | **-0.003 [-0.003, -0.003]** | 0.000–0.000 |
| hand-crafted dynamics (cut-on-beat + motion/energy) | sync (exact, or the shift still lines up) | 0.296 [0.185, 0.426] | 0.314 | **-0.017 [-0.135, +0.107]** | 0.296–0.296 |

Paired, same segments, on (hit − chance):

- shiftneg − hand: exact +0.001 [+0.000, +0.004]; within 0.5 s +0.002 [+0.000, +0.006]; sync -0.030 [-0.157, +0.090]
- noshift − hand: exact +0.002 [+0.000, +0.007]; within 0.5 s +0.005 [+0.000, +0.011]; sync -0.021 [-0.151, +0.105]
- fused − hand: exact +0.003 [+0.000, +0.009]; within 0.5 s +0.005 [+0.000, +0.011]; sync -0.014 [-0.144, +0.110]

Median absolute error (s): shiftneg 88.3, noshift 96.0, fused 107.3, hand 102.3. Fused queries with tied top offsets: 52%.

