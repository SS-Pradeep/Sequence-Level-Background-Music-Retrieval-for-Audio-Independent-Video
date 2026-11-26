# Phase 3: decision and paired tests

**Outcome: no new method meets the pre-registered rule against two-stage; two-stage change-point + retrieve is the main method and the HMM / CRF / semi-Markov rows are ablations.**

**Status (2026-11-22, D150).** The rule above only decides whether a *new* decoder (semi-Markov 2a / 2b / 2c, segmental Viterbi, cross-fitted CRF) replaces two-stage. It did not pick the system used later: from D146 on, the supplementary experiments take row 12 (HMM, π from gold paths, λ tuned) as the 'final system' for their Δs. That label is not supported by this table. Two-stage (row 3) and the segmental Viterbi beat row 12 on track accuracy on both datasets: original 0.083 / 0.069 vs 0.044; cleaned 1,011 pairs 0.081 / 0.081 vs 0.048, Δ +0.033 [+0.012, +0.053] / [+0.013, +0.053], film-cluster bootstrap. Switch F1 is equal or higher (cleaned: +0.026 [−0.000, +0.052] / +0.036 [+0.008, +0.061]). Row 12 has the highest TBIoU and alignment success, because it switches more often (4.1 / min vs 3.4 true). Which decoder the paper presents as the main method is open; see docs/PROGRESS.md D150. Cleaned re-run: exp_supp/stitch_clean_table.md, runs/exp_supp/stitch_clean/results/{main_table,decision}.md.

## Rule check per candidate

- **2a Semi-Markov, sum-max emission**: fail
  - vs Two-stage change-point + retrieve: Δacc -0.042 (p 0.00011), ΔF1 -0.040 (p 0.00024) → fail
  - vs Two-stage, re-tuned (768 configs, z rule): Δacc -0.043 (p 0.00025), ΔF1 -0.036 (p 0.002) → fail
- **2b Semi-Markov, pooled emission**: fail
  - vs Two-stage change-point + retrieve: Δacc -0.044 (p 0.0002), ΔF1 -0.179 (p 2.8e-13) → fail
  - vs Two-stage, re-tuned (768 configs, z rule): Δacc -0.044 (p 0.00023), ΔF1 -0.176 (p 7.8e-13) → fail
- **2c Semi-Markov, pooled + change predictor**: fail
  - vs Two-stage change-point + retrieve: Δacc -0.042 (p 0.00016), ΔF1 -0.096 (p 5.1e-06) → fail
  - vs Two-stage, re-tuned (3,072 configs, z rule): Δacc -0.046 (p 0.00013), ΔF1 -0.096 (p 6.4e-06) → fail
- **Segmental Viterbi (semi-Markov)**: fail
  - vs Two-stage change-point + retrieve: Δacc -0.013 (p 0.33), ΔF1 -0.014 (p 0.56) → fail
  - vs Two-stage, re-tuned (768 configs, z rule): Δacc -0.014 (p 0.26), ΔF1 -0.010 (p 0.92) → fail

## Wilcoxon, every new row vs two-stage (row 3) and greedy (row 1)

| row | vs | Δ track acc | p | Δ switch F1 | p | Δ precision | Δ recall |
|---|---|---|---|---|---|---|---|
| Switch at every cut (calibration) | Two-stage change-point + retrieve | -0.041 | 8.8e-05 | -0.038 | 0.00017 | -0.139 | +0.099 |
| Switch at every cut (calibration) | Greedy top-1 | +0.002 | 0.69 | -0.014 | 0.0055 | -0.063 | +0.067 |
| Never switch (calibration) | Two-stage change-point + retrieve | -0.054 | 2.5e-05 | -0.724 | 5.9e-20 | +nan | -0.754 |
| Never switch (calibration) | Greedy top-1 | -0.011 | 0.086 | -0.701 | 6e-20 | +nan | -0.787 |
| 2a Semi-Markov, sum-max emission | Two-stage change-point + retrieve | -0.042 | 0.00011 | -0.040 | 0.00024 | -0.027 | -0.036 |
| 2a Semi-Markov, sum-max emission | Greedy top-1 | +0.000 | 0.78 | -0.016 | 0.72 | +0.048 | -0.068 |
| 2a Semi-Markov, sum-max emission | Two-stage, re-tuned (768 configs, z rule) | -0.043 | 0.00025 | -0.036 | 0.002 | -0.015 | -0.041 |
| 2b Semi-Markov, pooled emission | Two-stage change-point + retrieve | -0.044 | 0.0002 | -0.179 | 2.8e-13 | -0.099 | -0.208 |
| 2b Semi-Markov, pooled emission | Greedy top-1 | -0.001 | 0.86 | -0.155 | 1.2e-11 | -0.022 | -0.241 |
| 2b Semi-Markov, pooled emission | Two-stage, re-tuned (768 configs, z rule) | -0.044 | 0.00023 | -0.176 | 7.8e-13 | -0.087 | -0.213 |
| 2c Semi-Markov, pooled + change predictor | Two-stage change-point + retrieve | -0.042 | 0.00016 | -0.096 | 5.1e-06 | -0.007 | -0.134 |
| 2c Semi-Markov, pooled + change predictor | Greedy top-1 | +0.001 | 0.86 | -0.072 | 0.0022 | +0.070 | -0.166 |
| 2c Semi-Markov, pooled + change predictor | Two-stage, re-tuned (3,072 configs, z rule) | -0.046 | 0.00013 | -0.096 | 6.4e-06 | +0.005 | -0.143 |
| Segmental Viterbi (semi-Markov) | Two-stage change-point + retrieve | -0.013 | 0.33 | -0.014 | 0.56 | +0.022 | -0.045 |
| Segmental Viterbi (semi-Markov) | Greedy top-1 | +0.029 | 0.0022 | +0.010 | 0.061 | +0.096 | -0.077 |
| Segmental Viterbi (semi-Markov) | Two-stage, re-tuned (768 configs, z rule) | -0.014 | 0.26 | -0.010 | 0.92 | +0.033 | -0.049 |
| Two-stage, re-tuned (768 configs, z rule) | Two-stage change-point + retrieve | +0.001 | 0.72 | -0.004 | 0.25 | -0.012 | +0.005 |
| Two-stage, re-tuned (768 configs, z rule) | Greedy top-1 | +0.044 | 0.00011 | +0.020 | 0.0028 | +0.064 | -0.028 |
| Two-stage, re-tuned (3,072 configs, z rule) | Two-stage change-point + retrieve | +0.004 | 0.35 | +0.000 | 0.43 | -0.012 | +0.009 |
| Two-stage, re-tuned (3,072 configs, z rule) | Greedy top-1 | +0.047 | 3.3e-05 | +0.024 | 0.00071 | +0.063 | -0.023 |
