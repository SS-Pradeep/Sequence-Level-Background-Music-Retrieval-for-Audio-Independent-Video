# Submitted paper vs repo: correction log

Paper: `Sequence_final.pdf` (CVPR 2027 submission). The PDF is deliberately **not** edited (user, 2026-11-23). This file records
where its text differs from the code and saved results. When writing the supplement, rebuttal or a later version, use the
"Repo record" column. Logged as D155.

## Claims that differ from the repo

| # | Paper | Paper says | Repo record | Source |
|---|---|---|---|---|
| 1 | l.523–525 | The LLM was not used to design the method, run experiments or produce results | All 94 commits at the time of this check carry a Claude co-author line, and the decision log records Claude implementing and running experiments. The statement is not accurate for the experiments and results | `git log`, `docs/PROGRESS.md` |
| 2 | l.456–457 | Trained BaSSL segmentation is used in all later experiments | BaSSL is used by the decoder study (Table 3). Retrieval, Table 1, the change predictor and the 2,401 segments use PySceneDetect + SigLIP merge | `results/hyperparameters.md` |
| 3 | l.458–459 | Table 2 tests real music changes inside continuous film | Every true change in Table 2 is a clip join, so it does **not** test this. The in-clip probe is at chance | `results/segmentation_comparison.md`, `exp_supp/cut_probe_retrained.md` |
| 4 | l.17–20 | Viterbi places changes better than greedy, with a closer switching rate | True on the 250 concatenations (0.739 vs 0.650). In Table 3 greedy is higher (0.703 vs 0.694) and closer to the true rate (3.81 vs 4.16, true 3.10). On cleaned sequences they tie (greedy − HMM −0.007 [−0.022, +0.008]) | `results/cleaned/ablation.md`, `exp_supp/stitch_clean_table.md` |
| 5 | Tables 3, 4 | Both read as one final system | Table 3: original 1,248-pair lattices, BaSSL scenes, no learned predictor. Table 4: cleaned 1,011 pairs, PySceneDetect segments, learned predictor in the boundary-F1 rows only (the greedy-vs-Viterbi row is a separate concatenation test) | `exp_supp/hsmm_table.md`, `results/hyperparameters.md` |
| 6 | l.487–490 | The predictor beats fixed penalties "partly" through a shortcut | Without duration it ties them (−0.010 [−0.035, +0.010]); the full predictor ties a duration-only prior (+0.007 [−0.015, +0.029]) | `results/cleaned/cue_switch.md`, `results/cleaned/oracle_p_table.md` |
| 7 | l.221 | Heads have 1.3M parameters | 1,645,313 (1.65M) | `results/hyperparameters.md` |
| 8 | l.387, Fig. 2 | Clips come from public YouTube uploads | 499 of the 1,011 cleaned pairs are SunNXT | `results/dataset_stats.md` |
| 9 | l.391, Fig. 2 | Clips are 10–60 s / 10–20 s | Original 9.84–171 s, median 15.0 s; cleaned 1.07–171 s, median 13.0 s | manifests |
| 10 | l.213 | Music is encoded in one-second segments | Features are at 1 Hz, but CLAP uses 10 s windows and MERT 30 s chunks + 5 s context | `results/hyperparameters.md` |
| 11 | l.250 | The coarse stage keeps the 10 best tracks | Retrieval uses the union of the top 10 files for the whole video and for each segment. The decoder-study lattice uses K = 10 | `results/hyperparameters.md` |
| 12 | l.492 | RTX 4080 | RTX 4080 SUPER, 16 GB | `nvidia-smi` |
| 13 | Table 4 | No development label | The encoder rows use one ensemble run per fold, not 5 folds × 3 seeds | `results/cleaned/ablation.md` |
| 14 | References | [1, 6, 16, 33, 36, 43, 54] are listed | They are not cited in the text | PDF |
| 15 | l.391, l.426 | "228 several movies"; unclosed parenthesis | Typos | PDF |

## Checked and consistent

- About 228 movies is valid for the original data: 228 real films + 15 compilations (445 film keys, 219 film groups). Cleaned: 420 keys, 211 groups.
- Konjum Salangai has 266 clips before cleaning and 117 after, and is train-only.
- Table 1, the 278 / 725 / 245 cleaning counts, 224 title links, 69.5% and 82.6% music, 0.9% speech and 894 test pairs.

## Not checked

- CVPR's rules on correcting a submission or adding post-deadline results.
- Whether each reference's authors, title and venue are accurate.
