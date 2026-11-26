# src/bgm/sweep/configs/

One YAML file per loss configuration of the sweep. Fields: `name`, `main` (the main contrastive loss from
`bgm.losses.registry`), `aux` (auxiliary terms and their weights) and `weighting`.

| Config | Loss |
|---|---|
| `infonce.yaml` | Symmetric InfoNCE (the final system's loss). |
| `siglip.yaml` | Sigmoid (SigLIP-style) pairwise loss. |
| `siglip+<term>.yaml` | SigLIP loss plus one auxiliary term: `center`, `csrd`, `intra`, `orth`, `rdrop`, `recon`, `supcon`, `vicreg`. |
| `soft.yaml` | Soft-target contrastive loss (targets from music–music similarity). |
| `soft+tag_teacher.yaml` | Soft targets mixed with LLM tag similarity. |
| `debiased.yaml` | Debiased contrastive loss (Chuang et al. 2020). |
| `hardneg.yaml` | Hard-negative-weighted contrastive loss. |
| `vsepp.yaml` | VSE++ max-violation triplet loss. |
| `hgm.yaml` | Distance-margin loss: positives pulled within one margin, negatives pushed beyond another. |
| `pq_tau.yaml` | InfoNCE with a learned per-query temperature. |

Run one with the sweep driver, e.g. `python -m bgm.sweep.titlesplit run --seeds 0`.
