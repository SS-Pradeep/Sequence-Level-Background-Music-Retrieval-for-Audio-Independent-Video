"""Attribute distillation as an auxiliary loss on the semantic heads (architecture plan Phase 3.1; PROGRESS.md D62).

The video head's output also has to predict the paired music span's profile through one linear layer:
  16 CLAP mood logits (span mean of the music file's attr_logits_1hz, span rule applied),
  loudness (mean rms_db over the span), onset density (mean log_onset over the span), log tempo (file level).
Targets are z-scored with the training split's statistics; the loss is the mean squared error, added as
lambda x MSE to InfoNCE. Vocal presence (in the plan) is not in the feature files, so it is left out.
The profile comes from the music and is used only as a training target on the train split (rule 8 holds: no
video feature reads the soundtrack). The linear layer is dropped after training; checkpoints are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from bgm.eval.store import FeatureStore
from bgm.features.dynamics import M_CHANNELS

RATE = 25
RMS, ONSET = M_CHANNELS.index("rms_db"), M_CHANNELS.index("log_onset")


def music_profile(store: FeatureStore, m: str, a: float, b: float) -> np.ndarray:
    """Raw profile [M + 3] of music file m over [a, b) s: mood logits, rms_db, log_onset, log tempo (nan if unknown)."""
    d = store.music(m)
    mood = store.pooled("music", m, "attr_logits_1hz", a, b)
    dyn = d["dyn_25hz"]
    lo = int(np.clip(round(a * RATE), 0, len(dyn) - 1))
    hi = int(np.clip(round(b * RATE), lo + 1, len(dyn)))
    tempo = float(d["tempo_bpm"])
    return np.concatenate([mood, [dyn[lo:hi, RMS].mean(), dyn[lo:hi, ONSET].mean(), np.log(tempo) if tempo > 0 else np.nan]]).astype(np.float64)


class ProfileTargets:
    """z-scored profiles; statistics from every segment of the training pairs."""

    def __init__(self, store: FeatureStore, train_ids: list[str]):
        self.store = store
        X = np.stack([music_profile(store, store.music_of[p], a, b) for p in train_ids for a, b in store.segments[p]])
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        self.dim = X.shape[1]

    def __call__(self, p: str, a: float, b: float) -> np.ndarray:
        z = (music_profile(self.store, self.store.music_of[p], a, b) - self.mu) / self.sd
        return np.nan_to_num(z, nan=0.0).astype(np.float32)          # unknown tempo -> the train mean


def distill_loss(head: torch.nn.Module, zv: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(head(zv), target)
