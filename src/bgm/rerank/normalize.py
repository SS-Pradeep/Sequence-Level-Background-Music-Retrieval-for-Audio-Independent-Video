"""Training-free re-scoring of a V2M score matrix (D140). All functions are pure.

Shapes: S [Q, G] scene queries x gallery windows (window j = test segment j's aligned music window, so the correct
window of query i is j = i); masked entries (other videos of the query's track) are -inf and stay -inf.
S_m2v [Q, G]: the same pairs scored with each window at its own duration (the reverse direction, ranked by columns).
S_bank [B, G]: query-bank scenes (the fold's training segments) x the same windows.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, logsumexp


def dsl(S: np.ndarray, S_m2v: np.ndarray, T: float) -> np.ndarray:
    """Dual softmax in log space: log softmax over windows (V2M rows) + log softmax over scenes (M2V columns).
    Transductive: the column term uses every other test scene."""
    return log_softmax(np.asarray(S, np.float64) / T, axis=1) + log_softmax(np.asarray(S_m2v, np.float64) / T, axis=0)


def hubness_z(S_bank: np.ndarray, T: float) -> np.ndarray:
    """[G] per-window hubness Z[c] = logsumexp over bank scenes of S_bank[b, c] / T."""
    return logsumexp(np.asarray(S_bank, np.float64) / T, axis=0)


def inverted_softmax(S: np.ndarray, S_bank: np.ndarray, T: float, beta: float) -> np.ndarray:
    """S / T - beta * Z[None, :] with Z from the query bank (inductive: no other test query is used)."""
    return np.asarray(S, np.float64) / T - beta * hubness_z(S_bank, T)[None, :]


def activation_set(S_bank: np.ndarray) -> np.ndarray:
    """[G] bool: windows that are the top-1 of at least one bank scene."""
    act = np.zeros(S_bank.shape[1], bool)
    act[np.argmax(S_bank, axis=1)] = True
    return act


def qbnorm(S: np.ndarray, S_bank: np.ndarray, T: float, beta: float, k: int = 1) -> np.ndarray:
    """QB-Norm with the dynamic inverted softmax (Bogolin et al., CVPR 2022): a query's row is replaced by its
    inverted-softmax row only when one of its top-k windows is in the bank's activation set; other rows are unchanged.
    Rows are ranked independently, so mixing raw and normalised rows is safe."""
    S = np.asarray(S, np.float64)
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]
    use = activation_set(S_bank)[top].any(axis=1)
    out = S.copy()
    out[use] = inverted_softmax(S[use], S_bank, T, beta)
    return out


def mutual_nn(S: np.ndarray, S_m2v: np.ndarray, m: int, eta: float) -> np.ndarray:
    """S + eta * [scene i is among the top-m scenes of window j in the M2V direction]. Transductive."""
    S = np.asarray(S, np.float64)
    order = np.argsort(-np.asarray(S_m2v, np.float64), axis=0, kind="stable")[:m]       # [m, G] top scenes per window
    bonus = np.zeros_like(S)
    bonus[order, np.arange(S.shape[1])[None, :]] = 1.0
    return S + eta * np.where(np.isfinite(S), bonus, 0.0)
