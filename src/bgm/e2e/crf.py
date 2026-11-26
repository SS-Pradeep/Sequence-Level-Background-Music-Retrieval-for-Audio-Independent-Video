"""Linear-chain CRF over the stitching lattice (D141).

Energy of a path y (one candidate per segment), lower is better, exactly the Viterbi objective of stitch_eval:
    E(y) = sum_i u_i(y_i) + sum_i T_i(y_{i-1}, y_i)
u_i = unary cost (-emission); T_i = 0 to continue (same file, contiguous), jump_i (same file, not contiguous) or
switch_i (another file). Here jump_i / switch_i depend on the boundary (video) and on the music where the outgoing window
ends and the incoming one starts (cross-modal boundary agreement):
    switch_i(a, b) = b_sw + base_i + A_i(a) + B_i(b) + mu * (fifths(a, b) + tempo(a, b))
    jump_i(a, b)   = b_jp + base_i + A_i(a) + B_i(b)
p(y) = exp(-E(y)) / Z (forward algorithm). Training loss: -log p(Y), Y = the paths that stay on the gold track group
(offsets latent; crf_marginal_nll), which is what part track accuracy and switch F1 score. With base = 0, A = B = 0 this is the fixed-penalty
decoder with lam_switch = b_sw, lam_jump = b_jp.
"""
from __future__ import annotations

import numpy as np
import torch

from bgm.retrieval.stitch_eval import fifths_matrix, tempo_matrix


def keytempo(key_p: np.ndarray, tempo_p: np.ndarray, key_c: np.ndarray, tempo_c: np.ndarray) -> np.ndarray:
    """[n_prev, n_cur] fifths + tempo distance (stitch_eval's mu term)."""
    return fifths_matrix(key_p, key_c) + tempo_matrix(tempo_p, tempo_c)


def structure(mid_p, off_p, dur_p, mid_c, off_c, tol_s: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """(same file, contiguous) boolean [n_prev, n_cur]."""
    same = mid_p[:, None] == mid_c[None, :]
    contiguous = np.abs(off_c[None, :] - (off_p + dur_p)[:, None]) <= tol_s
    return same, contiguous


def transition(same: torch.Tensor, contiguous: torch.Tensor, kt: torch.Tensor, sw: torch.Tensor, jp: torch.Tensor,
               A: torch.Tensor, B: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """[n_prev, n_cur] transition costs. sw, jp: scalars (b + base_i); A: [n_prev] exit terms; B: [n_cur] entry terms."""
    seam = A[:, None] + B[None, :]
    switch = sw + seam + mu * kt
    jump = jp + seam
    zero = torch.zeros((), dtype=switch.dtype, device=switch.device)
    return torch.where(same, torch.where(contiguous, zero, jump), switch)


def path_energy(u: list[torch.Tensor], T: list[torch.Tensor], path: list[int]) -> torch.Tensor:
    e = u[0][path[0]]
    for i in range(1, len(u)):
        e = e + T[i - 1][path[i - 1], path[i]] + u[i][path[i]]
    return e


def log_partition(u: list[torch.Tensor], T: list[torch.Tensor]) -> torch.Tensor:
    """log sum_y exp(-E(y)) by the forward algorithm. u: per layer [n_i]; T: per boundary [n_{i-1}, n_i]."""
    alpha = -u[0]
    for i in range(1, len(u)):
        alpha = torch.logsumexp(alpha[:, None] - T[i - 1], 0) - u[i]
    return torch.logsumexp(alpha, 0)


def crf_nll(u: list[torch.Tensor], T: list[torch.Tensor], gold: list[int]) -> torch.Tensor:
    """-log p(gold) = E(gold) + log Z >= 0."""
    return path_energy(u, T, gold) + log_partition(u, T)


def crf_marginal_nll(u: list[torch.Tensor], T: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
    """-log sum_{y in Y} p(y), Y = the paths whose every pick is allowed by masks (per layer [n_i] bool, each with at
    least one True): the gold track group's windows, offsets latent. One allowed candidate per layer = crf_nll."""
    inf = torch.tensor(float("inf"), dtype=u[0].dtype, device=u[0].device)
    return log_partition(u, T) - log_partition([torch.where(m, x, inf) for x, m in zip(u, masks)], T)


def viterbi_beam(costs: list[np.ndarray], n_top: list[int], cont_of: list, T_fn, beam: int | None) -> list[int]:
    """stitch_eval.decode_lattice with the transition matrix supplied by T_fn(i, prev_idx, cur_idx) -> [len(prev), len(cur)]
    (boundary i between layers i - 1 and i). Same candidate rule and beam: top n_top plus the continuations of the
    previous survivors; the beam keeps the lowest accumulated cost."""
    act = np.arange(len(costs[0]))
    D = costs[0][act].astype(float).copy()
    acts, backs = [act], []
    for i in range(1, len(costs)):
        conts = cont_of[i][acts[-1]]
        cur = np.unique(np.r_[np.arange(n_top[i]), conts[conts >= 0]])
        tot = D[:, None] + T_fn(i, acts[-1], cur)
        bk = tot.argmin(0)
        Dc = tot.min(0) + costs[i][cur]
        if beam is not None and len(cur) > beam:
            keep = np.sort(np.argsort(Dc)[:beam])
            cur, Dc, bk = cur[keep], Dc[keep], bk[keep]
        acts.append(cur), backs.append(bk)
        D = Dc
    assert np.isfinite(D.min()), "path cost is not finite"
    j = int(np.argmin(D))
    path = [j]
    for bk in reversed(backs):
        j = int(bk[j])
        path.append(j)
    path = path[::-1]
    return [int(acts[i][path[i]]) for i in range(len(costs))]
