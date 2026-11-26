"""Contrastive losses (SPEC §5.6), ported as-is."""
import torch
import torch.nn.functional as F


def info_nce(zv, zm, track_ids, log_tau, tau_min: float = 0.01):
    """Symmetric InfoNCE. zv, zm: [B, D], L2-normalised; track_ids: LongTensor [B];
    log_tau: learnable scalar, start log(0.07), floored at tau_min (train.semantic.tau_min, D138).
    Same-track pairs are removed from the negatives."""
    tau = log_tau.exp().clamp(min=tau_min)
    logits = zv @ zm.T / tau
    same = track_ids[:, None] == track_ids[None, :]
    eye = torch.eye(len(zv), dtype=torch.bool, device=zv.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    y = torch.arange(len(zv), device=zv.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def info_nce_multi(zv, zm, group_ids, log_tau, tau_min: float = 0.01):
    """Multi-positive symmetric InfoNCE (D66). group_ids: LongTensor [B]; every pair (i, j) with the same group is a
    positive: row i's loss is -log( sum_{j in pos(i)} exp(s_ij / tau) / sum_j exp(s_ij / tau) ). With one item per
    group in the batch this equals info_nce."""
    tau = log_tau.exp().clamp(min=tau_min)
    logits = zv @ zm.T / tau
    pos = group_ids[:, None] == group_ids[None, :]

    def side(lg):
        return -(torch.logsumexp(lg.masked_fill(~pos, float("-inf")), 1) - torch.logsumexp(lg, 1)).mean()
    return 0.5 * (side(logits) + side(logits.T))


def info_nce_weighted(zv, zm, group_ids, log_tau, w_soft: float = 0.5):
    """Weighted multi-positive InfoNCE (D74). For row i with positive set P(i) (same group; the paired item is the
    primary positive, weight 1; the others are soft positives, weight w_soft):
        L_i = -(1 / sum_m w_m) * sum_{m in P(i)} w_m * log( exp(s_im / tau) / (exp(s_im / tau) + sum_{j not in P(i)} exp(s_ij / tau)) )
    so each positive competes only with the non-positives: the other positives are masked out of its denominator.
    Symmetric over the two directions. With one item per group this equals info_nce."""
    tau = log_tau.exp().clamp(min=0.01)
    logits = zv @ zm.T / tau
    pos = group_ids[:, None] == group_ids[None, :]
    eye = torch.eye(len(zv), dtype=torch.bool, device=zv.device)
    w = torch.where(eye, torch.ones_like(logits), torch.full_like(logits, w_soft)) * pos

    def side(lg):
        neg = torch.logsumexp(lg.masked_fill(pos, float("-inf")), 1, keepdim=True)        # [B, 1]; -inf if a row has no negative
        term = lg - torch.logaddexp(lg, neg)                                               # log p of each candidate vs the negatives
        return -((w * term).sum(1) / w.sum(1)).mean()
    return 0.5 * (side(logits) + side(logits.T))


def _same_track_mask(zv, track_ids):
    """[B, B] bool: off-diagonal pairs that share a track (never negatives)."""
    eye = torch.eye(len(zv), dtype=torch.bool, device=zv.device)
    return (track_ids[:, None] == track_ids[None, :]) & ~eye, eye


def sigmoid_loss(zv, zm, track_ids, log_tau, bias):
    """SigLIP pairwise sigmoid loss (D124). Every (i, j) is an independent binary match: label +1 on the diagonal,
    -1 off it; same-track off-diagonal pairs are left out. logits = s_ij / tau + bias; bias: learnable scalar (init -10
    as in SigLIP). Mean over the kept pairs of -log sigmoid(label * logit), times B (SigLIP's per-row sum)."""
    tau = log_tau.exp().clamp(min=0.01)
    same, eye = _same_track_mask(zv, track_ids)
    logits = zv @ zm.T / tau + bias
    lab = 2.0 * eye.float() - 1.0
    ll = F.logsigmoid(lab * logits).masked_fill(same, 0.0)
    return -ll.sum() / len(zv)


def hn_nce(zv, zm, track_ids, log_tau, beta: float = 0.5):
    """Hard-negative InfoNCE (HN-NCE, Radenovic et al. 2023; D124). Each negative's exp(s_ij / tau) in the
    denominator is weighted by w_ij = (n_neg) * softmax_j(beta * s_ij / tau) over the row's negatives (weights
    detached, mean 1), so harder negatives count more. beta = 0 gives info_nce. Symmetric; same-track pairs masked."""
    tau = log_tau.exp().clamp(min=0.01)
    same, eye = _same_track_mask(zv, track_ids)
    logits = zv @ zm.T / tau

    def side(lg):
        neg = ~(same | eye)
        n_neg = neg.sum(1, keepdim=True).clamp(min=1).float()
        lw = torch.log_softmax((beta * lg).masked_fill(~neg, float("-inf")), 1).detach() + n_neg.log()
        den = torch.logaddexp(lg.diagonal(), torch.logsumexp((lg + lw).masked_fill(~neg, float("-inf")), 1))
        return (den - lg.diagonal()).mean()
    return 0.5 * (side(logits) + side(logits.T))


def dcl(zv, zm, track_ids, log_tau):
    """Decoupled contrastive loss (Yeh et al. 2022; D124): InfoNCE with the positive removed from the denominator,
    -s_ii / tau + logsumexp_{j negative} s_ij / tau. Symmetric; same-track pairs masked."""
    tau = log_tau.exp().clamp(min=0.01)
    same, eye = _same_track_mask(zv, track_ids)
    logits = zv @ zm.T / tau

    def side(lg):
        return (torch.logsumexp(lg.masked_fill(same | eye, float("-inf")), 1) - lg.diagonal()).mean()
    return 0.5 * (side(logits) + side(logits.T))


def vse_max(zv, zm, track_ids, margin: float = 0.2):
    """VSE++ max-violation hinge (Faghri et al. 2018; D124): [margin - s_ii + max_j s_ij]_+ over the hardest
    negative in each direction, on raw cosines (no temperature). Same-track pairs masked."""
    same, eye = _same_track_mask(zv, track_ids)
    s = (zv @ zm.T).masked_fill(same | eye, float("-inf"))
    pos = (zv * zm).sum(1)
    hv = (margin - pos + s.max(1).values).clamp(min=0)
    hm = (margin - pos + s.max(0).values).clamp(min=0)
    return 0.5 * (hv.mean() + hm.mean())


def distill_kl(zv, zm, teacher_sim, track_ids, log_tau, tau_teacher: float = 0.07):
    """Ensemble distillation (D76): KL(teacher || student) between the in-batch match distributions, both directions.
    teacher_sim: [B, B] the teachers' mean cosine for the same inputs. Same-track pairs are masked in both."""
    tau = log_tau.exp().clamp(min=0.01)
    same = track_ids[:, None] == track_ids[None, :]
    eye = torch.eye(len(zv), dtype=torch.bool, device=zv.device)
    mask = same & ~eye
    s = (zv @ zm.T / tau).masked_fill(mask, float("-inf"))
    t = (teacher_sim / tau_teacher).masked_fill(mask, float("-inf"))

    def side(a, b):
        p = torch.softmax(b, 1)
        return (p * (torch.log_softmax(b, 1) - torch.log_softmax(a, 1)).masked_fill(mask, 0.0)).sum(1).mean()
    return 0.5 * (side(s, t) + side(s.T, t.T))


def dynamics_loss(hv, hm, hm_shift, shift_valid, track_ids, log_tau):
    """hv, hm: [B, T, D] per-step unit vectors of aligned video / music windows (cropped from context).
    hm_shift: [B, K, T, D] shifted music windows sliced from the same music encoding;
    shift_valid: [B, K] bool. track_ids: LongTensor [B]. (Index logic checked against a loop.)

    Note: the temperature divides BEFORE masking. Masking with -inf first and dividing after gives the learnable
    temperature a NaN gradient (-inf / tau has d/dtau = inf * 0); found in the legacy implementation."""
    tau = log_tau.exp().clamp(min=0.01)
    T = hv.shape[1]
    s_batch = torch.einsum("btd,ctd->bc", hv, hm) / T / tau          # [B, B], positives on the diagonal
    same = track_ids[:, None] == track_ids[None, :]
    eye = torch.eye(len(hv), dtype=torch.bool, device=hv.device)
    s_batch = s_batch.masked_fill(same & ~eye, float("-inf"))
    s_shift = torch.einsum("btd,bktd->bk", hv, hm_shift) / T / tau    # [B, K]
    s_shift = s_shift.masked_fill(~shift_valid, float("-inf"))
    logits = torch.cat([s_batch, s_shift], dim=1)
    y = torch.arange(len(hv), device=hv.device)
    return F.cross_entropy(logits, y)
