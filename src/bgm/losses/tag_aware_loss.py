"""
Tag-Aware version of the Score-Aware loss.

Adds the LLM tag probabilities (Qwen2-Audio for music, Qwen2.5-VL for scenes)
to training in two independent ways, each with its own switch:

  1. Tag teacher  (beta)   : the soft targets mix the CLAP music-music teacher
                             with a scene-tag vs music-tag similarity.
  2. Tag heads    (lam_tag): the embeddings must predict tag distributions:
                             v -> scene tags, m -> music tags, and (cross)
                             v -> its paired music's tags.

beta = 0 and lam_tag = 0  ->  identical to ScoreAwareLoss (the no-tag baseline).

Ablation runs (5 folds each, same seeds):
    A  beta=0.0, lam_tag=0.0   baseline
    B  beta=0.5, lam_tag=0.0   teacher only
    C  beta=0.0, lam_tag=0.3   heads only
    D  beta=0.5, lam_tag=0.3   both
Keep tags only if B/C/D beat A on R@10 by more than one fold-std.

Tag vectors: one row per item, 42 columns (all options of all 5 groups),
each group normalised to sum to 1. Use the FULL probabilities from
data/features/tags/raw/, averaged over prompts A and B and prior-corrected
(see load_tag_probs). Do not use the thresholded tag lists.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bgm.losses.score_aware_loss import ScoreAwareLoss, same_track_mask


# ---------------------------------------------------------------- data
def load_tag_probs(
    df: pd.DataFrame,
    item_col="item", prompt_col="prompt", group_col="group",
    tag_col="tag", value_col="confidence",
    value_is_logprob=False, prior_correct=True, floor=1e-6,
):
    """
    df: long table, one row per (item, prompt, group, tag, value).
        Build it from data/features/tags/raw/ so every option of every group
        is present (missing options get `floor`).
    returns: items (list), vocab (list of (group, tag)),
             group_slices (list of index lists), probs (np.float32 [N, n_tags])
    """
    d = df[[item_col, prompt_col, group_col, tag_col, value_col]].copy()
    if value_is_logprob:
        d[value_col] = np.exp(d[value_col].astype(np.float64))
    # average the two prompt wordings
    d = d.groupby([item_col, group_col, tag_col], as_index=False)[value_col].mean()

    vocab = sorted(set(zip(d[group_col], d[tag_col])))
    col = {gt: i for i, gt in enumerate(vocab)}
    items = sorted(d[item_col].unique().tolist())
    row = {it: i for i, it in enumerate(items)}

    X = np.full((len(items), len(vocab)), floor, dtype=np.float64)
    r = d[item_col].map(row).to_numpy()
    c = np.array([col[(g, t)] for g, t in zip(d[group_col], d[tag_col])])
    X[r, c] = np.maximum(d[value_col].to_numpy(dtype=np.float64), floor)

    groups = sorted({g for g, _ in vocab})
    group_slices = [[i for i, (g, _) in enumerate(vocab) if g == gname] for gname in groups]
    for idx in group_slices:
        X[:, idx] /= X[:, idx].sum(1, keepdims=True)
        if prior_correct:  # removes "angry 53%", "strings 91%" style bias
            X[:, idx] /= X[:, idx].mean(0, keepdims=True)
            X[:, idx] /= X[:, idx].sum(1, keepdims=True)
    return items, vocab, group_slices, X.astype(np.float32)


def pool_windows(window_probs, window_start, window_end, seg_start, seg_end, group_slices):
    """
    Music tags are per 10 s window (hop 5 s). Average the windows that overlap
    a music segment [seg_start, seg_end) to get that segment's tag vector.
    window_probs: [W, n_tags] for ONE track, window_start/end: [W] seconds.
    """
    ov = np.clip(np.minimum(window_end, seg_end) - np.maximum(window_start, seg_start), 0, None)
    if ov.sum() == 0:
        ov = np.ones_like(ov)
    p = (window_probs * ov[:, None]).sum(0) / ov.sum()
    for idx in group_slices:
        p[idx] /= p[idx].sum()
    return p


# ---------------------------------------------------------------- loss
class TagAwareLoss(ScoreAwareLoss):
    def __init__(self, group_slices, d_embed, beta=0.5, tau_tag=0.1,
                 lam_tag=0.3, use_cross=True, **kw):
        super().__init__(**kw)
        self.group_slices = [list(s) for s in group_slices]
        n_tags = sum(len(s) for s in self.group_slices)
        self.beta, self.tau_tag = beta, tau_tag
        self.lam_tag, self.use_cross = lam_tag, use_cross
        self.head_v = nn.Linear(d_embed, n_tags)   # v -> scene tags
        self.head_m = nn.Linear(d_embed, n_tags)   # m -> music tags (also used on v for cross)

    def tag_similarity(self, t_scene, t_music):
        """Bhattacharyya coefficient per group, averaged. [B, B] in [0, 1]."""
        sims = [t_scene[:, g].clamp_min(0).sqrt() @ t_music[:, g].clamp_min(0).sqrt().T
                for g in self.group_slices]
        return torch.stack(sims).mean(0)

    def targets(self, m_raw, t_scene, t_music, same=None):
        """Sweep fix (reported bug): the original built q_m2v from teach_clap.T, whose rows do not sum to 1, so
        beta = 0 was not ScoreAwareLoss. Now: each direction's teacher is row-stochastic, q is symmetrised the way
        ScoreAwareLoss does it (q_v2m <- (q_v2m + q_m2v.T) / 2, q_m2v = its transpose), same-track pairs (decision 3)
        are masked out of every teacher and get zero mass, and rows are renormalised. beta = 0 reproduces
        ScoreAwareLoss.soft_targets exactly (tests/test_loss_registry.py)."""
        m = F.normalize(m_raw, dim=-1)
        sim = m @ m.T
        if same is not None:
            sim = sim.masked_fill(same, float("-inf"))
        teach_clap = F.softmax(sim / self.tau_teacher, dim=1)
        eye = torch.eye(m.size(0), device=m.device, dtype=m.dtype)
        if self.beta > 0:
            tag_sim = self.tag_similarity(t_scene, t_music)           # rows: scenes, cols: music
            if same is not None:
                tag_sim = tag_sim.masked_fill(same, float("-inf"))
            t_v2m = F.softmax(tag_sim / self.tau_tag, dim=1)
            t_m2v = F.softmax(tag_sim.T / self.tau_tag, dim=1)
            teach_v2m = (1 - self.beta) * teach_clap + self.beta * t_v2m
            teach_m2v = (1 - self.beta) * teach_clap + self.beta * t_m2v
        else:
            teach_v2m, teach_m2v = teach_clap, teach_clap
        q_v2m = (1 - self.eps) * eye + self.eps * teach_v2m
        q_m2v = (1 - self.eps) * eye + self.eps * teach_m2v
        q = 0.5 * (q_v2m + q_m2v.T)
        if same is not None:
            q = q.masked_fill(same, 0.0)
        q_v2m, q_m2v = q, q.T.contiguous()
        return q_v2m / q_v2m.sum(1, keepdim=True), q_m2v / q_m2v.sum(1, keepdim=True)

    def contrastive(self, v, m, q_v2m, q_m2v, same=None):
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * (v @ m.T)
        l1 = self.masked_ce(q_v2m, logits, same)
        l2 = self.masked_ce(q_m2v, logits.T, None if same is None else same.T)
        return 0.5 * (l1 + l2)

    def group_ce(self, logits, target):
        return sum(-(target[:, g] * F.log_softmax(logits[:, g], dim=1)).sum(1).mean()
                   for g in self.group_slices) / len(self.group_slices)

    def forward(self, v, m, v_raw, m_raw, t_scene, t_music, pairs=None, cont=None, track=None, pair_same_track=None):
        v = F.normalize(v, dim=-1)
        m = F.normalize(m, dim=-1)

        same = same_track_mask(track) if track is not None else None
        q_v2m, q_m2v = self.targets(m_raw, t_scene, t_music, same)
        l_soft = self.contrastive(v, m, q_v2m, q_m2v, same)
        l_intra = 0.5 * (self.intra(v_raw, v) + self.intra(m_raw, m))

        l_trans = v.new_zeros(())
        if pairs is not None and pairs.numel() > 0 and self.lam_trans > 0:
            if cont is None:
                cont = self.derive_continue_labels(m_raw, pairs)
                if pair_same_track is not None:
                    cont = torch.where(pair_same_track, torch.ones_like(cont), cont)
            l_trans = self.transition(v, pairs, cont.float())

        l_tag = v.new_zeros(())
        if self.lam_tag > 0:
            l_tag = self.group_ce(self.head_v(v), t_scene) + self.group_ce(self.head_m(m), t_music)
            if self.use_cross:
                l_tag = l_tag + self.group_ce(self.head_m(v), t_music)

        total = (l_soft + self.lam_intra * l_intra + self.lam_trans * l_trans
                 + self.lam_tag * l_tag)
        return total, {
            "soft": l_soft.detach(), "intra": l_intra.detach(),
            "trans": l_trans.detach(), "tag": l_tag.detach(),
            "tau": (1.0 / self.logit_scale.exp()).detach(),
        }


if __name__ == "__main__":
    # smoke test with random data (no real files needed)
    torch.manual_seed(0)
    B, D, Dv, Dm = 32, 128, 768, 512
    sizes = [12, 3, 10, 14, 3]                      # mood, energy, genre, instruments, tempo
    slices, start = [], 0
    for s in sizes:
        slices.append(list(range(start, start + s))); start += s

    def rand_tags(n):
        t = torch.rand(n, start)
        for g in slices:
            t[:, g] /= t[:, g].sum(1, keepdim=True)
        return t

    v_raw, m_raw = torch.randn(B, Dv), torch.randn(B, Dm)
    t_scene, t_music = rand_tags(B), rand_tags(B)
    head_v, head_m = nn.Linear(Dv, D), nn.Linear(Dm, D)
    from bgm.losses.score_aware_loss import consecutive_pairs
    pairs = consecutive_pairs(torch.arange(B) // 4, torch.arange(B) % 4)

    for beta, lam in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.3), (0.5, 0.3)]:
        loss_fn = TagAwareLoss(slices, D, beta=beta, lam_tag=lam)
        opt = torch.optim.AdamW(list(head_v.parameters()) + list(head_m.parameters())
                                + list(loss_fn.parameters()), lr=1e-3)
        for _ in range(50):
            total, parts = loss_fn(head_v(v_raw), head_m(m_raw), v_raw, m_raw, t_scene, t_music, pairs)
            opt.zero_grad(); total.backward(); opt.step()
        print(f"beta={beta} lam_tag={lam}  total={total.item():.3f}",
              {k: round(float(x), 3) for k, x in parts.items()})
