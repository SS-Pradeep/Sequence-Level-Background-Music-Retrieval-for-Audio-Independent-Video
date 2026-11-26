"""
Score-Aware Contrastive Loss for video -> BGM retrieval.

    L = L_soft + lam_intra * L_intra + lam_trans * L_trans

L_soft  : symmetric contrastive loss with soft targets built from the CLEAN BGM.
          Two clips whose music sounds alike are not pushed apart as hard negatives.
          (Handles one-to-many matches and repeated cues within a film.)
L_intra : keeps the projection heads close to the frozen encoders' similarity
          structure (main guard against overfitting on small data).
L_trans : NEW. For consecutive scenes from the same film, the composer either
          continued the music or changed it (label read from the BGM stem).
          The video embeddings must predict that decision:
          close videos  <-> music continues, far videos <-> music changes.

Inputs per batch
    v      : [B, D] projected video embeddings (trainable heads)
    m      : [B, D] projected music embeddings (trainable heads)
    v_raw  : [B, Dv] frozen video features (SigLIP)
    m_raw  : [B, Dm] frozen music features of the CLEAN BGM (CLAP / MERT)
    pairs  : [P, 2] long tensor, batch indices (a, b) of consecutive scenes
             from the same film (a directly before b). Can be empty.
    cont   : [P] float tensor, 1 = music continues across the cut, 0 = changes.
             If None, it is derived from m_raw with `cont_threshold`.

Batch sampling: draw films, then 2-4 consecutive segments per film, so that
`pairs` is non-empty. See `consecutive_pairs` below.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreAwareLoss(nn.Module):
    def __init__(
        self,
        eps: float = 0.2,            # weight of soft targets (0 = plain InfoNCE)
        tau_teacher: float = 0.05,   # sharpness of the music-music teacher
        lam_intra: float = 0.5,
        lam_trans: float = 0.5,
        init_tau: float = 0.07,
        cont_threshold: float = 0.8, # BGM cos-sim above this = "music continues"
    ):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))
        self.eps = eps
        self.tau_teacher = tau_teacher
        self.lam_intra = lam_intra
        self.lam_trans = lam_trans
        self.cont_threshold = cont_threshold
        # calibration of the transition predictor: P(continue) = sigmoid(a * cos + b)
        self.trans_a = nn.Parameter(torch.tensor(10.0))
        self.trans_b = nn.Parameter(torch.tensor(-5.0))

    # ---------- parts ----------
    def soft_targets(self, m_raw: torch.Tensor, same: torch.Tensor | None = None) -> torch.Tensor:
        """same: [B, B] bool, off-diagonal same-track pairs (sweep decision 3): masked out of the teacher and given
        zero target mass; soft_contrastive renormalises the rows."""
        m = F.normalize(m_raw, dim=-1)
        sim = m @ m.T                                       # [B, B], frozen, clean BGM
        if same is not None:
            sim = sim.masked_fill(same, float("-inf"))
        teacher = F.softmax(sim / self.tau_teacher, dim=1)  # rows sum to 1
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        q = (1.0 - self.eps) * eye + self.eps * teacher
        q = 0.5 * (q + q.T)                                 # symmetric, rows ~ sum to 1
        return q if same is None else q.masked_fill(same, 0.0)

    @staticmethod
    def masked_ce(q, logits, same=None):
        """-(sum_j q_ij log softmax_j logits_ij), mean over rows; masked entries are -inf logits with zero mass."""
        if same is not None:
            logits = logits.masked_fill(same, float("-inf"))
        logp = F.log_softmax(logits, dim=1)
        if same is not None:
            logp = logp.masked_fill(same, 0.0)
        return -(q * logp).sum(1).mean()

    def soft_contrastive(self, v, m, q, same=None):
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * (v @ m.T)
        q_row = q / q.sum(1, keepdim=True)
        same_t = None if same is None else same.T
        l_v2m = self.masked_ce(q_row, logits, same)
        l_m2v = self.masked_ce(q_row, logits.T, same_t)
        return 0.5 * (l_v2m + l_m2v)

    @staticmethod
    def intra(raw, enc):
        s_raw = F.normalize(raw, dim=-1) @ F.normalize(raw, dim=-1).T
        s_enc = enc @ enc.T
        return (1.0 - F.cosine_similarity(s_raw, s_enc, dim=1)).mean()

    def transition(self, v, pairs, cont):
        a, b = pairs[:, 0], pairs[:, 1]
        cos = (v[a] * v[b]).sum(-1)
        logit = self.trans_a * cos + self.trans_b
        return F.binary_cross_entropy_with_logits(logit, cont)

    @torch.no_grad()
    def derive_continue_labels(self, m_raw, pairs):
        m = F.normalize(m_raw, dim=-1)
        cos = (m[pairs[:, 0]] * m[pairs[:, 1]]).sum(-1)
        return (cos > self.cont_threshold).float()

    @torch.no_grad()
    def p_continue(self, v_prev, v_next):
        """Use at inference inside Viterbi as the transition prior."""
        v_prev, v_next = F.normalize(v_prev, dim=-1), F.normalize(v_next, dim=-1)
        return torch.sigmoid(self.trans_a * (v_prev * v_next).sum(-1) + self.trans_b)

    # ---------- total ----------
    def forward(self, v, m, v_raw, m_raw, pairs=None, cont=None, track=None, pair_same_track=None):
        """track: [B] long track ids (sweep decision 3), or None for no masking. pair_same_track: [P] bool, pairs
        sharing a track; they are labelled "continue" when the labels are derived."""
        v = F.normalize(v, dim=-1)
        m = F.normalize(m, dim=-1)

        same = same_track_mask(track) if track is not None else None
        q = self.soft_targets(m_raw, same)
        l_soft = self.soft_contrastive(v, m, q, same)
        l_intra = 0.5 * (self.intra(v_raw, v) + self.intra(m_raw, m))

        l_trans = v.new_zeros(())
        if pairs is not None and pairs.numel() > 0 and self.lam_trans > 0:
            if cont is None:
                cont = self.derive_continue_labels(m_raw, pairs)
                if pair_same_track is not None:
                    cont = torch.where(pair_same_track, torch.ones_like(cont), cont)
            l_trans = self.transition(v, pairs, cont.float())

        total = l_soft + self.lam_intra * l_intra + self.lam_trans * l_trans
        return total, {
            "soft": l_soft.detach(),
            "intra": l_intra.detach(),
            "trans": l_trans.detach(),
            "tau": (1.0 / self.logit_scale.exp()).detach(),
        }


def same_track_mask(track: torch.Tensor) -> torch.Tensor:
    """[B, B] bool: off-diagonal pairs that share a track_id (sweep decision 3: never negatives)."""
    eye = torch.eye(len(track), dtype=torch.bool, device=track.device)
    return (track[:, None] == track[None, :]) & ~eye


def consecutive_pairs(film_ids, seg_index):
    """
    film_ids  : [B] film id per item in the batch
    seg_index : [B] position of the segment inside its film (0, 1, 2, ...)
    returns   : [P, 2] batch indices (a, b) where b is the segment right after a.
    """
    film_ids, seg_index = film_ids.tolist(), seg_index.tolist()
    lookup = {(f, s): i for i, (f, s) in enumerate(zip(film_ids, seg_index))}
    out = [(i, lookup[(f, s + 1)]) for i, (f, s) in enumerate(zip(film_ids, seg_index))
           if (f, s + 1) in lookup]
    if not out:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, Dv, Dm = 32, 128, 768, 512
    v_raw, m_raw = torch.randn(B, Dv), torch.randn(B, Dm)
    head_v, head_m = nn.Linear(Dv, D), nn.Linear(Dm, D)
    film = torch.arange(B) // 4                 # 8 films x 4 consecutive segments
    seg = torch.arange(B) % 4
    pairs = consecutive_pairs(film, seg)

    loss_fn = ScoreAwareLoss()
    params = list(head_v.parameters()) + list(head_m.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)

    for step in range(201):
        total, parts = loss_fn(head_v(v_raw), head_m(m_raw), v_raw, m_raw, pairs)
        opt.zero_grad(); total.backward(); opt.step()
        if step % 50 == 0:
            print(step, f"total={total.item():.3f}",
                  {k: round(float(x), 3) for k, x in parts.items()})

    with torch.no_grad():
        v = F.normalize(head_v(v_raw), dim=-1); m = F.normalize(head_m(m_raw), dim=-1)
        r1 = ((v @ m.T).argmax(1) == torch.arange(B)).float().mean().item()
    print("train R@1 (sanity, should rise toward 1.0):", r1)
    print("pairs used:", len(pairs))
