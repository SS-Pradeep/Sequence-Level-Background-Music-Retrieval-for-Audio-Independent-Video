"""Loss registry for the loss sweep (results/loss_sweep).

Every term maps (v, m, batch) -> scalar. v, m: [B, D] projected, L2-normalised embeddings; s_ij = v_i . m_j.
`batch` keys:
    v_raw [B, dv], m_raw [B, dm]   the pooled frozen inputs (SigLIP; CLAP+MERT of the BGM file)
    v_pre, m_pre [B, D]            head outputs before normalisation (vicreg, barlow)
    track [B] long                 track ids: off-diagonal same-track pairs are never negatives (decision 3)
    t_scene, t_music [B, 42]       tag probabilities (tag terms only); genre [B] long (supcon, center)
    v2, m2 [B, D]                  a second dropout pass (rdrop only)
    epoch int
Main losses are symmetric (v->m and m->v averaged) unless the definition says otherwise.

Sweep-specific choices (documented in results/loss_sweep/SUMMARY.md):
* infonce, hardneg, debiased, vsepp's warm-up use the trainer's learnable log_tau (SemanticModel.log_tau), clamped
  to [0.01, 0.5], so `infonce` is the current trainer's loss (bgm.train.losses.info_nce) bit for bit below tau 0.5.
* orth uses W W^T - I on the last head layers (W: [D_out, hidden], rows orthonormal), since W^T W ([hidden, hidden],
  rank <= D_out < hidden) can never be I.
* uncert: aux term k becomes L_k / (2 sigma_k^2) + log sigma_k, with log sigma_k^2 initialised at -log(2 lambda_k),
  so training starts at the fixed weights. dwa and gradnorm modulate the default lambdas (weights start at 1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bgm.losses.score_aware_loss import ScoreAwareLoss, same_track_mask
from bgm.losses.tag_aware_loss import TagAwareLoss

MAIN = ("infonce", "siglip", "soft", "vsepp", "hardneg", "debiased", "hgm", "pq_tau", "multipos")
AUX_DEFAULT = {"intra": 0.5, "rdrop": 1.0, "vicreg": 0.04, "barlow": 0.005, "supcon": 0.1, "center": 0.01,
               "orth": 1e-3, "csrd": 0.1, "recon": 0.1, "trans": 0.5, "tag_teacher": 0.0, "tag_heads": 0.3,
               "intra_nce": 0.5, "supcon_music": 0.5, "vv_nce": 0.5}
WEIGHTING = ("fixed", "uncert", "dwa", "gradnorm")
TAG_BASED = {"tag_teacher", "tag_heads", "supcon", "center"}      # decision 6: supcon / center use tag genre labels
TAU_MIN, TAU_MAX = 0.01, 0.5
NEG_INF = float("-inf")


def _tau(log_tau):
    return log_tau.exp().clamp(TAU_MIN, TAU_MAX)


def _masked_ce_diag(logits, same):
    """Cross-entropy with the diagonal as target; same-track off-diagonal entries removed (-inf)."""
    y = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits.masked_fill(same, NEG_INF), y)


# ================================================================ main losses
class InfoNCE(nn.Module):
    def __init__(self, log_tau):
        super().__init__()
        self.log_tau = log_tau            # shared with the model (not registered here)

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        logits = v @ m.T / _tau(self.log_tau)
        return 0.5 * (_masked_ce_diag(logits, same) + _masked_ce_diag(logits.T, same.T))


class SigLIP(nn.Module):
    """-(1/B) sum_ij log sigmoid(y_ij (t s_ij + b)); same-track off-diagonal pairs left out."""

    def __init__(self):
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(math.log(10.0)))
        self.b = nn.Parameter(torch.tensor(-10.0))

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        eye = torch.eye(len(v), dtype=torch.bool, device=v.device)
        y = eye.float() * 2 - 1
        ll = F.logsigmoid(y * (self.t_prime.exp() * (v @ m.T) + self.b)).masked_fill(same, 0.0)
        return -ll.sum() / len(v)


class Soft(nn.Module):
    """ScoreAwareLoss soft-contrastive part only (lam_intra = lam_trans = 0), eps 0.2, tau_teacher 0.05; with
    tag_teacher on, TagAwareLoss's targets with beta (beta = 0 is identical, see tag_aware_loss.targets)."""

    def __init__(self, beta: float = 0.0, group_slices=None, d_embed: int = 256, eps: float = 0.2,
                 learn_tau: bool = True):
        """eps: soft-target weight (0 = one-hot targets, ASAC no_eps); learn_tau False = logit scale fixed at
        1 / 0.07 (ASAC no_tau)."""
        super().__init__()
        self.beta = beta
        if beta > 0:
            self.fn = TagAwareLoss(group_slices, d_embed, beta=beta, lam_tag=0.0, eps=eps, tau_teacher=0.05,
                                   lam_intra=0.0, lam_trans=0.0)
        else:
            self.fn = ScoreAwareLoss(eps=eps, tau_teacher=0.05, lam_intra=0.0, lam_trans=0.0)
        self.fn.logit_scale.requires_grad_(learn_tau)

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        if self.beta > 0:
            q_v2m, q_m2v = self.fn.targets(batch["m_raw"], batch["t_scene"], batch["t_music"], same)
            return self.fn.contrastive(v, m, q_v2m, q_m2v, same)
        return self.fn.soft_contrastive(v, m, self.fn.soft_targets(batch["m_raw"], same), same)


class VSEpp(nn.Module):
    """sum_i [a - s_ii + max_{j!=i} s_ij]_+ + [a - s_ii + max_{j!=i} s_ji]_+, a = 0.2; InfoNCE for the first 10 epochs."""

    def __init__(self, log_tau, margin: float = 0.2, warmup_epochs: int = 10):
        super().__init__()
        self.nce, self.margin, self.warm = InfoNCE(log_tau), margin, warmup_epochs

    def forward(self, v, m, batch):
        if batch["epoch"] < self.warm:
            return self.nce(v, m, batch)
        s = v @ m.T
        same = same_track_mask(batch["track"])
        eye = torch.eye(len(v), dtype=torch.bool, device=v.device)
        off = s.masked_fill(same | eye, NEG_INF)
        pos = s.diag()
        h1 = F.relu(self.margin - pos + off.max(1).values)
        h2 = F.relu(self.margin - pos + off.max(0).values)
        return (torch.nan_to_num(h1, neginf=0.0) + torch.nan_to_num(h2, neginf=0.0)).sum()


class HardNeg(nn.Module):
    """InfoNCE with each negative weighted w_ij ∝ exp(beta s_ij), normalised to mean 1 over the row's negatives."""

    def __init__(self, log_tau, beta: float = 0.5):
        super().__init__()
        self.log_tau, self.beta = log_tau, beta

    def side(self, s, same):
        eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
        neg = ~(same | eye)
        t = _tau(self.log_tau)
        lw = (self.beta * s).masked_fill(~neg, NEG_INF)
        lw = lw - torch.logsumexp(lw, 1, keepdim=True) + torch.log(neg.sum(1, keepdim=True).float())   # mean 1
        neg_term = torch.logsumexp((s / t + lw).masked_fill(~neg, NEG_INF), 1)
        pos = s.diag() / t
        return -(pos - torch.logaddexp(pos, neg_term)).mean()

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        s = v @ m.T
        return 0.5 * (self.side(s, same) + self.side(s.T, same.T))


class Debiased(nn.Module):
    """Debiased contrastive loss (Chuang et al. 2020), class prior tau_plus = 0.1. In a shifted exp space
    (exp(s / t - 1 / t) <= 1) for stability; the clamp N e^{-1/t} becomes N e^{-2/t}."""

    def __init__(self, log_tau, tau_plus: float = 0.1):
        super().__init__()
        self.log_tau, self.tp = log_tau, tau_plus

    def side(self, s, same):
        t = _tau(self.log_tau)
        eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
        neg = ~(same | eye)
        e = torch.exp((s - 1.0) / t)
        n = neg.sum(1).float()
        pos = e.diag()
        ng = ((e * neg).sum(1) - n * self.tp * pos) / (1 - self.tp)
        ng = torch.maximum(ng, n * torch.exp(-2.0 / t))
        return -torch.log(pos / (pos + ng)).mean()

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        s = v @ m.T
        return 0.5 * (self.side(s, same) + self.side(s.T, same.T))


class HGM(nn.Module):
    """[|v_i - m_i|^2 - g_high]_+ + mean_{j != i} [g_low - |v_i - m_j|^2]_+, g_high 0.5, g_low 1.5; both directions."""

    def __init__(self, g_high: float = 0.5, g_low: float = 1.5):
        super().__init__()
        self.hi, self.lo = g_high, g_low

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        eye = torch.eye(len(v), dtype=torch.bool, device=v.device)
        d2 = 2.0 - 2.0 * (v @ m.T)                       # unit vectors
        pos = F.relu(d2.diag() - self.hi)
        neg_mask = ~(same | eye)
        hinge = F.relu(self.lo - d2) * neg_mask
        n1 = hinge.sum(1) / neg_mask.sum(1).clamp_min(1)
        n2 = hinge.sum(0) / neg_mask.sum(0).clamp_min(1)
        return (pos + 0.5 * (n1 + n2)).mean()


class PQTau(nn.Module):
    """InfoNCE with a per-query temperature tau_i = 0.01 + softplus(MLP(v_raw_i)), MLP in -> 64 -> 1; the logits
    L_ij = s_ij / tau_i are used in both directions. Output bias starts at tau = 0.07."""

    def __init__(self, dv: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dv, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, math.log(math.expm1(0.06)))

    def forward(self, v, m, batch):
        same = same_track_mask(batch["track"])
        tau = TAU_MIN + F.softplus(self.mlp(batch["v_raw"]))          # [B, 1]
        logits = (v @ m.T) / tau
        return 0.5 * (_masked_ce_diag(logits, same) + _masked_ce_diag(logits.T, same.T))


# ================================================================ auxiliary terms
class Intra(nn.Module):
    def forward(self, v, m, batch):
        return 0.5 * (ScoreAwareLoss.intra(batch["v_raw"], v) + ScoreAwareLoss.intra(batch["m_raw"], m))


class RDrop(nn.Module):
    """Two dropout passes -> row-softmax similarities P1, P2 (rows and columns) -> symmetric KL."""

    def __init__(self, log_tau):
        super().__init__()
        self.log_tau = log_tau

    @staticmethod
    def sym_kl(a, b):
        la, lb = F.log_softmax(a, 1), F.log_softmax(b, 1)
        return 0.5 * (F.kl_div(lb, la, log_target=True, reduction="batchmean")
                      + F.kl_div(la, lb, log_target=True, reduction="batchmean"))

    def forward(self, v, m, batch):
        t = _tau(self.log_tau).detach()
        s1, s2 = v @ m.T / t, batch["v2"] @ batch["m2"].T / t
        return 0.5 * (self.sym_kl(s1, s2) + self.sym_kl(s1.T, s2.T))


def _vicreg_side(z, gamma=1.0):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(gamma - std).mean()
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + off.pow(2).sum() / z.shape[1]


class VICReg(nn.Module):
    def forward(self, v, m, batch):
        return _vicreg_side(batch["v_pre"]) + _vicreg_side(batch["m_pre"])


class Barlow(nn.Module):
    def forward(self, v, m, batch):
        V, M = batch["v_pre"], batch["m_pre"]
        B, D = V.shape
        assert D <= B, f"barlow needs D <= B (D={D}, B={B})"
        V = (V - V.mean(0)) / (V.std(0) + 1e-5)
        M = (M - M.mean(0)) / (M.std(0) + 1e-5)
        C = V.T @ M / B
        on = (torch.diag(C) - 1).pow(2).sum()
        off = (C - torch.diag(torch.diag(C))).pow(2).sum()
        return on + off / D


class SupCon(nn.Module):
    """Cross-modal supervised contrastive: anchors v_i, candidates m_j, positives = same genre label; both
    directions; fixed temperature 0.1."""

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    @staticmethod
    def side(logits, pos):
        logp = F.log_softmax(logits, 1)
        return -((logp * pos).sum(1) / pos.sum(1)).mean()

    def forward(self, v, m, batch):
        g = batch["genre"]
        pos = (g[:, None] == g[None, :]).float()
        logits = v @ m.T / self.tau
        return 0.5 * (self.side(logits, pos) + self.side(logits.T, pos.T))


class Center(nn.Module):
    def __init__(self, n_classes: int, d: int):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_classes, d) * 0.01)

    def forward(self, v, m, batch):
        c = self.c[batch["genre"]]
        return (0.5 * (v - c).pow(2).sum(1) + 0.5 * (m - c).pow(2).sum(1)).mean()


class Orth(nn.Module):
    def __init__(self, w_v: nn.Parameter, w_m: nn.Parameter):
        super().__init__()
        self._w = [w_v, w_m]              # not registered: they belong to the heads

    def forward(self, v, m, batch):
        out = 0.0
        for W in self._w:
            I = torch.eye(W.shape[0], device=W.device)
            out = out + (W @ W.T - I).pow(2).sum()
        return out


class CSRD(nn.Module):
    """||sigma(D_v) - sigma(D_m)||^2, singular values of the in-batch Euclidean distance matrices."""

    def forward(self, v, m, batch):
        dv = torch.cdist(v, v)
        dm = torch.cdist(m, m)
        return (torch.linalg.svdvals(dv) - torch.linalg.svdvals(dm)).pow(2).sum() / len(v)


class Recon(nn.Module):
    def __init__(self, d: int, dv: int, dm: int):
        super().__init__()
        self.dec_v, self.dec_m = nn.Linear(d, dv), nn.Linear(d, dm)

    def forward(self, v, m, batch):
        return F.mse_loss(self.dec_v(v), batch["v_raw"]) + F.mse_loss(self.dec_m(m), batch["m_raw"])


class Trans(nn.Module):
    """ScoreAwareLoss transition term on within-video consecutive segments (decision 2). batch['v_all'] holds the
    contrastive rows followed by the extra segments; batch['pairs'] indexes it; batch['cont'] the labels."""

    def __init__(self):
        super().__init__()
        self.fn = ScoreAwareLoss(lam_intra=0.0, lam_trans=1.0)

    def forward(self, v, m, batch):
        if batch.get("pairs") is None or batch["pairs"].numel() == 0:
            return v.new_zeros(())
        return self.fn.transition(batch["v_all"], batch["pairs"], batch["cont"])


class IntraNCE(nn.Module):
    """Within-modality InfoNCE between two augmented views (user, title-split request 2026-11-16):
    InfoNCE(v', v'') + InfoNCE(m', m''); video views = two independent 25% frame-drop draws, music views = two
    independent +/- 1 s jitter crops of the same segment, both through the trained heads. The trainer's tau; same-track
    off-diagonal pairs masked. Used as main `infonce` + 0.5 * this = `inter_intra_nce`."""

    def __init__(self, log_tau):
        super().__init__()
        self.nce = InfoNCE(log_tau)

    def forward(self, v, m, batch):
        return (self.nce(batch["v_view1"], batch["v_view2"], batch)
                + self.nce(batch["m_view1"], batch["m_view2"], batch))


class TagHeads(nn.Module):
    def __init__(self, group_slices, d: int):
        super().__init__()
        self.fn = TagAwareLoss(group_slices, d, beta=0.0, lam_tag=1.0, lam_intra=0.0, lam_trans=0.0)

    def forward(self, v, m, batch):
        f = self.fn
        l = f.group_ce(f.head_v(v), batch["t_scene"]) + f.group_ce(f.head_m(m), batch["t_music"])
        return l + f.group_ce(f.head_m(v), batch["t_music"])

# ================================================================ tag-free additions (user 2026-11-20, loss table)
def _multipos_side(logits, pos):
    """-log(sum_{pos} exp / sum_all exp) per row, averaged. logits [R, C], pos [R, C] bool (>= 1 per row)."""
    return -(torch.logsumexp(logits.masked_fill(~pos, NEG_INF), 1) - torch.logsumexp(logits, 1)).mean()


class MultiPos(nn.Module):
    """Multi-positive InfoNCE, no tags. Gallery = [m; m'] (the aligned window and a second independent +/- 1 s
    jitter crop, batch['m_view1']); positives of v_i = every gallery window whose track_id equals track_i (its own two
    windows plus other videos of the same track). Music->video: rows [m; m'], positives = same-track videos."""

    def __init__(self, log_tau):
        super().__init__()
        self.log_tau = log_tau

    def forward(self, v, m, batch):
        t = _tau(self.log_tau)
        tr = batch["track"]
        g = torch.cat([m, batch["m_view1"]])                      # [2B, d]
        same = tr[:, None] == torch.cat([tr, tr])[None, :]       # [B, 2B]
        logits = v @ g.T / t
        return 0.5 * (_multipos_side(logits, same) + _multipos_side(logits.T, same.T))


class SupConMusic(nn.Module):
    """Music-music supervised contrastive (Khosla et al. 2020), no tags: Z = [m'; m''] (two jitter crops of each
    window), positives = other rows with the same track_id (always includes the row's other view); tau 0.1."""

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, v, m, batch):
        z = torch.cat([batch["m_view1"], batch["m_view2"]])
        tr = torch.cat([batch["track"], batch["track"]])
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        pos = (tr[:, None] == tr[None, :]) & ~eye
        logp = F.log_softmax((z @ z.T / self.tau).masked_fill(eye, NEG_INF), 1)
        return -((logp.masked_fill(~pos, 0.0)).sum(1) / pos.sum(1)).mean()


class VVNCE(nn.Module):
    """Video-video InfoNCE between two independent 25% frame-drop views (intra_nce without the music half)."""

    def __init__(self, log_tau):
        super().__init__()
        self.nce = InfoNCE(log_tau)

    def forward(self, v, m, batch):
        return self.nce(batch["v_view1"], batch["v_view2"], batch)


# ================================================================ combination and weighting
class SweepLoss(nn.Module):
    """main + weighted aux terms. forward -> (total, parts, monitored); monitored = main + sum lambda_k L_k at the
    fixed lambdas (used by the divergence guard, comparable across weighting schemes)."""

    def __init__(self, main: str, aux: dict[str, float], weighting: str, ctx: dict, main_kw: dict | None = None):
        super().__init__()
        assert main in MAIN, main
        assert weighting in WEIGHTING, weighting
        unknown = set(aux) - set(AUX_DEFAULT)
        assert not unknown, unknown
        assert not ("vicreg" in aux and "barlow" in aux), "never combine vicreg and barlow"
        lt = ctx["log_tau"]
        self._shared = {id(lt)}                        # the trainer's log_tau: trained with the model, not twice
        beta = 0.5 if "tag_teacher" in aux else 0.0
        if beta:
            assert main == "soft", "tag_teacher is only valid with main = soft"
        self.main_name = main
        self.main = {"infonce": lambda: InfoNCE(lt), "siglip": SigLIP,
                     "soft": lambda: Soft(beta, ctx.get("group_slices"), ctx["d"], **(main_kw or {})),
                     "vsepp": lambda: VSEpp(lt), "hardneg": lambda: HardNeg(lt), "debiased": lambda: Debiased(lt),
                     "hgm": HGM, "pq_tau": lambda: PQTau(ctx["dv"]), "multipos": lambda: MultiPos(lt)}[main]()
        make = {"intra": Intra, "rdrop": lambda: RDrop(lt), "vicreg": VICReg, "barlow": Barlow, "supcon": SupCon,
                "center": lambda: Center(ctx.get("n_genres", 1), ctx["d"]),
                "orth": lambda: Orth(*ctx["last_layers"]), "csrd": CSRD,
                "recon": lambda: Recon(ctx["d"], ctx["dv"], ctx["dm"]), "trans": Trans,
                "tag_heads": lambda: TagHeads(ctx.get("group_slices"), ctx["d"]), "intra_nce": lambda: IntraNCE(lt),
                "supcon_music": SupConMusic, "vv_nce": lambda: VVNCE(lt)}
        self.aux_names = [k for k in aux if k != "tag_teacher"]          # tag_teacher lives inside the main loss
        self.lam = {k: float(aux[k]) for k in self.aux_names}
        self.aux = nn.ModuleDict({k: make[k]() for k in self.aux_names})
        self.weighting = weighting
        K = len(self.aux_names)
        if weighting == "uncert":
            self.log_var = nn.Parameter(torch.tensor([-math.log(2 * self.lam[k]) for k in self.aux_names]))
        self.w = torch.ones(K)                            # dwa / gradnorm multipliers on lambda (start at 1)
        if weighting == "gradnorm":
            self.gn_w = nn.Parameter(torch.ones(K))       # trained by its own optimiser (see GradNorm in train.py)
        self._epoch_losses: list[torch.Tensor] = []
        self._hist: list[torch.Tensor] = []

    @property
    def needs(self) -> set[str]:
        n = set(self.aux_names) | ({"tag_teacher"} if isinstance(self.main, Soft) and self.main.beta > 0 else set())
        return n

    def model_params(self):
        """Parameters trained with the heads (everything except the GradNorm weights)."""
        return [p for n, p in self.named_parameters()
                if not n.startswith("gn_w") and id(p) not in self._shared and p.requires_grad]

    def aux_values(self, v, m, batch) -> dict[str, torch.Tensor]:
        return {k: self.aux[k](v, m, batch) for k in self.aux_names}

    def forward(self, v, m, batch):
        main = self.main(v, m, batch)
        vals = self.aux_values(v, m, batch)
        parts = {"main": main.detach()} | {k: x.detach() for k, x in vals.items()}
        monitored = main.detach() + sum((self.lam[k] * vals[k].detach() for k in vals), torch.zeros((), device=main.device))
        total = main
        for i, k in enumerate(self.aux_names):
            if self.weighting == "fixed":
                total = total + self.lam[k] * vals[k]
            elif self.weighting == "uncert":
                lv = self.log_var[i]
                total = total + vals[k] * torch.exp(-lv) / 2 + 0.5 * lv           # L/(2 sigma^2) + log sigma
            elif self.weighting == "dwa":
                total = total + float(self.w[i]) * self.lam[k] * vals[k]
            else:                                                              # gradnorm
                total = total + self.gn_w[i].detach() * self.lam[k] * vals[k]
        self._last_vals = vals
        if self.aux_names:
            self._epoch_losses.append(torch.stack([x.detach() for x in vals.values()]).cpu())
        return total, parts, monitored

    def end_epoch(self, T: float = 2.0) -> None:
        """DWA (Liu et al. 2019): w_k = K exp(r_k / T) / sum exp(r_j / T), r_k = L_k(t-1) / L_k(t-2), per epoch means."""
        if self._epoch_losses:
            self._hist.append(torch.stack(self._epoch_losses).mean(0))
        self._epoch_losses = []
        if self.weighting == "dwa" and len(self._hist) >= 2:
            r = self._hist[-1] / self._hist[-2].clamp_min(1e-12)
            K = len(r)
            self.w = K * torch.softmax(r / T, 0)

    def sigmas(self) -> dict[str, float]:
        if self.weighting != "uncert":
            return {}
        return {k: float(torch.exp(0.5 * self.log_var[i].detach())) for i, k in enumerate(self.aux_names)}

    def weights_now(self) -> dict[str, float]:
        if self.weighting == "uncert":
            return {k: float(torch.exp(-self.log_var[i].detach()) / 2) for i, k in enumerate(self.aux_names)}
        if self.weighting == "gradnorm":
            return {k: float(self.gn_w[i].detach()) * self.lam[k] for i, k in enumerate(self.aux_names)}
        return {k: float(self.w[i]) * self.lam[k] for i, k in enumerate(self.aux_names)}


class GradNorm:
    """GradNorm (Chen et al. 2018), alpha 1.5, aux terms only, on the last shared layer of each head (the two output
    Linear weights). Weights w_k (on lambda_k L_k) are trained to equalise G_k = ||grad_W w_k lambda_k L_k|| toward
    mean(G) * r_k^alpha, r_k = (L_k / L_k(0)) / mean; renormalised to sum K after each step."""

    def __init__(self, loss: SweepLoss, shared: list[torch.Tensor], alpha: float = 1.5, lr: float = 1e-3):
        self.loss, self.shared, self.alpha = loss, shared, alpha
        self.opt = torch.optim.Adam([loss.gn_w], lr=lr) if loss.aux_names else None
        self.L0 = None

    def step(self) -> None:
        if self.opt is None:
            return
        L = self.loss
        vals = [L._last_vals[k] for k in L.aux_names]
        G = []
        for i, (k, x) in enumerate(zip(L.aux_names, vals)):
            if not x.requires_grad:
                G.append(torch.zeros((), device=L.gn_w.device))
                continue
            g = torch.autograd.grad(L.lam[k] * x, self.shared, retain_graph=True, allow_unused=True)
            n = torch.sqrt(sum((gi.pow(2).sum() for gi in g if gi is not None), torch.zeros((), device=L.gn_w.device)))
            G.append(L.gn_w[i] * n.detach())
        G = torch.stack(G)
        cur = torch.stack([x.detach() for x in vals])
        if self.L0 is None:
            self.L0 = cur.clamp_min(1e-12)
        r = cur / self.L0
        r = r / r.mean().clamp_min(1e-12)
        target = (G.mean() * r.pow(self.alpha)).detach()
        self.opt.zero_grad()
        (G - target).abs().sum().backward()
        self.opt.step()
        with torch.no_grad():
            L.gn_w.clamp_(min=1e-3)
            L.gn_w.mul_(len(L.gn_w) / L.gn_w.sum())
