"""
Learnable Viterbi stitching = linear-chain CRF over candidate music windows.

Emission  : alpha * sim(video_t, music_k)          (alpha learnable, = 1/temperature)
Transition: learned score s(j -> k | boundary context), built from features
Training  : maximise log P(gold path | video) = score(gold) - logZ  (forward algorithm)
Inference : Viterbi with the learned scores (exact MAP)

No continue/jump/switch labels are needed: the "gold path" is the real film BGM window
under each training scene, and the model learns which joins look like real film scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionScorer(nn.Module):
    """
    pair_feats [T-1, K, K, P] : features of the join  j (seg t-1) -> k (seg t)
        e.g. same_track, contiguous, |jump distance|, key distance, log tempo ratio,
             loudness diff (dB), MERT/CLAP distance end(j)<->start(k), beat-phase error
    ctx_feats  [T-1, C]       : features of the scene boundary
        e.g. SigLIP distance between scenes, mood/emotion shift, cut strength,
             time since video start, segment durations
    The bilinear term lets context change how much each join feature matters,
    e.g. "an abrupt audio change is fine when the visual change is large".
    """

    def __init__(self, n_pair, n_ctx, hidden=16):
        super().__init__()
        self.base = nn.Linear(n_pair, 1)                    # context-free join cost
        self.ctx_gate = nn.Sequential(                      # context-dependent re-weighting
            nn.Linear(n_ctx, hidden), nn.ReLU(), nn.Linear(hidden, n_pair)
        )

    def forward(self, pair_feats, ctx_feats):
        w_ctx = self.ctx_gate(ctx_feats)[:, None, None, :]          # [T-1,1,1,P]
        return self.base(pair_feats).squeeze(-1) + (w_ctx * pair_feats).sum(-1)  # [T-1,K,K]


class LearnableStitcher(nn.Module):
    def __init__(self, n_pair, n_ctx):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(2.3))   # emission scale (~1/0.1)
        self.log_lam = nn.Parameter(torch.tensor(0.0))     # transition weight, kept > 0
        self.trans = TransitionScorer(n_pair, n_ctx)

    def scores(self, sim, pair_feats, ctx_feats):
        emit = self.log_alpha.exp() * sim                                   # [T,K]
        trans = self.log_lam.exp() * self.trans(pair_feats, ctx_feats)      # [T-1,K,K]
        return emit, trans

    @staticmethod
    def nll(emit, trans, gold):
        """CRF negative log-likelihood of the gold path. gold: [T] candidate indices."""
        T = emit.shape[0]
        a = emit[0]
        for t in range(1, T):
            a = torch.logsumexp(a[:, None] + trans[t - 1], dim=0) + emit[t]
        log_z = torch.logsumexp(a, dim=0)
        idx = torch.arange(T)
        gold_score = emit[idx, gold].sum() + trans[idx[:-1], gold[:-1], gold[1:]].sum()
        return log_z - gold_score

    @staticmethod
    @torch.no_grad()
    def viterbi(emit, trans):
        T, K = emit.shape
        d, back = emit[0], []
        for t in range(1, T):
            s = d[:, None] + trans[t - 1]          # [K_prev, K]
            d, bp = s.max(0)
            d = d + emit[t]
            back.append(bp)
        path = [int(d.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        return path[::-1]


if __name__ == "__main__":
    # Smoke test with random data: T segments, K candidates (gold placed at index 0).
    torch.manual_seed(0)
    T, K, P, C = 12, 20, 8, 4
    model = LearnableStitcher(P, C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    sim = torch.randn(T, K) * 0.1
    sim[:, 0] += 0.3                                   # gold is somewhat more similar
    pair = torch.randn(T - 1, K, K, P)
    pair[:, 0, 0, :] += 1.0                           # gold joins have a distinct signature
    ctx = torch.randn(T - 1, C)
    gold = torch.zeros(T, dtype=torch.long)
    for step in range(200):
        emit, trans = model.scores(sim, pair, ctx)
        loss = model.nll(emit, trans, gold)
        opt.zero_grad(); loss.backward(); opt.step()
    emit, trans = model.scores(sim, pair, ctx)
    print("final NLL:", round(loss.item(), 3))
    print("decoded  :", model.viterbi(emit, trans))
    print("alpha, lambda:", model.log_alpha.exp().item(), model.log_lam.exp().item())
