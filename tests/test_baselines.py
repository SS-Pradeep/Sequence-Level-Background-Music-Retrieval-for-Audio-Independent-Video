"""Published-method baselines on our features (PROGRESS.md D43): encoder masking, context / pool masks, per-second
dynamics statistics, and the dual-path score."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from bgm.train.baselines import BaselineModel, SeqEncoder, pad_batch, per_second_stats, seq_item, span_indices


class _Feats:
    """Stand-in for SeqFeatures: token k of an item is [k, k, ...] so indices can be read back."""

    def tokens(self, side, item, names):
        return np.repeat(np.arange(30, dtype=np.float32)[:, None], 4, 1)


def test_encoder_ignores_padding_and_non_pooled_tokens_only_through_attention():
    torch.manual_seed(0)
    enc = SeqEncoder(4, d_model=16, layers=2, heads=2, p=0.0, d_out=8).eval()
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(9, 4)).astype(np.float32)
    alone = enc(*pad_batch([(a, np.ones(5, bool))], "cpu"))
    padded = enc(*pad_batch([(a, np.ones(5, bool)), (b, np.ones(9, bool))], "cpu"))[:1]
    assert torch.allclose(alone, padded, atol=1e-5)
    assert torch.allclose(alone.norm(dim=-1), torch.ones(1))


def test_seq_item_context_and_pool_mask():
    f = _Feats()
    x, pool = seq_item(f, "video", "v", 10.0, 15.0, ("siglip",), context_s=0.0)
    assert x[:, 0].tolist() == [10, 11, 12, 13, 14] and pool.all()
    x, pool = seq_item(f, "video", "v", 10.0, 15.0, ("siglip",), context_s=3.0)
    assert x[:, 0].tolist() == list(range(7, 18)) and x[pool, 0].tolist() == [10, 11, 12, 13, 14]
    x, pool = seq_item(f, "video", "v", 1.0, 4.0, ("siglip",), context_s=5.0)        # context clipped at the start
    assert x[0, 0] == 0 and x[pool, 0].tolist() == [1, 2, 3]
    x, pool = seq_item(f, "video", "v", 10.0, 20.0, ("siglip",), 3.0, drop=0.5, rng=np.random.default_rng(1))
    kept = x[pool, 0]
    assert 1 <= len(kept) < 10 and set(kept) <= set(range(10, 20))
    assert not set(range(10, 20)) - set(kept) & set(x[:, 0])   # dropped span tokens are gone, context stays
    assert span_indices(30, 4.2, 4.4).tolist() == [4]           # no centre inside: the nearest sample


def test_per_second_stats():
    dyn = np.zeros((25 * 3 + 2, 2), np.float32)
    dyn[25:50, 0] = np.arange(25)
    s = per_second_stats(dyn, 3)
    assert s.shape == (3, 4)
    assert s[1, 0] == 12.0 and s[1, 2] == 24.0 and s[0, 0] == 0.0
    assert per_second_stats(dyn[:70], 3).shape == (3, 4)        # a slightly short curve is edge-padded


def test_dual_path_score_is_the_mean_of_path_cosines():
    c = OmegaConf.create({"d_model": 16, "heads": 2, "dropout": 0.0, "out_dim": 8, "max_len": 64, "tau_init": 0.07})
    dims = {("video", "content"): 6, ("music", "content"): 5, ("video", "emotion"): 3, ("music", "emotion"): 3}
    torch.manual_seed(0)
    m = BaselineModel("cfd", dims, c, n_tags=16).eval()
    rng = np.random.default_rng(0)
    mk = lambda d, n: pad_batch([(rng.normal(size=(n, d)).astype(np.float32), np.ones(n, bool))], "cpu")  # noqa: E731
    zv, pv = m.embed("video", {"content": mk(6, 7), "emotion": mk(3, 7)})
    zm, pm = m.embed("music", {"content": mk(5, 9), "emotion": mk(3, 9)})
    want = 0.5 * (F.cosine_similarity(pv["content"], pm["content"]) + F.cosine_similarity(pv["emotion"], pm["emotion"]))
    assert torch.allclose((zv * zm).sum(-1), want, atol=1e-6)
    assert torch.allclose(zv.norm(dim=-1), torch.ones(1), atol=1e-6)
