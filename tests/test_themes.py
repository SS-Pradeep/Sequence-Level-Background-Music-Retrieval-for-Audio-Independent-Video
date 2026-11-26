"""D66: theme groups and the multi-positive loss."""
import numpy as np
import torch

from bgm.data.themes import link_groups, theme_links
from bgm.train.losses import info_nce, info_nce_multi


def test_link_groups_union_find():
    g = link_groups(6, [(0, 1), (1, 2), (4, 5)])
    assert g.tolist() == [0, 0, 0, 3, 4, 4]


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_theme_links_need_same_film_and_both_similarities():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 16))
    mert, clap = _unit(base.copy()), _unit(rng.standard_normal((40, 16)))
    film = np.array([f"f{i // 4}" for i in range(40)])
    mert[1], clap[1] = mert[0], clap[0]              # same film, near-identical on both: linked
    mert[2] = mert[0]                                # same film, MERT only: not linked
    mert[4], clap[4] = mert[0], clap[0]              # identical but another film: not linked
    mert[9], clap[9] = mert[8], clap[8]              # same film, identical, but the film is skipped
    link, thr = theme_links(mert, clap, film, 90.0, film == "f2")
    assert link[0, 1] and link[1, 0] and not link[0, 2] and not link[0, 4] and not link[8, 9]
    assert not link.diagonal().any() and thr["pct"] == 90.0


def test_multi_positive_loss():
    torch.manual_seed(0)
    zv = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    zm = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    lt = torch.tensor(np.log(0.07))
    assert torch.allclose(info_nce_multi(zv, zm, torch.arange(6), lt), info_nce(zv, zm, torch.arange(6), lt), atol=1e-6)
    g = torch.tensor([0, 0, 1, 2, 3, 4])             # items 0 and 1 are theme mates
    one, multi = info_nce(zv, zm, torch.arange(6), lt), info_nce_multi(zv, zm, g, lt)
    assert multi < one                               # the mate's score now counts for the positive, not against it
    zm2 = zm.clone(); zm2[1] = zv[0]                 # the mate matches query 0 perfectly: its row loss drops further
    assert info_nce_multi(zv, zm2, g, lt) < multi


def test_weighted_multi_positive_loss():
    from bgm.train.losses import info_nce_weighted
    torch.manual_seed(1)
    zv = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    zm = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    lt = torch.tensor(np.log(0.07))
    assert torch.allclose(info_nce_weighted(zv, zm, torch.arange(6), lt), info_nce(zv, zm, torch.arange(6), lt), atol=1e-5)
    g = torch.tensor([0, 0, 1, 2, 3, 4])
    # brute force for row 0, video -> music direction
    lg = (zv @ zm.T / 0.07)
    neg = torch.logsumexp(lg[0, 2:], 0)
    want0 = -(1.0 * (lg[0, 0] - torch.logaddexp(lg[0, 0], neg)) + 0.5 * (lg[0, 1] - torch.logaddexp(lg[0, 1], neg))) / 1.5
    w0 = info_nce_weighted(zv, zm, g, lt, 0.5)
    w1 = info_nce_weighted(zv, zm, g, lt, 0.0)                      # soft weight 0 = mates only masked from the negatives
    assert torch.isfinite(w0) and torch.isfinite(w1) and not torch.allclose(w0, w1)
    rows = []
    for d in (lg, lg.T):
        pos = g[:, None] == g[None, :]
        for i in range(6):
            n = torch.logsumexp(d[i][~pos[i]], 0)
            ws = torch.where(torch.arange(6) == i, 1.0, 0.5) * pos[i]
            rows.append(-(ws * (d[i] - torch.logaddexp(d[i], n))).sum() / ws.sum())
    assert torch.allclose(w0, torch.stack(rows).mean(), atol=1e-5) and torch.allclose(rows[0], want0, atol=1e-5)


def test_one_to_many_metrics_and_rerank():
    from bgm.analysis.arch_one_to_many import rerank, track_metrics
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    primary = np.array([0, 0, 1, 0, 0], bool)                 # the paired track is ranked 3rd
    anyhit = np.array([0, 1, 1, 0, 0], bool)                  # a theme mate is ranked 2nd
    m = track_metrics(scores, primary, anyhit)
    assert m["R@10 primary"] == 1.0 and m["R@10 any-hit"] == 1.0
    assert np.isclose(m["mAP"], (1 / 2 + 2 / 3) / 2)
    d = 1 / np.log2(np.arange(2, 7))
    assert np.isclose(m["nDCG@10"], (0.5 * d[1] + 1.0 * d[2]) / (1.0 * d[0] + 0.5 * d[1]))
    out = rerank(scores, lambda top: -scores[top], k=3)       # the reranker reverses the top 3 only
    assert np.argsort(-out).tolist() == [2, 1, 0, 3, 4]


def test_distill_kl_is_zero_when_student_matches_teacher():
    from bgm.train.losses import distill_kl
    torch.manual_seed(0)
    zv = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    zm = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    lt, ids = torch.tensor(np.log(0.07)), torch.tensor([0, 0, 1, 2, 3])
    assert distill_kl(zv, zm, zv @ zm.T, ids, lt, 0.07).abs() < 1e-6
    other = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1) @ zm.T
    assert distill_kl(zv, zm, other, ids, lt, 0.07) > 1e-3
