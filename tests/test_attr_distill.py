"""3.1 (D62): attribute-distillation targets and loss (train/attr_distill.py)."""
from dataclasses import dataclass

import numpy as np
import torch

from bgm.train.attr_distill import ONSET, RMS, ProfileTargets, distill_loss, music_profile


@dataclass
class _Store:
    files: dict
    music_of: dict
    segments: dict

    def music(self, m):
        return self.files[m]

    def pooled(self, side, m, key, a, b):
        x = self.files[m][key]
        return x[int(a):int(np.ceil(b))].mean(0)


def _file(rng, T, tempo):
    return {"attr_logits_1hz": rng.standard_normal((T, 4)).astype(np.float32), "dyn_25hz": rng.standard_normal((T * 25, 5)).astype(np.float32),
            "tempo_bpm": np.float32(tempo)}


def _store():
    rng = np.random.default_rng(0)
    files = {"m1": _file(rng, 10, 120.0), "m2": _file(rng, 10, 90.0), "m3": _file(rng, 10, -1.0)}
    return _Store(files, {"p1": "m1", "p2": "m2", "p3": "m3"}, {"p1": [(0.0, 5.0), (5.0, 10.0)], "p2": [(0.0, 10.0)], "p3": [(2.0, 8.0)]})


def test_profile_reads_the_span():
    st = _store()
    x = music_profile(st, "m1", 2.0, 6.0)
    d = st.files["m1"]["dyn_25hz"]
    assert x.shape == (7,)
    assert np.allclose(x[:4], st.files["m1"]["attr_logits_1hz"][2:6].mean(0))
    assert np.isclose(x[4], d[50:150, RMS].mean()) and np.isclose(x[5], d[50:150, ONSET].mean()) and np.isclose(x[6], np.log(120.0))
    assert np.isnan(music_profile(st, "m3", 0.0, 5.0)[6])                 # unknown tempo


def test_targets_are_z_scored_on_train_and_fill_unknown_tempo():
    st = _store()
    t = ProfileTargets(st, ["p1", "p2", "p3"])
    Z = np.stack([t(p, a, b) for p in ("p1", "p2") for a, b in st.segments[p]])
    assert t.dim == 7 and Z.dtype == np.float32 and np.isfinite(Z).all()
    assert t("p3", 2.0, 8.0)[6] == 0.0                                    # nan tempo -> the train mean
    X = np.stack([music_profile(st, st.music_of[p], a, b) for p in st.segments for a, b in st.segments[p]])
    assert np.allclose(t.mu[:6], X[:, :6].mean(0)) and np.isclose(t.mu[6], np.nanmean(X[:, 6]))


def test_distill_loss_trains_the_video_embedding():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 7)
    zv = torch.randn(5, 8, requires_grad=True)
    loss = distill_loss(head, zv, torch.randn(5, 7))
    loss.backward()
    assert loss.item() > 0 and zv.grad.abs().sum() > 0


def test_semantic_head_depth():
    from bgm.models.heads import SemanticHead
    h1, h3 = SemanticHead(16, 8, 4), SemanticHead(16, 8, 4, n_hidden=3)
    assert [k for k in h1.state_dict()] == ["net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", "net.4.weight", "net.4.bias"]
    assert sum(isinstance(m, torch.nn.Linear) for m in h3.net) == 4
    y = h3(torch.randn(5, 16))
    assert y.shape == (5, 4) and torch.allclose(y.norm(dim=-1), torch.ones(5), atol=1e-5)
