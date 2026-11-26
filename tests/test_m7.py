"""M7: context encoding equals whole-file encoding; window rules; synthetic acceptance (val sync >= 0.9)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from bgm.config import load_config
from bgm.models.heads import DynamicsEncoder
from bgm.train.dynamics import encode_grouped, window_starts


def test_context_crop_equals_whole_file_slice():
    torch.manual_seed(0)
    enc = DynamicsEncoder().eval()
    x = np.random.default_rng(0).standard_normal((1000, 5)).astype(np.float32)
    x[600:] *= 3
    with torch.no_grad():
        full = enc(torch.from_numpy(x.T)[None])[0]
        for s in (0, 5, 300, 770, 800):                        # interior and clamped-at-edge windows
            a, b = max(0, s - 30), min(1000, s + 200 + 30)
            crop = encode_grouped(enc, [x[a:b]], "cpu")[0][s - a:s - a + 200]
            assert torch.allclose(crop, full[s:s + 200], atol=1e-5), s
        # music: one encoding over +/- 4 s + C serves the aligned and every shifted window
        s = 400
        a, b = s - 100 - 30, s + 200 + 100 + 30
        h = encode_grouped(enc, [x[a:b]], "cpu")[0]
        for d in (-100, -37, 0, 52, 100):
            assert torch.allclose(h[s + d - a:s + d - a + 200], full[s + d:s + d + 200], atol=1e-5)


def test_window_rules():
    c = load_config("configs/default.yaml").train.dynamics
    rng = np.random.default_rng(0)
    assert window_starts(int(10.4 * 25), c, rng) == []           # too short: 8 s + 2 x 1.25 s margins
    st = window_starts(25 * 25, c, rng)
    assert len(st) == 2 and all(32 <= s <= 25 * 25 - 32 - 200 for s in st)
    assert len(window_starts(int(10.6 * 25), c, rng)) == 1
    from bgm.train.dynamics import window_length
    assert window_length(int(10.4 * 25), c, buckets=False) is None
    assert window_length(int(10.4 * 25), c) == 150 and window_length(int(7.0 * 25), c) == 100   # 6 s / 4 s buckets
    assert window_length(int(6.0 * 25), c) is None


@pytest.mark.slow
def test_synthetic_pairs_reach_sync_accuracy(tmp_path):
    """SPEC M7 acceptance: on synthetic pairs a trained encoder reaches val sync accuracy >= 0.9."""
    from bgm.data.segments import run_segment
    from bgm.eval.store import FeatureStore
    from bgm.train.dynamics import train_dynamics
    from tests.fixtures.pipeline import run_through_extract
    from tests.fixtures.synth import make_pairs
    pairs = make_pairs(tmp_path / "raw", n=36, seed=7, durations=(16.0, 26.0), embed_every=0)
    cfg, df = run_through_extract(tmp_path, pairs, what="sem_video,sem_music,dyn_video,dyn_music")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["train.dynamics.batch=32", "train.dynamics.patience=15"]))
    run_segment(cfg)
    store = FeatureStore(cfg)
    ids = list(df.pair_id)
    model, val = train_dynamics(cfg, store, ids[:28], ids[28:], seed=0, device="cuda" if torch.cuda.is_available() else "cpu",
                                k_shift=4, max_epochs=80)
    print(val)
    assert val["val_sync"] >= 0.9, val


def test_loss_stopping_keeps_a_checkpoint_when_losses_exceed_one(tmp_path):
    """M8 pretraining stops on held-out loss (criterion = -loss). The first epoch must count as an improvement even
    when the loss is above 1 (a best of -1 left no checkpoint and crashed the M8 run)."""
    from bgm.data.segments import run_segment
    from bgm.eval.store import FeatureStore
    from bgm.train.dynamics import train_dynamics
    from tests.fixtures.pipeline import run_through_extract
    from tests.fixtures.synth import make_pairs
    pairs = make_pairs(tmp_path / "raw", n=8, seed=3, durations=(16.0, 20.0), embed_every=0)
    cfg, df = run_through_extract(tmp_path, pairs, what="sem_video,sem_music,dyn_video,dyn_music")
    run_segment(cfg)
    ids = list(df.pair_id)
    model, val = train_dynamics(cfg, FeatureStore(cfg), ids[:6], ids[6:], seed=0, device="cpu", k_shift=4, max_epochs=2,
                                stop_on="loss", patience=5)
    assert val["best_epoch"] >= 0 and val["val_sync"] < -1.0, val       # held-out loss > 1, and a checkpoint was kept
