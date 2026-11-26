"""D117: the re-implemented LanguageBind towers and preprocessing. The spatial block against transformers' own
CLIPEncoderLayer, the temporal block against the official einops formulation, the mel chunking and the scores.
Random weights only (no download)."""
import numpy as np
import pytest
import torch
from einops import rearrange

from bgm.features import languagebind as LB

VC = {"hidden_size": 32, "layer_norm_eps": 1e-5, "patch_size": 14, "num_attention_heads": 4, "intermediate_size": 64,
      "num_hidden_layers": 2}


def _rand_(m: torch.nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)


def test_spatial_block_matches_transformers_clip_layer():
    from transformers import CLIPVisionConfig
    from transformers.models.clip.modeling_clip import CLIPEncoderLayer
    hf_cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4, hidden_act="gelu",
                              layer_norm_eps=1e-5, attention_dropout=0.0)
    hf_cfg._attn_implementation = "eager"
    ref = CLIPEncoderLayer(hf_cfg).eval()
    ours = LB._Layer(32, 4, 64, 1e-5, t=1).eval()
    _rand_(ref)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(3, 7, 32)
    out = ref(x, None)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(ours(x), out, atol=1e-5)


def test_temporal_block_matches_official_formulation():
    t, B, N, D = 4, 2, 5, 32
    layer = LB._Layer(D, 4, 64, 1e-5, t=t).eval()
    _rand_(layer, 1)
    x = torch.randn(B * t, N, D)
    # official: '(b t) n d -> (b n) t d', + temporal embedding, residual, temporal LN + attention, back; then spatial
    h = rearrange(x, "(b t) n d -> (b n) t d", t=t) + layer.temporal_embedding[:, :t]
    h = h + layer.temporal_attn(layer.temporal_layer_norm1(h))
    h = rearrange(h, "(b n) t d -> (b t) n d", n=N)
    h = h + layer.self_attn(layer.layer_norm1(h))
    h = h + layer.mlp(layer.layer_norm2(h))
    assert torch.allclose(layer(x), h, atol=1e-5)


def test_tower_shapes_and_frame_mean():
    tower = LB.LanguageBindTower(VC, 16, t=2, grid=(2, 2)).eval()
    _rand_(tower, 2)
    px = torch.randn(3 * 2, 3, 28, 28)
    z = tower(px)
    assert z.shape == (3, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_mel_chunks_short_tiles_long_takes_three_fixed_chunks():
    short = torch.arange(300 * LB.N_MEL, dtype=torch.float32).view(300, LB.N_MEL)
    x = LB.mel_chunks(short)
    assert x.shape == (3, LB.N_MEL, LB.TARGET_LEN)
    raw = x * (LB.AUDIO_STD * 2) + LB.AUDIO_MEAN
    assert torch.allclose(raw[0, :, 300], short[0], atol=1e-2) and torch.equal(x[0], x[2])
    T = 3000
    long = torch.arange(T, dtype=torch.float32)[:, None].repeat(1, LB.N_MEL)
    raw = LB.mel_chunks(long) * (LB.AUDIO_STD * 2) + LB.AUDIO_MEAN
    starts = [int(r[0]) for r in np.array_split(np.arange(T - LB.TARGET_LEN + 1), 3)]
    assert [round(float(raw[c, 0, 0])) for c in range(3)] == starts


def test_audio_mel_asserts_rate_and_shape():
    y = np.random.default_rng(0).standard_normal(LB.SR * 3).astype(np.float32) * 0.1
    assert LB.audio_mel(y, LB.SR).shape == (3, LB.N_MEL, LB.TARGET_LEN)
    with pytest.raises(AssertionError):
        LB.audio_mel(y, 48000)


def test_frame_indices_and_tw_score():
    idx = LB.frame_indices(2.0, 4.0)
    assert len(idx) == 8 and idx[0] == 50 and idx[-1] == 99 and np.all(np.diff(idx) >= 0)
    assert LB.frame_indices(0.0, 10.0, n_total=120)[-1] == 119
    assert LB.tw_score(np.array([1.0, 0.0]), np.array([3.0, 1.0])) == pytest.approx(0.75)


def test_video_pixels_crop():
    fr = np.random.default_rng(1).integers(0, 255, (8, 180, 320, 3), dtype=np.uint8)
    x = LB.video_pixels(fr)
    assert x.shape == (8, 3, 224, 224)
