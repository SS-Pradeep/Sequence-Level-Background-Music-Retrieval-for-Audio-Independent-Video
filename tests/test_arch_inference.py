"""D64: hubness-correction score transforms (analysis/arch_inference.py)."""
import numpy as np

from bgm.analysis.arch_inference import csls, inverted_softmax


def test_csls_penalises_a_hub():
    S = np.array([[0.50, 0.55]])                      # candidate 1 narrowly wins raw
    hub = np.array([[0.1, 0.2, 0.0], [0.9, 0.8, 0.9]])   # ... but it is close to every bank query
    assert S.argmax() == 1 and csls(S, hub, k=2).argmax() == 0
    assert np.allclose(csls(S, hub, k=2), [[0.50 - 0.15, 0.55 - 0.9]])
    assert np.allclose(csls(S, hub, k=99), S - hub.mean(1)[None])      # k larger than the bank: all of it


def test_inverted_softmax_matches_the_definition_and_keeps_rank_without_hubs():
    rng = np.random.default_rng(0)
    S, hub, tau = rng.standard_normal((4, 5)), rng.standard_normal((5, 7)), 0.07
    ref = np.log(np.exp(S / tau) / np.exp(hub / tau).sum(1)[None])
    assert np.allclose(inverted_softmax(S, hub, tau), ref)
    flat = np.tile(rng.standard_normal(7), (5, 1))                     # every candidate equally "hubby"
    assert (inverted_softmax(S, flat, tau).argsort(1) == S.argsort(1)).all()
