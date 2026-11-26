"""Shared fixtures. Tests never touch the real dataset; they use tests/fixtures/synth.py."""
from __future__ import annotations

import pytest

from tests.fixtures.synth import make_pairs


@pytest.fixture(scope="session")
def synth_pairs(tmp_path_factory):
    """Four short synthetic pairs; every second one carries embedded audio offset by 0.12 s."""
    out = tmp_path_factory.mktemp("synth")
    return make_pairs(out, n=4, seed=123, durations=(12.0, 14.0), embed_every=2)
