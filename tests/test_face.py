"""D77: facial-expression feature vector and three-branch fusion helpers."""
import numpy as np

from bgm.analysis.arch_three_branch import fuse, weight_grid
from bgm.features.face import DIM, frame_vector


def test_frame_vector_counts_faces_and_expressions():
    probs = np.array([[0.1, 0, 0, 0.7, 0.2, 0, 0], [0, 0, 0, 0.2, 0.1, 0.7, 0]], np.float32)   # one happy, one sad face
    v = frame_vector(probs, np.array([100.0, 300.0]), 4000.0)
    assert v.shape == (DIM,) and np.allclose(v[0:7], probs.sum(0))
    assert v[7:14].tolist() == [0, 0, 0, 1, 0, 1, 0] and v[14] == 2 and np.isclose(v[15], 0.1) and v[16] == 0
    e = frame_vector(np.zeros((0, 7), np.float32), np.zeros(0), 4000.0)
    assert e[16] == 1 and e[:16].sum() == 0


def test_fuse_and_weight_grid():
    g = weight_grid(0.1)
    assert len(g) == 66 and all(abs(sum(w.values()) - 1) < 1e-9 for w in g)
    a, b = np.array([0.0, 1.0, 2.0]), np.array([5.0, 3.0, 4.0])
    assert fuse({"sem": a, "face": b}, {"sem": 1.0, "face": 0.0}).argmax() == 2
    assert fuse({"sem": a, "face": b}, {"sem": 0.0, "face": 1.0}).argmax() == 0
