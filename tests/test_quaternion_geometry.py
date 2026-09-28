"""Regression for upstream HumanML3D's antiparallel-vector singularity."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_opposing_bone_vectors_produce_finite_unit_rotations() -> None:
    """A vertical raised arm must not create NaNs in motion features."""
    from src.motion_gpt.quaternion import quaternion_between

    source = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = quaternion_between(source, target)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result, axis=-1) == pytest.approx(np.ones(3))
    rotation = Rotation.from_quat(result[:, [1, 2, 3, 0]])
    assert rotation.apply(source) == pytest.approx(target, abs=1e-6)
    with pytest.raises(ValueError):
        quaternion_between(np.zeros((1, 3)), np.ones((1, 3)))
