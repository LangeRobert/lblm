"""Stable HumanML3D wxyz rotations, including exactly opposing vectors."""

import numpy as np
from numpy.typing import NDArray


def quaternion_between(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Return minimal rotations with a deterministic 180-degree fallback axis.

    :param source: Nonzero source vectors with trailing dimension three.
    :param target: Broadcast-compatible nonzero target vectors.
    :returns: Normalized wxyz quaternions compatible with upstream HumanML3D.
    :raises ValueError: For nonfinite or degenerate input vectors.
    """
    a, b = np.broadcast_arrays(
        np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    )
    if a.shape[-1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected finite three-dimensional vectors")
    a_norm, b_norm = (
        np.linalg.norm(a, axis=-1, keepdims=True),
        np.linalg.norm(b, axis=-1, keepdims=True),
    )
    if np.any(a_norm < 1e-10) or np.any(b_norm < 1e-10):
        raise ValueError("Cannot rotate a zero-length bone")
    a, b = a / a_norm, b / b_norm
    scalar = 1 + np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    vector = np.cross(a, b)
    opposing = scalar[..., 0] < 1e-10
    basis = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    fallback = np.cross(a, basis)
    fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
    vector = np.where(opposing[..., None], fallback, vector)
    scalar = np.where(opposing[..., None], 0.0, scalar)
    result = np.concatenate((scalar, vector), axis=-1)
    return np.asarray(result / np.linalg.norm(result, axis=-1, keepdims=True), dtype=np.float32)
