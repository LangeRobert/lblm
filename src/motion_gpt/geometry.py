"""Checked HumanML3D topology and timeline conversion, independent of model weights."""

import numpy as np
from numpy.typing import NDArray

from src.contract import SkeletonDefinition

type Positions = NDArray[np.float32]

JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def validate_skeleton(skeleton: SkeletonDefinition) -> None:
    """Require HumanML3D ordering and canonical metric coordinates.

    :param skeleton: Skeleton accompanying observed or requested motion.
    :raises ValueError: If topology or axes require an upstream conversion.
    """
    expected = tuple(
        (name, JOINT_NAMES[parent] if index else None)
        for index, (name, parent) in enumerate(zip(JOINT_NAMES, PARENTS))
    )
    if tuple((joint.name, joint.parent) for joint in skeleton.joints) != expected:
        raise ValueError("MotionGPT requires the 22-joint HumanML3D topology and order")
    coordinates = skeleton.coordinates
    if (
        coordinates.units,
        coordinates.handedness,
        coordinates.positive_x,
        coordinates.positive_y,
        coordinates.positive_z,
    ) != ("meters", "right", "right", "up", "backward"):
        raise ValueError(
            "MotionGPT adapter requires canonical metric right/up/backward coordinates"
        )


def resample_positions(
    positions: Positions, source_times: NDArray[np.float64], target_times: NDArray[np.float64]
) -> Positions:
    """Interpolate every joint coordinate on a validated increasing timeline.

    :param positions: Finite frame-by-22-by-3 joint positions.
    :param source_times: Strictly increasing source times, one per frame.
    :param target_times: Increasing desired times inside the source interval.
    :returns: Float32 interpolated poses with the same joint ordering.
    """
    if (
        positions.ndim != 3
        or positions.shape[1:] != (22, 3)
        or len(positions) < 1
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Expected finite motion with shape (frames, 22, 3)")
    if (
        len(source_times) != len(positions)
        or not np.isfinite(source_times).all()
        or np.any(np.diff(source_times) <= 0)
    ):
        raise ValueError("Source timestamps must increase strictly")
    if (
        len(target_times) < 1
        or not np.isfinite(target_times).all()
        or np.any(np.diff(target_times) <= 0)
    ):
        raise ValueError("Target timestamps must increase strictly")
    if target_times[0] < source_times[0] or target_times[-1] > source_times[-1] + 1e-9:
        raise ValueError("Resampling cannot extrapolate motion")
    result = np.empty((len(target_times), 22, 3), dtype=np.float32)
    for joint in range(22):
        for axis in range(3):
            result[:, joint, axis] = np.interp(
                target_times, source_times, positions[:, joint, axis]
            )
    return result
