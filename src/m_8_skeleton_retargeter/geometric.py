"""Geometric retargeting to the canonical humanoid's fixed bone lengths."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.contract import MotionChunk, Quaternion
from src.m_8_skeleton_retargeter.contract import (
    BoneTransform,
    HumanoidRig,
    RigFrame,
    RigMotionChunk,
    SkeletonRetargeter,
)
from src.motion_gpt.geometry import PARENTS, validate_skeleton


class GeometricSkeletonRetargeter(SkeletonRetargeter):
    """Solve bone swing and branching-joint orientation; leaf twist is inherited."""

    def __init__(self) -> None:
        """Create an unloaded stateless solver.

        :returns: None.
        """
        self._opened = False

    async def open(self) -> None:
        """Enable retargeting.

        :returns: None.
        """
        self._opened = True

    async def reset(self) -> None:
        """Retain no temporal solver state.

        :returns: None.
        """

    async def close(self) -> None:
        """Disable retargeting.

        :returns: None.
        """
        self._opened = False

    async def retarget(self, chunk: MotionChunk, rig: HumanoidRig) -> RigMotionChunk:
        """Fit fixed-length target bones to canonical joint directions.

        :param chunk: Canonical motion with complete positions.
        :param rig: HumanML3D target with identity rest rotations.
        :returns: Absolute parent-local transforms; no rest deltas.
        """
        if not self._opened:
            raise RuntimeError("Call open before retarget")
        validate_skeleton(chunk.skeleton)
        validate_skeleton(rig.skeleton)
        if len(rig.rest_transforms) != 22 or any(
            abs(b.rotation.w - 1.0) > 1e-6 for b in rig.rest_transforms
        ):
            raise ValueError("The geometric solver requires 22 identity-oriented rest bones")
        rest = np.array(
            [[j.rest_position.x, j.rest_position.y, j.rest_position.z] for j in rig.skeleton.joints]
        )
        frames: list[RigFrame] = []
        for frame in chunk.frames:
            if len(frame.positions) != 22:
                raise ValueError("Expected 22 canonical positions")
            points = np.array([[p.x, p.y, p.z] for p in frame.positions])
            global_rotations: list[Rotation] = []
            bones: list[BoneTransform] = []
            for i, parent in enumerate(PARENTS):
                children = [j for j, p in enumerate(PARENTS) if p == i]
                parent_rotation = global_rotations[parent] if i else Rotation.identity()
                if children:
                    source = rest[children] - rest[i]
                    target = points[children] - points[i]
                    valid = (np.linalg.norm(source, axis=1) > 1e-8) & (
                        np.linalg.norm(target, axis=1) > 1e-8
                    )
                    source, target = source[valid], target[valid]
                    if not len(source):
                        rotation = parent_rotation
                    elif (
                        len(source) == 1
                        or np.linalg.matrix_rank(source) < 2
                        or np.linalg.matrix_rank(target) < 2
                    ):
                        swing, _ = Rotation.align_vectors(
                            target[:1], parent_rotation.apply(source[:1])
                        )
                        rotation = swing * parent_rotation
                    else:
                        source /= np.linalg.norm(source, axis=1)[:, None]
                        target /= np.linalg.norm(target, axis=1)[:, None]
                        rotation, _ = Rotation.align_vectors(target, source)
                else:
                    rotation = parent_rotation
                global_rotations.append(rotation)
                q = (parent_rotation.inv() * rotation).as_quat()
                bones.append(
                    BoneTransform(
                        translation=frame.positions[0]
                        if i == 0
                        else rig.rest_transforms[i].translation,
                        rotation=Quaternion(
                            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
                        ),
                    )
                )
            frames.append(RigFrame(time_s=frame.time_s, bones=tuple(bones)))
        return RigMotionChunk(
            response_id=chunk.response_id,
            sequence=chunk.sequence,
            rig_id=rig.rig_id,
            frames=tuple(frames),
            is_final=chunk.is_final,
        )
