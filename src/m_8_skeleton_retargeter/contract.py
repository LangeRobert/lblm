"""Canonical joint positions to engine-neutral humanoid bone transforms."""

from abc import abstractmethod
from typing import Annotated

from pydantic import Field

from src.contract import (
    ContractModel,
    Identifier,
    MotionChunk,
    NonNegativeInt,
    PipelineModule,
    Quaternion,
    Seconds,
    SkeletonDefinition,
    Vector3,
)


class BoneTransform(ContractModel):
    """Absolute local transform relative to the parent, not a rest-pose delta.

    Root translation is in rig space, children in parent space. Translation
    uses meters; quaternions use the target rig's axes. Scale is fixed at one.
    """

    translation: Vector3
    rotation: Quaternion


class HumanoidRig(ContractModel):
    """Target skeleton and local rest transforms in matching joint order."""

    rig_id: Identifier
    skeleton: SkeletonDefinition
    rest_transforms: Annotated[tuple[BoneTransform, ...], Field(min_length=1)]


class RigFrame(ContractModel):
    """Complete pose in rig joint order on the response-relative timeline."""

    time_s: Seconds
    bones: Annotated[tuple[BoneTransform, ...], Field(min_length=1)]


class RigMotionChunk(ContractModel):
    """Retargeted frames with the same ordering and completion rules as MotionChunk."""

    response_id: Identifier
    sequence: NonNegativeInt
    rig_id: Identifier
    frames: tuple[RigFrame, ...]
    is_final: bool


class SkeletonRetargeter(PipelineModule):
    """Abstract bone mapping and inverse kinematics for a target humanoid rig."""

    @abstractmethod
    async def retarget(self, chunk: MotionChunk, rig: HumanoidRig) -> RigMotionChunk:
        """Solve target rotations and root movement from canonical joint positions.

        :param chunk: Processed motion with an explicit source skeleton.
        :param rig: Target hierarchy, coordinate conventions and bind pose.
        :returns: Rig-local transforms preserving timing and chunk identity.
        """
        ...
