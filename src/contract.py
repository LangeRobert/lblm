"""Shared, backend-independent data types and module lifecycle contracts."""

from abc import ABC, abstractmethod
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type Identifier = Annotated[str, Field(min_length=1)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveInt = Annotated[int, Field(gt=0)]
type Seconds = Annotated[float, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0)]
type Confidence = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    """Immutable payload with strict field types and finite numeric values."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class Vector3(ContractModel):
    """Three coordinates in the space declared by the enclosing payload."""

    x: float
    y: float
    z: float


class Quaternion(ContractModel):
    """Rotation in xyzw order; producers must supply a unit quaternion."""

    x: Annotated[float, Field(ge=-1, le=1)]
    y: Annotated[float, Field(ge=-1, le=1)]
    z: Annotated[float, Field(ge=-1, le=1)]
    w: Annotated[float, Field(ge=-1, le=1)]


class CoordinateSystem(ContractModel):
    """Explicit source axes, units and origin; no implicit engine convention."""

    units: Literal["meters", "normalized_image"]
    handedness: Literal["left", "right"]
    positive_x: Identifier
    positive_y: Identifier
    positive_z: Identifier
    origin: Identifier


class JointDefinition(ContractModel):
    """Named joint; parent is another name in the skeleton or None for root."""

    name: Identifier
    parent: Identifier | None
    rest_position: Vector3


class SkeletonDefinition(ContractModel):
    """Ordered unique joints, one root and an acyclic parent hierarchy.

    Rest positions are absolute positions in the declared coordinate system.
    Every pose uses this exact joint order. Identifiers are versioned by the
    producer whenever topology, rest pose or coordinate conventions change.
    """

    skeleton_id: Identifier
    coordinates: CoordinateSystem
    joints: Annotated[tuple[JointDefinition, ...], Field(min_length=1)]


class JointObservation(ContractModel):
    """Observed position and confidence; None explicitly represents occlusion."""

    position: Vector3 | None
    confidence: Confidence


class PoseFrame(ContractModel):
    """One tracked person's pose on the session's monotonic capture clock.

    Joint count must match the accompanying skeleton. A missing person is
    represented by None at the module boundary, not by a zero-valued pose.
    """

    frame_id: NonNegativeInt
    captured_at_s: Seconds
    person_id: Identifier
    joints: Annotated[tuple[JointObservation, ...], Field(min_length=1)]


class MotionFrame(ContractModel):
    """Complete generated pose; time is relative to the start of its response.

    Positions are absolute in the canonical space, including root motion.
    Their count and order must match the accompanying skeleton.
    """

    time_s: Seconds
    positions: Annotated[tuple[Vector3, ...], Field(min_length=1)]


class MotionChunk(ContractModel):
    """Contiguous generated frames, usable as either a full clip or a stream.

    Sequence numbers start at zero per response. Frame times strictly increase
    within and across chunks. Emit exactly one final chunk; an empty final
    chunk is allowed. Nonfinal chunks must contain frames. Chunks do not overlap.
    """

    response_id: Identifier
    sequence: NonNegativeInt
    skeleton: SkeletonDefinition
    frames: tuple[MotionFrame, ...]
    is_final: bool


class PipelineModule(ABC):
    """Lifecycle shared by all modules, with no concrete behavior.

    Call open once before use and close in a finally block. Operations are
    serialized per instance. Implementations must release resources on close,
    propagate cancellation and avoid blocking the event loop during inference.
    """

    @abstractmethod
    async def open(self) -> None:
        """Acquire camera, model or engine resources.

        :returns: None after the module is ready.
        """
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Clear temporal state and pending output while retaining loaded resources.

        :returns: None after state from the previous interaction is discarded.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release owned resources; repeated calls must be safe.

        :returns: None after resources are released.
        """
        ...
