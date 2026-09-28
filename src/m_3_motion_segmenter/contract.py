"""Temporal boundary between live poses and motion understanding."""

from abc import abstractmethod
from typing import Annotated, Literal

from pydantic import Field

from src.contract import (
    ContractModel,
    Identifier,
    PipelineModule,
    PoseFrame,
    SkeletonDefinition,
)
from src.m_2_pose_normalizer.contract import NormalizedPose


class MotionSegment(ContractModel):
    """Bounded window for one person, with strictly increasing capture times.

    All frames share the skeleton and person ID. Window end permits streaming
    recognition without asserting that the person has finished their turn.
    """

    segment_id: Identifier
    skeleton: SkeletonDefinition
    frames: Annotated[tuple[PoseFrame, ...], Field(min_length=1)]
    end_reason: Literal["window", "pause", "tracking_lost", "flush"]


class MotionSegmenter(PipelineModule):
    """Abstract incremental segmenter with a bounded internal pose buffer."""

    @abstractmethod
    async def push(self, pose: NormalizedPose) -> tuple[MotionSegment, ...]:
        """Consume one event and emit zero or more completed windows.

        :param pose: Canonical pose or explicit tracking-loss event.
        :returns: Ordered segments ready for captioning.
        """
        ...

    @abstractmethod
    async def flush(self) -> tuple[MotionSegment, ...]:
        """Emit pending frames at end of capture, then clear the buffer.

        :returns: Remaining segments, or an empty tuple if no frames remain.
        """
        ...
