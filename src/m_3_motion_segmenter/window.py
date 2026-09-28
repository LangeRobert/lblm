"""Bounded motion windows with tracking-loss boundaries."""

from typing import Literal
from uuid import uuid4

from pydantic import Field

from src.contract import ContractModel, PoseFrame, SkeletonDefinition
from src.m_2_pose_normalizer.contract import NormalizedPose
from src.m_3_motion_segmenter.contract import MotionSegment, MotionSegmenter


class SegmenterConfig(ContractModel):
    """Window bounds compatible with MotionGPT's input duration."""

    window_s: float = Field(default=1.5, ge=0.2, le=9.8)
    minimum_s: float = Field(default=0.3, ge=0.2, le=9.8)
    max_frames: int = Field(default=300, ge=2, le=2000)
    max_gap_s: float = Field(default=0.3, gt=0)
    allow_upper_body: bool = True


class WindowMotionSegmenter(MotionSegmenter):
    """Emit visible-arm windows while tolerating brief landmark dropouts."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        """Initialize bounded buffers.

        :param config: Duration, frame-count and capture-gap limits.
        """
        self.config = config or SegmenterConfig()
        if self.config.minimum_s > self.config.window_s:
            raise ValueError("Minimum duration exceeds window")
        self._frames: list[PoseFrame] = []
        self._skeleton: SkeletonDefinition | None = None
        self._opened = False

    async def open(self) -> None:
        """Enable segmentation.

        :returns: None.
        """
        self._opened = True

    async def reset(self) -> None:
        """Discard buffered observations.

        :returns: None.
        """
        self._frames.clear()
        self._skeleton = None

    async def close(self) -> None:
        """Discard state and disable segmentation.

        :returns: None.
        """
        await self.reset()
        self._opened = False

    def _emit(
        self, reason: Literal["window", "tracking_lost", "flush"]
    ) -> tuple[MotionSegment, ...]:
        """Drain the buffer if it contains sufficient observed motion.

        :param reason: Boundary causing emission.
        :returns: At most one segment.
        """
        frames, self._frames = tuple(self._frames), []
        if (
            not frames
            or frames[-1].captured_at_s - frames[0].captured_at_s < self.config.minimum_s - 1e-9
        ):
            return ()
        assert self._skeleton is not None
        return (
            MotionSegment(
                segment_id=str(uuid4()), skeleton=self._skeleton, frames=frames, end_reason=reason
            ),
        )

    async def push(self, pose: NormalizedPose) -> tuple[MotionSegment, ...]:
        """Buffer usable frames and split on gaps, subjects or duration bounds.

        :param pose: Canonical observations, possibly absent or occluded.
        :returns: Completed, nonoverlapping windows.
        """
        if not self._opened:
            raise RuntimeError("Call open before push")
        frame = pose.pose
        if frame is None:
            return self._emit("tracking_lost")
        required = tuple(
            i
            for i, j in enumerate(pose.skeleton.joints)
            if not self.config.allow_upper_body
            or j.name
            in (
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
            )
        )
        if len(frame.joints) != len(pose.skeleton.joints) or not required:
            raise ValueError("Unexpected pose topology")
        if any(
            frame.joints[i].position is None or frame.joints[i].confidence <= 0 for i in required
        ):
            if self._frames and (
                frame.person_id != self._frames[-1].person_id
                or pose.skeleton != self._skeleton
                or frame.captured_at_s - self._frames[-1].captured_at_s > self.config.max_gap_s
            ):
                return self._emit("tracking_lost")
            return ()  # Brief landmark dropouts must not erase the entire window.

        emitted: tuple[MotionSegment, ...] = ()
        if self._frames:
            previous = self._frames[-1]
            delta = frame.captured_at_s - previous.captured_at_s
            if delta <= 0:
                raise ValueError("Capture timestamps must increase")
            if (
                frame.person_id != previous.person_id
                or pose.skeleton != self._skeleton
                or delta > self.config.max_gap_s
            ):
                emitted = self._emit("tracking_lost")
            elif (
                frame.captured_at_s - self._frames[0].captured_at_s > self.config.window_s + 1e-9
                or len(self._frames) >= self.config.max_frames
            ):
                emitted = self._emit("window")
        self._skeleton = pose.skeleton
        self._frames.append(frame)
        if frame.captured_at_s - self._frames[0].captured_at_s >= self.config.window_s - 1e-9:
            emitted += self._emit("window")
        return emitted

    async def flush(self) -> tuple[MotionSegment, ...]:
        """Drain sufficient observations at end of input.

        :returns: The final window, if long enough for captioning.
        """
        return self._emit("flush")
