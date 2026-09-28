"""Camera pixels to a single tracked person's source skeleton."""

from abc import abstractmethod

from src.contract import (
    ContractModel,
    JointObservation,
    PipelineModule,
    PoseFrame,
    SkeletonDefinition,
)
from src.m_0_camera.contract import CameraFrame, CameraFrameMetadata


class PoseEstimate(ContractModel):
    """Source skeleton and pose; None means no person was detected."""

    source: CameraFrameMetadata
    skeleton: SkeletonDefinition
    pose: PoseFrame | None
    # Normalized image coordinates in the original captured (possibly mirrored) image.
    # Same joint order as skeleton; empty when unavailable. Never project world XYZ as pixels.
    image_joints: tuple[JointObservation, ...] = ()


class PoseEstimator(PipelineModule):
    """Abstract pose estimator responsible for stable single-person tracking."""

    @abstractmethod
    async def estimate(self, frame: CameraFrame) -> PoseEstimate:
        """Estimate positions while preserving the input ID and timestamp.

        :param frame: Captured image; mirrored images must be accounted for.
        :returns: Explicit source coordinates, confidence and optional pose.
        """
        ...
