"""Source poses to a stable canonical skeleton and coordinate space."""

from abc import abstractmethod

from src.contract import ContractModel, PipelineModule, PoseFrame, SkeletonDefinition
from src.m_0_camera.contract import CameraFrameMetadata
from src.m_1_pose_estimator.contract import PoseEstimate


class NormalizedPose(ContractModel):
    """Canonical pose in meters: right-handed, +X right, +Y up, +Z backward.

    Fix origin at the initial pelvis position and orientation at calibration;
    preserve later root translation and heading. Partial-body implementations may
    calibrate to visible shoulders; declare that origin in the returned skeleton
    and keep unobserved pelvis/leg joints missing. Reset when changing subjects.
    Missing joints remain explicit; do not silently replace them with zeros.
    """

    source: CameraFrameMetadata
    skeleton: SkeletonDefinition
    pose: PoseFrame | None


class PoseNormalizer(PipelineModule):
    """Abstract skeleton conversion, calibration and optional filtering."""

    @abstractmethod
    async def normalize(self, estimate: PoseEstimate) -> NormalizedPose:
        """Map source joints and coordinates into the configured canonical space.

        :param estimate: Raw tracked pose, including no-detection events.
        :returns: Canonical pose retaining capture identity and person ID.
        """
        ...
