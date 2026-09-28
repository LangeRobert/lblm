"""MediaPipe landmark mapping and fixed initial-heading calibration."""

import numpy as np
from numpy.typing import NDArray

from src.contract import JointObservation, PoseFrame, Vector3
from src.m_1_pose_estimator.contract import PoseEstimate
from src.m_2_pose_normalizer.contract import NormalizedPose, PoseNormalizer
from src.motion_gpt.geometry import validate_skeleton
from src.skeleton import CANONICAL_SKELETON

# Each output uses a convex combination of observed MediaPipe world landmarks.
MAPPING: tuple[dict[int, float], ...] = (
    {23: 0.5, 24: 0.5},
    {23: 1.0},
    {24: 1.0},
    {23: 0.375, 24: 0.375, 11: 0.125, 12: 0.125},
    {25: 1.0},
    {26: 1.0},
    {23: 0.25, 24: 0.25, 11: 0.25, 12: 0.25},
    {27: 1.0},
    {28: 1.0},
    {11: 0.5, 12: 0.5},
    {31: 1.0},
    {32: 1.0},
    {11: 0.25, 12: 0.25, 7: 0.25, 8: 0.25},
    {11: 0.7, 12: 0.3},
    {11: 0.3, 12: 0.7},
    {7: 0.5, 8: 0.5},
    {11: 1.0},
    {12: 1.0},
    {13: 1.0},
    {14: 1.0},
    {15: 1.0},
    {16: 1.0},
)


class CanonicalPoseNormalizer(PoseNormalizer):
    """Map 33 world landmarks or canonical poses without hiding occlusion."""

    def __init__(self, minimum_confidence: float = 0.3) -> None:
        """Configure the visibility floor.

        :param minimum_confidence: Minimum accepted landmark confidence.
        """
        if not 0 < minimum_confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")
        self.minimum_confidence = minimum_confidence
        self._opened = False
        self._skeleton = CANONICAL_SKELETON
        self._person: str | None = None
        self._origin: NDArray[np.float64] | None = None
        self._basis = np.eye(3)

    async def open(self) -> None:
        """Enable normalization.

        :returns: None.
        """
        self._opened = True

    async def reset(self) -> None:
        """Discard subject calibration.

        :returns: None.
        """
        self._person = None
        self._skeleton = CANONICAL_SKELETON
        self._origin = None
        self._basis = np.eye(3)

    async def close(self) -> None:
        """Release temporal state; safe to repeat.

        :returns: None.
        """
        await self.reset()
        self._opened = False

    async def normalize(self, estimate: PoseEstimate) -> NormalizedPose:
        """Map coordinates and calibrate once per tracked subject.

        :param estimate: MediaPipe world landmarks or canonical observations.
        :returns: A canonical pose with explicit missing observations.
        """
        if not self._opened:
            raise RuntimeError("Call open before normalize")
        pose = estimate.pose
        if pose is None:
            return NormalizedPose(source=estimate.source, skeleton=self._skeleton, pose=None)
        if len(pose.joints) != len(estimate.skeleton.joints):
            raise ValueError("Joint count does not match source skeleton")
        mediapipe = estimate.skeleton.skeleton_id == "mediapipe-world-33-v1"
        if not mediapipe:
            validate_skeleton(estimate.skeleton)
        elif len(pose.joints) != 33:
            raise ValueError("MediaPipe requires 33 landmarks")
        mapped: list[JointObservation] = []
        weights = MAPPING if mediapipe else tuple({i: 1.0} for i in range(22))
        for mapping in weights:
            confidence = min(pose.joints[i].confidence for i in mapping)
            if confidence < self.minimum_confidence or any(
                pose.joints[i].position is None for i in mapping
            ):
                mapped.append(JointObservation(position=None, confidence=0.0))
                continue
            xyz = np.zeros(3)
            for i, weight in mapping.items():
                point = pose.joints[i].position
                assert point is not None
                xyz += weight * np.array([point.x, point.y, point.z])
            if mediapipe:
                xyz *= [1.0, -1.0, -1.0]  # Camera Y-down/Z-forward -> right-handed Y-up.
            mapped.append(
                JointObservation(
                    position=Vector3(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                    confidence=confidence,
                )
            )
        if self._person != pose.person_id:
            await self.reset()
            self._person = pose.person_id
        if self._origin is None:
            root, left, right = (mapped[i].position for i in (0, 1, 2))
            if root is None or left is None or right is None:
                left, right = mapped[16].position, mapped[17].position
                if left is not None and right is not None:
                    # Calibrate from the visible shoulder midpoint, keeping hidden joints missing.
                    self._skeleton = CANONICAL_SKELETON.model_copy(
                        update={
                            "skeleton_id": "humanml3d-shoulder-calibrated-v1",
                            "coordinates": CANONICAL_SKELETON.coordinates.model_copy(
                                update={"origin": "initial shoulder midpoint"}
                            ),
                        }
                    )
                    root = Vector3(
                        x=(left.x + right.x) / 2, y=(left.y + right.y) / 2, z=(left.z + right.z) / 2
                    )
            if root is None or left is None or right is None:
                return NormalizedPose(source=estimate.source, skeleton=self._skeleton, pose=None)
            self._origin = np.array([root.x, root.y, root.z])
            axis = np.array([right.x - left.x, 0.0, right.z - left.z])
            if np.linalg.norm(axis) < 1e-6:
                self._origin = None
                return NormalizedPose(source=estimate.source, skeleton=self._skeleton, pose=None)
            axis /= np.linalg.norm(axis)
            self._basis = np.column_stack(
                (axis, np.array([0.0, 1.0, 0.0]), np.cross(axis, [0.0, 1.0, 0.0]))
            )
        calibrated: list[JointObservation] = []
        for joint in mapped:
            if joint.position is None:
                calibrated.append(joint)
                continue
            p = joint.position
            xyz = (np.array([p.x, p.y, p.z]) - self._origin) @ self._basis
            calibrated.append(
                JointObservation(
                    position=Vector3(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                    confidence=joint.confidence,
                )
            )
        return NormalizedPose(
            source=estimate.source,
            skeleton=self._skeleton,
            pose=PoseFrame(
                frame_id=pose.frame_id,
                captured_at_s=pose.captured_at_s,
                person_id=pose.person_id,
                joints=tuple(calibrated),
            ),
        )
