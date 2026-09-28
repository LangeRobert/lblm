"""CPU MediaPipe Pose Landmarker with pinned, automatically cached weights."""

import asyncio
import hashlib
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import Field

from src.assets import HubFile, resolve_file
from src.contract import (
    ContractModel,
    CoordinateSystem,
    JointDefinition,
    JointObservation,
    PoseFrame,
    SkeletonDefinition,
    Vector3,
)
from src.m_0_camera.contract import CameraFrame, CameraFrameMetadata
from src.m_1_pose_estimator.contract import PoseEstimate, PoseEstimator
from src.runtime import run_blocking

MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
SOURCE_SKELETON = SkeletonDefinition(
    skeleton_id="mediapipe-world-33-v1",
    coordinates=CoordinateSystem(
        units="meters",
        handedness="right",
        positive_x="camera right",
        positive_y="down",
        positive_z="forward",
        origin="current pelvis",
    ),
    joints=tuple(
        JointDefinition(
            name=f"landmark_{i}",
            parent="landmark_0" if i else None,
            rest_position=Vector3(x=0.0, y=0.0, z=0.0),
        )
        for i in range(33)
    ),
)


class MediaPipeConfig(ContractModel):
    """Pose model location and conservative single-subject tracking settings."""

    model: HubFile = HubFile(
        repo_id="imman12431/tennis-backhand-detector",
        repo_type="space",
        filename="pose_landmarker_lite.task",
        revision="e5cdeb5799ccd8f049e99006ab1d821fbf8719cf",
    )
    minimum_confidence: float = Field(default=0.5, gt=0, le=1)
    tracking_reset_s: float = Field(default=0.75, gt=0)


class MediaPipePoseEstimator(PoseEstimator):
    """Estimate one body's hip-relative world landmarks on a worker thread."""

    def __init__(self, config: MediaPipeConfig | None = None) -> None:
        """Create an unloaded detector.

        :param config: Model and tracking settings.
        """
        self.config = config or MediaPipeConfig()
        self._detector: Any = None
        self._lock = asyncio.Lock()
        self._timestamp = -1
        self._last_seen = -1.0
        self._person = str(uuid4())

    async def open(self) -> None:
        """Download verified weights and create the video detector.

        :returns: None after the native model is ready.
        """
        async with self._lock:
            if self._detector is not None:
                return
            from mediapipe.tasks.python import BaseOptions, vision

            path = await run_blocking(lambda: resolve_file(self.config.model))
            if self.config.model.local_path is None:
                digest = await run_blocking(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
                if digest != MODEL_SHA256:
                    raise ValueError("Pose checkpoint does not match the official Google model")
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.config.minimum_confidence,
                min_pose_presence_confidence=self.config.minimum_confidence,
                min_tracking_confidence=self.config.minimum_confidence,
            )
            self._detector = await run_blocking(
                lambda: vision.PoseLandmarker.create_from_options(options),
                on_cancel=lambda detector: detector.close(),
            )
            self._timestamp = -1

    async def estimate(self, frame: CameraFrame) -> PoseEstimate:
        """Decode packed pixels, undo mirroring and infer hip-relative landmarks.

        :param frame: Camera frame on the monotonic session clock.
        :returns: Explicit no-detection or 33 confidence-scored world landmarks.
        """
        async with self._lock:
            if self._detector is None:
                raise RuntimeError("Call open before estimate")
            import mediapipe as mp

            channels = 4 if frame.pixel_format == "rgba8" else 3
            if len(frame.data) != frame.width * frame.height * channels:
                raise ValueError("Camera byte count does not match dimensions")
            pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, channels
            )[:, :, :3]
            if frame.pixel_format == "bgr8":
                pixels = pixels[:, :, ::-1]
            if frame.mirrored:
                pixels = pixels[:, ::-1]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))
            self._timestamp = max(self._timestamp + 1, round(frame.captured_at_s * 1000))
            result = await run_blocking(
                lambda: self._detector.detect_for_video(image, self._timestamp)
            )
            pose = None
            image_joints: tuple[JointObservation, ...] = ()
            if result.pose_world_landmarks:
                if frame.captured_at_s - self._last_seen > self.config.tracking_reset_s:
                    self._person = str(uuid4())
                self._last_seen = frame.captured_at_s
                landmarks = result.pose_world_landmarks[0]
                if len(landmarks) != 33:
                    raise ValueError("Unexpected MediaPipe landmark count")
                pose = PoseFrame(
                    frame_id=frame.frame_id,
                    captured_at_s=frame.captured_at_s,
                    person_id=self._person,
                    joints=tuple(
                        JointObservation(
                            position=Vector3(x=float(j.x), y=float(j.y), z=float(j.z)),
                            confidence=float(min(j.visibility or 0.0, j.presence or 0.0)),
                        )
                        for j in landmarks
                    ),
                )
            if result.pose_landmarks:
                image_joints = tuple(
                    JointObservation(
                        position=Vector3(
                            x=float(1 - j.x if frame.mirrored else j.x), y=float(j.y), z=float(j.z)
                        ),
                        confidence=float(min(j.visibility or 0.0, j.presence or 0.0)),
                    )
                    for j in result.pose_landmarks[0]
                )
            return PoseEstimate(
                source=CameraFrameMetadata(
                    frame_id=frame.frame_id, captured_at_s=frame.captured_at_s
                ),
                skeleton=SOURCE_SKELETON,
                pose=pose,
                image_joints=image_joints,
            )

    async def reset(self) -> None:
        """Create a new subject identity without resetting native video time.

        :returns: None.
        """
        async with self._lock:
            self._last_seen = -1.0
            self._person = str(uuid4())

    async def close(self) -> None:
        """Wait for inference and release the native detector.

        :returns: None; safe to repeat.
        """
        async with self._lock:
            if self._detector is not None:
                detector, self._detector = self._detector, None
                await run_blocking(detector.close)
