"""MediaPipe adapter tests without loading native model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.m_0_camera.contract import CameraFrame


async def test_pose_decodes_mirror_and_retains_tracking_identity() -> None:
    """The inference input is unmirrored and missing detections remain explicit."""
    from pathlib import Path

    from src.assets import HubFile
    from src.m_1_pose_estimator.mediapipe import MediaPipeConfig, MediaPipePoseEstimator

    stage = MediaPipePoseEstimator(
        MediaPipeConfig(
            model=HubFile(repo_id="test", filename="test", local_path=Path("/tmp/test.task"))
        )
    )
    landmarks = [
        SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9, presence=0.8) for _ in range(33)
    ]
    detector = MagicMock()
    detector.detect_for_video.side_effect = [
        SimpleNamespace(pose_world_landmarks=[landmarks], pose_landmarks=[landmarks]),
        SimpleNamespace(pose_world_landmarks=[], pose_landmarks=[]),
    ]
    pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    with (
        patch("src.m_1_pose_estimator.mediapipe.resolve_file"),
        patch(
            "mediapipe.tasks.python.vision.PoseLandmarker.create_from_options",
            return_value=detector,
        ),
    ):
        await stage.open()
        first = await stage.estimate(
            CameraFrame(
                frame_id=1,
                captured_at_s=1.0,
                width=2,
                height=1,
                pixel_format="bgr8",
                data=pixels.tobytes(),
                mirrored=True,
            )
        )
        assert first.pose is not None and len(first.pose.joints) == 33
        assert first.pose.joints[0].confidence == 0.8
        assert first.image_joints[0].position.x == 0.9
        assert first.image_joints[0].position.y == 0.2
        image, timestamp = detector.detect_for_video.call_args.args
        assert image.numpy_view().tolist() == [[[6, 5, 4], [3, 2, 1]]]
        assert timestamp >= 0
        second = await stage.estimate(
            CameraFrame(
                frame_id=2,
                captured_at_s=1.0001,
                width=2,
                height=1,
                pixel_format="rgb8",
                data=pixels.tobytes(),
            )
        )
        assert second.pose is None
        assert detector.detect_for_video.call_args.args[1] > timestamp
        await stage.close()
        detector.close.assert_called_once()
