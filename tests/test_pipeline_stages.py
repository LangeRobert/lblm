"""Concrete pose and animation stage behavior."""

import pytest

from src.contract import JointObservation, MotionChunk, MotionFrame, PoseFrame, Vector3
from src.m_0_camera.contract import CameraFrameMetadata
from src.m_1_pose_estimator.contract import PoseEstimate
from src.m_2_pose_normalizer.contract import NormalizedPose


async def test_normalizer_preserves_translation_and_missing_joints() -> None:
    """Calibration is fixed, and occlusion must not become a fabricated joint."""
    from src.m_2_pose_normalizer.canonical import CanonicalPoseNormalizer
    from src.skeleton import CANONICAL_SKELETON

    stage = CanonicalPoseNormalizer()
    await stage.open()
    frames = []
    for i in range(2):
        source = CameraFrameMetadata(frame_id=i, captured_at_s=float(i))
        pose = PoseFrame(
            frame_id=i,
            captured_at_s=float(i),
            person_id="p",
            joints=tuple(
                JointObservation(
                    position=Vector3(
                        x=j.rest_position.x + i, y=j.rest_position.y, z=j.rest_position.z
                    ),
                    confidence=1.0,
                )
                for j in CANONICAL_SKELETON.joints
            ),
        )
        frames.append(
            await stage.normalize(
                PoseEstimate(source=source, skeleton=CANONICAL_SKELETON, pose=pose)
            )
        )
    assert frames[0].pose.joints[0].position.x == pytest.approx(0)
    assert frames[1].pose.joints[0].position.x == pytest.approx(1)
    missing = pose.model_copy(
        update={"joints": pose.joints[:-1] + (JointObservation(position=None, confidence=0.0),)}
    )
    result = await stage.normalize(
        PoseEstimate(source=source, skeleton=CANONICAL_SKELETON, pose=missing)
    )
    assert result.pose.joints[-1].position is None
    await stage.close()


async def test_segmenter_bounds_windows_and_flushes_tracking_loss() -> None:
    """Windows remain bounded and never combine different tracked people."""
    from src.m_3_motion_segmenter.window import SegmenterConfig, WindowMotionSegmenter
    from src.skeleton import CANONICAL_SKELETON

    stage = WindowMotionSegmenter(SegmenterConfig(window_s=0.5, minimum_s=0.2))
    await stage.open()
    emitted = []
    for i in range(20):
        source = CameraFrameMetadata(frame_id=i, captured_at_s=i / 20)
        pose = PoseFrame(
            frame_id=i,
            captured_at_s=i / 20,
            person_id="p",
            joints=tuple(
                JointObservation(position=j.rest_position, confidence=1.0)
                for j in CANONICAL_SKELETON.joints
            ),
        )
        emitted.extend(
            await stage.push(NormalizedPose(source=source, skeleton=CANONICAL_SKELETON, pose=pose))
        )
    emitted.extend(
        await stage.push(NormalizedPose(source=source, skeleton=CANONICAL_SKELETON, pose=None))
    )
    assert emitted
    assert all(
        s.frames[-1].captured_at_s - s.frames[0].captured_at_s <= 0.5 + 1e-9 for s in emitted
    )
    assert emitted[-1].end_reason == "tracking_lost"
    assert await stage.flush() == ()


async def test_processor_and_retargeter_preserve_timing_and_unit_rotations() -> None:
    """Smoothing reduces a spike and retargeting produces valid local transforms."""
    from src.m_7_motion_processor.causal import CausalMotionProcessor
    from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
    from src.skeleton import CANONICAL_RIG, CANONICAL_SKELETON

    processor = CausalMotionProcessor()
    retargeter = GeometricSkeletonRetargeter()
    await processor.open()
    await retargeter.open()
    frames = tuple(
        MotionFrame(
            time_s=i / 20,
            positions=tuple(
                Vector3(
                    x=j.rest_position.x + float(i == 1), y=j.rest_position.y, z=j.rest_position.z
                )
                for j in CANONICAL_SKELETON.joints
            ),
        )
        for i in range(3)
    )
    chunk = MotionChunk(
        response_id="r", sequence=0, skeleton=CANONICAL_SKELETON, frames=frames, is_final=True
    )
    (output,) = await processor.process(chunk)
    assert 0 < output.frames[1].positions[0].x < 1
    result = await retargeter.retarget(output, CANONICAL_RIG)
    assert result.is_final and result.sequence == 0
    assert [f.time_s for f in result.frames] == [f.time_s for f in frames]
    for frame in result.frames:
        for bone in frame.bones:
            q = bone.rotation
            assert q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w == pytest.approx(1)
    with pytest.raises(ValueError):
        await processor.process(chunk)


async def test_mediapipe_mapping_retains_low_confidence_occlusion() -> None:
    """Inferred spine joints are visible only when all contributing landmarks are."""
    from src.m_1_pose_estimator.mediapipe import SOURCE_SKELETON
    from src.m_2_pose_normalizer.canonical import CanonicalPoseNormalizer

    stage = CanonicalPoseNormalizer()
    await stage.open()
    landmarks = [
        JointObservation(
            position=Vector3(x=float(i % 2) * 0.2, y=float(i) * 0.01, z=0.0), confidence=1.0
        )
        for i in range(33)
    ]
    landmarks[15] = JointObservation(position=Vector3(x=0.0, y=0.0, z=0.0), confidence=0.1)
    source = CameraFrameMetadata(frame_id=0, captured_at_s=0.0)
    result = await stage.normalize(
        PoseEstimate(
            source=source,
            skeleton=SOURCE_SKELETON,
            pose=PoseFrame(frame_id=0, captured_at_s=0.0, person_id="p", joints=tuple(landmarks)),
        )
    )
    assert result.pose is not None
    assert len(result.pose.joints) == 22
    assert result.pose.joints[20].position is None
    assert result.pose.joints[0].position == Vector3(x=0.0, y=0.0, z=0.0)


async def test_retargeted_forward_kinematics_preserves_rig_lengths() -> None:
    """Recovered bones reproduce a rigidly rotated rest skeleton without stretching."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
    from src.motion_gpt.geometry import PARENTS
    from src.skeleton import CANONICAL_RIG, CANONICAL_SKELETON

    stage = GeometricSkeletonRetargeter()
    await stage.open()
    rotation = Rotation.from_euler("y", 45, degrees=True)
    points = rotation.apply(
        [
            [j.rest_position.x, j.rest_position.y, j.rest_position.z]
            for j in CANONICAL_SKELETON.joints
        ]
    )
    chunk = MotionChunk(
        response_id="r",
        sequence=0,
        skeleton=CANONICAL_SKELETON,
        frames=(
            MotionFrame(
                time_s=0.0,
                positions=tuple(
                    Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points
                ),
            ),
        ),
        is_final=True,
    )
    result = await stage.retarget(chunk, CANONICAL_RIG)
    globals_ = []
    recovered = []
    for i, bone in enumerate(result.frames[0].bones):
        q, t = bone.rotation, bone.translation
        local = Rotation.from_quat([q.x, q.y, q.z, q.w])
        parent = PARENTS[i]
        globals_.append(globals_[parent] * local if i else local)
        recovered.append(
            recovered[parent] + globals_[parent].apply([t.x, t.y, t.z])
            if i
            else np.array([t.x, t.y, t.z])
        )
    assert np.asarray(recovered) == pytest.approx(points, abs=1e-6)
    # A vertical spine inherits root heading instead of cancelling its yaw.
    assert result.frames[0].bones[3].rotation.w == pytest.approx(1.0)


async def test_segmenter_tolerates_brief_landmark_dropout() -> None:
    """A weak frame must not erase every short run of good camera observations."""
    from src.m_3_motion_segmenter.window import SegmenterConfig, WindowMotionSegmenter
    from src.skeleton import CANONICAL_SKELETON

    stage = WindowMotionSegmenter(SegmenterConfig(window_s=0.5, minimum_s=0.3))
    await stage.open()
    emitted = []
    for i in range(21):
        source = CameraFrameMetadata(frame_id=i, captured_at_s=i / 20)
        joints = tuple(
            JointObservation(position=j.rest_position, confidence=1.0)
            for j in CANONICAL_SKELETON.joints
        )
        if i % 4 == 3:
            joints = joints[:-1] + (JointObservation(position=None, confidence=0.0),)
        pose = PoseFrame(frame_id=i, captured_at_s=i / 20, person_id="p", joints=joints)
        emitted.extend(
            await stage.push(NormalizedPose(source=source, skeleton=CANONICAL_SKELETON, pose=pose))
        )
    assert emitted
    assert all(all(j.position is not None for j in f.joints) for s in emitted for f in s.frames)


async def test_normalizer_calibrates_upper_body_without_inventing_hips() -> None:
    """Visible shoulders establish heading while unseen lower joints stay absent."""
    from src.m_1_pose_estimator.mediapipe import SOURCE_SKELETON
    from src.m_2_pose_normalizer.canonical import CanonicalPoseNormalizer
    from src.m_3_motion_segmenter.window import SegmenterConfig, WindowMotionSegmenter

    stage = CanonicalPoseNormalizer()
    segmenter = WindowMotionSegmenter(SegmenterConfig(window_s=0.5))
    await stage.open()
    await segmenter.open()
    landmarks = [JointObservation(position=None, confidence=0.0) for _ in range(33)]
    for i, x, y in (
        (11, -0.2, 0.0),
        (12, 0.2, 0.0),
        (13, -0.4, 0.1),
        (14, 0.4, 0.1),
        (15, -0.5, 0.2),
        (16, 0.5, 0.2),
    ):
        landmarks[i] = JointObservation(position=Vector3(x=x, y=y, z=0.0), confidence=1.0)
    emitted = []
    for i in range(12):
        source = CameraFrameMetadata(frame_id=i, captured_at_s=i / 20)
        estimate = PoseEstimate(
            source=source,
            skeleton=SOURCE_SKELETON,
            pose=PoseFrame(
                frame_id=i, captured_at_s=i / 20, person_id="upper", joints=tuple(landmarks)
            ),
        )
        result = await stage.normalize(estimate)
        assert result.pose is not None
        assert result.pose.joints[0].position is None
        assert result.skeleton.coordinates.origin == "initial shoulder midpoint"
        emitted.extend(await segmenter.push(result))
    assert emitted
