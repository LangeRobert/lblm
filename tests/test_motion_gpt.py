"""Check real adapter conversion and shared-runtime ownership without large weights."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.contract import (
    CoordinateSystem,
    JointDefinition,
    JointObservation,
    PoseFrame,
    SkeletonDefinition,
    Vector3,
)
from src.m_3_motion_segmenter.contract import MotionSegment
from src.m_6_language_to_motion.contract import MotionRequest


@pytest.fixture
def skeleton() -> SkeletonDefinition:
    """Return the documented HumanML3D topology in canonical coordinates."""
    JOINT_NAMES = (
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )
    PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

    return SkeletonDefinition(
        skeleton_id="test-humanml3d",
        coordinates=CoordinateSystem(
            units="meters",
            handedness="right",
            positive_x="right",
            positive_y="up",
            positive_z="backward",
            origin="initial pelvis",
        ),
        joints=tuple(
            JointDefinition(
                name=name,
                parent=JOINT_NAMES[PARENTS[index]] if index else None,
                rest_position=Vector3(x=float(index % 3), y=float(index // 3), z=0.0),
            )
            for index, name in enumerate(JOINT_NAMES)
        ),
    )


@pytest.mark.asyncio
async def test_generation_chunks_retime_and_preserve_response(skeleton: SkeletonDefinition) -> None:
    """Generated joints use the requested timing, identifiers and chunk completion."""
    from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion

    runtime = MagicMock()
    runtime.acquire = AsyncMock()
    runtime.release = AsyncMock()
    # Engine returns canonical, not upstream model-space, positions.
    positions = np.zeros((12, 22, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(12)[:, None]
    runtime.generate = AsyncMock(return_value=positions)
    adapter = MotionGPTLanguageToMotion(runtime, chunk_frames=3)
    await adapter.open()
    request = MotionRequest(
        response_id="r",
        prompt="Wave.",
        skeleton=skeleton,
        duration_s=0.5,
        frame_rate_hz=20.0,
        start_time_s=1.0,
        first_sequence=4,
    )
    chunks = [chunk async for chunk in adapter.generate(request)]
    frames = [frame for chunk in chunks for frame in chunk.frames]
    assert len(frames) == 10
    assert [c.sequence for c in chunks] == [4, 5, 6, 7]
    assert frames[0].time_s == 1.0 and frames[-1].time_s == 1.45
    assert frames[-1].positions[0].x == pytest.approx(11)
    assert sum(c.is_final for c in chunks) == 1 and chunks[-1].is_final
    assert all(c.response_id == "r" for c in chunks)
    await adapter.close()
    await adapter.close()
    runtime.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_caption_preserves_segment_and_missing_joints_fail(
    skeleton: SkeletonDefinition,
) -> None:
    """Reject missing observations rather than silently constructing zero poses."""
    from src.m_4_motion_to_language.motion_gpt import MotionGPTMotionToLanguage

    runtime = MagicMock()
    runtime.acquire = AsyncMock()
    runtime.release = AsyncMock()
    runtime.caption = AsyncMock(return_value="A person waves.")
    adapter = MotionGPTMotionToLanguage(runtime)
    await adapter.open()
    segment = MotionSegment(
        segment_id="s",
        skeleton=skeleton,
        end_reason="pause",
        frames=tuple(
            PoseFrame(
                frame_id=i,
                captured_at_s=i / 20,
                person_id="p",
                joints=tuple(
                    JointObservation(position=j.rest_position, confidence=1.0)
                    for j in skeleton.joints
                ),
            )
            for i in range(10)
        ),
    )
    result = await adapter.describe(segment)
    assert result.text == "A person waves." and result.segment_id == "s"
    assert result.confidence is None
    invalid = segment.model_copy(
        update={
            "frames": (
                segment.frames[0].model_copy(
                    update={"joints": (JointObservation(position=None, confidence=0.0),) * 22}
                ),
            )
        }
    )
    with pytest.raises(ValueError):
        await adapter.describe(invalid)
    await adapter.close()


@pytest.mark.asyncio
async def test_segment_conversion_resamples_and_checks_tracking(
    skeleton: SkeletonDefinition,
) -> None:
    """Timestamps drive 20 Hz resampling and a segment cannot switch subjects."""
    from src.m_4_motion_to_language.motion_gpt import MotionGPTMotionToLanguage

    runtime = MagicMock()
    runtime.acquire = AsyncMock()
    runtime.release = AsyncMock()
    runtime.caption = AsyncMock(return_value="A person moves.")
    adapter = MotionGPTMotionToLanguage(runtime)
    await adapter.open()

    frames = tuple(
        PoseFrame(
            frame_id=i,
            captured_at_s=i / 10,
            person_id="p",
            joints=tuple(
                JointObservation(
                    position=Vector3(
                        x=j.rest_position.x + i, y=j.rest_position.y, z=j.rest_position.z
                    ),
                    confidence=1.0,
                )
                for j in skeleton.joints
            ),
        )
        for i in range(6)
    )
    segment = MotionSegment(segment_id="s", skeleton=skeleton, end_reason="pause", frames=frames)
    await adapter.describe(segment)
    positions = runtime.caption.call_args.args[0]
    assert positions.shape == (11, 22, 3)
    assert positions[1, 0, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="person"):
        await adapter.describe(
            segment.model_copy(
                update={
                    "frames": frames[:-1] + (frames[-1].model_copy(update={"person_id": "other"}),)
                }
            )
        )


def test_wrong_skeleton_is_not_silently_relabelled(skeleton: SkeletonDefinition) -> None:
    """HumanML3D ordering must be explicit; other topologies need normalization."""
    from src.motion_gpt.geometry import validate_skeleton

    with pytest.raises(ValueError, match="HumanML3D"):
        validate_skeleton(skeleton.model_copy(update={"joints": tuple(reversed(skeleton.joints))}))


@pytest.mark.asyncio
async def test_shared_runtime_loads_once_and_survives_one_adapter_closing() -> None:
    """The captioner can close while the generator retains the loaded checkpoint."""
    from src.motion_gpt.runtime import MotionGPTRuntime

    engine = MagicMock()
    engine.caption.return_value = "A person waves."
    runtime = MotionGPTRuntime()
    with patch("src.motion_gpt.runtime.MotionGPTEngine", return_value=engine) as loader:
        await runtime.acquire()
        await runtime.acquire()
        loader.assert_called_once()
        await runtime.release()
        assert await runtime.caption(np.zeros((5, 22, 3), dtype=np.float32)) == "A person waves."
        await runtime.release()
        with pytest.raises(RuntimeError, match="Acquire"):
            await runtime.caption(np.zeros((5, 22, 3), dtype=np.float32))


@pytest.mark.asyncio
async def test_nonfinal_motion_request_does_not_end_response(skeleton: SkeletonDefinition) -> None:
    """Multiple action requests may share a response without premature completion."""
    from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion

    runtime = MagicMock()
    runtime.acquire = AsyncMock()
    runtime.release = AsyncMock()
    runtime.generate = AsyncMock(return_value=np.zeros((8, 22, 3), dtype=np.float32))
    adapter = MotionGPTLanguageToMotion(runtime)
    await adapter.open()
    request = MotionRequest(
        response_id="r",
        prompt="Wave.",
        skeleton=skeleton,
        duration_s=0.4,
        frame_rate_hz=20.0,
        ends_response=False,
    )
    chunks = [chunk async for chunk in adapter.generate(request)]
    assert chunks and not any(chunk.is_final for chunk in chunks)
    await adapter.close()


@pytest.mark.parametrize("stage", [4, 6])
@pytest.mark.asyncio
async def test_cancelled_close_can_be_retried(stage: int) -> None:
    """Cancellation while waiting to release must not lose adapter ownership."""
    from src.m_4_motion_to_language.motion_gpt import MotionGPTMotionToLanguage
    from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion

    runtime = MagicMock()
    runtime.acquire = AsyncMock()
    runtime.release = AsyncMock(side_effect=[asyncio.CancelledError(), None])
    adapter = (
        MotionGPTMotionToLanguage(runtime) if stage == 4 else MotionGPTLanguageToMotion(runtime)
    )
    await adapter.open()
    with pytest.raises(asyncio.CancelledError):
        await adapter.close()
    await adapter.close()
    assert runtime.release.await_count == 2
