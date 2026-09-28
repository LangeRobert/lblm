"""Conversation routing and cleanup behavior through concrete orchestration."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from src.contract import JointObservation, MotionChunk, MotionFrame, PoseFrame
from src.m_3_motion_segmenter.contract import MotionSegment
from src.m_4_motion_to_language.contract import MotionDescription
from src.m_5_llm.contract import DialogueChunk, DialogueRequest
from src.m_6_language_to_motion.contract import MotionRequest
from src.skeleton import CANONICAL_RIG, CANONICAL_SKELETON


async def test_conversation_routes_action_and_closes_streams() -> None:
    """A caption drives text and motion, which is processed and sent to playback."""
    from src.m_7_motion_processor.causal import CausalMotionProcessor
    from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
    from src.pipeline import ConversationPipeline

    events = []
    captioner = MagicMock()
    captioner.describe = AsyncMock(
        return_value=MotionDescription(segment_id="s", text="A person waves.")
    )
    llm = MagicMock()
    motion = MagicMock()
    renderer = MagicMock()
    renderer.get_rig = AsyncMock(return_value=CANONICAL_RIG)
    renderer.submit = AsyncMock()
    renderer.cancel = AsyncMock()

    async def reply(request: DialogueRequest) -> AsyncGenerator[DialogueChunk]:
        """Supply a streamed dialogue response at the native model boundary."""
        yield DialogueChunk(
            response_id=request.response_id,
            sequence=0,
            text_delta="Hello!",
            motion_prompt="Wave back.",
            is_final=True,
        )

    async def generate(request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Supply canonical generated motion at the native model boundary."""
        yield MotionChunk(
            response_id=request.response_id,
            sequence=0,
            skeleton=CANONICAL_SKELETON,
            frames=(
                MotionFrame(
                    time_s=0.0, positions=tuple(j.rest_position for j in CANONICAL_SKELETON.joints)
                ),
            ),
            is_final=True,
        )

    llm.respond = reply
    motion.generate = generate
    processor = CausalMotionProcessor()
    retargeter = GeometricSkeletonRetargeter()
    await processor.open()
    await retargeter.open()
    pipeline = ConversationPipeline(
        camera=MagicMock(),
        estimator=MagicMock(),
        normalizer=MagicMock(),
        segmenter=MagicMock(),
        captioner=captioner,
        llm=llm,
        generator=motion,
        processor=processor,
        retargeter=retargeter,
        renderer=renderer,
        on_event=events.append,
    )
    segment = MotionSegment(
        segment_id="s",
        skeleton=CANONICAL_SKELETON,
        end_reason="window",
        frames=(
            PoseFrame(
                frame_id=0,
                captured_at_s=0.0,
                person_id="p",
                joints=tuple(
                    JointObservation(position=j.rest_position, confidence=1.0)
                    for j in CANONICAL_SKELETON.joints
                ),
            ),
        ),
    )
    await pipeline.reply_to(segment)
    assert [event.kind for event in events] == [
        "stage",
        "observation",
        "text",
        "reaction",
        "response_done",
    ]
    assert events[3].text == "Wave back."
    assert renderer.submit.await_count == 1
    request = renderer.submit.call_args.args[0]
    assert request.chunk.is_final and len(request.chunk.frames[0].bones) == 22
    assert request.play_at_s > 0
    assert len(pipeline.history) == 2


async def test_worker_failure_closes_camera_stream_and_every_open_stage() -> None:
    """TaskGroup failures must finish generators before releasing camera resources."""
    import asyncio
    from collections.abc import AsyncGenerator

    import pytest

    from src.m_0_camera.contract import CameraFrame
    from src.pipeline import ConversationPipeline

    closed = []
    names = (
        "camera",
        "estimator",
        "normalizer",
        "segmenter",
        "captioner",
        "llm",
        "generator",
        "processor",
        "retargeter",
        "renderer",
    )
    stages = {}
    for name in names:
        stage = MagicMock()
        stage.open = AsyncMock()
        stage.close = AsyncMock()
        stages[name] = stage

    async def frames() -> AsyncGenerator[CameraFrame]:
        """Keep capture alive until the failing pose worker cancels it."""
        try:
            yield CameraFrame(
                frame_id=0, captured_at_s=0.0, width=1, height=1, pixel_format="rgb8", data=bytes(3)
            )
            await asyncio.sleep(10)
        finally:
            closed.append("stream")

    stages["camera"].frames = frames
    stages["estimator"].estimate = AsyncMock(side_effect=ValueError("pose failure"))
    observed = []
    pipeline = ConversationPipeline(**stages, on_frame=observed.append)
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await pipeline.run()
    assert closed == ["stream"]
    assert len(observed) == 1 and observed[0].frame_id == 0
    for stage in stages.values():
        stage.close.assert_awaited_once()
