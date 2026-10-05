"""Generation failures are recoverable at the shared playback boundary."""

import asyncio
import time
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.contract import MotionChunk, MotionFrame
from src.m_6_language_to_motion.contract import MotionRequest
from src.m_7_motion_processor.causal import CausalMotionProcessor
from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
from src.pipeline import PipelineConfig
from src.playback import play_motion
from src.skeleton import CANONICAL_RIG, CANONICAL_SKELETON


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, asyncio.CancelledError])
async def test_failed_generation_cleans_up_and_accepts_next_gesture(
    partial: bool,
    error_type: type[BaseException],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip failed clips and discard partial playback while preserving cancellation.

    :param partial: Whether to queue a nonfinal chunk before the failure.
    :param error_type: Recoverable inference failure or session cancellation.
    :param caplog: Captured diagnostic messages.
    :returns: None after recovery and subsequent playback assertions.
    """
    closed: list[str] = []
    generator = MagicMock()
    renderer = MagicMock()
    renderer.get_rig = AsyncMock(return_value=CANONICAL_RIG)
    renderer.submit = AsyncMock()
    renderer.cancel = AsyncMock()
    processor = CausalMotionProcessor()
    retargeter = GeometricSkeletonRetargeter()
    await processor.open()
    await retargeter.open()

    async def generate(request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Supply a failed stream followed by a valid clip.

        :param request: Gesture and response identifier.
        :yields: A canonical rest frame when requested.
        """
        try:
            if partial or request.prompt == "wave":
                yield MotionChunk(
                    response_id=request.response_id,
                    sequence=0,
                    skeleton=CANONICAL_SKELETON,
                    frames=(
                        MotionFrame(
                            time_s=0.0,
                            positions=tuple(j.rest_position for j in CANONICAL_SKELETON.joints),
                        ),
                    ),
                    is_final=request.prompt == "wave",
                )
            if request.prompt == "fail":
                raise error_type("no usable motion")
        finally:
            closed.append(request.response_id)

    generator.generate = generate
    kwargs = {
        "generator": generator,
        "processor": processor,
        "retargeter": retargeter,
        "renderer": renderer,
        "config": PipelineConfig(),
    }
    if error_type is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await play_motion("fail", "failed", **kwargs)
        assert closed == ["failed"]
        assert "Skipping motion generation" not in caplog.text
        return
    end = await play_motion("fail", "failed", **kwargs)
    assert end <= time.monotonic()
    renderer.cancel.assert_awaited_once_with("failed")
    assert "Skipping motion generation: no usable motion" in caplog.text
    await play_motion("wave", "next", **kwargs)
    assert closed == ["failed", "next"]
    assert renderer.submit.await_count == int(partial) + 1
    assert renderer.submit.call_args.args[0].chunk.response_id == "next"


async def test_motion_processing_failure_still_propagates() -> None:
    """Confine recovery to generation rather than hiding later stage failures.

    :returns: None after checking propagation and stream closure.
    """
    closed: list[str] = []
    generator = MagicMock()
    renderer = MagicMock()
    renderer.get_rig = AsyncMock(return_value=CANONICAL_RIG)
    renderer.cancel = AsyncMock()
    processor = MagicMock()
    processor.reset = AsyncMock()
    processor.process = AsyncMock(side_effect=ValueError("invalid processing state"))

    async def generate(request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Supply a valid empty final chunk to the failing processor.

        :param request: Requested response identifier.
        :yields: One final chunk.
        """
        try:
            yield MotionChunk(
                response_id=request.response_id,
                sequence=0,
                skeleton=CANONICAL_SKELETON,
                frames=(),
                is_final=True,
            )
        finally:
            closed.append(request.response_id)

    generator.generate = generate
    with pytest.raises(ValueError, match="invalid processing state"):
        await play_motion(
            "wave",
            "failed",
            generator=generator,
            processor=processor,
            retargeter=MagicMock(),
            renderer=renderer,
            config=PipelineConfig(),
        )
    assert closed == ["failed"]
    renderer.cancel.assert_not_awaited()
