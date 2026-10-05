"""Shared text-to-motion generation, processing and rig playback."""

import logging
import time
from contextlib import aclosing
from typing import TYPE_CHECKING

from src.m_6_language_to_motion.contract import LanguageToMotion, MotionRequest
from src.m_7_motion_processor.contract import MotionProcessor
from src.m_8_skeleton_retargeter.contract import SkeletonRetargeter
from src.m_9_renderer.contract import PlaybackRequest, Renderer
from src.skeleton import CANONICAL_SKELETON

if TYPE_CHECKING:
    from src.pipeline import PipelineConfig


async def play_motion(
    prompt: str,
    response_id: str,
    *,
    generator: LanguageToMotion,
    processor: MotionProcessor,
    retargeter: SkeletonRetargeter,
    renderer: Renderer,
    config: "PipelineConfig",
) -> float:
    """Generate a gesture and queue it, skipping failed generation requests.

    Generation exceptions cancel partial playback and reset processing so the
    next gesture can run. Cancellation and failures in processing, retargeting
    or renderer transport still propagate to the owning session.

    :param prompt: Direct action description for the motion model.
    :param response_id: Unique identifier for this playback.
    :param generator: Canonical motion generator.
    :param processor: Temporal filter and stream validator.
    :param retargeter: Avatar bone solver.
    :param renderer: Open engine transport.
    :param config: Duration, frame rate and playback buffer settings.
    :returns: Playback completion time, or the current time for skipped generation.
    """
    play_until = time.monotonic()
    await processor.reset()
    rig = await renderer.get_rig()
    motion_request = MotionRequest(
        response_id=response_id,
        prompt=prompt,
        skeleton=CANONICAL_SKELETON,
        duration_s=config.motion_duration_s,
        frame_rate_hz=config.frame_rate_hz,
    )
    play_at: float | None = None
    async with aclosing(generator.generate(motion_request)) as motion_stream:
        while True:
            try:
                raw = await anext(motion_stream)
            except StopAsyncIteration:
                break
            except Exception as error:  # noqa: BLE001 - generation is a failable stage
                await renderer.cancel(response_id)
                await processor.reset()
                logging.getLogger(__name__).warning("Skipping motion generation: %s", error)
                return time.monotonic()
            for processed in await processor.process(raw):
                retargeted = await retargeter.retarget(processed, rig)
                if play_at is None:
                    play_at = time.monotonic() + config.playback_buffer_s
                await renderer.submit(PlaybackRequest(chunk=retargeted, play_at_s=play_at))
                if retargeted.frames:
                    play_until = play_at + retargeted.frames[-1].time_s
    return play_until
