"""Bounded asynchronous orchestration of the ten typed pipeline stages."""

import asyncio
import time
from collections.abc import Callable
from contextlib import AsyncExitStack, aclosing, suppress
from uuid import uuid4

from pydantic import Field

from src.contract import ContractModel
from src.events import PipelineEvent
from src.m_0_camera.contract import Camera, CameraFrame
from src.m_1_pose_estimator.contract import PoseEstimator
from src.m_2_pose_normalizer.contract import PoseNormalizer
from src.m_3_motion_segmenter.contract import MotionSegment, MotionSegmenter
from src.m_4_motion_to_language.contract import MotionToLanguage
from src.m_5_llm.contract import ChatMessage, DialogueRequest, LanguageModel
from src.m_6_language_to_motion.contract import LanguageToMotion, MotionRequest
from src.m_7_motion_processor.contract import MotionProcessor
from src.m_8_skeleton_retargeter.contract import SkeletonRetargeter
from src.m_9_renderer.contract import PlaybackRequest, Renderer
from src.skeleton import CANONICAL_SKELETON

__all__ = ["ConversationPipeline", "PipelineConfig", "PipelineEvent", "replace_latest"]


class PipelineConfig(ContractModel):
    """Latency bounds and conversation policy."""

    system_prompt: str = "You are a friendly embodied companion. Respond briefly to the observed body language. Describe one simple visible full-body action in your motion prompt. Do not infer sensitive traits or medical conditions."
    user_prompt: str = ""
    motion_duration_s: float = Field(default=2.0, ge=0.2, le=9.8)
    frame_rate_hz: float = Field(default=20.0, ge=1, le=60)
    capture_rate_hz: float = Field(default=30.0, gt=0, le=120)
    playback_buffer_s: float = Field(default=0.1, ge=0, le=1)
    history_pairs: int = Field(default=3, ge=0, le=10)
    repeat_cooldown_s: float = Field(default=5.0, ge=0)


def replace_latest[T](queue: asyncio.Queue[T], value: T) -> None:
    """Replace stale pending work without blocking its producer.

    :param queue: A bounded single-consumer mailbox.
    :param value: Newest pending item.
    :returns: None.
    """
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(value)


class ConversationPipeline:
    """Own lifecycle and route typed payloads while bounding latency and history."""

    def __init__(
        self,
        *,
        camera: Camera,
        estimator: PoseEstimator,
        normalizer: PoseNormalizer,
        segmenter: MotionSegmenter,
        captioner: MotionToLanguage,
        llm: LanguageModel,
        generator: LanguageToMotion,
        processor: MotionProcessor,
        retargeter: SkeletonRetargeter,
        renderer: Renderer,
        config: PipelineConfig | None = None,
        on_event: Callable[[PipelineEvent], None] | None = None,
    ) -> None:
        """Attach injectable implementations of each pipeline contract.

        :param camera: Frame source.
        :param estimator: Single-person landmark estimator.
        :param normalizer: Canonical skeleton conversion.
        :param segmenter: Bounded motion window producer.
        :param captioner: Motion-to-text model.
        :param llm: Local conversational model.
        :param generator: Text-to-motion model.
        :param processor: Temporal motion filter.
        :param retargeter: Target bone solver.
        :param renderer: Engine playback transport.
        :param config: Conversation and latency policy.
        :param on_event: Optional synchronous progress callback.
        """
        self.camera, self.estimator, self.normalizer = camera, estimator, normalizer
        self.segmenter, self.captioner, self.llm = segmenter, captioner, llm
        self.generator, self.processor, self.retargeter, self.renderer = (
            generator,
            processor,
            retargeter,
            renderer,
        )
        self.config = config or PipelineConfig()
        self.on_event = on_event
        self.history: tuple[ChatMessage, ...] = ()
        self._frames: asyncio.Queue[CameraFrame | None] = asyncio.Queue(maxsize=1)
        self._segments: asyncio.Queue[MotionSegment | None] = asyncio.Queue(maxsize=1)
        self._response: str | None = None
        self._last_caption = ""
        self._last_reply = -float("inf")
        self._play_until = 0.0
        self._reply_lock = asyncio.Lock()

    async def emit_event(self, event: PipelineEvent) -> None:
        """Deliver ordered dialogue updates to the renderer and optional observer.

        :param event: Camera-free conversation notification.
        :returns: None after the display acknowledges the update.
        """
        await self.renderer.publish_event(event)
        if self.on_event:
            self.on_event(event)

    async def run(self) -> None:
        """Open all stages, run bounded workers, then close resources in reverse order.

        :returns: None on EOF; cancellation interrupts workers and playback.
        """
        async with AsyncExitStack() as stack:
            for module in (
                self.renderer,
                self.estimator,
                self.normalizer,
                self.segmenter,
                self.captioner,
                self.llm,
                self.generator,
                self.processor,
                self.retargeter,
                self.camera,
            ):
                stack.push_async_callback(module.close)
                await module.open()
            await self.emit_event(
                PipelineEvent(
                    kind="ready",
                    text="Camera and models ready. Keep shoulders, elbows and wrists in view.",
                )
            )
            async with asyncio.TaskGroup() as group:
                group.create_task(self.capture(), name="camera")
                group.create_task(self.observe(), name="pose")
                group.create_task(self.converse(), name="conversation")
            await asyncio.sleep(max(0.0, self._play_until - time.monotonic()))

    async def capture(self) -> None:
        """Preview captured input and replace stale unprocessed images at a bounded rate.

        :returns: None after placing an EOF sentinel behind the last frame.
        """
        stream = self.camera.frames()
        next_capture = time.monotonic()
        async with aclosing(stream):
            async for frame in stream:
                await self.renderer.publish_frame(frame)
                replace_latest(self._frames, frame)
                next_capture = max(next_capture + 1 / self.config.capture_rate_hz, time.monotonic())
                await asyncio.sleep(max(0.0, next_capture - time.monotonic()))
        await self._frames.put(None)

    async def observe(self) -> None:
        """Estimate, normalize and segment the latest images without model backlog.

        :returns: None after draining final complete observations.
        """
        while (frame := await self._frames.get()) is not None:
            estimate = await self.estimator.estimate(frame)
            pose = await self.normalizer.normalize(estimate)
            for segment in await self.segmenter.push(pose):
                replace_latest(self._segments, segment)
        for segment in await self.segmenter.flush():
            replace_latest(self._segments, segment)
        await self._segments.put(None)

    async def converse(self) -> None:
        """Finish each reply and then consume only the most recent pending window.

        :returns: None after capture ends and pending work is drained.
        """
        while (segment := await self._segments.get()) is not None:
            await self.reply_to(segment)

    async def reply_to(self, segment: MotionSegment) -> None:
        """Route one observed action through both language models and animation.

        :param segment: Complete canonical motion window.
        :returns: None after the response is queued for playback.
        """
        async with self._reply_lock:
            started = time.monotonic()
            observation = await self.captioner.describe(segment)
            if (
                observation.text == self._last_caption
                and started - self._last_reply < self.config.repeat_cooldown_s
            ):
                return
            self._last_caption, self._last_reply = observation.text, started
            if self._response is not None:
                await self.renderer.cancel(self._response)
            response_id = str(uuid4())
            self._response = response_id
            await self.emit_event(
                PipelineEvent(kind="observation", text=observation.text, response_id=response_id)
            )
            text = ""
            action: str | None = None
            try:
                request = DialogueRequest(
                    response_id=response_id,
                    system_prompt=self.config.system_prompt,
                    user_prompt=self.config.user_prompt,
                    observation=observation,
                    history=self.history,
                )
                async with aclosing(self.llm.respond(request)) as stream:
                    async for delta in stream:
                        text += delta.text_delta
                        if delta.motion_prompt:
                            action = delta.motion_prompt
                        if delta.text_delta:
                            await self.emit_event(
                                PipelineEvent(
                                    kind="text", text=delta.text_delta, response_id=response_id
                                )
                            )
                if not action:
                    raise ValueError("Language model did not produce a complete motion prompt")
                await self.emit_event(
                    PipelineEvent(kind="reaction", text=action, response_id=response_id)
                )
                await self.processor.reset()
                rig = await self.renderer.get_rig()
                motion_request = MotionRequest(
                    response_id=response_id,
                    prompt=action,
                    skeleton=CANONICAL_SKELETON,
                    duration_s=self.config.motion_duration_s,
                    frame_rate_hz=self.config.frame_rate_hz,
                )
                play_at: float | None = None
                async with aclosing(self.generator.generate(motion_request)) as motion_stream:
                    async for raw in motion_stream:
                        for processed in await self.processor.process(raw):
                            retargeted = await self.retargeter.retarget(processed, rig)
                            if play_at is None:
                                play_at = time.monotonic() + self.config.playback_buffer_s
                            await self.renderer.submit(
                                PlaybackRequest(chunk=retargeted, play_at_s=play_at)
                            )
                            if retargeted.frames:
                                self._play_until = play_at + retargeted.frames[-1].time_s
                if self.config.history_pairs:
                    self.history = (
                        self.history
                        + (
                            ChatMessage(role="user", text=observation.text),
                            ChatMessage(role="assistant", text=text or action),
                        )
                    )[-2 * self.config.history_pairs :]
                await self.emit_event(
                    PipelineEvent(
                        kind="response_done",
                        response_id=response_id,
                        elapsed_s=time.monotonic() - started,
                    )
                )
            except BaseException:
                with suppress(ConnectionError, RuntimeError, TimeoutError, ValueError):
                    await self.renderer.cancel(response_id)
                await self.processor.reset()
                raise
