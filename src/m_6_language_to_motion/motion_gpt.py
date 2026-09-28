"""MotionGPT implementation of text-to-motion (stage 6)."""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np

from src.contract import MotionChunk, MotionFrame, PositiveInt, Vector3
from src.m_6_language_to_motion.contract import LanguageToMotion, MotionRequest
from src.motion_gpt.geometry import resample_positions, validate_skeleton
from src.motion_gpt.runtime import MotionGPTRuntime


class MotionGPTLanguageToMotion(LanguageToMotion):
    """Generate full clips, then adapt duration and emit bounded playback chunks."""

    def __init__(self, runtime: MotionGPTRuntime, chunk_frames: PositiveInt = 10) -> None:
        """Attach an unloaded generator to a shared runtime.

        :param runtime: Same host used by the stage 4 captioner.
        :param chunk_frames: Maximum frames emitted per chunk.
        """
        if isinstance(chunk_frames, bool) or chunk_frames < 1:
            raise ValueError("chunk_frames must be a positive integer")
        self.runtime = runtime
        self.chunk_frames = chunk_frames
        self._opened = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Acquire model ownership once.

        :returns: None after the shared runtime is ready.
        """
        async with self._lock:
            if not self._opened:
                await self.runtime.acquire()
                self._opened = True

    async def generate(self, request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Generate an action and stream its decoded frames in canonical coordinates.

        :param request: HumanML3D skeleton, duration, timing and action text.
        :returns: Bounded chunks after full-clip inference has completed.
        :raises ValueError: For unsupported skeletons, continuation or timing.
        """
        async with self._lock:
            if not self._opened:
                raise RuntimeError("Call open before generating motion")
            validate_skeleton(request.skeleton)
            if request.continuation:
                raise ValueError(
                    "This MotionGPT text-to-motion adapter does not support conditioned continuation; use motion processing for clip transitions"
                )
            if not 0.2 <= request.duration_s <= 9.8 or request.frame_rate_hz > 120:
                raise ValueError("Supported duration is 0.2–9.8 seconds at up to 120 Hz")
            count = max(1, round(request.duration_s * request.frame_rate_hz))
            positions = await self.runtime.generate(
                request.prompt, round(request.duration_s * 20), request.seed
            )
            if len(positions) < 2:
                raise ValueError("MotionGPT must generate at least two frames")
            # Length is a learned prompt constraint, so retime the full decoded clip.
            positions = resample_positions(
                positions,
                np.arange(len(positions), dtype=np.float64),
                np.linspace(0, len(positions) - 1, count, dtype=np.float64),
            )
            for offset in range(0, count, self.chunk_frames):
                stop = min(offset + self.chunk_frames, count)
                frames = tuple(
                    MotionFrame(
                        time_s=request.start_time_s + index / request.frame_rate_hz,
                        positions=tuple(
                            Vector3(x=float(x), y=float(y), z=float(z))
                            for x, y, z in positions[index]
                        ),
                    )
                    for index in range(offset, stop)
                )
                yield MotionChunk(
                    response_id=request.response_id,
                    sequence=request.first_sequence + offset // self.chunk_frames,
                    skeleton=request.skeleton,
                    frames=frames,
                    is_final=request.ends_response and stop == count,
                )
                await asyncio.sleep(0)

    async def reset(self) -> None:
        """Wait for active work; no generated clip is retained between calls.

        :returns: None after the active iterator is closed or exhausted.
        """
        async with self._lock:
            pass

    async def close(self) -> None:
        """Release this generator's reference to the shared model.

        :returns: None after release; safe to repeat.
        """
        async with self._lock:
            if self._opened:
                await self.runtime.release()
                self._opened = False
