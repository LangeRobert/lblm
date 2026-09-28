"""Causal temporal filtering with strict stream validation."""

import math

import numpy as np
from numpy.typing import NDArray

from src.contract import MotionChunk, MotionFrame, Vector3
from src.m_7_motion_processor.contract import MotionProcessor
from src.motion_gpt.geometry import validate_skeleton


class CausalMotionProcessor(MotionProcessor):
    """Smooth jitter across chunk boundaries without lookahead or added buffers."""

    def __init__(self, smoothing_s: float = 0.06) -> None:
        """Configure a time-based low-pass filter.

        :param smoothing_s: Filter time constant in seconds; zero disables smoothing.
        """
        if not math.isfinite(smoothing_s) or smoothing_s < 0:
            raise ValueError("Smoothing must be finite and nonnegative")
        self.smoothing_s = smoothing_s
        self._opened = False
        self._response: str | None = None
        self._sequence = 0
        self._last_time = -1.0
        self._positions: NDArray[np.float64] | None = None
        self._final = False

    async def open(self) -> None:
        """Enable processing.

        :returns: None.
        """
        self._opened = True

    async def reset(self) -> None:
        """Discard filter history and response ordering.

        :returns: None.
        """
        self._response = None
        self._sequence = 0
        self._last_time = -1.0
        self._positions = None
        self._final = False

    async def close(self) -> None:
        """Discard filter state and close.

        :returns: None.
        """
        await self.reset()
        self._opened = False

    async def process(self, chunk: MotionChunk) -> tuple[MotionChunk, ...]:
        """Filter positions continuously while preserving frame times and completion.

        :param chunk: Contiguous canonical animation frames.
        :returns: One processed chunk, including an empty final if supplied.
        """
        if not self._opened:
            raise RuntimeError("Call open before process")
        validate_skeleton(chunk.skeleton)
        if chunk.response_id != self._response:
            if self._response is not None and not self._final:
                raise ValueError("Reset before interrupting an unfinished response")
            if chunk.sequence != 0:
                raise ValueError("New response sequence must start at zero")
            await self.reset()
            self._response = chunk.response_id
        if self._final or chunk.sequence != self._sequence:
            raise ValueError("Duplicate, completed or out-of-order chunk")
        if not chunk.frames and not chunk.is_final:
            raise ValueError("Only final chunks may be empty")
        last_time = self._last_time
        for frame in chunk.frames:
            if len(frame.positions) != 22 or frame.time_s <= last_time:
                raise ValueError("Expected 22 joints and increasing frame times")
            last_time = frame.time_s
        result: list[MotionFrame] = []
        for frame in chunk.frames:
            positions = np.array([[p.x, p.y, p.z] for p in frame.positions])
            if self._positions is not None and self.smoothing_s:
                alpha = -math.expm1(-(frame.time_s - self._last_time) / self.smoothing_s)
                positions = self._positions + alpha * (positions - self._positions)
            self._positions = positions
            self._last_time = frame.time_s
            result.append(
                MotionFrame(
                    time_s=frame.time_s,
                    positions=tuple(
                        Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions
                    ),
                )
            )
        self._sequence += 1
        self._final = chunk.is_final
        return (chunk.model_copy(update={"frames": tuple(result)}),)
