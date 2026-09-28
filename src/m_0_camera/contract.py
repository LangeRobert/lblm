"""Camera capture boundary; no dependency on a capture library."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal

from src.contract import (
    ContractModel,
    NonNegativeInt,
    PipelineModule,
    PositiveInt,
    Seconds,
)


class CameraFrameMetadata(ContractModel):
    """Capture identity retained downstream without retaining image pixels."""

    frame_id: NonNegativeInt
    captured_at_s: Seconds


class CameraFrame(CameraFrameMetadata):
    """Packed uint8 pixels, row-major, without row padding.

    Data length must equal width * height * channels (3 for RGB/BGR, 4 for RGBA).
    Capture timestamps use one monotonic session clock, never wall-clock time.
    Frame IDs increase even when an implementation drops stale frames.
    """

    width: PositiveInt
    height: PositiveInt
    pixel_format: Literal["rgb8", "bgr8", "rgba8"]
    data: bytes
    mirrored: bool = False


class Camera(PipelineModule):
    """Abstract live source of camera frames."""

    @abstractmethod
    def frames(self) -> AsyncGenerator[CameraFrame]:
        """Stream frames until cancellation or end of capture.

        :returns: Async iterator of frames; consume directly with async for.
        """
        ...
