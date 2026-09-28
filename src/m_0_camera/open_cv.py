"""OpenCV capture of a camera device or local video file."""

import asyncio
from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from src.contract import ContractModel, NonNegativeInt, PositiveFloat, PositiveInt
from src.m_0_camera.contract import Camera, CameraFrame
from src.runtime import run_blocking

if TYPE_CHECKING:
    import cv2


class OpenCVConfig(ContractModel):
    """Capture preferences; hardware may negotiate different dimensions or FPS."""

    source: NonNegativeInt | Path = 0
    width: PositiveInt = 640
    height: PositiveInt = 480
    frame_rate_hz: PositiveFloat = 30.0
    mirrored: bool = False


class OpenCVCamera(Camera):
    """Pull-based BGR capture; one consumer, no unbounded Python frame queue."""

    def __init__(self, config: OpenCVConfig | None = None) -> None:
        """Configure capture without opening a device.

        :param config: Device index or local file and capture preferences.
        """
        self.config = config or OpenCVConfig()
        self._capture: cv2.VideoCapture | None = None
        self._lock = asyncio.Lock()
        self._streaming = False
        self._frame_id = 0

    async def open(self) -> None:
        """Open the configured source; repeated calls retain an open capture.

        :returns: None once the source is ready.
        :raises OSError: If OpenCV cannot open the device or file.
        """
        import cv2

        async with self._lock:
            if self._capture is not None:
                return
            source = self.config.source
            if isinstance(source, Path) and not source.is_file():
                raise FileNotFoundError(source)
            capture = cv2.VideoCapture()
            try:
                opened = await run_blocking(
                    partial(capture.open, str(source) if isinstance(source, Path) else source)
                )
                if not opened:
                    raise OSError(f"Cannot open OpenCV source {source}")
                if isinstance(source, int):
                    for key, value in (
                        (cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width)),
                        (cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height)),
                        (cv2.CAP_PROP_FPS, self.config.frame_rate_hz),
                        (cv2.CAP_PROP_BUFFERSIZE, 1.0),
                    ):
                        await run_blocking(partial(capture.set, key, value))
                self._capture = capture
            except BaseException:
                await run_blocking(capture.release)
                raise

    async def frames(self) -> AsyncGenerator[CameraFrame]:
        """Read frames off the event loop until EOF, closure or cancellation.

        :returns: Packed BGR uint8 frames with monotonic timestamps.
        :raises RuntimeError: If unopened or another consumer is active.
        :raises OSError: On live-device read failure; file EOF ends normally.
        """
        import cv2
        import numpy as np

        if self._capture is None:
            raise RuntimeError("Call open before consuming camera frames")
        if self._streaming:
            raise RuntimeError("Only one camera stream may be active")
        self._streaming = True
        capture = self._capture
        try:
            while True:
                async with self._lock:
                    if self._capture is not capture:
                        return
                    success, pixels = await run_blocking(capture.read)
                    captured_at = monotonic()
                    if not success:
                        if isinstance(self.config.source, Path):
                            return
                        raise OSError("OpenCV camera read failed")
                    if (
                        pixels is None
                        or pixels.dtype != np.uint8
                        or pixels.ndim != 3
                        or pixels.shape[2] != 3
                    ):
                        raise ValueError("OpenCV must produce uint8 BGR images")
                    if self.config.mirrored:
                        pixels = cv2.flip(pixels, 1)
                    frame = CameraFrame(
                        frame_id=self._frame_id,
                        captured_at_s=captured_at,
                        width=int(pixels.shape[1]),
                        height=int(pixels.shape[0]),
                        pixel_format="bgr8",
                        data=pixels.tobytes(),
                        mirrored=self.config.mirrored,
                    )
                    self._frame_id += 1
                yield frame
        finally:
            self._streaming = False

    async def reset(self) -> None:
        """Retain capture and monotonic identity; no Python frame queue is retained.

        :returns: None after any active read finishes.
        """
        async with self._lock:
            pass

    async def close(self) -> None:
        """Release capture after any in-flight read; repeated calls are safe.

        :returns: None after OpenCV resources are released.
        """
        async with self._lock:
            capture, self._capture = self._capture, None
            if capture is not None:
                await run_blocking(capture.release)
