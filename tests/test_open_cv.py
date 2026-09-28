"""Exercise actual OpenCV decoding without requesting camera access."""

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.asyncio
async def test_video_frames_preserve_pixels_and_monotonic_identity(tmp_path: Path) -> None:
    """Decode a real video and verify mirroring, EOF and resource reopening."""
    from src.m_0_camera.open_cv import OpenCVCamera, OpenCVConfig

    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
    assert writer.isOpened()
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[:, :16] = (0, 0, 255)
    for _ in range(3):
        writer.write(pixels)
    writer.release()
    camera = OpenCVCamera(OpenCVConfig(source=path, mirrored=True))
    with pytest.raises(RuntimeError, match="open"):
        await anext(camera.frames())
    await camera.open()
    frames = [frame async for frame in camera.frames()]
    assert len(frames) == 3
    assert [frame.frame_id for frame in frames] == [0, 1, 2]
    assert all(a.captured_at_s < b.captured_at_s for a, b in pairwise(frames))
    image = np.frombuffer(frames[0].data, dtype=np.uint8).reshape(24, 32, 3)
    assert image[:, 20:, 2].mean() > 240
    assert image[:, :10].mean() < 10
    assert frames[0].mirrored and frames[0].pixel_format == "bgr8"
    await camera.reset()
    await camera.close()
    await camera.close()
    await camera.open()
    reopened = camera.frames()
    assert (await anext(reopened)).frame_id > frames[-1].frame_id
    await reopened.aclose()
    await camera.close()


@pytest.mark.asyncio
async def test_missing_video_fails_without_opening_device(tmp_path: Path) -> None:
    """Reject a nonexistent file instead of falling back to a camera."""
    from src.m_0_camera.open_cv import OpenCVCamera, OpenCVConfig

    camera = OpenCVCamera(OpenCVConfig(source=tmp_path / "missing.avi"))
    with pytest.raises(FileNotFoundError):
        await camera.open()
    await camera.close()
