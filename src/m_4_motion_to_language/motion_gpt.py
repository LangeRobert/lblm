"""MotionGPT implementation of motion captioning (stage 4)."""

import asyncio

import numpy as np

from src.m_3_motion_segmenter.contract import MotionSegment
from src.m_4_motion_to_language.contract import MotionDescription, MotionToLanguage
from src.motion_gpt.geometry import resample_positions, validate_skeleton
from src.motion_gpt.runtime import MotionGPTRuntime


class MotionGPTMotionToLanguage(MotionToLanguage):
    """Caption complete HumanML3D motion windows through a shared model host."""

    def __init__(self, runtime: MotionGPTRuntime) -> None:
        """Attach an unloaded adapter to a shared runtime.

        :param runtime: Same instance used by the stage 6 adapter.
        """
        self.runtime = runtime
        self._opened = False
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """Acquire model ownership once.

        :returns: None after the runtime is ready.
        """
        async with self._lock:
            if not self._opened:
                await self.runtime.acquire()
                self._opened = True

    async def describe(self, segment: MotionSegment) -> MotionDescription:
        """Validate and caption a motion window.

        :param segment: Complete canonical 22-joint observations.
        :returns: Caption preserving the segment identity; confidence is unknown.
        """
        async with self._lock:
            if not self._opened:
                raise RuntimeError("Call open before describing motion")
            validate_skeleton(segment.skeleton)
            person = segment.frames[0].person_id
            observed: list[list[tuple[float, float, float]]] = []
            for frame in segment.frames:
                if frame.person_id != person:
                    raise ValueError("A motion segment must contain one person")
                if len(frame.joints) != 22:
                    raise ValueError("Every pose must contain 22 joints")
                row: list[tuple[float, float, float]] = []
                for joint in frame.joints:
                    if joint.position is None or joint.confidence <= 0:
                        raise ValueError("MotionGPT requires complete observed joints")
                    row.append((joint.position.x, joint.position.y, joint.position.z))
                observed.append(row)
            times = np.asarray([f.captured_at_s for f in segment.frames], dtype=np.float64)
            times -= times[0]
            if len(times) < 2 or not 0.2 - 1e-9 <= times[-1] <= 9.8 + 1e-9:
                raise ValueError("MotionGPT requires a window of 0.2 to 9.8 seconds")
            target = np.arange(int(np.floor(times[-1] * 20 + 1e-8)) + 1, dtype=np.float64) / 20
            positions = resample_positions(np.asarray(observed, dtype=np.float32), times, target)
            text = await self.runtime.caption(positions)
            return MotionDescription(segment_id=segment.segment_id, text=text)

    async def reset(self) -> None:
        """Wait for ongoing work; this adapter retains no temporal model state.

        :returns: None after the active operation finishes.
        """
        async with self._lock:
            pass

    async def close(self) -> None:
        """Release this adapter's ownership without unloading another adapter's model.

        :returns: None after the reference is released; safe to repeat.
        """
        async with self._lock:
            if self._opened:
                await self.runtime.release()
                self._opened = False
