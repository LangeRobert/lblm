"""Reference-counted shared MotionGPT host for both motion-language directions."""

import asyncio
from functools import partial

from src.motion_gpt.config import MotionGPTConfig
from src.motion_gpt.engine import MotionGPTEngine
from src.motion_gpt.geometry import Positions
from src.runtime import run_blocking


class MotionGPTRuntime:
    """Share one engine across adapters and serialize native inference and teardown."""

    def __init__(self, config: MotionGPTConfig | None = None) -> None:
        """Create an unloaded shared runtime.

        :param config: Pinned assets and execution device.
        """
        self.config = config or MotionGPTConfig()
        self._engine: MotionGPTEngine | None = None
        self._users = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Load once and acquire an adapter's ownership reference.

        :returns: None when inference can begin.
        """
        async with self._lock:
            if self._engine is None:
                self._engine = await run_blocking(partial(MotionGPTEngine, self.config))
            self._users += 1

    async def release(self) -> None:
        """Release one reference and unload only when the last adapter closes.

        :returns: None after any active inference finishes.
        """
        async with self._lock:
            if self._users > 0:
                self._users -= 1
            if self._users == 0:
                self._engine = None
            # No await after changing ownership: cancellation cannot partially
            # release a reference. PyTorch's global allocator may retain reusable
            # cache, but this runtime no longer owns the model or its tensors.

    async def caption(self, positions: Positions) -> str:
        """Caption canonical 20 Hz positions using the shared engine.

        :param positions: Validated complete HumanML3D joint positions.
        :returns: Generated movement description.
        """
        async with self._lock:
            if self._engine is None:
                raise RuntimeError("Acquire the MotionGPT runtime before captioning")
            return await run_blocking(partial(self._engine.caption, positions))

    async def generate(self, prompt: str, frame_count: int, seed: int | None) -> Positions:
        """Generate canonical positions using the shared engine.

        :param prompt: Complete action description.
        :param frame_count: Desired native-rate clip length.
        :param seed: Optional sampling seed.
        :returns: Complete generated positions; not incremental model inference.
        """
        async with self._lock:
            if self._engine is None:
                raise RuntimeError("Acquire the MotionGPT runtime before generation")
            return await run_blocking(partial(self._engine.generate, prompt, frame_count, seed))
