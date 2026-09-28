"""Native work must finish before a cancelled caller releases its resource lock."""

import asyncio
from threading import Event

import pytest


@pytest.mark.asyncio
async def test_cancellation_waits_for_native_operation() -> None:
    """Resource cleanup cannot race an uninterruptible native read or inference."""
    from src.runtime import run_blocking

    started, release = Event(), Event()
    disposed: list[bool] = []

    def native_call() -> bool:
        """Represent a native call that completes after the caller cancels."""
        started.set()
        return release.wait(timeout=3)

    task = asyncio.create_task(run_blocking(native_call, on_cancel=disposed.append))
    await asyncio.to_thread(started.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert disposed == [True]
