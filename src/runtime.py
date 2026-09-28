"""Cancellation-safe execution of serialized native operations off the event loop."""

import asyncio
from collections.abc import Callable


async def run_blocking[T](
    operation: Callable[[], T], *, on_cancel: Callable[[T], None] | None = None
) -> T:
    """Run native work and finish it before propagating cancellation.

    Python cannot kill a running native thread safely. Callers must hold their
    resource lock across this await; cleanup cannot race an unfinished call.

    :param operation: Zero-argument blocking operation, usually a partial.
    :param on_cancel: Optional disposer for an allocated result not handed back.
    :returns: Operation result when the caller has not been cancelled.
    :raises asyncio.CancelledError: After in-flight native work has finished.
    """
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - retrieve native failure; preserve caller cancellation
                break
        if not task.cancelled() and task.exception() is None and on_cancel is not None:
            await run_blocking(lambda: on_cancel(task.result()))
        raise
