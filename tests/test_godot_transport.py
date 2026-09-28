"""Backend transport validation without interface or scene assertions."""

import asyncio
import json
from unittest.mock import patch

import pytest

from src.m_8_skeleton_retargeter.contract import RigFrame, RigMotionChunk
from src.m_9_renderer.contract import PlaybackRequest
from src.skeleton import CANONICAL_RIG


async def test_renderer_transport_schedules_and_cancels() -> None:
    """Real TCP exchanges preserve response IDs and convert the clock once."""
    from src.m_9_renderer.godot import GodotConfig, GodotRenderer

    received = []
    tasks = []

    class Process:
        """Minimal process boundary for a protocol peer."""

        returncode = None

        def terminate(self) -> None:
            """Mark the peer process as terminated."""
            self.returncode = 0

        async def wait(self) -> int:
            """Return its successful exit status."""
            return 0

    async def peer(port: int, token: str) -> None:
        """Exercise the actual renderer wire protocol."""
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write((json.dumps({"token": token, "protocol": 1}) + "\n").encode())
        await writer.drain()
        while data := await reader.readline():
            message = json.loads(data)
            received.append(message)
            writer.write((json.dumps({"ok": True, "engine_time_s": 10.0}) + "\n").encode())
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def launch(*args: str, **kwargs: object) -> Process:
        """Start a local protocol peer in place of an engine process."""
        tasks.append(asyncio.create_task(peer(int(args[-2]), args[-1])))
        return Process()

    renderer = GodotRenderer(GodotConfig(executable="/bin/echo"))
    with patch("asyncio.create_subprocess_exec", side_effect=launch):
        await renderer.open()
        assert await renderer.get_rig() == CANONICAL_RIG
        chunk = RigMotionChunk(
            response_id="r",
            sequence=0,
            rig_id=CANONICAL_RIG.rig_id,
            frames=(RigFrame(time_s=0.0, bones=CANONICAL_RIG.rest_transforms),),
            is_final=True,
        )
        await renderer.submit(PlaybackRequest(chunk=chunk, play_at_s=1.0))
        await renderer.cancel("r")
        await renderer.close()
        await renderer.close()
    await asyncio.gather(*tasks)
    assert [m["command"] for m in received] == ["init", "ping", "submit", "cancel"]
    assert received[2]["request"]["chunk"]["response_id"] == "r"
    assert received[3]["response_id"] == "r"
    with pytest.raises(RuntimeError):
        await renderer.get_rig()
