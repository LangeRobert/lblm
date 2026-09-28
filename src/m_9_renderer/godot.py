"""Owned Godot process with an authenticated loopback playback connection."""

import asyncio
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.contract import ContractModel
from src.m_8_skeleton_retargeter.contract import HumanoidRig
from src.m_9_renderer.contract import PlaybackRequest, Renderer
from src.skeleton import CANONICAL_RIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GodotConfig(ContractModel):
    """Local engine launch settings; no listening socket is exposed remotely."""

    executable: str | None = None
    project_path: Path = PROJECT_ROOT / "godot"
    headless: bool = False
    timeout_s: float = Field(default=15.0, gt=0, le=120)


class GodotRenderer(Renderer):
    """Send bounded animation chunks to a separately clocked Godot process."""

    def __init__(self, config: GodotConfig | None = None) -> None:
        """Configure process ownership without starting the engine.

        :param config: Executable, project directory and startup timeout.
        """
        self.config = config or GodotConfig()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._server: asyncio.Server | None = None
        self._connections: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
            asyncio.Queue(maxsize=1)
        )
        self._token = ""
        self._offset = 0.0
        self._lock = asyncio.Lock()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Authenticate the one engine connection and reject other peers.

        :param reader: Accepted loopback stream.
        :param writer: Accepted loopback stream writer.
        :returns: None after transferring ownership or closing the peer.
        """
        accepted = False
        try:
            async with asyncio.timeout(2.0):
                hello = json.loads(await reader.readline())
                if (
                    isinstance(hello, dict)
                    and hello.get("protocol") == 1
                    and secrets.compare_digest(str(hello.get("token", "")), self._token)
                    and self._connections.empty()
                    and self._writer is None
                ):
                    self._connections.put_nowait((reader, writer))
                    accepted = True
        except (TimeoutError, ValueError, ConnectionError):
            pass
        finally:
            if not accepted:
                writer.close()
                await writer.wait_closed()

    async def open(self) -> None:
        """Launch the engine, transfer its rig and synchronize monotonic clocks.

        :returns: None after the scene acknowledges readiness.
        """
        if self._writer is not None:
            return
        executable = self.config.executable or shutil.which("godot") or shutil.which("godot4")
        if executable is None:
            for path in (
                PROJECT_ROOT / "models/tools/Godot.app/Contents/MacOS/Godot",
                Path("/Applications/Godot.app/Contents/MacOS/Godot"),
            ):
                if path.is_file():
                    executable = str(path)
                    break
        if executable is None:
            raise FileNotFoundError("Install Godot 4.5+ or supply --godot /path/to/Godot")
        self._token = secrets.token_urlsafe(32)
        try:
            self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0, limit=262144)
            port = self._server.sockets[0].getsockname()[1]
            args = [executable, "--path", str(self.config.project_path)]
            if self.config.headless:
                args.append("--headless")
            args += ["--", "--lblm", str(port), self._token]
            self._process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.DEVNULL
            )
            async with asyncio.timeout(self.config.timeout_s):
                self._reader, self._writer = await self._connections.get()
            self._server.close()
            await self._exchange({"command": "init", "rig": CANONICAL_RIG.model_dump(mode="json")})
            before = time.monotonic()
            reply = await self._exchange({"command": "ping"})
            self._offset = float(reply["engine_time_s"]) - (before + time.monotonic()) / 2
        except BaseException:
            await self.close()
            raise

    async def _exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        """Serialize a bounded request/acknowledgement transaction.

        :param message: One protocol command.
        :returns: The engine acknowledgement.
        """
        async with self._lock:
            if self._writer is None or self._reader is None:
                raise RuntimeError("Call open before using the Godot renderer")
            payload = (json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n").encode()
            if len(payload) > 262144:
                raise ValueError("Godot message exceeds 256 KiB; use smaller chunks")
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    self._writer.write(payload)
                    await self._writer.drain()
                    line = await self._reader.readline()
                    if not line:
                        raise ConnectionError("Godot disconnected or its window was closed")
                    reply: dict[str, Any] = json.loads(line)
                    if not reply.get("ok") and reply.get("error") != "full":
                        raise ValueError(f"Godot rejected playback: {reply.get('error')}")
                    return reply
            except BaseException:
                self._writer.close()
                raise

    async def get_rig(self) -> HumanoidRig:
        """Return the acknowledged mannequin bind pose.

        :returns: Ordered canonical rig.
        """
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError("Call open before get_rig")
        return CANONICAL_RIG

    async def submit(self, request: PlaybackRequest) -> None:
        """Submit frames with clock conversion and bounded engine backpressure.

        :param request: Rig-local animation on the Python monotonic clock.
        :returns: None when queued, without waiting for playback to finish.
        """
        if request.chunk.rig_id != CANONICAL_RIG.rig_id or len(request.chunk.frames) > 120:
            raise ValueError("Unexpected rig or chunk exceeds 120 frames")
        if any(len(f.bones) != 22 for f in request.chunk.frames):
            raise ValueError("Every rig frame needs 22 bones")
        payload = request.model_dump(mode="json")
        payload["play_at_s"] = request.play_at_s + self._offset
        deadline = time.monotonic() + self.config.timeout_s
        while True:
            reply = await self._exchange({"command": "submit", "request": payload})
            if reply.get("ok"):
                return
            if time.monotonic() > deadline:
                raise TimeoutError("Godot playback queue remained full")
            await asyncio.sleep(0.02)

    async def cancel(self, response_id: str) -> None:
        """Remove all queued frames and stop the matching response.

        :param response_id: Response identifier to interrupt.
        :returns: None when acknowledged.
        """
        await self._exchange({"command": "cancel", "response_id": response_id})

    async def reset(self) -> None:
        """Clear playback and restore the mannequin rest pose.

        :returns: None when acknowledged.
        """
        await self._exchange({"command": "reset"})

    async def close(self) -> None:
        """Close transport and terminate only the engine process owned here.

        :returns: None after process exit; safe to repeat.
        """
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
            self._reader, self._writer = None, None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._process is not None:
            process, self._process = self._process, None
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 3.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
