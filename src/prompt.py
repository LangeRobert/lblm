"""Interactive terminal gestures using only motion generation and playback."""

import asyncio
import sys
import time
from contextlib import AsyncExitStack, suppress
from uuid import uuid4

import click

from src.app import AppConfig
from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion
from src.m_7_motion_processor.causal import CausalMotionProcessor
from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
from src.m_9_renderer.godot import GodotRenderer
from src.motion_gpt.runtime import MotionGPTRuntime
from src.playback import play_motion


async def run_prompt(config: AppConfig) -> None:
    """Read terminal actions sequentially and play each before accepting the next.

    :param config: Motion model, renderer and playback settings.
    :returns: None on terminal EOF or an exit command, after resources close.
    """
    generator = MotionGPTLanguageToMotion(MotionGPTRuntime(config.motiongpt))
    processor = CausalMotionProcessor()
    retargeter = GeometricSkeletonRetargeter()
    renderer = GodotRenderer(config.renderer)
    async with AsyncExitStack() as stack:
        for module in (renderer, generator, processor, retargeter):
            stack.push_async_callback(module.close)
            await module.open()
        # StreamReader keeps piped lines buffered and input waits cancellable.
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )
        stack.callback(transport.close)
        click.echo(
            "Prompt mode ready. Type a gesture; use /quit or /exit to stop (Ctrl-C also works)."
        )
        while True:
            click.echo("Gesture> ", nl=False)
            line = await reader.readline()
            if not line:
                break
            action = line.decode(sys.getdefaultencoding()).strip()
            if action.lower() in ("/quit", "/exit"):
                break
            if not action:
                continue
            response_id = str(uuid4())
            try:
                play_until = await play_motion(
                    action,
                    response_id,
                    generator=generator,
                    processor=processor,
                    retargeter=retargeter,
                    renderer=renderer,
                    config=config.pipeline,
                )
                await asyncio.sleep(max(0.0, play_until - time.monotonic()))
            except BaseException:
                with suppress(ConnectionError, RuntimeError, TimeoutError, ValueError):
                    await renderer.cancel(response_id)
                await processor.reset()
                raise
