"""Opt-in all-stage smoke run against an explicitly supplied recorded video."""

import asyncio
import os
from pathlib import Path

import pytest

from src.app import AppConfig, create_pipeline
from src.m_0_camera.open_cv import OpenCVConfig
from src.m_9_renderer.godot import GodotConfig
from src.pipeline import PipelineEvent


@pytest.mark.models
@pytest.mark.skipif(
    os.environ.get("LBLM_RUN_MODEL_TESTS") != "1" or not os.environ.get("LBLM_TEST_VIDEO"),
    reason="Set LBLM_RUN_MODEL_TESTS=1 and LBLM_TEST_VIDEO to a full-body video fixture",
)
async def test_recorded_camera_to_godot_pipeline() -> None:
    """Run all ten actual backends without opening a physical camera.

    :returns: None after at least one response reaches the real Godot process.
    """
    events: list[PipelineEvent] = []
    config = AppConfig(
        camera=OpenCVConfig(source=Path(os.environ["LBLM_TEST_VIDEO"])),
        renderer=GodotConfig(headless=True),
    )
    pipeline = create_pipeline(config, events.append)
    async with asyncio.timeout(90):
        await pipeline.run()
    assert any(event.kind == "observation" for event in events)
    assert any(event.kind == "text" for event in events)
    assert any(event.kind == "response_done" for event in events)
