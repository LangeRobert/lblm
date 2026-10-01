"""Native idle-motion math and blending regression without interface tests."""

import subprocess
from pathlib import Path

import pytest

from src.m_9_renderer.godot import PROJECT_ROOT


def test_idle_loop_and_response_blending() -> None:
    """Verify looping, relaxed posture, response priority and smooth return.

    :returns: None after the native motion layer passes its assertions.
    """
    executable = PROJECT_ROOT / "models/tools/Godot.app/Contents/MacOS/Godot"
    if not executable.is_file():
        pytest.skip("Requires the local Godot runtime")
    script = Path(__file__).parent / "godot/test_idle_motion.gd"
    result = subprocess.run(
        [
            str(executable),
            "--headless",
            "--path",
            str(PROJECT_ROOT / "godot"),
            "--script",
            str(script),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IDLE_MOTION_OK" in result.stdout, result.stdout + result.stderr
    assert "SCRIPT ERROR" not in result.stderr, result.stderr
