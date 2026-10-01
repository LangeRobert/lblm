"""Real native FBX loader and retargeting regression, without interface assertions."""

import subprocess
from pathlib import Path

import pytest

from src.m_9_renderer.godot import PROJECT_ROOT


def test_mixamo_bind_pose_and_motion_mapping() -> None:
    """Exercise the supplied rig's scale, limb motion, fingers and reset.

    :returns: None after Godot reports successful rig assertions.
    """
    executable = PROJECT_ROOT / "models/tools/Godot.app/Contents/MacOS/Godot"
    asset = PROJECT_ROOT / "models/mixamo-t-pose.fbx"
    if not executable.is_file() or not asset.is_file():
        pytest.skip("Requires the local Mixamo FBX and Godot runtime")
    script = Path(__file__).parent / "godot/test_mixamo_rig.gd"
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
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MIXAMO_RIG_OK" in result.stdout, result.stdout + result.stderr
    assert "SCRIPT ERROR" not in result.stderr, result.stderr
