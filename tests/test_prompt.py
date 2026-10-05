"""Direct terminal gestures bypass camera observation and dialogue inference."""

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

import main as cli
from src.app import AppConfig
from src.contract import MotionChunk, MotionFrame
from src.m_6_language_to_motion.contract import MotionRequest
from src.skeleton import CANONICAL_RIG, CANONICAL_SKELETON


def test_prompt_flag_starts_direct_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bare flag must select interactive gestures without a text argument."""
    run = AsyncMock()
    monkeypatch.setattr(cli, "run", run)
    result = CliRunner().invoke(cli.main, ["--prompt", "--offline"])
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["prompt"] is True


@pytest.mark.parametrize("option", ["--demo", "--camera", "--video"])
def test_prompt_rejects_observation_options(option: str) -> None:
    """Direct mode rejects options that imply an observation source."""
    args = ["--prompt", option]
    if option == "--camera":
        args.append("0")
    elif option == "--video":
        args.append(__file__)
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ("  wave hello  \n\nraise both arms\n", ["wave hello", "raise both arms"]),
        ("wave hello\n/quit\nignored\n", ["wave hello"]),
        ("body role\nwave hello\n/quit\n", ["body role", "wave hello"]),
        ("/exit\nignored\n", []),
        (None, []),
    ],
)
async def test_prompt_plays_multiple_gestures_and_closes_on_eof(
    monkeypatch: pytest.MonkeyPatch,
    input_text: str | None,
    expected: list[str],
) -> None:
    """Play successive gestures, including after invalid generated output.

    :param monkeypatch: Backend and terminal substitutions.
    :param input_text: Buffered gestures, or None to test the session timeout.
    :param expected: Gestures passed to generation before the session ends.
    :returns: None after playback and resource cleanup assertions.
    """
    import src.prompt as prompt_module

    requests: list[MotionRequest] = []
    renderer = MagicMock()
    renderer.open = AsyncMock()
    renderer.close = AsyncMock()
    renderer.get_rig = AsyncMock(return_value=CANONICAL_RIG)
    renderer.submit = AsyncMock()
    renderer.cancel = AsyncMock()
    renderer.publish_event = AsyncMock()
    generator = MagicMock()
    generator.open = AsyncMock()
    generator.close = AsyncMock()

    async def generate(request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Fail invalid output or supply a frame without loading model weights.

        :param request: Gesture description and response identifier.
        :yields: One final canonical rest frame for valid gestures.
        """
        requests.append(request)
        if request.prompt == "body role":
            raise ValueError("MotionGPT returned no complete motion token sequence")
        yield MotionChunk(
            response_id=request.response_id,
            sequence=0,
            skeleton=CANONICAL_SKELETON,
            frames=(
                MotionFrame(
                    time_s=0.0,
                    positions=tuple(j.rest_position for j in CANONICAL_SKELETON.joints),
                ),
            ),
            is_final=True,
        )

    generator.generate = generate
    monkeypatch.setattr(prompt_module, "GodotRenderer", lambda config: renderer)
    monkeypatch.setattr(prompt_module, "MotionGPTLanguageToMotion", lambda runtime: generator)
    monkeypatch.setattr(
        cli, "create_pipeline", MagicMock(side_effect=AssertionError("camera path"))
    )
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as terminal:
        if input_text is not None:
            os.write(write_fd, input_text.encode())
            os.close(write_fd)
        monkeypatch.setattr(prompt_module.sys, "stdin", terminal)
        try:
            await cli.run(AppConfig(), prompt=True, seconds=0.05 if input_text is None else None)
        finally:
            if input_text is None:
                os.close(write_fd)
    assert [request.prompt for request in requests] == expected
    assert len({request.response_id for request in requests}) == len(expected)
    assert renderer.submit.await_count == len([action for action in expected if action != "body role"])
    if expected:
        assert len(renderer.submit.call_args.args[0].chunk.frames[0].bones) == 22
    renderer.close.assert_awaited_once()
    generator.close.assert_awaited_once()
