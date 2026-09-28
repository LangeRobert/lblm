"""Run the local camera-to-Godot body-language conversation."""

import asyncio
import json
import math
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path

import click
from pydantic import ValidationError

from src.app import AppConfig, create_pipeline
from src.debug_preview import DebugPreview
from src.demo import waving_segment
from src.pipeline import PipelineEvent


def print_event(event: PipelineEvent, *, debug: bool = False) -> None:
    """Display streamed dialogue without printing raw camera observations.

    :param event: Structured pipeline progress notification.
    :param debug: Color detected input and the action sent to motion generation.
    :returns: None.
    """
    if event.kind in ("tracking", "stage"):
        if debug:
            click.echo(f"\n[{event.kind}] {event.text}")
    elif debug and event.kind == "observation":
        click.secho(f"\n[Detected input] {event.text}", fg="green")
    elif event.kind == "reaction":
        if debug:
            click.secho(f"\n[Reaction] {event.text}", fg="blue")
    elif event.kind == "text":
        print(event.text, end="", flush=True)
    elif event.kind == "response_done":
        print(f"\n[Motion queued after {event.elapsed_s:.2f}s]", flush=True)
    else:
        print(f"\n[{event.kind}] {event.text}", flush=True)


async def run(
    config: AppConfig, *, demo: bool = False, seconds: float | None = None, debug: bool = False
) -> None:
    """Run the configured application or a camera-free model/render smoke demo.

    :param config: Validated backend and conversation settings.
    :param demo: Use a synthetic wave as the captioner's input.
    :param seconds: Optional live-session time limit after startup begins.
    :param debug: Enable colored diagnostics and the captured-input preview.
    :returns: None after all owned resources close.
    """
    preview = DebugPreview() if debug and not demo else None
    pipeline = create_pipeline(
        config,
        on_event=partial(print_event, debug=debug),
        on_pose=preview.show if preview else None,
    )
    try:
        if demo:
            async with AsyncExitStack() as stack:
                for module in (
                    pipeline.renderer,
                    pipeline.captioner,
                    pipeline.llm,
                    pipeline.generator,
                    pipeline.processor,
                    pipeline.retargeter,
                ):
                    stack.push_async_callback(module.close)
                    await module.open()
                print(
                    "Demo input: synthetic waving motion; camera and pose estimation are skipped.",
                    flush=True,
                )
                await pipeline.reply_to(waving_segment())
                await asyncio.sleep(
                    config.pipeline.motion_duration_s + config.pipeline.playback_buffer_s
                )
        else:
            deadline = asyncio.timeout(seconds)
            try:
                async with deadline:
                    await pipeline.run()
            except TimeoutError:
                if not deadline.expired():
                    raise
                print("\nSession time limit reached; resources closed.", flush=True)
    finally:
        if preview:
            preview.close()


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run the local camera-to-Godot body-language conversation.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Strict AppConfig JSON file.",
)
@click.option("--camera", type=click.IntRange(min=0), help="Camera device index (default 0).")
@click.option(
    "--video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local video instead of a camera.",
)
@click.option("--godot", help="Godot 4.5+ executable path.")
@click.option("--headless", is_flag=True, help="Run Godot without a visible window.")
@click.option("--demo", is_flag=True, help="Use a synthetic wave; no camera or input preview.")
@click.option(
    "--seconds",
    type=click.FloatRange(min=0, min_open=True),
    help="Maximum live run time including startup.",
)
@click.option("--prompt", help="Text prompt accompanying observed movement.")
@click.option("--offline", is_flag=True, help="Require already cached model files.")
@click.option(
    "--debug",
    is_flag=True,
    help="Show input video, detected input in green and reenacted reaction in blue.",
)
def main(
    config_path: Path | None,
    camera: int | None,
    video: Path | None,
    godot: str | None,
    headless: bool,
    demo: bool,
    seconds: float | None,
    prompt: str | None,
    offline: bool,
    debug: bool,
) -> None:
    """Run the local camera-to-Godot body-language conversation.

    :param config_path: Optional strict JSON settings file.
    :param camera: Camera device override.
    :param video: Local input video override.
    :param godot: Godot executable override.
    :param headless: Hide the Godot window; debug input remains visible.
    :param demo: Run with synthetic motion instead of captured input.
    :param seconds: Optional finite, positive live-session timeout.
    :param prompt: Additional dialogue instructions.
    :param offline: Disable model downloads.
    :param debug: Show captured input and colored motion diagnostics.
    :returns: None after owned resources have closed.
    """
    if camera is not None and video is not None:
        raise click.UsageError("--camera and --video are mutually exclusive")
    if seconds is not None and not math.isfinite(seconds):
        raise click.BadParameter("must be finite", param_hint="--seconds")
    try:
        config = (
            AppConfig.model_validate_json(config_path.read_text()) if config_path else AppConfig()
        )
        values = config.model_dump(mode="json")
        if camera is not None or video is not None:
            values["camera"]["source"] = str(video) if video is not None else camera
        if godot:
            values["renderer"]["executable"] = godot
        if headless:
            values["renderer"]["headless"] = True
        if prompt is not None:
            values["pipeline"]["user_prompt"] = prompt
        if offline:
            values["pose"]["model"]["local_files_only"] = True
            values["llm"]["model"]["local_files_only"] = True
            values["motiongpt"]["local_files_only"] = True
            values["motiongpt"]["checkpoint"]["local_files_only"] = True
        config = AppConfig.model_validate_json(json.dumps(values))
    except (OSError, ValidationError) as error:
        raise click.ClickException(str(error)) from error
    try:
        asyncio.run(run(config, demo=demo, seconds=seconds, debug=debug))
    except KeyboardInterrupt:
        click.echo("\nStopped.")


if __name__ == "__main__":
    main()
