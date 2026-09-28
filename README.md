# LBLM — Large Body Language Model

A local Python 3.13+ camera → body motion → conversation → animated humanoid
application. All ten stages have concrete implementations behind typed Pydantic
contracts. Godot 4 displays a bundled white, skinned humanoid in a frontal view on a black
background. No separate avatar download is required.

## Run

From this directory on Apple silicon:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --all-extras --group dev
uv run --all-extras python main.py
# Preview camera input and color detected input/reaction diagnostics:
uv run --all-extras python main.py --debug
```

Godot is discovered on `PATH`, in `/Applications/Godot.app`, or in the local
`models/tools/Godot.app` used during development. Otherwise pass
`--godot /path/to/Godot`. Install the standard [Godot 4.5+ executable](https://godotengine.org/download/).
Grant camera access to the launching terminal when macOS requests it. Keep both shoulders, elbows and wrists visible. Full-body framing enables
MotionGPT captioning; upper-body framing uses conservative arm-motion descriptions. Ctrl-C closes capture, models and
the owned Godot process.

```bash
# Camera-free demo: real language/motion models and Godot, synthetic input wave.
uv run --all-extras python main.py --demo
# Local video as input; no physical camera access.
uv run --all-extras python main.py --video /absolute/path/person.mp4
# Reuse cached weights without network access; hide the engine window for a smoke run.
uv run --all-extras python main.py --demo --offline --headless
```

Missing model assets download through Hugging Face into ignored
`models/huggingface` on first open. Importing modules opens no devices and loads
no weights. The pose checkpoint is pinned and verified against Google's original
SHA-256. MediaPipe is pinned to 0.10.35 because 1.0.1 crashes during macOS pose
initialization. The camera extra uses the same OpenCV-contrib distribution as
MediaPipe to avoid two packages overwriting `cv2`.

`--config settings.json` loads the strict nested `AppConfig` schema in
`src/app.py`. CLI overrides include `--camera 0`, `--prompt "..."`, `--seconds 60`,
`--godot`, `--headless`, and `--offline`. The time limit includes model startup. The CLI uses Click; `--help` lists all options.

`--debug` opens an OpenCV input window with detected joints and skeleton connections
(green: visible, orange: uncertain), prints detected motion in green and the
exact action sent to MotionGPT in blue. Every two seconds it reports tracking
status and names missing arm joints; it also reports when captioning starts. Escape or closing the preview disables
only that window; Ctrl-C stops the application. `--headless` hides Godot but does
not hide the debug input window. `--demo --debug` prints colored diagnostics
without an input window, because the demo uses synthetic motion rather than video.

## Stages

| Stage | Implementation |
| --- | --- |
| 0 Capture | OpenCV, packed BGR frames, monotonic timestamps |
| 1 Pose | MediaPipe Pose Landmarker Lite, 33 world landmarks |
| 2 Normalize | Confidence-aware 33→22 mapping and fixed initial heading/origin |
| 3 Segment | Bounded 1.5-second windows; split on tracking loss, gaps and subject changes |
| 4 Describe | MotionGPT for complete poses; geometric arm-motion descriptions for upper-body input |
| 5 Respond | Gemma 4 E2B Q4_0 via llama-cpp-python/Metal, streamed text and action JSON |
| 6 Animate | The same MotionGPT runtime, full-clip generation delivered in chunks |
| 7 Process | Causal smoothing across chunk boundaries and stream validation |
| 8 Retarget | Fixed-length humanoid bone rotations and parent-local transforms |
| 9 Render | Godot Skeleton3D, interpolation, short entry blend and bounded TCP playback |

The orchestrator uses one-slot latest-frame and latest-segment mailboxes, bounded
history, repeat-caption suppression, shared MotionGPT ownership and cancellation-safe
cleanup. It finishes an active model response before consuming the newest pending
window. Playback for the preceding response is cancelled when a new reply begins.

## Verification and limits

```bash
uv run --all-extras pytest
uv run --all-extras mypy src main.py
uv run --all-extras ruff check src tests main.py
LBLM_RUN_MODEL_TESTS=1 uv run --all-extras pytest -q -s -m models
# Include all ten stages using a full-body recorded video:
LBLM_RUN_MODEL_TESTS=1 LBLM_TEST_VIDEO=/path/person.mp4 uv run --all-extras pytest -q -m models
```

A native smoke run exercised all ten stages with a local video made from Google's
public pose sample. It queued motion 2.06 seconds after captioning began, in addition
to the observation window, and Python peaked at approximately 4.15 GB RSS. A separate
synthetic-wave demo took 3.15 seconds. These single runs on an M4 Pro do **not** establish
sustained latency or total 16 GB unified-memory compliance, including Godot and macOS.

MediaPipe's root is pelvis-relative, so this backend cannot recover reliable room
translation. Missing landmarks remain explicit. Upper-body windows require both shoulders,
elbows and wrists; short dropouts up to 0.3 seconds are skipped and the model
resamples between retained observations. Longer gaps split the window.
The upper-body fallback describes measured arm position/movement using fixed
geometric rules, without reconstructing legs or inferring emotion/intent.
Spine/collar positions are geometric approximations; caption accuracy on camera poses
still needs evaluation. MotionGPT can misdescribe actions and does not generate poses
incrementally. Retiming changes motion speed. This baseline has no planted-foot IK,
hand/finger motion, arbitrary GLB rig support or conversational turn-end detector.

See [architecture](docs/architecture.md), [model backends](docs/backends.md), and
[Godot transport](docs/godot.md) for configuration and implementation details.
