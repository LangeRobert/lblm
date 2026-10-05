# LBLM — Large Body Language Model

A local Python 3.13+ camera → body motion → conversation → animated humanoid
application. All ten stages have concrete implementations behind typed Pydantic
contracts. Godot 4 displays the supplied Mixamo mannequin in matte white, viewed from the
front on black. The local avatar file is `models/mixamo-t-pose.fbx`.

## Run

From this directory on Apple silicon:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --all-extras --group dev
uv run --all-extras python main.py
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
# Direct gesture mode: type actions in the terminal; no camera, captioner or dialogue LLM.
uv run --extra motiongpt python main.py --prompt
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
`src/app.py`. CLI overrides include `--camera 0`, `--prompt`, `--seconds 60`,
`--godot`, `--headless`, and `--offline`. The time limit includes model startup. The CLI uses Click; `--help` lists all options.

`python main.py --prompt` starts an interactive gesture session. Enter an action
such as `wave hello` or `raise both arms` at `Gesture>`; it goes directly to
MotionGPT and plays on the avatar before the next prompt. Blank lines are ignored.
If motion generation fails, the application reports the error, cancels any partial
playback and skips that gesture. The session stays open for the next gesture;
camera mode uses the same recovery behavior for generated reactions.
Use `/quit`, `/exit`, terminal EOF (Ctrl-D), or Ctrl-C to close the session.
Only stages 6–9 open: motion generation, processing, retargeting and rendering.
The camera, pose detection, motion classification/captioning and dialogue LLM are
skipped. This mode needs the `motiongpt` extra and Godot, and supports `--offline`,
`--headless`, `--config` and `--seconds` (including time spent waiting for input).
It cannot be combined with `--demo`, `--camera` or `--video`.
`--prompt` is now a mode flag; conversation instructions for camera mode can be
set using `pipeline.user_prompt` in the JSON configuration.

Godot opens two native windows side by side: the avatar visualizer and a portrait
conversation window matching the supplied 1080×1920 design. The companion uses
Poppins on black, a camera preview at upper left, and the stacked
“Large body Language Model v2.0” title beneath it. The scrollable right column
shows observed movement in white and streamed system replies in gray, without
speaker labels or status headers. It retains the latest 100 messages and follows
new text when you are near the bottom. Scroll up to read without being pulled down.
The camera preview uses captured frames at up to 10 Hz, without landmarks or a
second camera connection; a white block remains until the first frame arrives.
`--headless` hides both windows and skips camera-preview encoding. Closing the
conversation window hides it; closing the visualizer stops the session.
There is no debug mode.

## Avatar format

The default avatar is `models/mixamo-t-pose.fbx`, loaded directly by Godot's
FBXDocument without editor imports or conversion. The supplied FBX imports in
meters; its Mixamo bone names map onto all 22 canonical joints. Fingers and end
bones retain their bind pose, including bones without skin bindings.

`renderer.avatar_path` in your JSON configuration can select another local
Mixamo-rigged `.fbx`, `.glb` or `.gltf`. This adapter requires every canonical
joint's corresponding Mixamo bone; arbitrary rigs are rejected. There is no file
picker. The supplied model stays in the ignored `models` directory, so a fresh
checkout needs that file or an explicit configured path. The older Mannequiny
asset remains attributed in `godot/assets`, but is no longer the default rig.

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
| 9 Render | Godot Skeleton3D, interpolation, continuous waiting loop, blended transitions and bounded TCP playback |

The orchestrator uses one-slot latest-frame and latest-segment mailboxes, bounded
history, repeat-caption suppression, shared MotionGPT ownership and cancellation-safe
cleanup. It finishes an active model response before consuming the newest pending
window. Playback for the preceding response is cancelled when a new reply begins. Between
responses the character relaxes its arms and loops subtle breathing and listening
movements, blending back over 600 ms after completion or cancellation.

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
