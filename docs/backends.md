# Model and capture backends

All implementations retain the Pydantic contracts and asynchronous lifecycle.
Inference and capture run off the event loop. Model assets are automatically
resolved through `huggingface-hub`; no manual checkpoint paths are required.

## Install and cache

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync --all-extras --group dev
```

Use Metal on Apple silicon. The base dependencies include Hugging Face and
NumPy and SciPy; select `--extra camera`, `--extra pose`, `--extra llm`, or `--extra motiongpt` to install
individual backends instead. Python 3.13 was used for verification.

Downloads use pinned revisions and the ignored `models/huggingface` directory:

| Asset | Repository | Purpose |
| --- | --- | --- |
| `pose_landmarker_lite.task` | `imman12431/tennis-backhand-detector` (Space) | Pinned mirror, SHA-256 verified against Google; approximately 5.8 MB |
| `gemma-4-E2B-it-Q4_0.gguf` | `ggml-org/gemma-4-E2B-it-GGUF` | Gemma text inference, approximately 2.84 GB |
| `motiongpt_s3_h3d.tar` | `OpenMotionLab/MotionGPT-base` | MotionGPT-base language and VQ weights, approximately 1.33 GB |
| FLAN-T5-base weights and tokenizer | `google/flan-t5-base` | Upstream MotionGPT backbone initialization |
| Selected Python modules, `mean.npy`, `std.npy` | `OpenMotionLab/MotionGPT` (Space) | Upstream architecture and HumanML3D feature conversion |

Projectors, audio models, renderer assets, training datasets and evaluator weights
are not required. MotionGPT loads only its language backbone and motion VQ model;
the upstream dataset package initializers are bypassed to avoid training imports.
The pinned Python source is executed locally as part of the requested backend.
Checkpoint loading uses `torch.load(weights_only=True)`; NumPy assets disable pickle.

Use `HubFile.local_path` to override a checkpoint. An absent override falls back
to the Hub cache/download. Set `HubFile.local_files_only=True` and
`MotionGPTConfig.local_files_only=True` for offline use. `cache_dir=None` selects
Hugging Face's standard cache. Authentication, if required by an alternative
repository, uses the existing Hugging Face login; tokens are not stored in configs.

## OpenCV camera

The requested implementation is `src/m_0_camera/open_cv.py`.

```python
from contextlib import aclosing
from src.m_0_camera.open_cv import OpenCVCamera, OpenCVConfig

camera = OpenCVCamera(OpenCVConfig(source=0, width=640, height=480, mirrored=False))
await camera.open()
try:
    async with aclosing(camera.frames()) as frames:
        async for frame in frames:
            # Pass the frame to a PoseEstimator.
            print(frame.frame_id, frame.width, frame.height)
            break
finally:
    await camera.close()
```

These snippets run inside an async function. A `pathlib.Path` source decodes a
local video; an integer selects a device. Images are packed BGR uint8. Live-device
read failures raise `OSError`; file EOF ends the iterator. Capture dimensions/FPS
are preferences, and each output reports the actual dimensions. Frame IDs remain
monotonic across reset/reopen. macOS grants camera access through its normal UI.

Capture is pull-based with no Python frame queue. The requested one-frame driver
buffer is best effort. The application continuously drains camera frames into a
one-slot mailbox and drops stale frames if pose inference falls behind.

## Gemma 4 E2B through llama-cpp-python

```python
from contextlib import aclosing
from src.m_4_motion_to_language.contract import MotionDescription
from src.m_5_llm.contract import DialogueRequest
from src.m_5_llm.llama_cpp import LlamaCppLanguageModel

llm = LlamaCppLanguageModel()
await llm.open()  # Downloads the pinned GGUF if it is missing.
try:
    request = DialogueRequest(
        response_id="turn-1",
        system_prompt="Respond warmly and briefly through body language.",
        observation=MotionDescription(segment_id="gesture-1", text="The person waves."),
        user_prompt="Greet the person.",
        max_new_tokens=128,
    )
    async with aclosing(llm.respond(request)) as response:
        async for chunk in response:
            print(chunk.text_delta, end="", flush=True)
            if chunk.motion_prompt:
                print("\nAction:", chunk.motion_prompt)
finally:
    await llm.close()
```

The backend uses the GGUF's embedded Gemma chat template, disables thinking,
and constrains output to JSON with `reply` and `motion_prompt`. The adapter emits
decoded reply deltas, then emits a complete motion description exactly once after
validating the finished response. Truncation or malformed JSON raises `ValueError`
without inventing a motion command. Context overflow propagates from llama.cpp;
callers must bound history. Defaults are a 2048-token context and 128-token batch.

## Shared MotionGPT stages

```python
from src.motion_gpt.runtime import MotionGPTRuntime
from src.m_4_motion_to_language.motion_gpt import MotionGPTMotionToLanguage
from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion

runtime = MotionGPTRuntime()
captioner = MotionGPTMotionToLanguage(runtime)
generator = MotionGPTLanguageToMotion(runtime, chunk_frames=10)
try:
    await captioner.open()
    await generator.open()  # Reuses the already-loaded checkpoint.
    # caption = await captioner.describe(segment)
    # async for chunk in generator.generate(request): ...
finally:
    await generator.close()
    await captioner.close()
```

Use the **same runtime instance** for both adapters. Reference counting unloads
the model only after the last adapter closes. Native calls are serialized. CPU
and MPS are supported; `device="auto"` selects MPS when available.

Stage 4 requires complete, positive-confidence observations in the exact
HumanML3D 22-joint order exported by `src.motion_gpt.geometry.JOINT_NAMES`, with
the corresponding `PARENTS`. Units and axes must be meters, right-handed,
X-right/Y-up/Z-backward. The adapter checks topology, person identity, timestamps
and finite coordinates, then resamples a 0.2–9.8-second window to 20 Hz. Feature
extraction aligns heading, centers/floors positions, computes the upstream
263-feature representation, applies checkpoint mean/std, and runs the VQ encoder.

The implemented normalizer maps MediaPipe landmarks into this topology, fixes
initial pelvis/heading, and propagates low-confidence landmarks as missing. The
segmenter retains windows with visible shoulders, elbows and wrists. The
`VisibleMotionToLanguage` wrapper uses geometric arm-motion descriptions for
partial windows; only complete windows are passed to MotionGPT. MediaPipe provides pelvis-relative estimated
meters; no calibrated room translation or training-skeleton proportion fitting is
claimed. The caption adapter never silently substitutes zero positions.
VQ downsampling discards at most three trailing feature frames.

Stage 6 validates native motion-token markers, decodes positions, converts axes
and places the initial pelvis at the canonical origin. Requested duration is
limited to 0.2–9.8 seconds, with output rates up to 120 Hz. MotionGPT's length
instruction is approximate: the first real request for 40 frames produced 160.
The adapter linearly retimes the decoded clip to the requested duration, which
can change action speed. It preserves response IDs, chunk sequences, start times
and final-event semantics. Conditioned continuation currently raises an explicit
error. The processor smooths within a response; Godot adds a short pose blend
between responses. Foot-contact IK is not implemented.

**This is full-clip generation followed by chunked delivery.** It is not a model
that generates one new pose per frame. First-frame latency includes the complete
inference. The first post-load MPS check measured approximately 4.52 s for
generation and 0.54 s for captioning. A repeated check with Gemma and MotionGPT
resident together measured 0.83 s for dialogue, 0.95 s for motion generation and
0.28 s for captioning. These are individual calls, not p50/p95 figures or a claim
of meeting the end-to-end conversation target. The complete pipeline has also passed a prerecorded-video smoke run: 2.06 s
from window delivery to queued animation, with approximately 4.15 GB Python RSS.
This excludes total engine/OS accounting and the preceding observation window.

## Cancellation and verification

Consume streams with `contextlib.aclosing`, especially when breaking early.
Cancel/close active iterators before calling reset/close. Python cannot safely kill
a native inference thread: cancellation waits for its current call to finish
before allowing cleanup, so resources cannot be freed underneath native code.
The orchestrator closes producer streams before releasing resources and cancels
obsolete renderer responses. Model responses finish before the newest pending
observation is consumed, avoiding starvation from continuous input.

```bash
uv run --all-extras pytest
uv run --all-extras mypy src
uv run --all-extras ruff check src tests
LBLM_RUN_MODEL_TESTS=1 uv run --all-extras pytest -q -s -m models
```

Ordinary tests use a real OpenCV video fixture and doubles at heavyweight inference
boundaries. The opt-in model test loads Gemma and MotionGPT together, generates
dialogue and motion, and captions the resulting positions. It uses cached weights
or downloads missing assets. No test opens the physical camera.

Full-system 16 GB compliance, sustained latency, camera-derived caption quality,
motion transitions and renderer performance remain separate acceptance checks.

Upstream references: [Google's Gemma llama.cpp guide](https://ai.google.dev/gemma/docs/integrations/llamacpp),
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python),
[MotionGPT](https://github.com/OpenMotionLab/MotionGPT),
[official MotionGPT Space](https://huggingface.co/spaces/OpenMotionLab/MotionGPT).

## MediaPipe compatibility

MediaPipe 0.10.35 is pinned for Apple silicon. Version 1.0.1 fatally aborts in
its macOS Tasks Vision graph initializer; this was reproduced during native
verification and is tracked in [upstream issue 6356](https://github.com/google-ai-edge/mediapipe/issues/6356).
MotionGPT geometry also receives a local, isolated replacement for its quaternion
between-vectors function so perfectly opposing bone directions produce a valid
180-degree rotation instead of NaNs. Downloaded upstream source is not modified.
