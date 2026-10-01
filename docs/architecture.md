# LBLM architecture

All ten stages are implemented in Python behind the existing ABC/Pydantic contracts.
`src/app.py` composes the defaults; `src/pipeline.py` manages their lifetime and
routes payloads. Godot is the selected renderer.

```text
OpenCV → MediaPipe Pose Lite → canonical 22-joint mapping → bounded motion window
                                                               ↓
                                                     MotionGPT caption
                                                               ↓
                                                observation + prompt + history
                                                               ↓
                                              Gemma 4 E2B / llama.cpp / Metal
                                                               ↓
                                               text deltas + complete action
                                                               ↓
Shared MotionGPT → causal smoothing → local bone transforms → Godot Skeleton3D
                                             ↑
                                      renderer.get_rig()
```

## Models and ownership

MediaPipe estimates 33 hip-relative world landmarks. A confidence-aware geometric
mapping synthesizes the HumanML3D spine/collar joints, calibrates initial heading
and origin, and retains later heading changes. It cannot reconstruct calibrated
room translation from hip-relative observations. Low-confidence joints remain
missing. If hips are unavailable, visible shoulders establish the calibration
frame. Windows can contain partial poses when both shoulders, elbows and wrists
are visible. The visible-motion captioner routes complete windows to MotionGPT;
partial windows receive conservative geometric arm-motion descriptions, with
hidden lower-body motion explicitly unknown. No missing legs are reconstructed.

MotionGPT-base supports captioning and text-to-motion through the **same loaded
runtime**. The model's language backbone and VQ encoder/decoder are serialized
behind a shared lock. Reference-counted ownership prevents one adapter from
unloading another's model. The 263-feature representation, checkpoint statistics,
native axes and motion-token conventions are handled at the inference boundary.
The quaternion conversion includes a stable antiparallel-vector case.

Gemma 4 E2B uses the pinned 2.84 GB Q4_0 text GGUF, its embedded chat template,
Metal layers and a 2048-token context. Thinking is disabled. JSON grammar and
Pydantic validation produce a streamed reply and one complete motion description.
MotionGPT generates the entire clip before delivering chunks; it is not a
frame-by-frame autoregressive motion service. See [backends](backends.md).

## Scheduling and lifecycle

Three workers separate capture, pose processing and conversation. Camera and
motion-window mailboxes each hold one pending item; newer work replaces stale
work. The camera worker is paced at up to 30 Hz. The default segmenter emits
nonoverlapping 1.5-second windows and splits on missing tracking, subject changes
or long gaps. Brief arm-landmark dropouts are skipped for up to 0.3 seconds
instead of repeatedly erasing the entire buffer. It enforces frame-count and duration limits; it does not infer
conversational turn endings.

An active model response finishes before the newest pending window is consumed.
Identical captions are suppressed for five seconds and history is capped at
three user/assistant pairs. A new response cancels preceding avatar playback.
Each response gets a UUID and a monotonic playback origin, assigned only once
its first motion chunk is ready. A 100 ms playback buffer absorbs transport jitter.

Every owned stage is closed in reverse order by `AsyncExitStack`, including
partial-startup failure. `TaskGroup` cancellation stops the other workers on a
failure. Producers are explicitly closed through `aclosing` before their native
resources are released. A native operation cannot safely be killed in mid-call:
cancellation waits for it to finish and then discards its result.

## Geometry and rendering

Canonical positions use meters, a right-handed frame and X-right/Y-up/Z-backward.
Joint order and hierarchy are explicit. Generated positions include root motion;
frame time is relative to the response. The processor validates sequences and
smooths across chunk boundaries without buffering future frames.

The geometric retargeter solves bone swing and branching-joint orientation while
preserving fixed rig lengths. It emits absolute parent-local translations and
unit quaternions. It supports the included canonical identity-rest rig; arbitrary
bind orientations or topologies are rejected. Leaf twist is inherited.

Godot owns the supplied Mixamo FBX humanoid, lighting and playback. Python starts an
owned process, authenticates its loopback TCP connection, sends the canonical rig and local avatar path, and
measures the monotonic clock offset. Godot bounds its queue, validates chunk order,
interpolates translations/rotations at display rate and blends into new responses.
[Transport and renderer details](godot.md).

## Boundaries and limitations

Payloads are immutable Pydantic models with strict field types, finite numbers
and forbidden extra fields. Concrete stages enforce their relational invariants.
Stream contracts use closable async generators. Camera bytes feed pose estimation and a bounded local JPEG preview in the
companion window; model stages after pose estimation receive only landmarks or text. Downloads occur
on first model open and use pinned Hugging Face revisions. No cloud inference or
image persistence is part of the live pipeline.

The supplied Mixamo FBX has a dedicated bind-pose adapter with all 22 canonical
joints mapped and finger/end bones preserved; arbitrary non-Mixamo rigs are not supported. There is no foot-contact IK, finger/facial animation,
robust person reidentification or learned
turn-taking policy. Body-language captions describe model predictions, not reliable
emotion, intent or sign-language translation. A static pose fixture was captioned
as movement during a smoke run, so camera-derived caption quality needs explicit
evaluation.

## Memory and responsiveness

The defaults avoid duplicate motion models, bound queues/history and use a small
quantized dialogue model. A native ten-stage smoke run on an M4 Pro measured
2.06 seconds from a completed observation window to queued animation and about
4.15 GB peak Python RSS. The window itself adds up to 1.5 seconds; Godot, shared
GPU allocations and macOS require separate total-memory accounting. A synthetic
wave demo measured 3.15 seconds for the same response path.

These results prove functional execution, not the original target of a convincing
real-time conversation within 16 GB. Sustained p50/p95 latency, camera-caption
quality, playback quality and total unified-memory pressure remain acceptance
work tracked in the project's Notion ToDos. No swap-based capacity assumption is
made.
