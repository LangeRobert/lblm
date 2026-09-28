# Godot renderer

The bundled scene targets Godot 4.5+ with the compatibility renderer. It displays
[GDQuest’s Mannequiny](https://github.com/gdquest-demos/godot-3d-mannequin), a
skinned neutral humanoid, in matte white against a pure black background. There
is no floor. The camera faces the character directly at torso height; neutral
lighting gives the white surfaces enough shading to remain readable.

Python sends `CANONICAL_RIG` from `src/skeleton.py` at startup. A hidden 22-bone
Skeleton3D receives the motion. `godot/humanoid.gd` maps its global rotation
deltas onto the avatar's bind axes while preserving the mesh's proportions.
The GLB is loaded through GLTFDocument at runtime, so no editor import step or
network download is required to launch. Embedded animation playback is disabled.

The source asset was exported in a posed state; the adapter reconstructs its
true bind transforms from the skin's inverse bind matrices before applying
motion. A half-turn aligns the model's forward direction with canonical -Z.
Fingers retain their bind pose and follow the wrists. Two avatar spine bones
follow canonical spine1 and spine3. Rest-pose limb angles and body proportions
remain those of the avatar, rather than stretching the mesh to landmark lengths.

Asset attribution and pinned source revision: [assets/README.md](../godot/assets/README.md).

## Transport

Python opens a temporary **127.0.0.1-only** TCP listener and launches its own
Godot process with the port and a random session token. Godot authenticates with
`{"protocol":1,"token":"..."}`. The listener closes after accepting the engine.
No other process is terminated during cleanup.

Messages are newline-delimited JSON, bounded to 256 KiB. Commands are `init`,
`ping`, `submit`, `cancel`, and `reset`. Each receives an acknowledgement before
the next transaction. `init` supplies the rig; `ping` supplies Godot monotonic
time. Python estimates the clock offset from the round-trip midpoint and
translates the common response playback origin once per submission.

A submit contains the serialized `PlaybackRequest`. Python limits individual
chunks to 120 frames; Godot keeps at most 240 pending frames. A full queue returns
`{"ok":false,"error":"full"}`; Python retries with a timeout while allowing
cancellation commands between retries. Acknowledgement means queued, not played.
Godot validates the rig, sequence, unchanged response clock, joint count and
increasing frame timestamps. New responses start at sequence zero. Exactly one
final chunk ends a response.

Godot interpolates translations and slerps unit rotations on each display frame.
Expired frames are consumed without slowing playback. The current pose blends
into a new response over 150 ms; the default Python playback buffer is 100 ms.
Cancellation clears only the matching response and freezes its current pose.
Reset clears playback and restores rest. Socket closure exits the owned scene;
Python also terminates the process on cleanup.

## Coordinate and retargeting scope

Positions are meters in a right-handed X-right/Y-up/Z-backward space. The avatar
origin is the avatar’s bind-pose pelvis (approximately 0.95 m above world zero). Transforms are **absolute local poses**,
not offsets to add to Godot bone rest transforms. The geometric retargeter fits
parent orientations to observed child directions, preserves rig segment lengths,
and inherits twist for terminal joints. It supports the canonical 22-joint rig
with identity bind rotations; other rigs are rejected explicitly.

This renderer has no foot-contact IK or general-purpose avatar rig adapter. Full-body observations and generated 22-joint poses omit fingers and
facial expression. Single-view camera depth and motion-caption quality need
separate evaluation before treating the avatar as motion capture.

References: [Skeleton3D](https://docs.godotengine.org/en/stable/classes/class_skeleton3d.html),
[GLTFDocument](https://docs.godotengine.org/en/stable/classes/class_gltfdocument.html),
[StreamPeerTCP](https://docs.godotengine.org/en/stable/classes/class_streampeertcp.html).
