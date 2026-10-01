# Godot renderer

The scene targets Godot 4.5+ with the Compatibility renderer. Its default avatar
is the supplied `models/mixamo-t-pose.fbx`, displayed in matte white on black.
There is no floor. The frontal camera and neutral lighting remain unchanged.

Python sends `CANONICAL_RIG` from `src/skeleton.py` and the resolved local avatar
path at startup. A hidden 22-bone Skeleton3D receives streamed motion.
`godot/humanoid.gd` loads `.fbx` through FBXDocument/FBXState, or a Mixamo-rigged
`.glb`/`.gltf` through GLTFDocument/GLTFState, at runtime without editor imports.
Embedded animation playback is disabled so it cannot overwrite streamed poses.

The supplied rig contains 65 bones, with 52 skin binds. Bone names lose their
Mixamo namespace/prefix before mapping to the canonical 22 joints. Global bind
transforms are recovered for skin-bound bones; unbound end bones retain imported
rest transforms. Parent-local bind transforms are then reconstructed. All 22
required source mappings must exist; incompatible rigs fail initialization.

The FBX loader already converts this asset to meters (mesh height ~1.77 m), so no
additional centimeter scaling is applied. A half-turn aligns its +Z-facing bind
pose with canonical -Z and maps its positive-X left arm into canonical left.
Global motion rotations are conjugated across those facing axes before applying
avatar bind orientations. Limb translations and proportions are preserved;
fingers and end bones follow their parents with their bind rotations. Root
motion adds to the avatar's ~1 m bind-pose pelvis in the avatar's own axes.

Configure a different local Mixamo avatar with `renderer.avatar_path` in your
AppConfig JSON. The default FBX remains in the ignored models directory; it must
be present locally. The historical Mannequiny asset and attribution remain in
[assets/README.md](../godot/assets/README.md), but are no longer loaded by default.

## Transport

Python opens a temporary **127.0.0.1-only** TCP listener and launches its own
Godot process with the port and a random session token. Godot authenticates with
`{"protocol":1,"token":"..."}`. The listener closes after accepting the engine.
No other process is terminated during cleanup.

Messages are newline-delimited JSON, bounded to 256 KiB. Commands are `init`,
`ping`, `submit`, `cancel`, `reset`, `event`, and `camera`. Each receives an acknowledgement before
the next transaction. `init` supplies the rig and local avatar path; `ping` supplies Godot monotonic
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
into a new response over 180 ms. A separate entry envelope starts from the displayed pose when a response
arrives, preserving the blend even when its first chunk is late. Animation frame
timestamps and the response clock supplied by Python remain unchanged. A display-rate exponential
filter (35 ms time constant) softens jumps after missed display frames. The default
Python playback buffer is 100 ms. The Compatibility renderer uses 4× MSAA for
smoother silhouette edges.
Cancellation clears only the matching response and blends from its displayed pose
back into the waiting loop.
Reset clears playback and restores rest. Socket closure exits the owned scene;
Python also terminates the process on cleanup.

## Coordinate and retargeting scope

Positions are meters in a right-handed X-right/Y-up/Z-backward space. The avatar
origin is the Mixamo avatar’s bind-pose pelvis (approximately 1.0 m above world zero). Transforms are **absolute local poses**,
not offsets to add to Godot bone rest transforms. The geometric retargeter fits
parent orientations to observed child directions, preserves rig segment lengths,
and inherits twist for terminal joints. It supports the canonical 22-joint rig
with identity bind rotations; other rigs are rejected explicitly.

This renderer has no foot-contact IK or general-purpose adapter for non-Mixamo rigs. Full-body observations and generated 22-joint poses omit fingers and
facial expression. Single-view camera depth and motion-caption quality need
separate evaluation before treating the avatar as motion capture.

References: [Skeleton3D](https://docs.godotengine.org/en/stable/classes/class_skeleton3d.html),
[GLTFDocument](https://docs.godotengine.org/en/stable/classes/class_gltfdocument.html),
[StreamPeerTCP](https://docs.godotengine.org/en/stable/classes/class_streampeertcp.html).

## Conversation window

`conversation_window.gd` creates a second native, resizable portrait window next
to the visualizer within the current display's usable area. Its logical canvas
is 1080×1920 and preserves the reference's proportions when resized. On a black
background, the camera occupies `(54, 47, 422, 517)`, the right-aligned stacked
80 px title starts at `(54, 609)`, and the scroll view occupies the right half
from x=540. Poppins Regular and its OFL license are bundled; startup needs no
network request or editor import for the font. The visualizer has no text overlay.

`event` carries typed conversation notifications through the authenticated
connection. Observations appear in white and streamed replies in #666666 gray,
using 32 px wrapping plain-text labels without prefixes. Reply tokens update one
label per response; history is bounded to 100 messages and each label to 16,384
characters. Readers who scroll up retain their position. The white camera block
is the empty state before input arrives. Live images fill it with a centered crop.

`camera` carries base64 JPEG pixels over that same temporary loopback connection.
Python sends previews at most 10 Hz, downsamples the longest edge to 480 pixels,
and encodes at quality 70. Godot reuses its texture, validates decoded dimensions,
and retains only the latest frame. Input pixels are neither saved nor uploaded.
Headless sessions skip encoding; both windows remain hidden. There are no debug
landmark overlays or separate preview cameras.

## Mixamo compatibility

Godot 4.5 supports FBX at runtime through `FBXDocument` and `FBXState`; conversion
to GLB is optional. Its ufbx importer supports binary and ASCII FBX from version
3000 onward, including 6.1 and 7.4. Prefer Mixamo's FBX Binary 7.4 character export
with its skin and a neutral T-pose. The Unity variant is still FBX, with a preset
that requires checking axes and units. COLLADA `.dae` is an editor-imported scene
format and would require an import step or conversion for this runtime loader.

LBLM now loads the supplied FBX directly and maps this Mixamo rig onto its
canonical motion. The native rig regression verifies all 22 mappings, meter-scale
pelvis height, arm raising, preserved limb lengths, stable finger bind rotations,
root-motion axes and reset. Other Mixamo exports can be selected by configuration,
but require checking their actual skeleton and proportions; these tests establish
compatibility with this supplied file.

References: [runtime FBX loading](https://docs.godotengine.org/en/4.5/tutorials/io/runtime_file_loading_and_saving.html),
[ufbx format support](https://docs.godotengine.org/en/4.5/classes/class_editorsceneformatimporterufbx.html),
[scene import formats](https://docs.godotengine.org/en/4.5/classes/class_resourceimporterscene.html).

## Continuous waiting motion

`idle_motion.gd` adds a procedural eight-second loop on the canonical rig. Arms
relax from the T-pose, the torso breathes and subtly leans, and the neck/head make
small listening movements. Its phase runs continuously. Completion of a final
chunk and matching cancellation blend from the displayed pose into idle over
600 ms, rather than holding a raised limb indefinitely. A paused non-final stream
retains its response pose with a small breathing layer while awaiting more data.

New response playback retains its existing entry blend and priority; only a
0.15-degree breathing offset is layered onto the chest/head during gestures.
The idle layer does not move the root, regenerate motion through an AI model, or
add playback buffering. Each idle pose is evaluated against a captured departure
pose, avoiding cumulative rotation drift. Explicit reset clears that anchor.
This is a procedural waiting animation, not a new MotionGPT-generated clip.
