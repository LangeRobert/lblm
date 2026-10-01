extends Node3D
## Bounded local animation transport and a skinned humanoid display.

const MAX_BYTES: int = 262144
const MAX_FRAMES: int = 240
var peer := StreamPeerTCP.new()
var buffer := PackedByteArray()
var authenticated: bool = false
var token: String = ""
var skeleton := Skeleton3D.new()
var humanoid := preload("res://humanoid.gd").new()
var idle_motion := preload("res://idle_motion.gd").new()
var rig: Dictionary = {}
var pending: Array[Dictionary] = []
var current: Dictionary = {}
var active_response: String = ""
var expected_sequence: int = 0
var last_frame_time: float = -1.0
var response_final: bool = false
var response_base: float = 0.0
var dialogue := preload("res://conversation_window.gd").new()
var entry_pose: Array[Dictionary] = []
var entry_started: float = -1.0


func _ready() -> void:
	## Connect to the owning Python process and construct the stage.
	var args: PackedStringArray = OS.get_cmdline_user_args()
	if args.size() != 3 or args[0] != "--lblm":
		push_error("Launch this scene with python main.py")
		get_tree().quit(1)
		return
	token = args[2]
	peer.connect_to_host("127.0.0.1", int(args[1]))
	add_child(skeleton)
	add_child(humanoid)
	var camera := Camera3D.new()
	add_child(camera)
	camera.position = Vector3(0, 0.9, -3.6)
	camera.fov = 40.0
	camera.look_at(Vector3(0, 0.9, 0))
	camera.current = true
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45, -30, 0)
	light.light_energy = 0.8
	add_child(light)
	var fill := OmniLight3D.new()
	fill.position = Vector3(-2, 2, -2)
	fill.light_color = Color.WHITE
	fill.light_energy = 0.6
	add_child(fill)
	var world := WorldEnvironment.new()
	world.environment = Environment.new()
	world.environment.background_mode = Environment.BG_COLOR
	world.environment.background_color = Color.BLACK
	world.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	world.environment.ambient_light_color = Color.WHITE
	world.environment.ambient_light_energy = 0.5
	add_child(world)
	add_child(dialogue)
	dialogue.configure(get_window())


func vector(value: Dictionary) -> Vector3:
	## Decode an xyz protocol object.
	return Vector3(float(value.x), float(value.y), float(value.z))


func quaternion(value: Dictionary) -> Quaternion:
	## Decode and normalize a protocol rotation.
	return Quaternion(float(value.x), float(value.y), float(value.z), float(value.w)).normalized()


func send_reply(value: Dictionary) -> void:
	## Send a newline-delimited acknowledgement.
	peer.put_data((JSON.stringify(value) + "\n").to_utf8_buffer())


func _process(delta: float) -> void:
	## Poll transport, interpolate responses, and blend continuous waiting motion.
	peer.poll()
	if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		if not authenticated:
			send_reply({"token": token, "protocol": 1})
			authenticated = true
		var available: int = peer.get_available_bytes()
		if buffer.size() + available > MAX_BYTES:
			get_tree().quit(2)
			return
		if available > 0:
			var data: Array = peer.get_data(available)
			if data[0] != OK:
				get_tree().quit(2)
				return
			buffer.append_array(data[1])
		var newline: int = buffer.find(10)
		while newline >= 0:
			var parsed: Variant = JSON.parse_string(buffer.slice(0, newline).get_string_from_utf8())
			buffer = buffer.slice(newline + 1)
			if not parsed is Dictionary:
				get_tree().quit(2)
				return
			handle_command(parsed)
			newline = buffer.find(10)
	elif authenticated:
		get_tree().quit()
		return
	var now: float = Time.get_ticks_usec() / 1000000.0
	while not pending.is_empty() and float(pending[0].time) <= now:
		current = pending.pop_front()
	if not current.is_empty():
		var following: Dictionary = pending[0] if not pending.is_empty() else current
		var duration: float = float(following.time) - float(current.time)
		var weight: float = clampf((now - float(current.time)) / duration, 0, 1) if duration > 0 else 0
		var smoothing: float = 1.0 - exp(-delta / 0.035)
		var entry_weight: float = smoothstep(0.0, 0.18, now - entry_started)
		for i: int in range(skeleton.get_bone_count()):
			var a: Dictionary = current.bones[i]
			var b: Dictionary = following.bones[i]
			var position_target: Vector3 = vector(a.translation).lerp(vector(b.translation), weight)
			var rotation_target: Quaternion = quaternion(a.rotation).slerp(quaternion(b.rotation), weight)
			if entry_weight < 1.0 and entry_pose.size() == skeleton.get_bone_count():
				position_target = vector(entry_pose[i].translation).lerp(position_target, entry_weight)
				rotation_target = quaternion(entry_pose[i].rotation).slerp(rotation_target, entry_weight)
			skeleton.set_bone_pose_position(i, skeleton.get_bone_pose_position(i).lerp(position_target, smoothing))
			skeleton.set_bone_pose_rotation(i, skeleton.get_bone_pose_rotation(i).slerp(rotation_target, smoothing))
	if skeleton.get_bone_count() > 0:
		var playing: bool = not pending.is_empty() or (not active_response.is_empty() and not response_final)
		idle_motion.apply_pose(skeleton, delta, playing)
		humanoid.apply_pose()


func handle_command(message: Dictionary) -> void:
	## Validate and acknowledge one bounded protocol operation.
	match str(message.get("command", "")):
		"init":
			if not rig.is_empty():
				send_reply({"ok": false, "error": "already initialized"})
				return
			rig = message.rig
			var joints: Array = rig.skeleton.joints
			for i: int in range(joints.size()):
				skeleton.add_bone(joints[i].name)
				if joints[i].parent != null:
					skeleton.set_bone_parent(i, skeleton.find_bone(joints[i].parent))
				var translation: Vector3 = vector(rig.rest_transforms[i].translation)
				skeleton.set_bone_rest(i, Transform3D(Basis.IDENTITY, translation))
				skeleton.set_bone_pose_position(i, translation)
			var loaded: Error = humanoid.configure(skeleton, str(message.get("avatar_path", "")))
			if loaded != OK:
				send_reply({"ok": false, "error": "humanoid asset failed to load: " + error_string(loaded)})
				return
		"camera":
			dialogue.receive_camera(str(message.get("jpeg", "")))
		"event":
			dialogue.receive_event(message.event)
		"ping":
			send_reply({"ok": true, "engine_time_s": Time.get_ticks_usec() / 1000000.0})
			return
		"submit":
			var request: Dictionary = message.request
			var chunk: Dictionary = request.chunk
			var frames: Array = chunk.frames
			if pending.size() + frames.size() > MAX_FRAMES:
				send_reply({"ok": false, "error": "full"})
				return
			if str(chunk.rig_id) != str(rig.get("rig_id", "")):
				send_reply({"ok": false, "error": "rig mismatch"})
				return
			var new_response: bool = str(chunk.response_id) != active_response
			var expected: int = 0 if new_response else expected_sequence
			var previous_time: float = -1.0 if new_response else last_frame_time
			if int(chunk.sequence) != expected or (not new_response and response_final):
				send_reply({"ok": false, "error": "sequence mismatch"})
				return
			if not new_response and not is_equal_approx(float(request.play_at_s), response_base):
				send_reply({"ok": false, "error": "response clock changed"})
				return
			for frame: Dictionary in frames:
				if frame.bones.size() != skeleton.get_bone_count() or float(frame.time_s) <= previous_time:
					send_reply({"ok": false, "error": "invalid frame"})
					return
				previous_time = float(frame.time_s)
			if new_response:
				pending.clear()
				var blend_bones: Array[Dictionary] = []
				for i: int in range(skeleton.get_bone_count()):
					var p: Vector3 = skeleton.get_bone_pose_position(i)
					var q: Quaternion = skeleton.get_bone_pose_rotation(i)
					blend_bones.append({"translation": {"x": p.x, "y": p.y, "z": p.z}, "rotation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w}})
				var now: float = Time.get_ticks_usec() / 1000000.0
				var first_time: float = float(frames[0].time_s) if not frames.is_empty() else 0.0
				entry_pose = blend_bones
				entry_started = maxf(now, float(request.play_at_s) + first_time - 0.18)
				current = {"time": float(request.play_at_s) + first_time - 0.18, "bones": blend_bones}
				active_response = str(chunk.response_id)
				response_base = float(request.play_at_s)
			for frame: Dictionary in frames:
				pending.append({"time": response_base + float(frame.time_s), "bones": frame.bones})
			last_frame_time = previous_time
			expected_sequence = expected + 1
			response_final = bool(chunk.is_final)
		"cancel", "reset":
			if message.command == "reset" or str(message.response_id) == active_response:
				pending.clear()
				current.clear()
				entry_pose.clear()
				active_response = ""
				expected_sequence = 0
				response_final = false
				if message.command == "reset":
					skeleton.reset_bone_poses()
					idle_motion.reset()
					humanoid.apply_pose()
		_:
			send_reply({"ok": false, "error": "unknown command"})
			return
	send_reply({"ok": true})
