extends RefCounted
## Continuous canonical waiting motion, smoothly layered around streamed responses.

const LOOP_SECONDS: float = 8.0
const RETURN_SECONDS: float = 0.6
var phase_time: float = 0.0
var return_time: float = 0.0
var initialized: bool = false
var was_playing: bool = false
var departure_rotations: Array[Quaternion] = []
var departure_positions: Array[Vector3] = []


func apply_pose(driver: Skeleton3D, delta: float, playback_active: bool) -> void:
	## Breathe during gestures; otherwise blend from the displayed pose into relaxed idle.
	phase_time = fposmod(phase_time + maxf(delta, 0.0), LOOP_SECONDS)
	if not initialized or (was_playing and not playback_active):
		departure_rotations.clear()
		departure_positions.clear()
		for i: int in range(driver.get_bone_count()):
			departure_rotations.append(driver.get_bone_pose_rotation(i))
			departure_positions.append(driver.get_bone_pose_position(i))
		return_time = 0.0
		initialized = true
	was_playing = playback_active
	var phase: float = TAU * phase_time / LOOP_SECONDS
	var breath: float = sin(phase * 2.0)
	var sway: float = sin(phase)
	if playback_active:
		# Main playback supplies a fresh base every frame, so these offsets cannot accumulate.
		for i: int in range(driver.get_bone_count()):
			var name: String = driver.get_bone_name(i)
			if name == "spine3" or name == "head":
				var axis: Vector3 = Vector3.RIGHT if name == "spine3" else Vector3.UP
				var offset := Quaternion(axis, deg_to_rad(0.15) * breath)
				driver.set_bone_pose_rotation(i, driver.get_bone_pose_rotation(i) * offset)
		return
	return_time = minf(return_time + maxf(delta, 0.0), RETURN_SECONDS)
	var blend: float = smoothstep(0.0, RETURN_SECONDS, return_time)
	var idle_rotations: Dictionary = {
		"spine1": Quaternion(Vector3.FORWARD, deg_to_rad(0.6) * sway),
		"spine2": Quaternion(Vector3.RIGHT, deg_to_rad(0.5) * breath),
		"spine3": Quaternion(Vector3.RIGHT, deg_to_rad(0.7) * breath),
		"neck": Quaternion(Vector3.UP, deg_to_rad(1.5) * sway),
		"head": Quaternion(Vector3.RIGHT, deg_to_rad(0.7) * sin(phase + 0.4)),
		"left_shoulder": Quaternion(Vector3(0, 0, 1), deg_to_rad(75.0 + 0.5 * breath)),
		"right_shoulder": Quaternion(Vector3(0, 0, 1), deg_to_rad(-75.0 - 0.5 * breath)),
		"left_elbow": Quaternion(Vector3.UP, deg_to_rad(-8.0 + 0.4 * sway)),
		"right_elbow": Quaternion(Vector3.UP, deg_to_rad(8.0 - 0.4 * sway)),
	}
	for i: int in range(driver.get_bone_count()):
		var target: Quaternion = idle_rotations.get(driver.get_bone_name(i), Quaternion.IDENTITY)
		driver.set_bone_pose_rotation(i, departure_rotations[i].slerp(target, blend))
		var position: Vector3 = departure_positions[i] if driver.get_bone_parent(i) < 0 else driver.get_bone_rest(i).origin
		driver.set_bone_pose_position(i, departure_positions[i].lerp(position, blend))


func reset() -> void:
	## Forget cached pose anchors after an explicit reset while keeping the loop's phase.
	initialized = false
	was_playing = false
	return_time = 0.0
	departure_rotations.clear()
	departure_positions.clear()
