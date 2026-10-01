extends SceneTree
## Motion-layer regression without interface or model assertions.

func _initialize() -> void:
	## Verify continuous looping, stable blending, and response pose preservation.
	await process_frame
	var driver := Skeleton3D.new()
	root.add_child(driver)
	var names: Array[String] = ["pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
	for name: String in names:
		driver.add_bone(name)
	var idle: Variant = load("res://idle_motion.gd").new()
	idle.apply_pose(driver, 0.0, false)
	assert(driver.get_bone_pose_rotation(6).is_equal_approx(Quaternion.IDENTITY), "Idle entry must start from the displayed pose")
	idle.apply_pose(driver, 1.0, false)
	var first: Quaternion = driver.get_bone_pose_rotation(3)
	assert(driver.get_bone_pose_rotation(6).get_angle() > 1.0, "Relax arms below the T-pose")
	idle.apply_pose(driver, 1.0, false)
	assert(not first.is_equal_approx(driver.get_bone_pose_rotation(3)), "Waiting pose must keep moving")
	var loop_pose: Quaternion = driver.get_bone_pose_rotation(3)
	idle.apply_pose(driver, 8.0, false)
	assert(loop_pose.is_equal_approx(driver.get_bone_pose_rotation(3)), "Idle loop closes without a seam")
	idle.apply_pose(driver, 0.0, false)
	assert(loop_pose.is_equal_approx(driver.get_bone_pose_rotation(3)), "Repeated evaluation must not accumulate rotations")
	var response: Quaternion = Quaternion(Vector3(0, 0, 1), -0.4)
	driver.set_bone_pose_rotation(6, response)
	idle.apply_pose(driver, 0.1, true)
	assert(driver.get_bone_pose_rotation(6).is_equal_approx(response), "Response gestures retain their shoulder pose")
	var before_exit: Quaternion = driver.get_bone_pose_rotation(6)
	idle.apply_pose(driver, 0.0, false)
	assert(before_exit.is_equal_approx(driver.get_bone_pose_rotation(6)), "Completion or cancellation must not snap to idle")
	idle.apply_pose(driver, 0.6, false)
	assert(driver.get_bone_pose_rotation(6).get_angle() > 1.0, "Return to relaxed waiting posture")
	assert(driver.get_bone_pose_position(0).is_equal_approx(Vector3.ZERO), "Idle must not drift or slide the root")
	driver.set_bone_pose_position(0, Vector3(0.3, 0, 0))
	idle.apply_pose(driver, 0.1, true)
	idle.apply_pose(driver, 0.6, false)
	driver.reset_bone_poses()
	idle.reset()
	idle.apply_pose(driver, 0.6, false)
	assert(driver.get_bone_pose_position(0).is_equal_approx(Vector3.ZERO), "Explicit reset must discard the old root anchor")
	print("IDLE_MOTION_OK")
	quit()
