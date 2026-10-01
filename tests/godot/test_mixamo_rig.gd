extends SceneTree
## Backend rig regression: real FBX bind pose and streamed limb rotations.

func _initialize() -> void:
	## Verify the supplied Mixamo rig without rendering an interface or loading models.
	await process_frame
	var driver := Skeleton3D.new()
	root.add_child(driver)
	var names: Array[String] = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
	var parents: Array[int] = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
	for i: int in range(names.size()):
		driver.add_bone(names[i])
		if parents[i] >= 0:
			driver.set_bone_parent(i, parents[i])
	var adapter: Variant = load("res://humanoid.gd").new()
	root.add_child(adapter)
	var result: Error = adapter.configure(driver, ProjectSettings.globalize_path("res://../models/mixamo-t-pose.fbx"))
	assert(result == OK, "Mixamo FBX must load with partial skin bindings")
	assert(adapter.body.get_bone_count() == 65, "Keep finger and end bones")
	assert(adapter.source_bones.count(-1) == 43, "Map all 22 canonical driver joints")
	var hips: int = adapter.body.find_bone("mixamorig1_Hips")
	var wrist: int = adapter.body.find_bone("mixamorig1_LeftHand")
	var elbow: int = adapter.body.find_bone("mixamorig1_LeftForeArm")
	var finger: int = adapter.body.find_bone("mixamorig1_LeftHandIndex1")
	var root_position: Vector3 = adapter.body.get_bone_pose_position(hips)
	assert(root_position.y > 0.9 and root_position.y < 1.1, "FBX units already meters")
	var initial_wrist: Vector3 = adapter.body.get_bone_global_pose(wrist).origin
	var segment_length: float = adapter.body.get_bone_rest(elbow).origin.length()
	var finger_pose: Quaternion = adapter.body.get_bone_pose_rotation(finger)
	driver.set_bone_pose_rotation(names.find("left_shoulder"), Quaternion(Vector3(0, 0, 1), -PI / 2))
	adapter.apply_pose()
	assert(adapter.body.get_bone_global_pose(wrist).origin.y > initial_wrist.y + 0.35, "Left arm must raise in avatar space")
	assert(is_equal_approx(segment_length, adapter.body.get_bone_rest(elbow).origin.length()), "Preserve limb proportions")
	assert(finger_pose.is_equal_approx(adapter.body.get_bone_pose_rotation(finger)), "Fingers retain bind rotations")
	driver.set_bone_pose_position(0, Vector3(0.2, 0.1, -0.3))
	adapter.apply_pose()
	assert(adapter.body.get_bone_pose_position(hips).is_equal_approx(root_position + adapter.FACING.inverse() * Vector3(0.2, 0.1, -0.3)), "Root translation uses avatar axes")
	driver.reset_bone_poses()
	adapter.apply_pose()
	assert(adapter.body.get_bone_global_pose(wrist).origin.is_equal_approx(initial_wrist), "Reset restores the bind pose")
	print("MIXAMO_RIG_OK")
	quit()
