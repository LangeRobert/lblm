extends Node3D
## Retarget canonical motion onto a Mixamo skinned humanoid, retaining bind proportions.

const BONE_MAP: Dictionary = {
	"Hips": "pelvis", "Spine": "spine1", "Spine1": "spine2", "Spine2": "spine3",
	"Neck": "neck", "Head": "head",
	"LeftUpLeg": "left_hip", "LeftLeg": "left_knee", "LeftFoot": "left_ankle", "LeftToeBase": "left_foot",
	"RightUpLeg": "right_hip", "RightLeg": "right_knee", "RightFoot": "right_ankle", "RightToeBase": "right_foot",
	"LeftShoulder": "left_collar", "LeftArm": "left_shoulder", "LeftForeArm": "left_elbow", "LeftHand": "left_wrist",
	"RightShoulder": "right_collar", "RightArm": "right_shoulder", "RightForeArm": "right_elbow", "RightHand": "right_wrist",
}
const FACING: Basis = Basis(Vector3.UP, PI)
var body: Skeleton3D
var driver: Skeleton3D
var bind_global: Array[Transform3D] = []
var source_bones: PackedInt32Array = []


func configure(source: Skeleton3D, asset_path: String) -> Error:
	## Load FBX or glTF directly and recover skin binds without discarding end bones.
	driver = source
	if not FileAccess.file_exists(asset_path):
		return ERR_FILE_NOT_FOUND
	var extension: String = asset_path.get_extension().to_lower()
	if extension not in ["fbx", "glb", "gltf"]:
		return ERR_FILE_UNRECOGNIZED
	var document: GLTFDocument = FBXDocument.new() if extension == "fbx" else GLTFDocument.new()
	var state: GLTFState = FBXState.new() if extension == "fbx" else GLTFState.new()
	var result: Error = document.append_from_file(asset_path, state)
	if result != OK:
		return result
	var model: Node3D = document.generate_scene(state)
	if model == null:
		return ERR_INVALID_DATA
	add_child(model)
	model.basis = FACING
	var skin: Skin
	var material := StandardMaterial3D.new()
	material.albedo_color = Color.WHITE
	material.roughness = 0.85
	for node: Node in model.find_children("*", "", true, false):
		if node is AnimationPlayer:
			node.stop()
			# Embedded animations must never overwrite the streamed poses.
			node.active = false
		elif node is Skeleton3D:
			body = node
		elif node is MeshInstance3D:
			node.material_override = material
			if node.skin != null:
				skin = node.skin
	if body == null or skin == null or skin.get_bind_count() == 0:
		return ERR_INVALID_DATA
	bind_global.resize(body.get_bone_count())
	for i: int in range(body.get_bone_count()):
		bind_global[i] = body.get_bone_global_rest(i)
	for i: int in range(skin.get_bind_count()):
		var bone: int = skin.get_bind_bone(i)
		if not skin.get_bind_name(i).is_empty():
			bone = body.find_bone(skin.get_bind_name(i))
		if bone < 0 or bone >= body.get_bone_count():
			return ERR_INVALID_DATA
		bind_global[bone] = skin.get_bind_pose(i).affine_inverse()
	for i: int in range(body.get_bone_count()):
		var parent: int = body.get_bone_parent(i)
		var rest: Transform3D = bind_global[i]
		if parent >= 0:
			rest = bind_global[parent].affine_inverse() * rest
		body.set_bone_rest(i, rest)
		var bone_name: String = body.get_bone_name(i)
		bone_name = bone_name.get_slice(":", bone_name.get_slice_count(":") - 1)
		bone_name = bone_name.get_slice("_", bone_name.get_slice_count("_") - 1)
		source_bones.append(driver.find_bone(str(BONE_MAP.get(bone_name, ""))))
	for i: int in range(driver.get_bone_count()):
		if not source_bones.has(i):
			return ERR_INVALID_DATA
	body.reset_bone_poses()
	apply_pose()
	return OK


func apply_pose() -> void:
	## Transfer global rotation deltas across bind axes, preserving mesh proportions and fingers.
	if body == null:
		return
	var rotations: Array[Basis] = []
	rotations.resize(body.get_bone_count())
	for i: int in range(body.get_bone_count()):
		var parent: int = body.get_bone_parent(i)
		var parent_rotation: Basis = rotations[parent] if parent >= 0 else Basis.IDENTITY
		var rest: Transform3D = body.get_bone_rest(i)
		var desired: Basis = parent_rotation * rest.basis.orthonormalized()
		if source_bones[i] >= 0:
			var motion: Basis = driver.get_bone_global_pose(source_bones[i]).basis.orthonormalized()
			desired = FACING.inverse() * motion * FACING * bind_global[i].basis.orthonormalized()
		rotations[i] = desired
		body.set_bone_pose_rotation(i, (parent_rotation.inverse() * desired).get_rotation_quaternion())
		if parent < 0:
			body.set_bone_pose_position(i, rest.origin + FACING.inverse() * driver.get_bone_pose_position(0))
