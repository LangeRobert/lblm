extends Node3D
## Retarget canonical motion onto Mannequiny's skinned, meter-scale bind pose.

const BONE_MAP: Dictionary = {
	"pelvis": "pelvis", "spine_01": "spine1", "spine_02": "spine3",
	"neck_01": "neck", "head": "head",
	"thigh.l": "left_hip", "calf.l": "left_knee", "foot.l": "left_ankle", "ball.l": "left_foot",
	"thigh.r": "right_hip", "calf.r": "right_knee", "foot.r": "right_ankle", "ball.r": "right_foot",
	"clavicle.l": "left_collar", "upperarm.l": "left_shoulder", "lowerarm.l": "left_elbow", "hand.l": "left_wrist",
	"clavicle.r": "right_collar", "upperarm.r": "right_shoulder", "lowerarm.r": "right_elbow", "hand.r": "right_wrist",
}
const FACING: Basis = Basis(Vector3.UP, PI)
var body: Skeleton3D
var driver: Skeleton3D
var bind_global: Array[Transform3D] = []
var source_bones: PackedInt32Array = []


func configure(source: Skeleton3D) -> Error:
	## Load the bundled GLB without editor imports and restore its actual skin bind pose.
	driver = source
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	var result: Error = document.append_from_file("res://assets/mannequiny.glb", state)
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
	if body == null or skin == null or skin.get_bind_count() != body.get_bone_count():
		return ERR_INVALID_DATA
	bind_global.resize(body.get_bone_count())
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
		source_bones.append(driver.find_bone(str(BONE_MAP.get(body.get_bone_name(i), ""))))
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
