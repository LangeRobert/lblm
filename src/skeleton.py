"""Shared HumanML3D topology and a meter-scale neutral humanoid rig."""

from src.contract import CoordinateSystem, JointDefinition, Quaternion, SkeletonDefinition, Vector3
from src.m_8_skeleton_retargeter.contract import BoneTransform, HumanoidRig
from src.motion_gpt.geometry import JOINT_NAMES, PARENTS

REST = (
    (0.0, 0.0, 0.0),
    (-0.10, -0.04, 0.0),
    (0.10, -0.04, 0.0),
    (0.0, 0.12, 0.0),
    (-0.10, -0.46, 0.0),
    (0.10, -0.46, 0.0),
    (0.0, 0.26, 0.0),
    (-0.10, -0.87, 0.0),
    (0.10, -0.87, 0.0),
    (0.0, 0.40, 0.0),
    (-0.10, -0.91, -0.15),
    (0.10, -0.91, -0.15),
    (0.0, 0.53, 0.0),
    (-0.08, 0.44, 0.0),
    (0.08, 0.44, 0.0),
    (0.0, 0.69, 0.0),
    (-0.21, 0.44, 0.0),
    (0.21, 0.44, 0.0),
    (-0.49, 0.44, 0.0),
    (0.49, 0.44, 0.0),
    (-0.74, 0.44, 0.0),
    (0.74, 0.44, 0.0),
)
CANONICAL_SKELETON = SkeletonDefinition(
    skeleton_id="humanml3d-canonical-v1",
    coordinates=CoordinateSystem(
        units="meters",
        handedness="right",
        positive_x="right",
        positive_y="up",
        positive_z="backward",
        origin="initial pelvis",
    ),
    joints=tuple(
        JointDefinition(
            name=name,
            parent=JOINT_NAMES[PARENTS[i]] if i else None,
            rest_position=Vector3(x=x, y=y, z=z),
        )
        for i, (name, (x, y, z)) in enumerate(zip(JOINT_NAMES, REST, strict=True))
    ),
)
CANONICAL_RIG = HumanoidRig(
    rig_id="lblm-mannequin-v1",
    skeleton=CANONICAL_SKELETON,
    rest_transforms=tuple(
        BoneTransform(
            translation=Vector3(
                x=p[0] - (REST[PARENTS[i]][0] if i else 0),
                y=p[1] - (REST[PARENTS[i]][1] if i else 0),
                z=p[2] - (REST[PARENTS[i]][2] if i else 0),
            ),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        for i, p in enumerate(REST)
    ),
)
