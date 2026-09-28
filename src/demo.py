"""Deterministic observed wave for exercising real models without a camera."""

import math

from src.contract import JointObservation, PoseFrame, Vector3
from src.m_3_motion_segmenter.contract import MotionSegment
from src.skeleton import CANONICAL_SKELETON


def waving_segment() -> MotionSegment:
    """Construct a two-second, complete canonical wave for a model smoke run.

    :returns: A synthetic observation clearly identified as demo input.
    """
    frames: list[PoseFrame] = []
    for i in range(41):
        joints = [
            JointObservation(position=j.rest_position, confidence=1.0)
            for j in CANONICAL_SKELETON.joints
        ]
        joints[19] = JointObservation(position=Vector3(x=0.38, y=0.61, z=0.0), confidence=1.0)
        angle = 0.4 * math.sin(i / 20 * math.tau * 2)
        joints[21] = JointObservation(
            position=Vector3(
                x=0.38 + 0.25 * math.sin(angle), y=0.61 + 0.25 * math.cos(angle), z=0.0
            ),
            confidence=1.0,
        )
        frames.append(
            PoseFrame(
                frame_id=i, captured_at_s=i / 20, person_id="synthetic-demo", joints=tuple(joints)
            )
        )
    return MotionSegment(
        segment_id="synthetic-wave",
        skeleton=CANONICAL_SKELETON,
        frames=tuple(frames),
        end_reason="flush",
    )
