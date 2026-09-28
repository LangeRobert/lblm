"""Full-body MotionGPT captioning with an explicit visible-arm fallback."""

import numpy as np

from src.m_3_motion_segmenter.contract import MotionSegment
from src.m_4_motion_to_language.contract import MotionDescription, MotionToLanguage
from src.motion_gpt.geometry import validate_skeleton


class VisibleMotionToLanguage(MotionToLanguage):
    """Use kinematic descriptions for partial poses, without fabricating hidden limbs."""

    def __init__(self, full_body: MotionToLanguage) -> None:
        """Attach the full-body model used for complete windows.

        :param full_body: MotionGPT adapter sharing the generation runtime.
        """
        self.full_body = full_body
        self._opened = False

    async def open(self) -> None:
        """Acquire the delegated model.

        :returns: None after readiness.
        """
        await self.full_body.open()
        self._opened = True

    async def reset(self) -> None:
        """Reset delegated temporal state.

        :returns: None.
        """
        await self.full_body.reset()

    async def close(self) -> None:
        """Release delegated model ownership.

        :returns: None; safe to repeat.
        """
        await self.full_body.close()
        self._opened = False

    async def describe(self, segment: MotionSegment) -> MotionDescription:
        """Describe full motion or only measured upper-body geometry.

        :param segment: One person's canonical observations with explicit occlusions.
        :returns: Model caption or conservative visible-arm description.
        """
        if not self._opened:
            raise RuntimeError("Call open before describe")
        validate_skeleton(segment.skeleton)
        if any(
            len(f.joints) != 22 or f.person_id != segment.frames[0].person_id
            for f in segment.frames
        ):
            raise ValueError("Expected one subject and 22 canonical joints")
        if all(
            j.position is not None and j.confidence > 0 for f in segment.frames for j in f.joints
        ):
            return await self.full_body.describe(segment)
        observations: list[list[list[float]]] = []
        for frame in segment.frames:
            row: list[list[float]] = []
            for index in (16, 17, 18, 19, 20, 21):
                joint = frame.joints[index]
                if joint.position is None or joint.confidence <= 0:
                    raise ValueError(
                        "Upper-body captioning needs visible shoulders, elbows and wrists"
                    )
                p = joint.position
                row.append([p.x, p.y, p.z])
            observations.append(row)
        points = np.asarray(observations)
        descriptions: list[str] = []
        for side, shoulder, wrist in (("left", 0, 4), ("right", 1, 5)):
            relative = points[:, wrist] - points[:, shoulder]
            recent = np.median(relative[-5:], axis=0)
            if recent[1] > 0.08:
                descriptions.append(f"holds the {side} hand above the shoulder")
            elif abs(recent[0]) > 0.20 and recent[1] > -0.15:
                descriptions.append(f"extends the {side} arm outward")
            else:
                descriptions.append(f"holds the {side} hand below shoulder height")
            if len(relative) >= 5:
                movement = np.ptp(relative, axis=0)
                if movement[0] > 0.12:
                    descriptions.append(f"moves the {side} hand sideways")
                if movement[1] > 0.12:
                    descriptions.append(f"moves the {side} hand up or down")
        text = (
            "Only the upper body is visible. The person "
            + ", and ".join(descriptions)
            + ". Hidden lower-body motion is unknown."
        )
        return MotionDescription(segment_id=segment.segment_id, text=text)
