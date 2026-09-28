"""Partial-body observations must reach a grounded caption without invented legs."""

from unittest.mock import AsyncMock

from src.contract import JointObservation, PoseFrame
from src.m_3_motion_segmenter.contract import MotionSegment
from src.skeleton import CANONICAL_SKELETON


async def test_upper_body_caption_does_not_call_full_body_model() -> None:
    """Missing legs retain explicit occlusion while visible arms produce a caption."""
    from src.m_4_motion_to_language.visible import VisibleMotionToLanguage

    delegate = AsyncMock()
    stage = VisibleMotionToLanguage(delegate)
    await stage.open()
    joints = tuple(
        JointObservation(
            position=j.rest_position if i >= 16 else None, confidence=1.0 if i >= 16 else 0.0
        )
        for i, j in enumerate(CANONICAL_SKELETON.joints)
    )
    segment = MotionSegment(
        segment_id="upper",
        skeleton=CANONICAL_SKELETON,
        end_reason="window",
        frames=tuple(
            PoseFrame(frame_id=i, captured_at_s=i / 20, person_id="p", joints=joints)
            for i in range(11)
        ),
    )
    caption = await stage.describe(segment)
    assert "upper body" in caption.text.lower()
    assert "arm" in caption.text.lower()
    delegate.describe.assert_not_awaited()
    await stage.close()
