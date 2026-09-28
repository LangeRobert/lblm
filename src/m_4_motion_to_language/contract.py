"""Motion windows to grounded descriptions, not inferred private intentions."""

from abc import abstractmethod

from src.contract import Confidence, ContractModel, Identifier, PipelineModule
from src.m_3_motion_segmenter.contract import MotionSegment


class MotionDescription(ContractModel):
    """Observable action caption; unknown confidence must be None."""

    segment_id: Identifier
    text: Identifier
    confidence: Confidence | None = None


class MotionToLanguage(PipelineModule):
    """Abstract motion captioner; model-specific feature conversion stays internal."""

    @abstractmethod
    async def describe(self, segment: MotionSegment) -> MotionDescription:
        """Describe observable movement in a completed motion window.

        :param segment: Canonical positions and capture timestamps.
        :returns: Caption tied to the original segment ID.
        """
        ...
