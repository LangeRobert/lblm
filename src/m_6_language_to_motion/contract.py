"""Action descriptions to complete clips or incrementally generated poses."""

from abc import abstractmethod
from collections.abc import AsyncGenerator

from src.contract import (
    ContractModel,
    Identifier,
    MotionChunk,
    MotionFrame,
    NonNegativeInt,
    PipelineModule,
    PositiveFloat,
    Seconds,
    SkeletonDefinition,
)


class MotionRequest(ContractModel):
    """One action on a response timeline with optional preceding pose context.

    Continuation poses use the requested skeleton and precede start_time_s.
    Multiple requests per response have nonoverlapping time ranges; the caller
    supplies the next chunk sequence and marks only the last request final.
    """

    response_id: Identifier
    prompt: Identifier
    skeleton: SkeletonDefinition
    duration_s: PositiveFloat
    frame_rate_hz: PositiveFloat
    start_time_s: Seconds = 0.0
    first_sequence: NonNegativeInt = 0
    ends_response: bool = True
    continuation: tuple[MotionFrame, ...] = ()
    seed: NonNegativeInt | None = None


class LanguageToMotion(PipelineModule):
    """Abstract motion generator; a clip backend may yield a single chunk."""

    @abstractmethod
    def generate(self, request: MotionRequest) -> AsyncGenerator[MotionChunk]:
        """Generate canonical motion, preserving the request's response timeline.

        :param request: Complete action description and continuity constraints.
        :returns: Ordered chunks; set is_final only when ends_response is true.
        :raises ValueError: If the requested skeleton is unsupported.
        """
        ...
