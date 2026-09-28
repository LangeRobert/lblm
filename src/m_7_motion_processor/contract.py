"""Generated motion to temporally consistent, render-ready joint positions."""

from abc import abstractmethod

from src.contract import MotionChunk, PipelineModule


class MotionProcessor(PipelineModule):
    """Abstract causal smoothing, resampling and foot-contact correction.

    Input and output share MotionChunk. Preserve response identity and skeleton;
    output sequence numbers are contiguous, timestamps increase and any buffered
    tail is drained on final input. Reset discards the tail on interruption.
    """

    @abstractmethod
    async def process(self, chunk: MotionChunk) -> tuple[MotionChunk, ...]:
        """Process a chunk, emitting zero or more chunks as buffering permits.

        :param chunk: Raw canonical motion, including final completion events.
        :returns: Processed chunks with exactly one final event per response.
        """
        ...
