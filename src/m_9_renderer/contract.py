"""Engine-neutral playback boundary; selection of a renderer is deferred."""

from abc import abstractmethod

from src.contract import ContractModel, Identifier, PipelineModule, Seconds
from src.events import PipelineEvent
from src.m_0_camera.contract import CameraFrame
from src.m_8_skeleton_retargeter.contract import HumanoidRig, RigMotionChunk


class PlaybackRequest(ContractModel):
    """Schedule a chunk against the same monotonic clock used for capture.

    Every chunk of a response uses the same play_at_s. Each frame's absolute
    presentation time is play_at_s + frame.time_s. Late frames may be dropped.
    """

    chunk: RigMotionChunk
    play_at_s: Seconds


class Renderer(PipelineModule):
    """Abstract humanoid playback; asset and transport details belong to backends."""

    @abstractmethod
    async def get_rig(self) -> HumanoidRig:
        """Describe the loaded avatar for the retargeter after opening the renderer.

        :returns: Active avatar hierarchy, axes and local rest transforms.
        """
        ...

    @abstractmethod
    async def submit(self, request: PlaybackRequest) -> None:
        """Queue animation with bounded backpressure, without awaiting playback.

        :param request: Timestamped transforms for the active rig.
        :returns: None when accepted into the bounded playback queue.
        """
        ...

    @abstractmethod
    async def cancel(self, response_id: Identifier) -> None:
        """Discard queued motion for an interrupted response and stop its playback.

        :param response_id: Response to cancel; unknown IDs are a safe no-op.
        :returns: None when obsolete animation can no longer be played.
        """
        ...

    async def publish_event(self, event: PipelineEvent) -> None:
        """Accept dialogue notifications; playback-only backends may ignore them.

        :param event: Structured conversation update without captured pixels.
        :returns: None after the backend accepts the update.
        """

    async def publish_frame(self, frame: CameraFrame) -> None:
        """Accept live preview pixels; renderers without a preview may ignore them.

        :param frame: Captured image shared with pose estimation.
        :returns: None after accepting or dropping the preview frame.
        """
