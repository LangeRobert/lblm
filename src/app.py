"""Application configuration and concrete implementation composition."""

from collections.abc import Callable

from src.contract import ContractModel
from src.m_0_camera.open_cv import OpenCVCamera, OpenCVConfig
from src.m_1_pose_estimator.mediapipe import MediaPipeConfig, MediaPipePoseEstimator
from src.m_2_pose_normalizer.canonical import CanonicalPoseNormalizer
from src.m_3_motion_segmenter.window import SegmenterConfig, WindowMotionSegmenter
from src.m_4_motion_to_language.motion_gpt import MotionGPTMotionToLanguage
from src.m_4_motion_to_language.visible import VisibleMotionToLanguage
from src.m_5_llm.llama_cpp import LlamaCppConfig, LlamaCppLanguageModel
from src.m_6_language_to_motion.motion_gpt import MotionGPTLanguageToMotion
from src.m_7_motion_processor.causal import CausalMotionProcessor
from src.m_8_skeleton_retargeter.geometric import GeometricSkeletonRetargeter
from src.m_9_renderer.godot import GodotConfig, GodotRenderer
from src.motion_gpt.config import MotionGPTConfig
from src.motion_gpt.runtime import MotionGPTRuntime
from src.pipeline import ConversationPipeline, PipelineConfig, PipelineEvent


class AppConfig(ContractModel):
    """Strict JSON-loadable configuration for every stateful backend."""

    camera: OpenCVConfig = OpenCVConfig()
    pose: MediaPipeConfig = MediaPipeConfig()
    segmenter: SegmenterConfig = SegmenterConfig()
    llm: LlamaCppConfig = LlamaCppConfig()
    motiongpt: MotionGPTConfig = MotionGPTConfig()
    renderer: GodotConfig = GodotConfig()
    pipeline: PipelineConfig = PipelineConfig()


def create_pipeline(
    config: AppConfig,
    on_event: Callable[[PipelineEvent], None] | None = None,
) -> ConversationPipeline:
    """Compose the ten stages with one shared MotionGPT model allocation.

    :param config: Validated application configuration.
    :param on_event: Optional UI or terminal progress callback.
    :returns: Unopened, fully wired conversation pipeline.
    """
    runtime = MotionGPTRuntime(config.motiongpt)
    return ConversationPipeline(
        camera=OpenCVCamera(config.camera),
        estimator=MediaPipePoseEstimator(config.pose),
        normalizer=CanonicalPoseNormalizer(),
        segmenter=WindowMotionSegmenter(config.segmenter),
        captioner=VisibleMotionToLanguage(MotionGPTMotionToLanguage(runtime)),
        llm=LlamaCppLanguageModel(config.llm),
        generator=MotionGPTLanguageToMotion(runtime),
        processor=CausalMotionProcessor(),
        retargeter=GeometricSkeletonRetargeter(),
        renderer=GodotRenderer(config.renderer),
        config=config.pipeline,
        on_event=on_event,
    )
