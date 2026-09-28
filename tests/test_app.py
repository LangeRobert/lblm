"""Default implementation wiring."""


def test_app_shares_motion_runtime_and_uses_all_concrete_stages() -> None:
    """The application must load one MotionGPT runtime for both directions."""
    from src.app import AppConfig, create_pipeline
    from src.m_0_camera.open_cv import OpenCVCamera
    from src.m_9_renderer.godot import GodotRenderer

    pipeline = create_pipeline(AppConfig())
    assert isinstance(pipeline.camera, OpenCVCamera)
    assert isinstance(pipeline.renderer, GodotRenderer)
    assert pipeline.captioner.full_body.runtime is pipeline.generator.runtime
    assert all(
        not getattr(type(stage), "__abstractmethods__", ())
        for stage in (
            pipeline.estimator,
            pipeline.normalizer,
            pipeline.segmenter,
            pipeline.processor,
            pipeline.retargeter,
        )
    )
