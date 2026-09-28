"""Check isolation of the upstream inference modules from training package imports."""

import sys
from pathlib import Path
from unittest.mock import patch


def test_geometry_import_skips_dataset_initializers_and_reuses_source(tmp_path: Path) -> None:
    """Geometry imports must not execute HumanML3D's spaCy-dependent initializer."""
    from src.motion_gpt.source import import_source_module

    root = tmp_path / "mGPT" / "data" / "humanml"
    (root / "scripts").mkdir(parents=True)
    (root / "__init__.py").write_text('raise RuntimeError("training package imported")\n')
    (root / "scripts" / "motion_process.py").write_text("VALUE = 263\n")
    with patch.dict(sys.modules):
        # Native integration tests may already have imported the real source.
        for name in tuple(sys.modules):
            if name == "_lblm_motiongpt_geometry" or name.startswith("_lblm_motiongpt_geometry."):
                del sys.modules[name]
        module = import_source_module(tmp_path, "geometry.scripts.motion_process")
        assert module.VALUE == 263
        assert import_source_module(tmp_path, "geometry.scripts.motion_process") is module
