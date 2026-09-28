"""Pinned model, source and preprocessing assets for MotionGPT-base."""

from pathlib import Path
from typing import Literal

from src.assets import DEFAULT_CACHE, HubFile
from src.contract import ContractModel, Identifier


class MotionGPTConfig(ContractModel):
    """Automatic Hub setup with optional local source and checkpoint overrides."""

    checkpoint: HubFile = HubFile(
        repo_id="OpenMotionLab/MotionGPT-base",
        filename="motiongpt_s3_h3d.tar",
        revision="a0a37a388137f15df8299c643a885a42b07772fe",
    )
    source_repo: Identifier = "OpenMotionLab/MotionGPT"
    source_revision: Identifier = "89e4bf56c6da6896b2bb13bb700c72803872d7d8"
    source_path: Path | None = None
    text_model_repo: Identifier = "google/flan-t5-base"
    text_model_revision: Identifier = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
    text_model_path: Path | None = None
    cache_dir: Path | None = DEFAULT_CACHE
    local_files_only: bool = False
    device: Literal["cpu", "mps", "auto"] = "auto"
