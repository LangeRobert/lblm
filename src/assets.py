"""Pinned Hugging Face assets with local overrides and offline cache support."""

from pathlib import Path
from typing import Literal

from src.contract import ContractModel, Identifier

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "models" / "huggingface"


class HubFile(ContractModel):
    """One cached model asset; authentication uses Hugging Face's existing login."""

    repo_id: Identifier
    filename: Identifier
    revision: Identifier = "main"
    repo_type: Literal["model", "dataset", "space"] = "model"
    local_path: Path | None = None
    cache_dir: Path | None = DEFAULT_CACHE
    local_files_only: bool = False


def resolve_file(asset: HubFile) -> Path:
    """Resolve a local file or download it through Hugging Face's shared cache.

    :param asset: Repository, revision, filename and cache policy.
    :returns: Existing local asset path; Hub authentication errors propagate.
    """
    if asset.local_path is not None and asset.local_path.is_file():
        return asset.local_path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.filename,
            revision=asset.revision,
            repo_type=asset.repo_type,
            cache_dir=asset.cache_dir,
            local_files_only=asset.local_files_only,
        )
    )
