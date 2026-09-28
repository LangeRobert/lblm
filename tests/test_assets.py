"""Verify lazy Hugging Face resolution and explicit offline behavior."""

from pathlib import Path
from unittest.mock import patch


def test_existing_file_skips_download(tmp_path: Path) -> None:
    """An explicit existing local file takes priority over the Hub."""
    from src.assets import HubFile, resolve_file

    local = tmp_path / "model.gguf"
    local.touch()
    with patch("huggingface_hub.hf_hub_download") as download:
        assert (
            resolve_file(HubFile(repo_id="test/model", filename="model.gguf", local_path=local))
            == local
        )
    download.assert_not_called()


def test_missing_file_uses_huggingface_cache(tmp_path: Path) -> None:
    """Missing assets resolve through the Hub with revision and offline policy."""
    from src.assets import HubFile, resolve_file

    cached = tmp_path / "cached.gguf"
    cached.touch()
    with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as download:
        result = resolve_file(
            HubFile(
                repo_id="test/model", filename="model.gguf", revision="abc", local_files_only=True
            )
        )
    assert result == cached
    assert download.call_args.kwargs["revision"] == "abc"
    assert download.call_args.kwargs["local_files_only"] is True
