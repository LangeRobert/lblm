"""Verify dialogue parsing and streaming around the native inference boundary."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.m_4_motion_to_language.contract import MotionDescription
from src.m_5_llm.contract import DialogueRequest


@pytest.mark.asyncio
async def test_llama_stream_decodes_json_and_emits_one_motion(tmp_path: Path) -> None:
    """Native token chunks become plain text deltas and one final action."""
    from src.assets import HubFile
    from src.m_5_llm.llama_cpp import LlamaCppConfig, LlamaCppLanguageModel

    model_path = tmp_path / "gemma.gguf"
    model_path.touch()
    backend = MagicMock()
    backend.metadata = {"general.architecture": "gemma4", "tokenizer.chat_template": "template"}
    backend.create_chat_completion.return_value = iter(
        [
            {"choices": [{"delta": {"content": part}, "finish_reason": None}]}
            for part in (
                '{"reply":"Hel',
                "lo \\u263a",
                '","motion_prompt":"Wave ',
                'the right hand."}',
            )
        ]
        + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    )
    config = LlamaCppConfig(
        model=HubFile(repo_id="test/model", filename="gemma.gguf", local_path=model_path)
    )
    llm = LlamaCppLanguageModel(config)
    with patch("llama_cpp.Llama", return_value=backend):
        await llm.open()
    request = DialogueRequest(
        response_id="r",
        system_prompt="Be concise.",
        observation=MotionDescription(segment_id="s", text="Waves."),
    )
    chunks = [chunk async for chunk in llm.respond(request)]
    assert "".join(c.text_delta for c in chunks) == "Hello ☺"
    assert len(chunks) > 1
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    assert [c.motion_prompt for c in chunks if c.motion_prompt] == ["Wave the right hand."]
    assert sum(c.is_final for c in chunks) == 1 and chunks[-1].is_final
    kwargs = backend.create_chat_completion.call_args.kwargs
    assert kwargs["stream"] is True and kwargs["max_tokens"] == 128
    assert "Waves." in kwargs["messages"][-1]["content"]
    await llm.reset()
    await llm.close()
    await llm.close()
    backend.close.assert_called_once()


@pytest.mark.asyncio
async def test_truncated_dialogue_never_emits_motion(tmp_path: Path) -> None:
    """A token budget exhausted inside JSON must not produce a fabricated action."""
    from src.assets import HubFile
    from src.m_5_llm.llama_cpp import LlamaCppConfig, LlamaCppLanguageModel

    model_path = tmp_path / "gemma.gguf"
    model_path.touch()
    backend = MagicMock()
    backend.metadata = {"general.architecture": "gemma4", "tokenizer.chat_template": "template"}
    backend.create_chat_completion.return_value = iter(
        [
            {"choices": [{"delta": {"content": '{"reply":"Hi'}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ]
    )
    llm = LlamaCppLanguageModel(
        LlamaCppConfig(
            model=HubFile(repo_id="test/model", filename="gemma.gguf", local_path=model_path)
        )
    )
    with patch("llama_cpp.Llama", return_value=backend):
        await llm.open()
    request = DialogueRequest(
        response_id="r",
        system_prompt="Reply.",
        observation=MotionDescription(segment_id="s", text="Waves."),
    )
    with pytest.raises(ValueError, match="truncated"):
        async for chunk in llm.respond(request):
            assert chunk.motion_prompt is None and not chunk.is_final
    await llm.close()
