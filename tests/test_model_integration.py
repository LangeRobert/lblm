"""Opt-in real checkpoint verification; never downloads models during ordinary tests."""

import os
from time import perf_counter

import numpy as np
import pytest

from src.m_4_motion_to_language.contract import MotionDescription
from src.m_5_llm.contract import DialogueRequest
from src.m_5_llm.llama_cpp import LlamaCppLanguageModel
from src.motion_gpt.runtime import MotionGPTRuntime


@pytest.mark.models
@pytest.mark.skipif(
    os.environ.get("LBLM_RUN_MODEL_TESTS") != "1",
    reason="Set LBLM_RUN_MODEL_TESTS=1 to load the real checkpoints",
)
@pytest.mark.asyncio
async def test_real_gemma_and_motiongpt_resident_together() -> None:
    """Verify cached native Gemma dialogue and both MotionGPT inference directions."""
    llm = LlamaCppLanguageModel()
    motion = MotionGPTRuntime()
    try:
        await llm.open()
        await motion.acquire()
        start = perf_counter()
        request = DialogueRequest(
            response_id="integration",
            system_prompt="Greet the person briefly.",
            observation=MotionDescription(segment_id="s", text="A person waves hello."),
            max_new_tokens=128,
        )
        chunks = [chunk async for chunk in llm.respond(request)]
        dialogue_seconds = perf_counter() - start
        assert "".join(chunk.text_delta for chunk in chunks).strip()
        assert chunks[-1].is_final and chunks[-1].motion_prompt
        start = perf_counter()
        positions = await motion.generate("A person waves their right hand.", 40, 42)
        generation_seconds = perf_counter() - start
        assert positions.ndim == 3 and positions.shape[1:] == (22, 3)
        assert len(positions) >= 4 and np.isfinite(positions).all()
        start = perf_counter()
        caption = await motion.caption(positions)
        caption_seconds = perf_counter() - start
        assert caption.strip() and "<motion_id_" not in caption
        print(
            f"Resident-model smoke timings: dialogue={dialogue_seconds:.2f}s, motion={generation_seconds:.2f}s, caption={caption_seconds:.2f}s; caption={caption!r}"
        )
    finally:
        await motion.release()
        await llm.close()
