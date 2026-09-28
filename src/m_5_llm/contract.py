"""Local dialogue boundary with incremental text and explicit motion intent."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal

from src.contract import (
    ContractModel,
    Identifier,
    NonNegativeInt,
    PipelineModule,
    PositiveInt,
)
from src.m_4_motion_to_language.contract import MotionDescription


class ChatMessage(ContractModel):
    """One prior conversational message; system instructions live in the request."""

    role: Literal["user", "assistant"]
    text: Identifier


class DialogueRequest(ContractModel):
    """Explicit prompt, observed movement and caller-managed bounded history."""

    response_id: Identifier
    system_prompt: Identifier
    observation: MotionDescription
    user_prompt: str = ""
    history: tuple[ChatMessage, ...] = ()
    max_new_tokens: PositiveInt = 128


class DialogueChunk(ContractModel):
    """Append-only text delta, optionally paired with a complete action description.

    Sequence starts at zero; exactly one chunk is final. A motion_prompt is a
    complete, actionable sentence emitted once, never an unfinished token delta.
    Empty text is allowed for motion-only or final events.
    """

    response_id: Identifier
    sequence: NonNegativeInt
    text_delta: str
    motion_prompt: Identifier | None = None
    is_final: bool


class LanguageModel(PipelineModule):
    """Abstract local dialogue model with cancellable incremental output."""

    @abstractmethod
    def respond(self, request: DialogueRequest) -> AsyncGenerator[DialogueChunk]:
        """Stream the response without waiting for the entire text to complete.

        :param request: Instructions, observation and conversation context.
        :returns: Async iterator of ordered text and action events.
        """
        ...
