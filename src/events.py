"""Typed conversation notifications shared by orchestration and display backends."""

from typing import Literal

from src.contract import ContractModel


class PipelineEvent(ContractModel):
    """Structured terminal or application notifications without camera pixels."""

    kind: Literal["ready", "observation", "text", "reaction", "response_done"]
    text: str = ""
    response_id: str = ""
    elapsed_s: float = 0.0
