"""Abstract inference backend interface and result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import Response

    from claas.core.types import ChoiceLogprobs

BackendKind = Literal["tinker", "local", "modal"]


@dataclass
class CompletionResult:
    """Result from a chat completion call."""

    content: str
    raw_prompt: str
    raw_response: str
    response_token_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    logprobs_content: ChoiceLogprobs | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class TextCompletionResult:
    """Result from a text completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ScoreResult:
    """Result from scoring a completion by computing per-token logprobs."""

    logprobs: list[float]
    tokens: list[str]
    prompt_tokens: int
    completion_tokens: int
    logprob_sum: float


class InferenceBackend(ABC):
    """Abstract base for inference backends (Tinker SDK or vLLM forwarding)."""

    @abstractmethod
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 1,
    ) -> CompletionResult:
        """Run a chat completion and return structured results."""

    @abstractmethod
    async def text_completion(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> TextCompletionResult:
        """Run a text completion and return structured results."""

    @abstractmethod
    async def list_models(self) -> dict[str, object] | Response:
        """List available models."""

    @abstractmethod
    async def score(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        completion: str,
    ) -> ScoreResult:
        """Score a completion by computing per-token logprobs."""

    def register_routes(self, app: FastAPI) -> None:
        """Register backend-specific routes on the FastAPI app.

        Default is a no-op; backends override to add custom endpoints.
        """
