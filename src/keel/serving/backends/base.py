"""The one interface every inference backend hides behind (mlx-lm, oMLX, MTPLX)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class InferenceBackend(ABC):
    """Strictly local: implementations must never make a network/API call to generate."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> GenerationResult:
        """Generate a completion. ``grammar`` carries a constrained-decoding spec (M2)."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> Iterator[str]:
        """Yield tokens as they decode."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> GenerationResult:
        """Generate the next assistant turn for a message history.

        The default here renders a plain role-labelled transcript and calls ``generate()``.
        That's fine for a base/completion model, but an instruct model trained on a chat
        template needs that exact template, or it tends to answer out of character — for
        example declaring a task done on the first turn without taking any action, because
        nothing in a raw-concatenated transcript looks like a conversation it's mid-way
        through. Override this method for any backend that has the model's real chat
        template available (see ``MLXBackend``); only fall back to this default when no
        template can be resolved.
        """
        prompt = (
            "\n\n".join(f"{m.role.upper()}:\n{m.content}" for m in messages) + "\n\nASSISTANT:\n"
        )
        return self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, grammar=grammar
        )
