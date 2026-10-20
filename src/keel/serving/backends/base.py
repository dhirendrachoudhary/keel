"""The one interface every inference backend hides behind (mlx-lm, oMLX, MTPLX)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


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
