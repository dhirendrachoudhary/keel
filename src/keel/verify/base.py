"""Verifier interface for test-time scaling (M7).

Verifiers are execution-based (run tests/lint/types) or execution-free (compare
candidates against each other); both are strictly local.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Verifier(ABC):
    name: str

    @abstractmethod
    def score(self, task: str, candidates: Sequence[str]) -> list[float]:
        """Return one score per candidate; higher is better."""

    def best(self, task: str, candidates: Sequence[str]) -> int:
        """Index of the highest-scoring candidate (best-of-N)."""
        scores = self.score(task, candidates)
        return max(range(len(scores)), key=scores.__getitem__)
