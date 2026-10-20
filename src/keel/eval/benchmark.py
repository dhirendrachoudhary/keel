"""A benchmark is anything that yields eval cases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from keel.eval.case import EvalCase


class Benchmark(ABC):
    name: str

    @abstractmethod
    def cases(self) -> Iterator[EvalCase]:
        """Yield the benchmark's cases, in a stable order."""
