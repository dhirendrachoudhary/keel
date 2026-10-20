"""Tool contract for the agent loop (M2).

Every tool returns an explicit evidence-first result — findings / evidence / unknowns
(REA pattern) — because small models reason better over explicit evidence than over
raw output they must re-interpret each turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    findings: str
    evidence: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    ok: bool = True


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def schema(self) -> Mapping[str, Any]:
        """JSON-schema for the tool's arguments; used to build the M2 decoding grammar."""

    @abstractmethod
    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute the tool. Must be safe under M9 (workspace-scoped, dry-run destructive)."""
