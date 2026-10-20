"""The one trace schema every Keel run writes (M8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    findings: str = ""
    evidence: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class ContextFold:
    reason: str
    tokens_before: int
    tokens_after: int
    summary: str


@dataclass
class TraceRecord:
    task_id: str
    model_id: str
    prompt: str
    outcome: str = "fail"  # "pass" | "fail" | "escalated"
    tool_calls: list[ToolCall] = field(default_factory=list)
    context_folds: list[ContextFold] = field(default_factory=list)
    verifier_scores: dict[str, float] = field(default_factory=dict)
    # pass, pct_opus, wall_clock_s, tokens, tool_call_error_rate, escalations, peak_memory_gb
    metrics: dict[str, Any] = field(default_factory=dict)
