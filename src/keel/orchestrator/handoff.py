"""Typed handoff contract the planner uses to dispatch a worker (M3).

Fields: task, inputs, allowed tools, success check, return schema
(the return schema is enforced by M2 constrained decoding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Handoff:
    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    success_check: str = ""
    return_schema: dict[str, Any] = field(default_factory=dict)
