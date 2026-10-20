"""The plain single-agent baseline loop — the yardstick.

Deliberately minimal (mini-swe-agent style): linear append-only history, one bash
action per step, stateless execution, no context folding and no skills. Every Keel
module must beat this baseline on the same local model at equal token budget.
"""

from __future__ import annotations

from keel.baseline.actions import Action, BashAction, ParseError, SubmitAction, parse_action
from keel.baseline.agent import AgentResult, BaselineAgent
from keel.baseline.bash import BashTool, Observation

__all__ = [
    "Action",
    "AgentResult",
    "BaselineAgent",
    "BashAction",
    "BashTool",
    "Observation",
    "ParseError",
    "SubmitAction",
    "parse_action",
]
