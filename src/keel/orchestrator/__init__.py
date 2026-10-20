"""M3 — Orchestrator.

A planner decomposes coding tasks into typed subtasks and dispatches workers in fresh
contexts. Only the planner thinks; workers run with thinking off to keep tool calls
stable (unrestricted thinking destabilizes small-agent tool orchestration).
"""

from __future__ import annotations

from keel.orchestrator.handoff import Handoff

__all__ = ["Handoff"]
