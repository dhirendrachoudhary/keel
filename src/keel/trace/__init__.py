"""M8 — Eval & trace store.

One trace schema (prompt, context folds, tool calls, verifier scores, outcome, metrics)
shared by the skill writer (M5), harness adaptation (M6) and v2 planner training. Every
claim in the README must map to a reproducible local run recorded here.
"""

from __future__ import annotations

from keel.trace.schema import ContextFold, ToolCall, TraceRecord
from keel.trace.store import TraceStore

__all__ = ["ContextFold", "ToolCall", "TraceRecord", "TraceStore"]
