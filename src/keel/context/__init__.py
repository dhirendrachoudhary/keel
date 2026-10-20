"""M4 — Context manager.

No agent ever exceeds 32K active tokens, and nothing important is lost: ReFold-style
training-free rendering (dedupe repeats, fold completed turns, append-only history),
Context-Folding branch/return-with-summary, and a per-agent budget meter.
"""

from __future__ import annotations

from keel.context.budget import BudgetMeter

__all__ = ["BudgetMeter"]
