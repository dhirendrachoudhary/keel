"""M7 — Verifier & test-time scaling.

Turn free local tokens into accuracy: sample K candidate patches and pick the right one.
Hybrid verifiers combine execution-based signals (tests, lint, type checks) with
execution-free ones (compare candidates against each other). K adapts to task difficulty
and battery/thermal state from M1.
"""

from __future__ import annotations

from keel.verify.base import Verifier

__all__ = ["Verifier"]
