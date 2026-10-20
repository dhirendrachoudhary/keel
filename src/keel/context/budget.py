"""Per-agent active-token budget meter (M4): warn at 70%, fold at 85%, hard-stop at the cap."""

from __future__ import annotations

from dataclasses import dataclass

from keel.config import CONTEXT_BUDGET_TOKENS


@dataclass
class BudgetMeter:
    limit: int = CONTEXT_BUDGET_TOKENS
    warn_ratio: float = 0.70
    fold_ratio: float = 0.85

    def status(self, used_tokens: int) -> str:
        """One of ``'ok'`` | ``'warn'`` | ``'fold'`` | ``'stop'``."""
        if used_tokens >= self.limit:
            return "stop"
        ratio = used_tokens / self.limit
        if ratio >= self.fold_ratio:
            return "fold"
        if ratio >= self.warn_ratio:
            return "warn"
        return "ok"

    def should_fold(self, used_tokens: int) -> bool:
        return self.status(used_tokens) in {"fold", "stop"}
