"""Per-case results and the aggregate report for an eval run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    agent_outcome: str  # "submitted" | "max_steps" | "agent_error" | "setup_error"
    steps: int
    tool_call_error_rate: float
    verify_exit_code: int
    duration_s: float
    error: str = ""  # harness-level error (setup/agent crash), not a normal task failure


@dataclass
class EvalReport:
    benchmark: str
    model_id: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0

    @property
    def mean_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.n_total if self.n_total else 0.0

    @property
    def mean_tool_call_error_rate(self) -> float:
        return (
            sum(r.tool_call_error_rate for r in self.results) / self.n_total
            if self.n_total
            else 0.0
        )

    def summary_line(self) -> str:
        return (
            f"{self.benchmark}: {self.n_passed}/{self.n_total} passed "
            f"({self.pass_rate * 100:.1f}%) · mean {self.mean_steps:.1f} steps · "
            f"tool-err {self.mean_tool_call_error_rate * 100:.1f}% · "
            f"model {self.model_id or 'n/a'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "model_id": self.model_id,
            "n_total": self.n_total,
            "n_passed": self.n_passed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_steps": round(self.mean_steps, 2),
            "mean_tool_call_error_rate": round(self.mean_tool_call_error_rate, 4),
            "results": [asdict(r) for r in self.results],
        }
