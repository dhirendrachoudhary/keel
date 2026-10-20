"""M8 — Eval harness.

Run a benchmark locally, score it by execution, and record a trace per case. The framework
is model-agnostic (it takes an ``AgentFn``); adapters turn a benchmark into eval cases:
``LocalBenchmark`` (a directory of cases), ``AiderPolyglotBenchmark``, and
``SWEBenchVerified``. Every number a report shows comes from a reproducible local run.
"""

from __future__ import annotations

from keel.eval.aider_polyglot import AiderPolyglotBenchmark
from keel.eval.benchmark import Benchmark
from keel.eval.case import EvalCase
from keel.eval.local import LocalBenchmark
from keel.eval.report import CaseResult, EvalReport
from keel.eval.runner import AgentFn, baseline_agent_fn, run_benchmark, run_case
from keel.eval.swebench import SWEBenchVerified

__all__ = [
    "AgentFn",
    "AiderPolyglotBenchmark",
    "Benchmark",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "LocalBenchmark",
    "SWEBenchVerified",
    "baseline_agent_fn",
    "run_benchmark",
    "run_case",
]
