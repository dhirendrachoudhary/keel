"""Run a benchmark: prepare a fresh workspace per case, run the agent, score by execution.

The runner takes an ``AgentFn`` (instructions + workspace -> AgentResult) rather than a
concrete agent, so the same harness drives the baseline loop, later orchestrated agents,
or a scripted fake in tests. Scoring is execution-based: the case's ``verify_command`` is
the ground truth (exit 0 == pass), with protected files restored first so tests can't be
gamed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from keel.baseline.agent import AgentResult, BaselineAgent
from keel.eval.benchmark import Benchmark
from keel.eval.case import EvalCase
from keel.eval.report import CaseResult, EvalReport
from keel.serving.backends.base import InferenceBackend
from keel.trace.store import TraceStore

AgentFn = Callable[[str, Path], AgentResult]


def _eval_env() -> dict[str, str]:
    """Environment for setup/verify commands: put the running interpreter on PATH so a
    bare ``python`` (used by pytest-based checks) resolves to it, and expose KEEL_PYTHON."""
    env = dict(os.environ)
    bindir = os.path.dirname(sys.executable)
    if bindir:
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["KEEL_PYTHON"] = sys.executable
    return env


def baseline_agent_fn(
    backend: InferenceBackend, *, model_id: str = "", max_steps: int = 30
) -> AgentFn:
    """Adapt the plain baseline loop into an AgentFn for the eval runner."""

    def fn(instructions: str, workspace: Path) -> AgentResult:
        agent = BaselineAgent(backend, workspace=workspace, max_steps=max_steps)
        return agent.run(instructions, task_id="eval", model_id=model_id)

    return fn


def prepare_workspace(workspace: Path, case: EvalCase) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if case.source_dir:
        _copy_tree(Path(case.source_dir), workspace, exclude=set(case.exclude_names))
    for rel, content in case.seed_files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for command in case.setup_commands:
        subprocess.run(
            command, shell=True, cwd=str(workspace), timeout=case.verify_timeout_s, env=_eval_env()
        )


def run_verify(workspace: Path, case: EvalCase) -> int:
    if not case.verify_command:
        return 1  # no ground-truth check means we cannot score this as a pass
    _restore_protected(workspace, case)
    try:
        proc = subprocess.run(
            case.verify_command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=case.verify_timeout_s,
            env=_eval_env(),
        )
    except subprocess.TimeoutExpired:
        return 124
    return proc.returncode


def run_case(
    case: EvalCase,
    agent_fn: AgentFn,
    *,
    runs_dir: Path | str,
    trace_store: TraceStore | None = None,
) -> CaseResult:
    workspace = Path(runs_dir) / case.case_id
    started = time.time()

    try:
        prepare_workspace(workspace, case)
    except (OSError, subprocess.SubprocessError) as exc:
        return _error_result(case.case_id, "setup_error", started, f"setup: {exc}")

    try:
        result = agent_fn(case.instructions, workspace)
    except Exception as exc:
        # A crashing agent is one failed case, not a crashed eval run.
        return _error_result(case.case_id, "agent_error", started, f"agent: {exc}")

    exit_code = run_verify(workspace, case)
    passed = exit_code == 0
    tool_err = float(result.trace.metrics.get("tool_call_error_rate", 0.0)) if result.trace else 0.0

    if trace_store is not None and result.trace is not None:
        result.trace.task_id = case.case_id
        result.trace.outcome = "pass" if passed else "fail"
        result.trace.metrics["verify_exit_code"] = exit_code
        trace_store.append(result.trace)

    return CaseResult(
        case_id=case.case_id,
        passed=passed,
        agent_outcome=result.outcome,
        steps=result.steps,
        tool_call_error_rate=tool_err,
        verify_exit_code=exit_code,
        duration_s=round(time.time() - started, 2),
    )


def run_benchmark(
    benchmark: Benchmark,
    agent_fn: AgentFn,
    *,
    runs_dir: Path | str,
    model_id: str = "",
    trace_store: TraceStore | None = None,
    limit: int | None = None,
) -> EvalReport:
    report = EvalReport(benchmark=benchmark.name, model_id=model_id)
    for index, case in enumerate(benchmark.cases()):
        if limit is not None and index >= limit:
            break
        report.results.append(run_case(case, agent_fn, runs_dir=runs_dir, trace_store=trace_store))
    return report


def _error_result(case_id: str, outcome: str, started: float, message: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=False,
        agent_outcome=outcome,
        steps=0,
        tool_call_error_rate=0.0,
        verify_exit_code=-1,
        duration_s=round(time.time() - started, 2),
        error=message,
    )


def _copy_tree(src: Path, dst: Path, exclude: set[str] | None = None) -> None:
    skip = {"case.json"} | (exclude or set())
    for item in src.iterdir():
        if item.name in skip:
            continue
        dest = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def _restore_protected(workspace: Path, case: EvalCase) -> None:
    if not case.source_dir:
        return
    src = Path(case.source_dir)
    for rel in case.protected_paths:
        origin = src / rel
        if origin.exists():
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
