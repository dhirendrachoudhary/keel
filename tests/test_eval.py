from __future__ import annotations

from pathlib import Path

from keel.baseline.agent import AgentResult
from keel.eval import EvalReport, LocalBenchmark, run_benchmark, run_case
from keel.eval.case import EvalCase
from keel.eval.report import CaseResult
from keel.eval.runner import prepare_workspace
from keel.trace import TraceStore
from keel.trace.schema import TraceRecord

REPO_SAMPLES = Path(__file__).resolve().parents[1] / "evals" / "samples"


def _agent_writing(files: dict[str, str]):
    """An AgentFn that drops fixed files into the workspace, then 'submits'."""

    def fn(instructions: str, workspace: Path) -> AgentResult:
        for rel, content in files.items():
            (workspace / rel).write_text(content)
        trace = TraceRecord(task_id="t", model_id="fake", prompt=instructions, outcome="submitted")
        trace.metrics["tool_call_error_rate"] = 0.0
        return AgentResult(outcome="submitted", final_message="done", steps=1, trace=trace)

    return fn


def test_local_benchmark_loads_sample_suite() -> None:
    cases = {c.case_id: c for c in LocalBenchmark(REPO_SAMPLES).cases()}
    assert set(cases) == {"add", "greet"}
    assert "add(a, b)" in cases["add"].instructions
    assert cases["add"].verify_command.startswith("python -c")
    assert cases["greet"].protected_paths == ["test_greet.py"]


def test_run_case_pass_and_fail(tmp_path: Path) -> None:
    case = EvalCase(
        case_id="solve",
        benchmark="local",
        instructions="write solution.txt",
        verify_command="test -f solution.txt",
    )
    good = run_case(case, _agent_writing({"solution.txt": "x"}), runs_dir=tmp_path / "good")
    assert good.passed is True and good.verify_exit_code == 0 and good.agent_outcome == "submitted"

    bad = run_case(case, _agent_writing({}), runs_dir=tmp_path / "bad")
    assert bad.passed is False and bad.verify_exit_code != 0


def test_run_case_writes_verified_trace(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.jsonl")
    case = EvalCase(case_id="c1", benchmark="local", instructions="do it", verify_command="true")
    run_case(case, _agent_writing({}), runs_dir=tmp_path / "ws", trace_store=store)
    records = list(store)
    assert len(records) == 1
    assert records[0].task_id == "c1"
    assert records[0].outcome == "pass"
    assert records[0].metrics["verify_exit_code"] == 0


def test_protected_path_is_restored_before_verify(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "marker.txt").write_text("ORIGINAL")
    case = EvalCase(
        case_id="prot",
        benchmark="local",
        instructions="",
        source_dir=str(source),
        protected_paths=["marker.txt"],
        verify_command="grep -q ORIGINAL marker.txt",
    )
    # Agent overwrites the protected file; the runner must restore it before verifying.
    result = run_case(case, _agent_writing({"marker.txt": "TAMPERED"}), runs_dir=tmp_path / "ws")
    assert result.passed is True


def test_agent_crash_is_a_failed_case_not_a_crash(tmp_path: Path) -> None:
    def boom(instructions: str, workspace: Path) -> AgentResult:
        raise RuntimeError("model exploded")

    case = EvalCase(case_id="x", benchmark="local", instructions="", verify_command="true")
    result = run_case(case, boom, runs_dir=tmp_path)
    assert result.passed is False
    assert result.agent_outcome == "agent_error"
    assert "model exploded" in result.error


def test_run_benchmark_report_and_limit(tmp_path: Path) -> None:
    report = run_benchmark(
        LocalBenchmark(REPO_SAMPLES),
        _agent_writing({"add.py": "def add(a, b):\n    return a + b\n"}),
        runs_dir=tmp_path,
        model_id="fake",
        limit=1,
    )
    assert isinstance(report, EvalReport)
    assert report.n_total == 1  # limit honored
    assert "add" in report.results[0].case_id


def test_prepare_workspace_excludes_named_dirs(tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / ".meta").mkdir(parents=True)
    (source / ".meta" / "example.py").write_text("# the answer")
    (source / "stub.py").write_text("def f(): ...")
    case = EvalCase(
        case_id="e",
        benchmark="local",
        instructions="",
        source_dir=str(source),
        exclude_names=[".meta"],
    )
    ws = tmp_path / "ws"
    prepare_workspace(ws, case)
    assert (ws / "stub.py").exists()
    assert not (ws / ".meta").exists()  # reference solution excluded


def test_report_aggregates() -> None:
    report = EvalReport(benchmark="local", model_id="m")
    report.results = [
        CaseResult("a", True, "submitted", 2, 0.0, 0, 1.0),
        CaseResult("b", False, "max_steps", 4, 0.5, 1, 2.0),
    ]
    assert report.n_passed == 1 and report.n_total == 2
    assert report.pass_rate == 0.5
    assert report.mean_steps == 3.0
    assert "1/2 passed" in report.summary_line()
    assert report.to_dict()["pass_rate"] == 0.5
