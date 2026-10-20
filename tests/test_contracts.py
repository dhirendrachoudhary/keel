from __future__ import annotations

from pathlib import Path

from keel.config import CONTEXT_BUDGET_TOKENS, KeelConfig
from keel.context.budget import BudgetMeter
from keel.loop.tools.base import ToolResult
from keel.trace import ContextFold, ToolCall, TraceRecord, TraceStore


def test_budget_meter_thresholds() -> None:
    m = BudgetMeter()
    assert m.limit == CONTEXT_BUDGET_TOKENS == 32_000
    assert m.status(0) == "ok"
    assert m.status(int(0.72 * m.limit)) == "warn"
    assert m.status(int(0.90 * m.limit)) == "fold"
    assert m.status(m.limit) == "stop"
    assert m.should_fold(int(0.90 * m.limit)) is True
    assert m.should_fold(0) is False


def test_tool_result_is_evidence_first() -> None:
    r = ToolResult(findings="found the bug")
    assert r.evidence == []
    assert r.unknowns == []
    assert r.ok is True


def test_config_defaults_encode_invariants_and_round_trip(tmp_path: Path) -> None:
    cfg = KeelConfig()
    assert cfg.context_budget_tokens == 32_000
    assert cfg.memory_budget_gb == 48.0

    cfg.serving.model_id = "qwen3.6-35b-a3b"
    cfg.serving.quantization = "int4"
    path = tmp_path / "config.json"
    cfg.save(path)

    loaded = KeelConfig.load(path)
    assert loaded.serving.model_id == "qwen3.6-35b-a3b"
    assert loaded.serving.quantization == "int4"
    assert loaded.context_budget_tokens == 32_000


def test_trace_store_round_trip(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.jsonl")
    rec = TraceRecord(
        task_id="t1",
        model_id="qwen3.6-35b-a3b",
        prompt="fix the failing test",
        outcome="pass",
        tool_calls=[ToolCall(name="shell", arguments={"cmd": "pytest"}, findings="2 passed")],
        context_folds=[
            ContextFold(
                reason="turn complete", tokens_before=30000, tokens_after=1200, summary="ok"
            )
        ],
        verifier_scores={"tests": 1.0},
        metrics={"tokens": 1234, "wall_clock_s": 5.0},
    )
    store.append(rec)

    back = list(store)
    assert len(back) == 1
    assert back[0].task_id == "t1"
    assert back[0].tool_calls[0].findings == "2 passed"
    assert back[0].context_folds[0].tokens_after == 1200
    assert back[0].verifier_scores["tests"] == 1.0
