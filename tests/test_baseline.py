from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from keel.baseline import (
    BaselineAgent,
    BashAction,
    BashTool,
    ParseError,
    SubmitAction,
    parse_action,
)
from keel.serving.backends.base import GenerationResult, InferenceBackend


class ScriptedBackend(InferenceBackend):
    """A fake backend that replays a fixed list of completions, for testing the loop."""

    name = "scripted"

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        text = self._responses.pop(0)
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
        )

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> Iterator[str]:
        yield self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, grammar=grammar
        ).text


# --- action parser ---------------------------------------------------------


def test_parse_bash_action() -> None:
    action = parse_action("thinking...\n```bash\nls -la\n```")
    assert isinstance(action, BashAction)
    assert action.command == "ls -la"


def test_parse_submit_action() -> None:
    action = parse_action("```submit\nall done\n```")
    assert isinstance(action, SubmitAction)
    assert action.message == "all done"


def test_parse_takes_last_block() -> None:
    action = parse_action("```bash\necho 1\n```\nand then\n```submit\nfin\n```")
    assert isinstance(action, SubmitAction)


def test_parse_no_block_raises() -> None:
    with pytest.raises(ParseError):
        parse_action("just prose, no code block at all")


# --- bash tool -------------------------------------------------------------


def test_bash_tool_executes_and_reports_exit_code(tmp_path: Path) -> None:
    tool = BashTool(workspace=tmp_path)
    ok = tool.run("echo hi")
    assert ok.exit_code == 0
    assert "hi" in ok.stdout
    assert tool.run("exit 3").exit_code == 3


def test_bash_tool_times_out(tmp_path: Path) -> None:
    tool = BashTool(workspace=tmp_path, timeout_s=0.2)
    obs = tool.run("sleep 5")
    assert obs.timed_out is True
    assert obs.exit_code == 124


# --- full loop -------------------------------------------------------------


def test_agent_runs_command_then_submits(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        [
            "I'll create the file.\n```bash\necho hello > out.txt && cat out.txt\n```",
            "Confirmed.\n```submit\nWrote 'hello' to out.txt\n```",
        ]
    )
    agent = BaselineAgent(backend, workspace=tmp_path, max_steps=5)
    result = agent.run("create out.txt containing hello", task_id="t1", model_id="fake")

    assert result.outcome == "submitted"
    assert result.steps == 2
    assert "Wrote 'hello'" in result.final_message
    assert (tmp_path / "out.txt").read_text().strip() == "hello"

    assert result.trace is not None
    assert len(result.trace.tool_calls) == 1
    assert result.trace.tool_calls[0].ok is True
    assert result.trace.metrics["tool_call_error_rate"] == 0.0


def test_agent_hits_max_steps_on_persistent_bad_output(tmp_path: Path) -> None:
    backend = ScriptedBackend(["no block", "still none", "nope"])
    agent = BaselineAgent(backend, workspace=tmp_path, max_steps=3)
    result = agent.run("do something")

    assert result.outcome == "max_steps"
    assert result.steps == 3
    assert result.trace is not None
    assert result.trace.metrics["parse_errors"] == 3
    assert result.trace.metrics["tool_call_error_rate"] == 1.0
