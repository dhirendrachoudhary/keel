from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

import keel.serving.backends as backends
import keel.serving.backends.mlx as mlxmod
from keel.baseline.agent import _first_turn
from keel.cli import main
from keel.serving.backends import MLXBackend, create_backend
from keel.serving.backends.base import GenerationResult, InferenceBackend


@dataclass
class _Resp:
    """Stand-in for an mlx-lm GenerationResponse."""

    text: str
    prompt_tokens: int
    generation_tokens: int


# --- factory ---------------------------------------------------------------


def test_create_backend_returns_mlx() -> None:
    assert isinstance(create_backend("mlx-lm", "some/model"), MLXBackend)
    assert isinstance(create_backend("mlx", "some/model"), MLXBackend)


def test_create_backend_unknown_raises() -> None:
    with pytest.raises(NotImplementedError):
        create_backend("omlx", "some/model")


# --- backend (mlx-lm calls mocked; no weights downloaded) ------------------


def test_generate_accumulates_text_and_uses_mlx_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mlxmod, "_load", lambda model_id: ("MODEL", "TOK"))

    def fake_stream(
        model: object, tokenizer: object, prompt: str, max_tokens: int, temperature: float
    ) -> Iterator[_Resp]:
        assert (model, tokenizer) == ("MODEL", "TOK")
        yield _Resp(text="hel", prompt_tokens=5, generation_tokens=1)
        yield _Resp(text="lo", prompt_tokens=5, generation_tokens=2)

    monkeypatch.setattr(mlxmod, "_stream_responses", fake_stream)

    result = MLXBackend("some/model").generate("hi", max_tokens=16, temperature=0.0)
    assert result.text == "hello"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2


def test_stream_yields_text_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlxmod, "_load", lambda model_id: ("M", "T"))
    monkeypatch.setattr(
        mlxmod,
        "_stream_responses",
        lambda *a, **k: iter([_Resp("a", 1, 1), _Resp("b", 1, 2)]),
    )
    assert list(MLXBackend("m").stream("x", max_tokens=8)) == ["a", "b"]


def test_generate_rejects_grammar_until_m2() -> None:
    with pytest.raises(NotImplementedError):
        MLXBackend("m").generate("x", max_tokens=8, grammar="{...}")


# --- raw-completion turn truncation ---------------------------------------


def test_first_turn_truncates_at_next_role_header() -> None:
    text = "```bash\nls\n```\nOBSERVATION:\nhallucinated\nASSISTANT:\nmore"
    assert _first_turn(text) == "```bash\nls\n```"


def test_first_turn_keeps_full_text_when_no_delimiter() -> None:
    assert _first_turn("just one turn") == "just one turn"


# --- `keel run` wiring (backend mocked) ------------------------------------


class _SubmitBackend(InferenceBackend):
    name = "fake"

    def generate(
        self, prompt: str, *, max_tokens: int, temperature: float = 0.0, grammar: str | None = None
    ) -> GenerationResult:
        return GenerationResult(
            text="done\n```submit\nall good\n```", prompt_tokens=1, completion_tokens=1
        )

    def stream(
        self, prompt: str, *, max_tokens: int, temperature: float = 0.0, grammar: str | None = None
    ) -> Iterator[str]:
        yield "x"


def test_run_executes_agent_and_writes_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backends, "create_backend", lambda name, model_id: _SubmitBackend())
    monkeypatch.chdir(tmp_path)

    code = main(["run", "do a thing", "--model", "fake/model", "--max-steps", "3"])
    out = capsys.readouterr().out

    assert code == 0
    assert "submitted" in out
    assert "all good" in out
    assert (tmp_path / "traces" / "keel.trace.jsonl").exists()


def test_run_without_model_or_config_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .keel/config.json here
    code = main(["run", "do a thing"])
    assert code == 2
    assert "no model configured" in capsys.readouterr().err


def test_eval_runs_local_suite_and_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A one-case local suite whose verify always passes, so we exercise the CLI/runner/report
    # path without a real model.
    suite = tmp_path / "suite" / "case1"
    suite.mkdir(parents=True)
    (suite / "case.json").write_text('{"instructions": "do it", "verify_command": "true"}')
    monkeypatch.setattr(backends, "create_backend", lambda name, model_id: _SubmitBackend())
    monkeypatch.chdir(tmp_path)

    code = main(
        ["eval", "--benchmark", "local", "--suite", str(tmp_path / "suite"), "--model", "fake"]
    )
    out = capsys.readouterr().out

    assert code == 0  # all (one) cases passed
    assert "1/1 passed" in out
    assert "PASS  case1" in out
