from __future__ import annotations

import json
from pathlib import Path

from keel.eval import AiderPolyglotBenchmark, SWEBenchVerified


def _make_polyglot_repo(root: Path) -> Path:
    ex = root / "python" / "exercises" / "practice" / "hello-world"
    (ex / ".docs").mkdir(parents=True)
    (ex / ".docs" / "instructions.md").write_text("# Hello World\n\nReturn the greeting.")
    (ex / "hello_world.py").write_text("def hello():\n    pass\n")
    (ex / "hello_world_test.py").write_text("def test_hello():\n    assert True\n")
    return root


def test_aider_polyglot_maps_python_exercises(tmp_path: Path) -> None:
    repo = _make_polyglot_repo(tmp_path)
    cases = list(AiderPolyglotBenchmark(repo).cases())
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "python/hello-world"
    assert case.benchmark == "aider-polyglot"
    assert "Hello World" in case.instructions
    assert "Do not edit the test files" in case.instructions
    assert case.verify_command == "python -m pytest -q"
    assert case.protected_paths == ["hello_world_test.py"]
    assert case.metadata["language"] == "python"


def test_aider_polyglot_skips_unknown_languages(tmp_path: Path) -> None:
    _make_polyglot_repo(tmp_path)
    # rust is not wired yet -> no cases, no crash
    assert list(AiderPolyglotBenchmark(tmp_path, languages=("rust",)).cases()) == []


def test_swebench_loads_instance_from_jsonl(tmp_path: Path) -> None:
    instance = {
        "instance_id": "proj__repo-123",
        "repo": "acme/widget",
        "base_commit": "abc123",
        "problem_statement": "Fix the crash when the list is empty.",
        "test_patch": "diff --git a/t.py b/t.py\n",
        "FAIL_TO_PASS": json.dumps(["tests/test_x.py::test_empty"]),
        "PASS_TO_PASS": ["tests/test_x.py::test_basic"],
    }
    jsonl = tmp_path / "swe.jsonl"
    jsonl.write_text(json.dumps(instance) + "\n")

    cases = list(SWEBenchVerified(jsonl).cases())
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "proj__repo-123"
    assert "Fix the crash" in case.instructions
    assert any(
        "git clone --quiet https://github.com/acme/widget widget" in c for c in case.setup_commands
    )
    assert any("checkout --quiet abc123" in c for c in case.setup_commands)
    assert any("git apply" in c for c in case.setup_commands)  # test patch applied
    assert "tests/test_x.py::test_empty" in case.verify_command
    assert case.metadata["fail_to_pass"] == ["tests/test_x.py::test_empty"]
    assert case.metadata["pass_to_pass"] == ["tests/test_x.py::test_basic"]
