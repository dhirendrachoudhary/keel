"""SWE-bench Verified adapter: loads instances from a local JSONL file into eval cases.

Each line is one instance dict with at least: instance_id, repo, base_commit,
problem_statement, test_patch, FAIL_TO_PASS, PASS_TO_PASS. Export the dataset to JSONL
first, e.g. with the `datasets` library:

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    ds.to_json("swebench_verified.jsonl")

EXECUTION IS EXPERIMENTAL. A case checks out the repo at ``base_commit``, applies the test
patch, and runs the FAIL_TO_PASS tests. Real SWE-bench scoring needs each repo's exact
environment — the official harness builds per-repo Docker images — which this adapter does
not provision yet, so cases only run where the toolchain and dependencies already exist.
Use it today to load and inspect instances; full containerised scoring is not implemented
yet. Cloning happens over the network at setup time (a dataset/repo fetch, not model
inference, so it does not break the strictly-local-at-inference rule).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from keel.eval.benchmark import Benchmark
from keel.eval.case import EvalCase


class SWEBenchVerified(Benchmark):
    name = "swebench-verified"

    def __init__(self, jsonl_path: Path | str) -> None:
        self.jsonl_path = Path(jsonl_path)

    def cases(self) -> Iterator[EvalCase]:
        for line in self.jsonl_path.read_text().splitlines():
            if line.strip():
                yield self._to_case(json.loads(line))

    @staticmethod
    def _to_case(inst: dict[str, Any]) -> EvalCase:
        repo = str(inst["repo"])
        base = str(inst["base_commit"])
        repo_dir = repo.split("/")[-1]
        fail_to_pass = _as_list(inst.get("FAIL_TO_PASS", []))
        test_patch = str(inst.get("test_patch", ""))

        setup = [
            f"git clone --quiet https://github.com/{repo} {repo_dir}",
            f"git -C {repo_dir} checkout --quiet {base}",
        ]
        if test_patch:
            setup.append(f"cd {repo_dir} && git apply - <<'KEEL_PATCH'\n{test_patch}\nKEEL_PATCH")

        verify = ""
        if fail_to_pass:
            verify = f"cd {repo_dir} && python -m pytest -q {' '.join(fail_to_pass)}"

        return EvalCase(
            case_id=str(inst["instance_id"]),
            benchmark="swebench-verified",
            instructions=str(inst.get("problem_statement", "")),
            setup_commands=setup,
            verify_command=verify,
            verify_timeout_s=float(inst.get("verify_timeout_s", 900.0)),
            metadata={
                "repo": repo,
                "base_commit": base,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": _as_list(inst.get("PASS_TO_PASS", [])),
            },
        )


def _as_list(value: Any) -> list[str]:
    # SWE-bench often stores these fields as a JSON-encoded string.
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(x) for x in parsed] if isinstance(parsed, list) else [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return []
