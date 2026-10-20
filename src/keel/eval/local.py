"""Local benchmark: a directory of self-contained cases.

Each subdirectory with a ``case.json`` manifest is a case; every other file in it is seeded
into the workspace. This is the format the offline sample suite (``evals/samples``) uses,
and the quickest way to add a regression case.

Manifest keys (all optional except a way to score):
  instructions        inline task text
  instructions_file   or read the task text from this file in the case dir
  verify_command      shell; exit 0 == pass
  verify_timeout_s    default 120
  protected_paths     files restored from the case dir before verifying
  setup_commands      shell run in the workspace before the agent
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from keel.eval.benchmark import Benchmark
from keel.eval.case import EvalCase


class LocalBenchmark(Benchmark):
    name = "local"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def cases(self) -> Iterator[EvalCase]:
        for case_dir in sorted(p for p in self.path.iterdir() if (p / "case.json").is_file()):
            manifest = json.loads((case_dir / "case.json").read_text())
            instructions = manifest.get("instructions", "")
            if not instructions and manifest.get("instructions_file"):
                instructions = (case_dir / manifest["instructions_file"]).read_text()
            yield EvalCase(
                case_id=case_dir.name,
                benchmark=self.name,
                instructions=instructions,
                source_dir=str(case_dir),
                setup_commands=list(manifest.get("setup_commands", [])),
                protected_paths=list(manifest.get("protected_paths", [])),
                verify_command=manifest.get("verify_command", ""),
                verify_timeout_s=float(manifest.get("verify_timeout_s", 120.0)),
                metadata={"source": str(case_dir)},
            )
