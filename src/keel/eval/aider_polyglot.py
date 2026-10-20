"""Aider polyglot adapter: maps the polyglot-benchmark repo into eval cases.

The benchmark is a set of Exercism exercises across languages. Clone it from
https://github.com/Aider-AI/polyglot-benchmark and pass the repo path. The layout is
``<repo>/<lang>/exercises/practice/<slug>/`` with ``.docs/instructions.md``, a solution
stub, and a test file.

v1 supports Python (scored with pytest); other languages are skipped until their runners
are wired. This is the cheap, fast regression check.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from keel.eval.benchmark import Benchmark
from keel.eval.case import EvalCase

# Execution-based ground truth per language.
_LANG_VERIFY: dict[str, str] = {
    "python": "python -m pytest -q",
}


class AiderPolyglotBenchmark(Benchmark):
    name = "aider-polyglot"

    def __init__(self, repo_path: Path | str, *, languages: Sequence[str] = ("python",)) -> None:
        self.repo_path = Path(repo_path)
        self.languages = [lang for lang in languages if lang in _LANG_VERIFY]

    def cases(self) -> Iterator[EvalCase]:
        for lang in self.languages:
            practice = self.repo_path / lang / "exercises" / "practice"
            if not practice.is_dir():
                continue
            for exercise in sorted(p for p in practice.iterdir() if p.is_dir()):
                instructions = self._instructions(exercise)
                if not instructions:
                    continue
                protected = [p.name for p in sorted(exercise.glob("*_test.py"))]
                yield EvalCase(
                    case_id=f"{lang}/{exercise.name}",
                    benchmark=self.name,
                    instructions=instructions,
                    source_dir=str(exercise),
                    protected_paths=protected,
                    verify_command=_LANG_VERIFY[lang],
                    metadata={"language": lang, "exercise": exercise.name},
                )

    @staticmethod
    def _instructions(exercise: Path) -> str:
        docs = exercise / ".docs"
        parts: list[str] = []
        for name in ("introduction.md", "instructions.md", "instructions.append.md"):
            doc = docs / name
            if doc.is_file():
                parts.append(doc.read_text())
        body = "\n\n".join(parts).strip()
        if not body:
            return ""
        return (
            body + "\n\nImplement the solution in the existing stub file(s). Do not edit the test "
            "files. Submit when the tests would pass."
        )
