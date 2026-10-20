"""Stateless bash execution for the baseline loop (mini-swe-agent style).

Each action runs in a fresh ``subprocess.run`` — no shell state persists between steps
(a ``cd`` in one step does not carry to the next), so the model must write self-contained
commands. Scoped to a workspace dir with a timeout; real sandboxing is M9.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 10_000
TIMEOUT_EXIT_CODE = 124


@dataclass
class Observation:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    def render(self) -> str:
        header = f"exit_code: {self.exit_code}" + (" (timed out)" if self.timed_out else "")
        parts = [header]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n... [{dropped} chars truncated] ...\n{text[-half:]}"


class BashTool:
    def __init__(self, workspace: Path | str = ".", timeout_s: float = 30.0) -> None:
        self.workspace = Path(workspace)
        self.timeout_s = timeout_s

    def run(self, command: str) -> Observation:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return Observation(
                stdout=_truncate(_as_text(exc.stdout)),
                stderr=_truncate(_as_text(exc.stderr)),
                exit_code=TIMEOUT_EXIT_CODE,
                timed_out=True,
            )
        return Observation(
            stdout=_truncate(proc.stdout),
            stderr=_truncate(proc.stderr),
            exit_code=proc.returncode,
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
