"""A single evaluation case: a task the agent runs in a fresh workspace, plus how to score it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    case_id: str
    benchmark: str
    instructions: str  # the prompt handed to the agent
    source_dir: str | None = None  # copied into the workspace before the agent runs
    seed_files: dict[str, str] = field(default_factory=dict)  # extra files: relpath -> content
    setup_commands: list[str] = field(default_factory=list)  # shell run in the workspace pre-agent
    # Files restored from source_dir just before verification, so the agent cannot game the
    # ground-truth tests by editing them.
    protected_paths: list[str] = field(default_factory=list)
    verify_command: str = ""  # execution-based ground truth; exit 0 == pass
    verify_timeout_s: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)
