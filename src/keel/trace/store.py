"""Append-only JSONL trace store (M8)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from keel.trace.schema import ContextFold, ToolCall, TraceRecord


class TraceStore:
    def __init__(self, path: Path | str = Path("traces/keel.trace.jsonl")) -> None:
        self.path = Path(path)

    def append(self, record: TraceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

    def __iter__(self) -> Iterator[TraceRecord]:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            data["tool_calls"] = [ToolCall(**tc) for tc in data.get("tool_calls", [])]
            data["context_folds"] = [ContextFold(**cf) for cf in data.get("context_folds", [])]
            yield TraceRecord(**data)
