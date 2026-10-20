"""Agent tools (shell, edit, search, test runner, git, MCP client), all returning
an evidence-first ``ToolResult``."""

from __future__ import annotations

from keel.loop.tools.base import Tool, ToolResult

__all__ = ["Tool", "ToolResult"]
