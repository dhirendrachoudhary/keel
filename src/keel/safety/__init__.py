"""M9 — Safety & sandboxing.

A local agent with shell access and self-written skills must not damage the user's machine
or get hijacked by a skill: skill scanning before admission, workspace-scoped file access,
deny-by-default network for tools, dry-run mode for destructive commands, command approval
with learned safe-command memory, and config backups before writing.
"""

from __future__ import annotations
