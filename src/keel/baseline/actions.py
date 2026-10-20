"""Action protocol for the plain baseline loop.

Each assistant turn must contain exactly one fenced code block whose info string marks
the action: a ``bash`` block (a shell command to run) or a ``submit`` block (the final
answer). The last fenced block in the message is taken as the action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
_BASH_LANGS = {"bash", "sh", "shell"}
_SUBMIT_LANGS = {"submit", "finish"}


class ParseError(ValueError):
    """Raised when an assistant message has no single valid action block."""


@dataclass
class BashAction:
    command: str


@dataclass
class SubmitAction:
    message: str


Action = BashAction | SubmitAction


def parse_action(text: str) -> Action:
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        raise ParseError("no fenced ```bash or ```submit block found")
    lang, body = blocks[-1]  # the action is the last fenced block
    lang = lang.lower()
    body = body.strip()
    if lang in _BASH_LANGS:
        if not body:
            raise ParseError("empty bash command")
        return BashAction(command=body)
    if lang in _SUBMIT_LANGS:
        return SubmitAction(message=body)
    raise ParseError(f"unknown action block language: {lang!r}")
