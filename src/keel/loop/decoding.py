"""M2 constrained decoding: wire XGrammar structural tags into the MLX sampler.

Free text until a tool trigger, then schema-locked arguments. Constrained decoding
removes *structural* tool-call failures in small (0.6B–4B) models but not *semantic*
ones, so it is always paired with argument validation and retries.
"""

from __future__ import annotations

from typing import Any


def grammar_from_schema(schema: dict[str, Any]) -> str:
    """Compile a JSON-schema of a tool's arguments into an XGrammar spec string."""
    raise NotImplementedError("constrained decoding not implemented yet")
