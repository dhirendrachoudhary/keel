"""M2 — Agent loop & tool layer.

A tiny, auditable loop where a small model never emits a broken tool call: constrained
decoding (XGrammar structural tags) + argument validation + retries, a tool router that
exposes only 3–5 tools per step, and an evidence-first result contract for every tool.
"""

from __future__ import annotations
