"""Keel — a strictly-local, small-model coding-agent framework.

The harness, not the parameter count, is what keeps a small model upright. Nothing
in this package may make a network/API call on the inference path: escalation means a
bigger *local* model, more *local* samples, or asking the user.
"""

from __future__ import annotations

__version__ = "0.0.1"
__all__ = ["__version__"]
