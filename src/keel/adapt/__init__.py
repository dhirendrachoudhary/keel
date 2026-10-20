"""M6 — Harness auto-adaptation.

The harness tunes itself to whichever local model the user runs instead of us hand-tuning
prompts per model: a meta-agent searches harness variants (prompts, tool descriptions,
step structure) offline on a dev task set when a new model is installed, and stores the
winning profile next to the M1 serving profile.
"""

from __future__ import annotations
