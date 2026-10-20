"""M1 auto-tuner: benchmark candidate configs (MTP depth, KV quant, batch size);
keep the fastest that passes a quality check. Speculative sampling stays lossless at
real temperatures (Leviathan / Chen 2023)."""

from __future__ import annotations

from keel.config import ServingProfile


def autotune() -> ServingProfile:
    """Search backend configs, benchmark each, return the fastest quality-preserving one."""
    raise NotImplementedError("auto-tuner not implemented yet")
