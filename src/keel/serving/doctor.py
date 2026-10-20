"""M1 ``keel doctor``: diagnose common causes of slow local inference."""

from __future__ import annotations

# Known slowness causes doctor must catch:
KNOWN_CAUSES: tuple[str, ...] = (
    "dense model where an MoE would fit",
    "GGUF on an old Metal path instead of MLX",
    "memory pressure and swap",
    "Low Power Mode or thermal throttling",
    "prefill-bound runs: huge prompts re-sent each turn with no prefix cache",
    "runaway thinking tokens counted as slowness",
)


def diagnose() -> list[str]:
    """Return the subset of KNOWN_CAUSES detected on this machine/run."""
    raise NotImplementedError("keel doctor not implemented yet")
