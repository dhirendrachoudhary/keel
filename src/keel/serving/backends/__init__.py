"""Inference backends, each hidden behind ``InferenceBackend`` (mlx-lm, oMLX, MTPLX)."""

from __future__ import annotations

from keel.serving.backends.base import ChatMessage, GenerationResult, InferenceBackend
from keel.serving.backends.mlx import MLXBackend

__all__ = ["ChatMessage", "GenerationResult", "InferenceBackend", "MLXBackend", "create_backend"]

_MLX_NAMES = {"mlx-lm", "mlx"}


def create_backend(name: str, model_id: str) -> InferenceBackend:
    """Build a backend by name. Only mlx-lm exists today; oMLX and MTPLX come later."""
    if name in _MLX_NAMES:
        return MLXBackend(model_id)
    raise NotImplementedError(
        f"backend {name!r} is not implemented yet (planned: oMLX, MTPLX). Use 'mlx-lm'."
    )
