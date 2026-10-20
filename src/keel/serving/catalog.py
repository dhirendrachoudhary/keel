"""Seed catalog of local models the picker chooses from (M1).

Each entry carries just enough to budget memory and estimate decode speed: total vs active
params (active < total for MoE), the architecture fields the KV-cache formula needs, and the
MLX model id per quant. These specs are best-effort estimates meant to be refined — verify a
model id and its dimensions before relying on the pick; the picker logic is the real product.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantOption:
    name: str  # "int4" | "int8" | "bf16"
    bits_per_weight: float  # effective bits incl. quantization overhead (int4 ~= 4.5)
    model_id: str  # the MLX model id for this quant


@dataclass(frozen=True)
class ModelSpec:
    key: str
    total_params_b: float
    active_params_b: float  # == total for dense; the few active experts for MoE
    is_moe: bool
    context_length: int
    n_layers: int
    n_kv_heads: int  # KV heads (GQA), which drive KV-cache size
    head_dim: int
    quants: tuple[QuantOption, ...]
    has_mtp: bool = False  # native MTP heads -> MTPLX speculative decoding (later)
    notes: str = ""


# Ordered roughly largest-to-smallest. The picker, not this order, decides the winner.
DEFAULT_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="Qwen3-30B-A3B",
        total_params_b=30.5,
        active_params_b=3.3,
        is_moe=True,
        context_length=32768,
        n_layers=48,
        n_kv_heads=4,
        head_dim=128,
        quants=(
            QuantOption("int4", 4.5, "mlx-community/Qwen3-30B-A3B-4bit"),
            QuantOption("int8", 8.5, "mlx-community/Qwen3-30B-A3B-8bit"),
        ),
        notes="MoE, ~3B active — the v1 default when it fits (fast per unit of capability).",
    ),
    ModelSpec(
        key="Qwen2.5-Coder-32B",
        total_params_b=32.5,
        active_params_b=32.5,
        is_moe=False,
        context_length=32768,
        n_layers=64,
        n_kv_heads=8,
        head_dim=128,
        quants=(QuantOption("int4", 4.5, "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"),),
    ),
    ModelSpec(
        key="Qwen2.5-Coder-14B",
        total_params_b=14.7,
        active_params_b=14.7,
        is_moe=False,
        context_length=32768,
        n_layers=48,
        n_kv_heads=8,
        head_dim=128,
        quants=(
            QuantOption("int4", 4.5, "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"),
            QuantOption("int8", 8.5, "mlx-community/Qwen2.5-Coder-14B-Instruct-8bit"),
        ),
    ),
    ModelSpec(
        key="Qwen2.5-Coder-7B",
        total_params_b=7.6,
        active_params_b=7.6,
        is_moe=False,
        context_length=32768,
        n_layers=28,
        n_kv_heads=4,
        head_dim=128,
        quants=(QuantOption("int4", 4.5, "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),),
    ),
    ModelSpec(
        key="Qwen2.5-Coder-3B",
        total_params_b=3.1,
        active_params_b=3.1,
        is_moe=False,
        context_length=32768,
        n_layers=36,
        n_kv_heads=2,
        head_dim=128,
        quants=(QuantOption("int4", 4.5, "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"),),
    ),
    ModelSpec(
        key="Qwen2.5-0.5B",
        total_params_b=0.5,
        active_params_b=0.5,
        is_moe=False,
        context_length=32768,
        n_layers=24,
        n_kv_heads=2,
        head_dim=64,
        quants=(QuantOption("int4", 4.5, "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),),
        notes="Tiny — for smoke tests and low-memory machines.",
    ),
)


def find_by_model_id(model_id: str) -> tuple[ModelSpec, QuantOption] | None:
    """Look up a (spec, quant) in the catalog by its MLX model id, or None."""
    for spec in DEFAULT_CATALOG:
        for quant in spec.quants:
            if quant.model_id == model_id:
                return spec, quant
    return None
