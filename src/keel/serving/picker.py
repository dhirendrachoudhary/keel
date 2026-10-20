"""M1 model/quant picker.

Given a hardware profile, choose the (model, quant) that fits in the usable memory budget
with KV-cache headroom for several concurrent agents, preferring MoE (few active params) for
speed. The memory and speed formulas are deliberate heuristics — the benchmark/auto-tune step
(not built yet) will replace the estimates with measured numbers.

Policy: keep only pairs that fit; among those, prefer ones that clear a minimum decode speed;
within the fast set prefer MoE, then capability (total params), then the faster quant (more
headroom). If nothing clears the speed bar, take the fastest that fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from keel.config import CONTEXT_BUDGET_TOKENS
from keel.serving.catalog import DEFAULT_CATALOG, ModelSpec, QuantOption
from keel.serving.profiler import HardwareProfile

ACTIVATION_OVERHEAD_GB = 2.0  # framework + activation slack beyond weights and KV
KV_BYTES_PER_ELEM = 2  # fp16 KV cache
MEM_BW_EFFICIENCY = 0.3  # fraction of peak bandwidth realized at decode (rough)
MIN_DECODE_TOK_S = 20.0  # below this a run feels too slow to iterate on
DEFAULT_AGENTS = 3  # one planner + a couple of workers sharing the model


@dataclass
class MemoryBreakdown:
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return round(self.weights_gb + self.kv_cache_gb + self.overhead_gb, 2)


@dataclass
class Candidate:
    spec: ModelSpec
    quant: QuantOption
    memory: MemoryBreakdown
    est_decode_tok_s: float
    fits: bool
    fast_enough: bool


@dataclass
class PickResult:
    model_id: str
    quantization: str
    backend: str
    spec_key: str
    est_decode_tok_s: float
    memory: MemoryBreakdown
    rationale: str


def weights_gb(spec: ModelSpec, quant: QuantOption) -> float:
    return spec.total_params_b * quant.bits_per_weight / 8.0


def kv_cache_gb(spec: ModelSpec, context_tokens: int, n_agents: int) -> float:
    per_token_bytes = 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * KV_BYTES_PER_ELEM
    return per_token_bytes * context_tokens * n_agents / 1e9


def estimate_decode_tok_s(
    spec: ModelSpec, quant: QuantOption, bandwidth_gbps: float | None
) -> float:
    if not bandwidth_gbps:
        return 0.0
    active_gb = spec.active_params_b * quant.bits_per_weight / 8.0
    if active_gb <= 0:
        return 0.0
    return round(MEM_BW_EFFICIENCY * bandwidth_gbps / active_gb, 1)


def evaluate(
    spec: ModelSpec,
    quant: QuantOption,
    profile: HardwareProfile,
    *,
    context_tokens: int,
    n_agents: int,
) -> Candidate:
    memory = MemoryBreakdown(
        weights_gb=round(weights_gb(spec, quant), 2),
        kv_cache_gb=round(kv_cache_gb(spec, context_tokens, n_agents), 2),
        overhead_gb=ACTIVATION_OVERHEAD_GB,
    )
    speed = estimate_decode_tok_s(spec, quant, profile.memory_bandwidth_gbps)
    fits = memory.total_gb <= profile.usable_model_memory_gb
    # With no bandwidth to estimate from, don't gate on speed.
    fast_enough = speed >= MIN_DECODE_TOK_S if profile.memory_bandwidth_gbps else True
    return Candidate(spec, quant, memory, speed, fits, fast_enough)


def pick_model(
    profile: HardwareProfile,
    *,
    context_tokens: int = CONTEXT_BUDGET_TOKENS,
    n_agents: int = DEFAULT_AGENTS,
    catalog: tuple[ModelSpec, ...] = DEFAULT_CATALOG,
) -> PickResult | None:
    candidates = [
        evaluate(spec, quant, profile, context_tokens=context_tokens, n_agents=n_agents)
        for spec in catalog
        for quant in spec.quants
    ]
    fitting = [c for c in candidates if c.fits]
    if not fitting:
        return None

    fast = [c for c in fitting if c.fast_enough]
    if fast:
        best = max(
            fast,
            key=lambda c: (
                c.spec.is_moe,
                c.spec.total_params_b,
                c.est_decode_tok_s,
                c.quant.bits_per_weight,
            ),
        )
    else:
        best = max(
            fitting, key=lambda c: (c.est_decode_tok_s, c.spec.is_moe, c.spec.total_params_b)
        )

    return PickResult(
        model_id=best.quant.model_id,
        quantization=best.quant.name,
        backend="mlx-lm",
        spec_key=best.spec.key,
        est_decode_tok_s=best.est_decode_tok_s,
        memory=best.memory,
        rationale=_rationale(
            best,
            fast_selected=bool(fast),
            n_agents=n_agents,
            context_tokens=context_tokens,
            usable_gb=profile.usable_model_memory_gb,
        ),
    )


def _rationale(
    best: Candidate,
    *,
    fast_selected: bool,
    n_agents: int,
    context_tokens: int,
    usable_gb: float,
) -> str:
    kind = "MoE" if best.spec.is_moe else "dense"
    size = (
        f"{best.spec.active_params_b:g}B active"
        if best.spec.is_moe
        else f"{best.spec.total_params_b:g}B"
    )
    speed = (
        f"~{best.est_decode_tok_s:g} tok/s est"
        if best.est_decode_tok_s
        else "speed n/a (bandwidth unknown)"
    )
    headroom = round(usable_gb - best.memory.total_gb, 1)
    tail = (
        "" if fast_selected else " (nothing cleared the speed target; chose the fastest that fits)"
    )
    return (
        f"{best.spec.key} {best.quant.name} ({kind}, {size}): {speed}; "
        f"{best.memory.total_gb} GB of {usable_gb:g} GB usable "
        f"(weights {best.memory.weights_gb} + KV {best.memory.kv_cache_gb} for {n_agents} agents "
        f"@ {context_tokens // 1000}K + overhead {best.memory.overhead_gb}), "
        f"{headroom} GB headroom{tail}."
    )
