"""``keel doctor``: diagnose common causes of slow local inference.

``diagnose`` is pure over its inputs (a hardware profile, an optional config, and an optional
measured swap figure), so it is unit-testable without the host; the CLI gathers those inputs
and calls it. It catches the configuration- and system-level causes that can be checked at
rest; prefill-bound prompts and runaway reasoning tokens only show up in a run, so those are
diagnosed from a trace, not here.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from keel.config import CONTEXT_BUDGET_TOKENS, KeelConfig
from keel.serving.catalog import find_by_model_id
from keel.serving.picker import DEFAULT_AGENTS, evaluate, pick_model
from keel.serving.profiler import HardwareProfile

# The slowness causes doctor aims to catch.
KNOWN_CAUSES: tuple[str, ...] = (
    "dense model where an MoE would fit",
    "GGUF on an old Metal path instead of MLX",
    "memory pressure and swap",
    "Low Power Mode or thermal throttling",
    "prefill-bound runs: huge prompts re-sent each turn with no prefix cache",
    "runaway thinking tokens counted as slowness",
)

SWAP_WARN_GB = 1.0
SWAP_CRITICAL_GB = 4.0

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    severity: str  # "critical" | "warning" | "info"
    cause: str
    detail: str
    remedy: str


def diagnose(
    profile: HardwareProfile,
    config: KeelConfig | None = None,
    *,
    swap_used_gb: float | None = None,
) -> list[Finding]:
    findings: list[Finding] = []

    if profile.low_power_mode:
        findings.append(
            Finding(
                "warning",
                "low-power-mode",
                "macOS Low Power Mode is on, which throttles the CPU and GPU.",
                "Turn off Low Power Mode (System Settings > Battery) and run on AC power.",
            )
        )

    if profile.thermal_throttled:
        findings.append(
            Finding(
                "warning",
                "thermal-throttling",
                f"CPU speed is limited to {profile.cpu_speed_limit_pct}% under thermal pressure.",
                "Let the machine cool, improve ventilation, and close heavy background apps.",
            )
        )

    if swap_used_gb is not None and swap_used_gb >= SWAP_WARN_GB:
        severity = "critical" if swap_used_gb >= SWAP_CRITICAL_GB else "warning"
        findings.append(
            Finding(
                severity,
                "memory-pressure",
                f"{swap_used_gb:g} GB of swap in use — memory pressure destroys decode throughput.",
                "Run a smaller model/quant, or fewer concurrent agents, via `keel setup`.",
            )
        )

    if config is not None and config.serving.model_id:
        findings.extend(_diagnose_config(profile, config))

    return findings


def _diagnose_config(profile: HardwareProfile, config: KeelConfig) -> list[Finding]:
    out: list[Finding] = []
    serving = config.serving
    model_id = serving.model_id

    on_mlx = (not serving.backend or serving.backend == "mlx-lm") and "gguf" not in model_id.lower()
    if not on_mlx:
        out.append(
            Finding(
                "warning",
                "not-on-mlx",
                f"Configured runtime ({serving.backend or 'unknown'}, {model_id}) is not the MLX "
                "path; GGUF on an old Metal path is much slower on Apple Silicon.",
                "Use an MLX (mlx-community) model on the mlx-lm backend via `keel setup`.",
            )
        )

    found = find_by_model_id(model_id)
    if found is None:
        out.append(
            Finding(
                "info",
                "unknown-model",
                f"Configured model {model_id} is not in the catalog; memory fit can't be checked.",
                "Add it to keel/serving/catalog.py, or run `keel setup` to pick a known model.",
            )
        )
        return out

    spec, quant = found
    memory = evaluate(
        spec, quant, profile, context_tokens=CONTEXT_BUDGET_TOKENS, n_agents=DEFAULT_AGENTS
    ).memory
    usable = profile.usable_model_memory_gb
    if memory.total_gb > usable:
        out.append(
            Finding(
                "critical",
                "over-budget",
                f"{spec.key} {quant.name} needs ~{memory.total_gb} GB but only ~{usable:g} GB is "
                "usable; the overflow swaps and collapses throughput.",
                "Pick a smaller model/quant or fewer agents via `keel setup`.",
            )
        )

    if not spec.is_moe:
        recommended = pick_model(profile)
        rec_spec = find_by_model_id(recommended.model_id) if recommended else None
        if recommended is not None and rec_spec is not None and rec_spec[0].is_moe:
            out.append(
                Finding(
                    "warning",
                    "dense-where-moe-fits",
                    f"{spec.key} is dense; the MoE {rec_spec[0].key} fits your budget and is much "
                    "faster (few active params). This is the most common cause of low tok/s.",
                    f"Switch via `keel setup` (recommends {recommended.model_id}).",
                )
            )

    return out


def probe_swap_gb() -> float | None:
    """Read current swap usage (macOS `vm.swapusage`), in GB, or None if unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_swap_used_gb(out)


def _parse_swap_used_gb(swapusage_output: str) -> float | None:
    # e.g. "total = 2048.00M  used = 123.00M  free = 1925.00M  (encrypted)"
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swapusage_output)
    if not match:
        return None
    value = float(match.group(1))
    factor = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1.0}[match.group(2)]
    return round(value * factor, 2)
