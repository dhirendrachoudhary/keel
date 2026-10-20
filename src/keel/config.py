"""The single config the Keel core reads. Written by ``keel setup``; users never edit it.

Defaults encode two hard invariants: a 32K active-token budget per agent and a ~48GB
resident-model memory budget on a 64GB M4 Max.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(".keel")
CONFIG_PATH = CONFIG_DIR / "config.json"

CONTEXT_BUDGET_TOKENS = 32_000
MEMORY_BUDGET_GB = 48.0


@dataclass
class ServingProfile:
    """Chosen by M1 ``setup``: which local model/quant/backend to serve, and how fast."""

    model_id: str = ""
    quantization: str = ""  # e.g. "int4"
    backend: str = ""  # "mlx-lm" | "omlx" | "mtplx"
    mtp_depth: int = 0  # 0 = no speculative decoding
    kv_quantization: str = ""
    decode_tok_s: float = 0.0
    prefill_tok_s: float = 0.0
    ttft_ms_8k: float = 0.0
    ttft_ms_32k: float = 0.0


@dataclass
class KeelConfig:
    version: int = 1
    context_budget_tokens: int = CONTEXT_BUDGET_TOKENS
    memory_budget_gb: float = MEMORY_BUDGET_GB
    serving: ServingProfile = field(default_factory=ServingProfile)
    harness_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> KeelConfig:
        data: dict[str, Any] = json.loads(path.read_text())
        serving = ServingProfile(**data.get("serving", {}))
        return cls(
            version=data.get("version", 1),
            context_budget_tokens=data.get("context_budget_tokens", CONTEXT_BUDGET_TOKENS),
            memory_budget_gb=data.get("memory_budget_gb", MEMORY_BUDGET_GB),
            serving=serving,
            harness_profile=data.get("harness_profile", {}),
        )
