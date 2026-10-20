"""M1 hardware profiler: chip, unified memory, bandwidth, free disk, power/thermal mode.

The pure parsers (``_parse_*``, ``_chip_bandwidth_gbps``) are kept separate from the
command runners so they can be unit-tested without the host. ``detect()`` shells out to
``sysctl`` / ``pmset`` (macOS) and tolerates missing fields.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

# Published Apple-Silicon unified-memory bandwidth (GB/s), best-effort by chip tier.
_BANDWIDTH_GBPS: dict[str, float] = {
    "m1": 68.0,
    "m1 pro": 200.0,
    "m1 max": 400.0,
    "m1 ultra": 800.0,
    "m2": 100.0,
    "m2 pro": 200.0,
    "m2 max": 400.0,
    "m2 ultra": 800.0,
    "m3": 100.0,
    "m3 pro": 150.0,
    "m3 max": 400.0,
    "m3 ultra": 800.0,
    "m4": 120.0,
    "m4 pro": 273.0,
    "m4 max": 546.0,
}

# Fraction of unified memory reserved for the OS and non-model processes. On a 64GB
# machine this leaves ~48GB for resident models + KV caches (the scope-doc budget).
OS_RESERVE_FRACTION = 0.25


@dataclass
class HardwareProfile:
    chip: str = ""
    is_apple_silicon: bool = False
    unified_memory_gb: float = 0.0
    performance_cores: int = 0
    efficiency_cores: int = 0
    memory_bandwidth_gbps: float | None = None
    free_disk_gb: float = 0.0
    low_power_mode: bool = False
    cpu_speed_limit_pct: int = 100  # 100 = not thermally throttled

    @property
    def thermal_throttled(self) -> bool:
        return self.cpu_speed_limit_pct < 100

    @property
    def usable_model_memory_gb(self) -> float:
        """Memory available for resident models + KV caches, after OS headroom."""
        return round(self.unified_memory_gb * (1.0 - OS_RESERVE_FRACTION), 1)


def _chip_bandwidth_gbps(chip: str) -> float | None:
    match = re.search(r"m(\d+)\s*(pro|max|ultra)?", chip.lower())
    if not match:
        return None
    key = f"m{match.group(1)}"
    if match.group(2):
        key += f" {match.group(2)}"
    return _BANDWIDTH_GBPS.get(key)


def _parse_low_power_mode(pmset_output: str) -> bool:
    match = re.search(r"lowpowermode\s+(\d+)", pmset_output)
    return bool(match and match.group(1) != "0")


def _parse_cpu_speed_limit(pmset_therm_output: str) -> int:
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", pmset_therm_output)
    return int(match.group(1)) if match else 100


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key]).strip()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect() -> HardwareProfile:
    """Probe the machine to build a HardwareProfile (macOS / Apple Silicon)."""
    chip = _sysctl("machdep.cpu.brand_string")
    memsize = _sysctl("hw.memsize")
    perf = _sysctl("hw.perflevel0.physicalcpu")
    eff = _sysctl("hw.perflevel1.physicalcpu")
    try:
        free_disk_gb = round(shutil.disk_usage("/").free / (1024**3), 1)
    except OSError:
        free_disk_gb = 0.0

    return HardwareProfile(
        chip=chip,
        is_apple_silicon=chip.lower().startswith("apple"),
        unified_memory_gb=round(int(memsize) / (1024**3), 1) if memsize.isdigit() else 0.0,
        performance_cores=int(perf) if perf.isdigit() else 0,
        efficiency_cores=int(eff) if eff.isdigit() else 0,
        memory_bandwidth_gbps=_chip_bandwidth_gbps(chip),
        free_disk_gb=free_disk_gb,
        low_power_mode=_parse_low_power_mode(_run(["pmset", "-g"])),
        cpu_speed_limit_pct=_parse_cpu_speed_limit(_run(["pmset", "-g", "therm"])),
    )
