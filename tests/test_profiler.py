from __future__ import annotations

import sys

import pytest

from keel.serving.profiler import (
    HardwareProfile,
    _chip_bandwidth_gbps,
    _parse_cpu_speed_limit,
    _parse_low_power_mode,
    detect,
)


def test_parse_low_power_mode() -> None:
    assert _parse_low_power_mode("lowpowermode         1") is True
    assert _parse_low_power_mode("lowpowermode         0") is False
    assert _parse_low_power_mode("no such field") is False


def test_parse_cpu_speed_limit() -> None:
    assert _parse_cpu_speed_limit("CPU_Speed_Limit = 100") == 100
    assert _parse_cpu_speed_limit("CPU_Speed_Limit \t= 70") == 70
    assert _parse_cpu_speed_limit("no thermal data") == 100


def test_chip_bandwidth_lookup() -> None:
    assert _chip_bandwidth_gbps("Apple M4 Max") == 546.0
    assert _chip_bandwidth_gbps("Apple M2") == 100.0
    assert _chip_bandwidth_gbps("Apple M1 Ultra") == 800.0
    assert _chip_bandwidth_gbps("Intel Core i9") is None


def test_profile_derived_properties() -> None:
    p = HardwareProfile(unified_memory_gb=64.0, cpu_speed_limit_pct=80)
    assert p.usable_model_memory_gb == 48.0
    assert p.thermal_throttled is True
    assert HardwareProfile(cpu_speed_limit_pct=100).thermal_throttled is False


@pytest.mark.skipif(sys.platform != "darwin", reason="profiler shells out to macOS tools")
def test_detect_smoke() -> None:
    p = detect()
    assert p.unified_memory_gb > 0
    assert p.free_disk_gb > 0
    assert isinstance(p.thermal_throttled, bool)
