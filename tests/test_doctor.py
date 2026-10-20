from __future__ import annotations

from pathlib import Path

import pytest

import keel.serving.doctor as doctor_mod
import keel.serving.profiler as profiler
from keel.cli import main
from keel.config import KeelConfig, ServingProfile
from keel.serving.doctor import _parse_swap_used_gb, diagnose, probe_swap_gb
from keel.serving.profiler import HardwareProfile


def _healthy_64gb() -> HardwareProfile:
    return HardwareProfile(
        chip="Apple M4 Max",
        is_apple_silicon=True,
        unified_memory_gb=64.0,
        memory_bandwidth_gbps=546.0,
        low_power_mode=False,
        cpu_speed_limit_pct=100,
    )


def _config(model_id: str, backend: str = "mlx-lm") -> KeelConfig:
    return KeelConfig(
        serving=ServingProfile(model_id=model_id, quantization="int4", backend=backend)
    )


def _causes(profile: HardwareProfile, config: KeelConfig | None = None, **kw: float) -> set[str]:
    return {f.cause for f in diagnose(profile, config, **kw)}


# --- swap parsing ----------------------------------------------------------


def test_parse_swap_used_gb() -> None:
    assert _parse_swap_used_gb("total = 4096.00M  used = 0.00M  free = 4096.00M") == 0.0
    assert _parse_swap_used_gb("total = 4096.00M  used = 2048.00M  free = 2048.00M") == 2.0
    assert _parse_swap_used_gb("total = 8.00G  used = 3.00G  free = 5.00G") == 3.0
    assert _parse_swap_used_gb("no swap info") is None


def test_probe_swap_gb_returns_float_or_none() -> None:
    value = probe_swap_gb()
    assert value is None or isinstance(value, float)


# --- system-level findings -------------------------------------------------


def test_healthy_machine_has_no_findings() -> None:
    assert diagnose(_healthy_64gb(), None, swap_used_gb=0.0) == []


def test_low_power_and_thermal_are_flagged() -> None:
    profile = HardwareProfile(
        unified_memory_gb=64.0,
        memory_bandwidth_gbps=546.0,
        low_power_mode=True,
        cpu_speed_limit_pct=70,
    )
    causes = _causes(profile, swap_used_gb=0.0)
    assert "low-power-mode" in causes
    assert "thermal-throttling" in causes


def test_swap_severity() -> None:
    warn = diagnose(_healthy_64gb(), None, swap_used_gb=2.0)
    assert [f.severity for f in warn] == ["warning"]
    crit = diagnose(_healthy_64gb(), None, swap_used_gb=5.0)
    assert crit[0].severity == "critical"
    assert diagnose(_healthy_64gb(), None, swap_used_gb=0.5) == []


# --- config-level findings -------------------------------------------------


def test_dense_model_where_moe_fits_is_flagged() -> None:
    config = _config("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")  # dense, fits on 64GB
    causes = _causes(_healthy_64gb(), config, swap_used_gb=0.0)
    assert "dense-where-moe-fits" in causes
    assert "over-budget" not in causes


def test_over_budget_model_is_critical() -> None:
    small = HardwareProfile(unified_memory_gb=16.0, memory_bandwidth_gbps=120.0)
    config = _config("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")  # won't fit in ~12GB
    findings = diagnose(small, config, swap_used_gb=0.0)
    over = [f for f in findings if f.cause == "over-budget"]
    assert over and over[0].severity == "critical"


def test_non_mlx_runtime_is_flagged() -> None:
    config = _config("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="llama.cpp")
    assert "not-on-mlx" in _causes(_healthy_64gb(), config, swap_used_gb=0.0)


def test_unknown_model_is_info() -> None:
    config = _config("someorg/mystery-model")
    causes = _causes(_healthy_64gb(), config, swap_used_gb=0.0)
    assert "unknown-model" in causes


# --- CLI wiring ------------------------------------------------------------


def test_cli_doctor_clean_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(profiler, "detect", _healthy_64gb)
    monkeypatch.setattr(doctor_mod, "probe_swap_gb", lambda: 0.0)
    monkeypatch.chdir(tmp_path)  # no .keel/config.json
    assert main(["doctor"]) == 0
    assert "no configuration" in capsys.readouterr().out


def test_cli_doctor_critical_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(profiler, "detect", _healthy_64gb)
    monkeypatch.setattr(doctor_mod, "probe_swap_gb", lambda: 6.0)
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 1
    assert "CRITICAL" in capsys.readouterr().out
