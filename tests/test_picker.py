from __future__ import annotations

from keel.serving.catalog import ModelSpec, QuantOption
from keel.serving.picker import (
    estimate_decode_tok_s,
    kv_cache_gb,
    pick_model,
    weights_gb,
)
from keel.serving.profiler import HardwareProfile

_INT4 = QuantOption("int4", 4.5, "test/moe-4bit")


def _moe(key: str = "test-moe", total: float = 30.0, active: float = 3.0) -> ModelSpec:
    return ModelSpec(
        key=key,
        total_params_b=total,
        active_params_b=active,
        is_moe=True,
        context_length=32768,
        n_layers=48,
        n_kv_heads=4,
        head_dim=128,
        quants=(_INT4,),
    )


def _dense(key: str, total: float) -> ModelSpec:
    return ModelSpec(
        key=key,
        total_params_b=total,
        active_params_b=total,
        is_moe=False,
        context_length=32768,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        quants=(QuantOption("int4", 4.5, f"test/{key}-4bit"),),
    )


def test_memory_and_speed_math() -> None:
    spec = _moe(total=30.0, active=3.0)
    # weights = 30B * 4.5 bits / 8 = 16.875 GB
    assert weights_gb(spec, _INT4) == 16.875
    # KV for one 1000-token context, one agent: 2*48*4*128*2 bytes/token * 1000 / 1e9
    assert round(kv_cache_gb(spec, 1000, 1), 4) == round(2 * 48 * 4 * 128 * 2 * 1000 / 1e9, 4)
    # decode speed ~ eff * bw / active_weight_gb; active = 3B*4.5/8 = 1.6875 GB
    assert estimate_decode_tok_s(spec, _INT4, 546.0) == round(0.3 * 546.0 / 1.6875, 1)
    assert estimate_decode_tok_s(spec, _INT4, None) == 0.0


def test_picks_moe_on_a_64gb_host() -> None:
    profile = HardwareProfile(unified_memory_gb=64.0, memory_bandwidth_gbps=546.0)
    pick = pick_model(profile)
    assert pick is not None
    assert "Qwen3-30B-A3B" in pick.model_id
    assert pick.backend == "mlx-lm"
    assert pick.est_decode_tok_s > 20.0
    assert pick.memory.total_gb <= profile.usable_model_memory_gb


def test_small_host_picks_a_small_model_that_fits() -> None:
    profile = HardwareProfile(unified_memory_gb=8.0, memory_bandwidth_gbps=120.0)
    pick = pick_model(profile)
    assert pick is not None
    assert pick.memory.total_gb <= profile.usable_model_memory_gb  # 6 GB
    assert "0.5B" in pick.model_id


def test_returns_none_when_nothing_fits() -> None:
    profile = HardwareProfile(unified_memory_gb=0.2, memory_bandwidth_gbps=68.0)
    assert pick_model(profile) is None


def test_moe_is_preferred_over_a_fitting_dense() -> None:
    catalog = (_dense("big-dense", total=20.0), _moe("small-moe", total=12.0, active=1.0))
    profile = HardwareProfile(unified_memory_gb=64.0, memory_bandwidth_gbps=546.0)
    pick = pick_model(profile, catalog=catalog)
    assert pick is not None
    assert pick.spec_key == "small-moe"


def test_unknown_bandwidth_still_picks_by_capacity() -> None:
    profile = HardwareProfile(unified_memory_gb=64.0, memory_bandwidth_gbps=None)
    pick = pick_model(profile)
    assert pick is not None
    assert pick.est_decode_tok_s == 0.0
    assert "speed n/a" in pick.rationale


def test_falls_back_to_fastest_when_none_clear_the_speed_bar() -> None:
    # A very low bandwidth means nothing meets MIN_DECODE_TOK_S; the fastest (smallest
    # active) fitting model should win.
    profile = HardwareProfile(unified_memory_gb=64.0, memory_bandwidth_gbps=5.0)
    pick = pick_model(profile)
    assert pick is not None
    assert "0.5B" in pick.model_id
    assert "fastest that fits" in pick.rationale
