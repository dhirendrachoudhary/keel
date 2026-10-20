from __future__ import annotations

import json
from pathlib import Path

import pytest

import keel.serving.profiler as profiler
from keel.cli import build_parser, main
from keel.serving.profiler import HardwareProfile

_M4_MAX = HardwareProfile(
    chip="Apple M4 Max",
    is_apple_silicon=True,
    unified_memory_gb=64.0,
    memory_bandwidth_gbps=546.0,
    free_disk_gb=500.0,
)


def test_no_args_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: keel" in capsys.readouterr().out


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "keel" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["doctor", "ablate"])
def test_unimplemented_subcommands_are_known_stubs(
    cmd: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([cmd]) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_setup_reports_hardware_and_picks_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(profiler, "detect", lambda: _M4_MAX)
    monkeypatch.chdir(tmp_path)
    assert main(["setup", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "Detected hardware" in out
    assert "Apple M4 Max" in out
    assert "~48" in out  # usable_model_memory_gb = 64 * 0.75
    assert "Qwen3-30B-A3B" in out  # the MoE is picked for a 64GB host
    assert not (tmp_path / ".keel" / "config.json").exists()  # --no-write


def test_setup_writes_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(profiler, "detect", lambda: _M4_MAX)
    monkeypatch.chdir(tmp_path)
    assert main(["setup"]) == 0
    config = tmp_path / ".keel" / "config.json"
    assert config.exists()
    data = json.loads(config.read_text())
    assert "Qwen3-30B-A3B" in data["serving"]["model_id"]
    assert data["serving"]["quantization"] in {"int4", "int8"}
    assert data["serving"]["backend"] == "mlx-lm"


def test_parser_prog_name() -> None:
    assert build_parser().prog == "keel"
