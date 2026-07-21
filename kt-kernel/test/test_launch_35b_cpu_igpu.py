"""Tests for the manual 35B server launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "perf-log" / "35b-test-cpu-igpu.sh"


def run_dry(
    backend: str, priority: str = "normal", telemetry_layer: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DRY_RUN"] = "1"
    environment["PORT"] = "31234"
    environment["ENGINE_PRIORITY"] = priority
    environment.pop("ENGINE_NICE", None)
    environment.pop("ENGINE_CPU_WEIGHT", None)
    if telemetry_layer is None:
        environment.pop("SCHEDULER_TELEMETRY_FILE", None)
        environment.pop("SCHEDULER_TELEMETRY_LAYER", None)
    else:
        environment["SCHEDULER_TELEMETRY_FILE"] = "/tmp/scheduler-telemetry.jsonl"
        environment["SCHEDULER_TELEMETRY_LAYER"] = telemetry_layer
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), backend],
        capture_output=True,
        text=True,
        env=environment,
        timeout=10.0,
        check=False,
    )


def test_dynamic_launch_command():
    result = run_dry("dynamic")

    assert result.returncode == 0, result.stderr
    assert "Backend: dynamic (kt-method=CPU_IGPU_GPTQ_INT4)" in result.stdout
    assert "ONEAPI_DEVICE_SELECTOR=level_zero:gpu" in result.stdout
    assert "--port 31234" in result.stdout
    assert "--disable-radix-cache" in result.stdout
    assert "--kt-method CPU_IGPU_GPTQ_INT4" in result.stdout
    assert "Scheduler: decode_policy=service-cost prefill_load=0.99/1.0" in result.stdout
    assert (
        "cost_ewma=0.20 margin=0.10 cost_load_delta=0.10 calibration=32 reprobe_samples=32 "
        "reprobe_interval=4096 load_delta=0.25 load_probe_max=0.20 load_grace=64"
    ) in result.stdout


def test_vnni_launch_command():
    result = run_dry("vnni-only")

    assert result.returncode == 0, result.stderr
    assert "Backend: vnni-only (kt-method=GPTQ_INT4)" in result.stdout
    assert "KT_GPTQ_INT4_BACKEND=avxvnni fused=1" in result.stdout
    assert "--kt-method GPTQ_INT4" in result.stdout


def test_packed_cpu_fixed_launch_command():
    result = run_dry("packed-cpu-fixed")

    assert result.returncode == 0, result.stderr
    assert "Backend: packed-cpu-fixed (kt-method=CPU_IGPU_GPTQ_INT4)" in result.stdout
    assert "Scheduler: policy=fixed igpu_ratio=0" in result.stdout
    assert "ONEAPI_DEVICE_SELECTOR=level_zero:gpu" in result.stdout
    assert "--kt-method CPU_IGPU_GPTQ_INT4" in result.stdout


def test_phase_fixed_launch_command():
    result = run_dry("phase-fixed")

    assert result.returncode == 0, result.stderr
    assert "Backend: phase-fixed (kt-method=CPU_IGPU_GPTQ_INT4)" in result.stdout
    assert "Scheduler: policy=phase-fixed igpu_ratio=0" in result.stdout
    assert "Scheduler: prefill_ratio=0 decode_ratio=1" in result.stdout
    assert "--kt-method CPU_IGPU_GPTQ_INT4" in result.stdout


def test_igpu_fixed_launch_command():
    result = run_dry("igpu-fixed")

    assert result.returncode == 0, result.stderr
    assert "Backend: igpu-fixed (kt-method=CPU_IGPU_GPTQ_INT4)" in result.stdout
    assert "Scheduler: policy=fixed igpu_ratio=1" in result.stdout
    assert "ONEAPI_DEVICE_SELECTOR=level_zero:gpu" in result.stdout
    assert "--kt-method CPU_IGPU_GPTQ_INT4" in result.stdout


def test_low_priority_engine_uses_a_weighted_systemd_scope():
    result = run_dry("dynamic", priority="low")

    assert result.returncode == 0, result.stderr
    assert "Engine priority: profile=low nice=5 CPUWeight=33" in result.stdout
    assert "systemd-run --user --scope --quiet -p CPUWeight=33 nice -n 5" in result.stdout
    assert "--kt-method CPU_IGPU_GPTQ_INT4" in result.stdout


def test_all_layer_scheduler_telemetry_is_accepted():
    result = run_dry("dynamic", telemetry_layer="all")

    assert result.returncode == 0, result.stderr
    assert "Scheduler telemetry: file=/tmp/scheduler-telemetry.jsonl layer=all" in result.stdout


def test_unknown_backend_is_rejected():
    result = run_dry("avx2")

    assert result.returncode != 0
    assert "expected dynamic, phase-fixed, packed-cpu-fixed, igpu-fixed, or vnni-only" in result.stderr
