"""Tests for the running-server steady CPU-load sweep."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "bench_running_server_load_sweep.py"
SPEC = importlib.util.spec_from_file_location("bench_running_server_load_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def test_parse_worker_counts_and_deterministic_schedule():
    assert sweep.parse_worker_counts("0,4,8,12") == [0, 4, 8, 12]
    first = sweep.make_load_schedule([0, 4, 8, 12], "shuffled", 42)
    second = sweep.make_load_schedule([0, 4, 8, 12], "shuffled", 42)
    assert first == second
    assert sorted(first) == [0, 4, 8, 12]
    assert sweep.make_load_schedule([0, 4], "listed", 42) == [0, 4]


@pytest.mark.parametrize("value", ("", "-1,4", "4,4", "four"))
def test_invalid_worker_counts_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_worker_counts(value)


def test_validate_dynamic_server_and_reject_telemetry():
    metadata = {
        "nice": 5,
        "command": ["python", "-m", "sglang.launch_server", "--kt-method", "CPU_IGPU_GPTQ_INT4"],
        "environment": {
            "KT_CPU_IGPU_POLICY": "dynamic",
            "KT_CPU_IGPU_RATIO": "0",
            "KT_CPU_IGPU_PREFILL_RATIO": None,
            "KT_CPU_IGPU_DECODE_RATIO": None,
            "SCHEDULER_TELEMETRY_FILE": "",
        },
    }
    sweep.validate_server(metadata, "dynamic", 5)

    metadata["environment"]["SCHEDULER_TELEMETRY_FILE"] = "events.jsonl"
    with pytest.raises(RuntimeError, match="must be empty"):
        sweep.validate_server(metadata, "dynamic", 5)


def test_validate_server_rejects_wrong_backend_and_priority():
    metadata = {
        "nice": 0,
        "command": ["python", "--kt-method", "CPU_IGPU_GPTQ_INT4"],
        "environment": {
            "KT_CPU_IGPU_POLICY": "fixed",
            "KT_CPU_IGPU_RATIO": "1",
            "KT_CPU_IGPU_PREFILL_RATIO": None,
            "KT_CPU_IGPU_DECODE_RATIO": None,
            "SCHEDULER_TELEMETRY_FILE": "",
        },
    }
    with pytest.raises(RuntimeError, match="server preflight failed") as error:
        sweep.validate_server(metadata, "packed-cpu-fixed", 5)
    assert "KT_CPU_IGPU_RATIO" in str(error.value)
    assert "server nice expected 5" in str(error.value)


def test_validate_server_identity_rejects_restarted_server(monkeypatch):
    monkeypatch.setattr(sweep, "resolve_server_pid", lambda *_args: 22)
    with pytest.raises(RuntimeError, match="server PID changed"):
        sweep.validate_server_identity("http://127.0.0.1:30100", 11, "dynamic", 5)


def test_background_load_command_uses_free_affinity_and_nice():
    args = SimpleNamespace(
        load_affinity="free",
        load_cpus=None,
        load_nice=0,
        background_ready_timeout=1.0,
    )
    controller = sweep.ManagedBackgroundLoad(args, 8)
    assert controller.command[controller.command.index("--workers") + 1] == "8"
    assert controller.command[controller.command.index("--affinity") + 1] == "free"
    assert controller.command[controller.command.index("--nice") + 1] == "0"
    assert "--cpus" not in controller.command


def test_background_load_startup_failure_cleans_up(monkeypatch):
    class FakeProcess:
        pid = 123
        stdout = io.StringIO("not-json\n")
        stderr = io.StringIO()

        def poll(self):
            return None

    fake_process = FakeProcess()
    stopped = []
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(sweep.select, "select", lambda *args, **kwargs: ([fake_process.stdout], [], []))
    monkeypatch.setattr(
        sweep,
        "terminate_process_group",
        lambda process, timeout: stopped.append((process, timeout)) or True,
    )
    args = SimpleNamespace(
        load_affinity="free",
        load_cpus=None,
        load_nice=0,
        background_ready_timeout=1.0,
    )

    with pytest.raises(json.JSONDecodeError):
        with sweep.ManagedBackgroundLoad(args, 1):
            pass

    assert stopped == [(fake_process, 10.0)]


def test_benchmark_command_never_enables_scheduler_telemetry(tmp_path):
    args = SimpleNamespace(
        base_url="http://127.0.0.1:30100",
        workloads="1024:600",
        warmups=3,
        warmup_prompt_tokens=256,
        warmup_output_tokens=128,
        repetitions=5,
        seed=7,
        bootstrap_samples=5000,
        request_timeout=900.0,
        server_wait_timeout=30.0,
    )
    command = sweep.benchmark_command(args, 8, "dynamic__compute8__b1", tmp_path)
    assert "--scheduler-telemetry-file" not in command
    assert command[command.index("--warmup-output-tokens") + 1] == "128"
    assert command[command.index("--output-dir") + 1] == str(tmp_path)


def test_dry_run_does_not_require_server_or_create_output(tmp_path, capsys):
    output = tmp_path / "sweep"
    result = sweep.main(
        [
            "--backend",
            "dynamic",
            "--block-label",
            "pilot-b1",
            "--load-workers",
            "0,8",
            "--load-order",
            "listed",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    assert result == 0
    assert not output.exists()
    stdout = capsys.readouterr().out
    assert "Realized order: [0, 8]" in stdout
    assert "--scheduler-telemetry-file" not in stdout
