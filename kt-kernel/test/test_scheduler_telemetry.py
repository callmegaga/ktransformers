"""Tests for optional CPU/iGPU scheduler JSONL telemetry."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "python" / "scheduler_telemetry.py"
SPEC = importlib.util.spec_from_file_location("scheduler_telemetry", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
telemetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = telemetry
SPEC.loader.exec_module(telemetry)


class FakeScheduler:
    def __init__(self):
        self.execution_calls = 1
        self.execution_ratio_units = 1_000_000

    def scheduler_igpu_ratio(self):
        return 0.0

    def scheduler_cpu_load(self):
        return 0.75

    def scheduler_debug(self, decode):
        assert decode is True
        return [1.0, 0.5, 0.25, 3.0, 10.0, 2.0, 1.0, 1.0, 0.35, 0.75, 0.90, 1.0]

    def scheduler_execution_debug(self, decode):
        assert decode is True
        return [self.execution_calls, self.execution_ratio_units]


def test_writer_records_named_scheduler_fields(monkeypatch, tmp_path):
    output = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_FILE_ENV, str(output))
    monkeypatch.setenv(telemetry.TELEMETRY_LAYER_ENV, "3")
    monkeypatch.setenv("KT_CPU_IGPU_POLICY", "dynamic")
    monkeypatch.setenv("KT_CPU_IGPU_RATIO", "0")

    writer = telemetry.SchedulerTelemetryWriter.from_environment(3, "CPU_IGPU_GPTQ_INT4")
    assert writer is not None
    writer.record(FakeScheduler(), qlen=1)
    writer.close()

    event = json.loads(output.read_text(encoding="utf-8"))
    assert event["sequence"] == 0
    assert event["layer"] == 3
    assert event["phase"] == "decode"
    assert event["igpu_ratio"] == 1.0
    assert event["igpu_ratio_snapshot"] == 0.0
    assert event["execution_calls_delta"] == 1
    assert event["policy_igpu_ratio"] == 1.0
    assert event["cpu_load"] == 0.75
    assert event["cpu_sample_load"] == pytest.approx(0.35)
    assert event["igpu_sample_load"] == pytest.approx(0.75)
    assert event["igpu_reference_load"] == pytest.approx(0.90)
    assert event["reprobe_reason"] == 1
    assert event["cpu_samples"] == 3
    assert event["igpu_samples"] == 10
    assert event["switch_count"] == 2
    assert event["high_load_epoch"] is True
    assert event["exploration"] is True


def test_writer_marks_ratio_missing_without_completed_execution(monkeypatch, tmp_path):
    output = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_FILE_ENV, str(output))
    scheduler = FakeScheduler()
    scheduler.execution_calls = 0
    scheduler.execution_ratio_units = 0

    writer = telemetry.SchedulerTelemetryWriter.from_environment(0, "CPU_IGPU_GPTQ_INT4")
    assert writer is not None
    writer.record(scheduler, qlen=1)
    writer.close()

    event = json.loads(output.read_text(encoding="utf-8"))
    assert event["execution_calls_delta"] == 0
    assert event["igpu_ratio"] is None
    assert event["igpu_ratio_snapshot"] == 0.0


def test_writer_is_disabled_for_other_layers_and_backends(monkeypatch, tmp_path):
    monkeypatch.setenv(telemetry.TELEMETRY_FILE_ENV, str(tmp_path / "scheduler.jsonl"))
    monkeypatch.setenv(telemetry.TELEMETRY_LAYER_ENV, "0")

    assert telemetry.SchedulerTelemetryWriter.from_environment(1, "CPU_IGPU_GPTQ_INT4") is None
    assert telemetry.SchedulerTelemetryWriter.from_environment(0, "GPTQ_INT4") is None


def test_all_layer_selector_records_each_layer(monkeypatch, tmp_path):
    output = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_FILE_ENV, str(output))
    monkeypatch.setenv(telemetry.TELEMETRY_LAYER_ENV, "all")

    writers = [telemetry.SchedulerTelemetryWriter.from_environment(layer, "CPU_IGPU_GPTQ_INT4") for layer in (0, 39)]
    for writer in writers:
        assert writer is not None
        writer.record(FakeScheduler(), qlen=1)
        writer.close()

    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [event["layer"] for event in events] == [0, 39]
    assert [event["sequence"] for event in events] == [0, 0]


def test_invalid_layer_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(telemetry.TELEMETRY_FILE_ENV, str(tmp_path / "scheduler.jsonl"))
    monkeypatch.setenv(telemetry.TELEMETRY_LAYER_ENV, "invalid")

    with pytest.raises(ValueError, match="non-negative integer or 'all'"):
        telemetry.SchedulerTelemetryWriter.from_environment(0, "CPU_IGPU_GPTQ_INT4")
