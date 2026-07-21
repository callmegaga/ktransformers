"""Tests for per-layer scheduler telemetry reports."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "report_scheduler_layers.py"
SPEC = importlib.util.spec_from_file_location("report_scheduler_layers", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def event(layer: int, phase: str, sequence: int) -> dict[str, object]:
    decode = phase == "decode"
    return {
        "layer": layer,
        "phase": phase,
        "sequence": sequence,
        "monotonic_ns": sequence * 100,
        "request_index": 0,
        "execution_calls_delta": 1,
        "igpu_ratio": 1.0 if decode else 0.0,
        "policy_igpu_ratio": 1.0 if decode else 0.0,
        "cpu_load": 0.8,
        "exploration": False,
        "reprobe_reason": 0,
        "switch_count": 1,
        "cpu_ms_per_row": 1.0,
        "igpu_ms_per_row": 0.1,
    }


def test_reports_each_scheduler_layer(tmp_path):
    artifact = tmp_path / "artifact"
    output = tmp_path / "report"
    artifact.mkdir()
    events = [
        event(layer, phase, sequence)
        for layer in (0, 1)
        for sequence, phase in ((1, "prefill"), (2, "decode"), (3, "decode"))
    ]
    (artifact / "scheduler-telemetry.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )

    assert reporter.main([str(artifact), "--output-dir", str(output)]) == 0

    with (output / "layers.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[0]["decode_calls"] == "2"
    assert float(rows[0]["decode_igpu_ratio"]) == 1.0
    assert rows[0]["exploration_calls"] == "0"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Decode fully iGPU layers: 2 / 2" in report
    assert "Prefill fully CPU layers: 2 / 2" in report


def test_rejects_events_without_layer_identifiers(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "scheduler-telemetry.jsonl").write_text(json.dumps({"phase": "decode"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing layer identifiers"):
        reporter.load_events(artifact)


def test_reports_missing_prefill_execution_sample(tmp_path):
    artifact = tmp_path / "artifact"
    output = tmp_path / "report"
    artifact.mkdir()
    prefill = event(0, "prefill", 1)
    prefill["execution_calls_delta"] = 0
    prefill["igpu_ratio"] = None
    prefill["policy_igpu_ratio"] = 0.0
    decode = event(0, "decode", 2)
    decode["igpu_ratio"] = 0.0
    decode["policy_igpu_ratio"] = 0.0
    (artifact / "scheduler-telemetry.jsonl").write_text(
        json.dumps(prefill) + "\n" + json.dumps(decode) + "\n",
        encoding="utf-8",
    )

    assert reporter.main([str(artifact), "--output-dir", str(output)]) == 0

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Decode fully CPU layers: 1 / 1" in report
    assert "Prefill fully CPU layers: 0 / 0 observed (coverage 0 / 1)" in report
    assert "Prefill CPU-policy layers: 1 / 1" in report
    assert "| 0 | n/a |" in report
