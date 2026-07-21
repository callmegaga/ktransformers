"""Tests for dynamic-load transition artifact aggregation."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bench" / "report_load_transition_cycles.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("report_load_transition_cycles", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def make_cycle(path: Path, decode_tps: float, delay_ms: float, direction: str = "high-to-low") -> None:
    path.mkdir()
    sample = {
        "status": "ok",
        "prefill_tps": 100.0,
        "decode_tps": decode_tps,
        "ttft_ms": 1000.0,
        "tpot_ms": 50.0,
        "e2e_ms": 2000.0,
        "completion_tokens": 10,
        "background_stopped": True,
    }
    if direction == "high-to-low":
        sample.update(
            {
                "background_stop_signal_ns": 350_000_000,
                "transition_first_cpu_execution_delay_calls": 1,
                "transition_first_cpu_execution_delay_ms": delay_ms,
                "transition_settled_cpu_delay_calls": 3,
                "transition_settled_cpu_delay_ms": delay_ms + 100.0,
                "transition_settled_cpu_sequence": 7,
            }
        )
    else:
        sample.update(
            {
                "stream_tokens": 10,
                "stream_chunks": 9,
                "transition_client_pre_tps": 30.0,
                "transition_client_post_tps": 15.0,
                "background_start_launch_ns": 350_000_000,
                "background_ready_ns": 375_000_000,
                "background_ready_delay_ms": 25.0,
                "transition_first_igpu_execution_delay_calls": 1,
                "transition_first_igpu_execution_delay_ms": delay_ms,
                "transition_first_igpu_execution_ready_delay_ms": delay_ms - 25.0,
                "transition_first_igpu_execution_sequence": 5,
                "transition_settled_igpu_delay_calls": 3,
                "transition_settled_igpu_delay_ms": delay_ms + 100.0,
                "transition_settled_igpu_ready_delay_ms": delay_ms + 75.0,
                "transition_settled_igpu_sequence": 7,
            }
        )
    (path / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    events = []
    for sequence in range(1, 11):
        ratio = 1.0 if sequence <= 4 else 0.0
        if direction == "low-to-high":
            ratio = 1.0 - ratio
        events.append(
            {
                "phase": "decode",
                "sequence": sequence,
                "monotonic_ns": sequence * 100_000_000,
                "execution_calls_delta": 1,
                "igpu_ratio": ratio,
                "exploration": 5 <= sequence <= 6,
                "switch_count": 1 if sequence <= 4 else 2,
            }
        )
    (path / "scheduler-telemetry.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def test_aggregates_transition_cycles(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "aggregate"
    make_cycle(first, 20.0, 50.0)
    make_cycle(second, 22.0, 70.0)

    assert (
        reporter.main(
            [
                str(first),
                str(second),
                "--output-dir",
                str(output),
                "--bootstrap-samples",
                "100",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    with (output / "cycles.csv").open(encoding="utf-8") as source:
        cycles = list(csv.DictReader(source))
    with (output / "summary.csv").open(encoding="utf-8") as source:
        summary = {row["metric"]: row for row in csv.DictReader(source)}

    assert len(cycles) == 2
    assert cycles[0]["exploration_calls"] == "2"
    assert cycles[0]["switch_delta"] == "1"
    assert float(summary["decode_tps"]["mean"]) == 21.0
    assert (output / "report.md").is_file()
    assert (output / "manifest.json").is_file()


def test_aggregates_low_to_high_cycles(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "aggregate"
    make_cycle(first, 18.0, 50.0, direction="low-to-high")
    make_cycle(second, 20.0, 70.0, direction="low-to-high")

    assert reporter.main([str(first), str(second), "--output-dir", str(output), "--bootstrap-samples", "100"]) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    with (output / "summary.csv").open(encoding="utf-8") as source:
        summary = {row["metric"]: row for row in csv.DictReader(source)}

    assert manifest["direction"] == "low-to-high"
    assert float(summary["background_ready_ms"]["mean"]) == 25.0
    assert float(summary["ready_to_first_target_calls"]["mean"]) == 1.0
    assert float(summary["ready_to_first_target_ms"]["mean"]) == 35.0
    assert float(summary["client_pre_transition_tps"]["mean"]) == 30.0
    assert float(summary["client_post_transition_tps"]["mean"]) == 15.0
    assert manifest["exact_token_timestamp_cycles"] == 2


def test_rejects_mismatched_stream_token_timestamps(tmp_path):
    cycle = tmp_path / "cycle"
    make_cycle(cycle, 18.0, 50.0, direction="low-to-high")
    sample_path = cycle / "samples.jsonl"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["stream_tokens"] = 9
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stream token timestamps do not match"):
        reporter.load_cycle(cycle, 1)
